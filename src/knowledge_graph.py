"""
Модуль работы с Knowledge Graph на базе Neo4j
Интеграция с LangChain для KAG (Knowledge Augmented Generation)
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Neo4j
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship, NodeMatcher

# NLP для извлечения сущностей
import spacy
from transformers import pipeline

# LangChain
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphQAChain
from langchain.llms import HuggingFacePipeline

# Local imports
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeEntity:
    """Класс для представления сущности в knowledge graph"""
    name: str
    type: str
    properties: Dict[str, Any]
    source_document: str
    confidence: float = 0.8

@dataclass
class KnowledgeRelation:
    """Класс для представления отношения в knowledge graph"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]
    source_document: str
    confidence: float = 0.8

class KnowledgeGraphManager:
    """Основной класс управления Knowledge Graph"""
    
    def __init__(self):
        self.neo4j_graph = None
        self.py2neo_graph = None
        self.nlp_model = None
        self.ner_pipeline = None
        self.relation_extractor = None
        
        # Инициализация
        self._init_neo4j()
        self._init_nlp_models()
        self._setup_constraints()
    
    def _init_neo4j(self):
        """Инициализация подключения к Neo4j"""
        try:
            # LangChain Neo4j Graph
            self.neo4j_graph = Neo4jGraph(
                url=config.knowledge_graph.uri,
                username=config.knowledge_graph.user,
                password=config.knowledge_graph.password,
                database=config.knowledge_graph.database
            )
            
            # Py2neo Graph
            self.py2neo_graph = Graph(
                config.knowledge_graph.uri,
                auth=(config.knowledge_graph.user, config.knowledge_graph.password),
                name=config.knowledge_graph.database
            )
            
            logger.info("Neo4j connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            raise
    
    def _init_nlp_models(self):
        """Инициализация NLP моделей для извлечения сущностей"""
        try:
            # Загрузка spaCy модели для украинского и русского языков
            try:
                self.nlp_model = spacy.load("xx_ent_wiki_sm")  # Multilingual model
            except OSError:
                logger.warning("spaCy multilingual model not found, using basic processing")
                self.nlp_model = None
            
            # Transformers для NER
            self.ner_pipeline = pipeline(
                "ner",
                model="Jean-Baptiste/roberta-large-ner-english",
                aggregation_strategy="simple"
            )
            
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            self.nlp_model = None
            self.ner_pipeline = None
    
    def _setup_constraints(self):
        """Настройка ограничений и индексов в Neo4j"""
        try:
            constraints = [
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.type)"
            ]
            
            for constraint in constraints:
                try:
                    self.neo4j_graph.query(constraint)
                except Exception as e:
                    logger.warning(f"Constraint already exists or failed: {e}")
            
            logger.info("Neo4j constraints and indexes set up")
            
        except Exception as e:
            logger.error(f"Error setting up constraints: {e}")
    
    def extract_entities_from_text(self, text: str, document_type: str = "general") -> List[KnowledgeEntity]:
        """Извлечение сущностей из текста"""
        entities = []
        
        try:
            if self.nlp_model:
                # Используем spaCy
                doc = self.nlp_model(text)
                for ent in doc.ents:
                    entity = KnowledgeEntity(
                        name=ent.text.strip(),
                        type=ent.label_,
                        properties={
                            "spacy_label": ent.label_,
                            "start_char": ent.start_char,
                            "end_char": ent.end_char,
                            "document_type": document_type
                        },
                        source_document="",
                        confidence=0.8
                    )
                    entities.append(entity)
            
            if self.ner_pipeline:
                # Используем transformers NER
                ner_results = self.ner_pipeline(text)
                for result in ner_results:
                    entity = KnowledgeEntity(
                        name=result['word'].strip(),
                        type=result['entity_group'],
                        properties={
                            "score": result['score'],
                            "start": result['start'],
                            "end": result['end'],
                            "document_type": document_type
                        },
                        source_document="",
                        confidence=result['score']
                    )
                    entities.append(entity)
            
            # Фильтрация дубликатов
            unique_entities = {}
            for entity in entities:
                key = f"{entity.name.lower()}_{entity.type}"
                if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                    unique_entities[key] = entity
            
            return list(unique_entities.values())
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_relations_from_text(self, text: str, entities: List[KnowledgeEntity]) -> List[KnowledgeRelation]:
        """Извлечение отношений между сущностями"""
        relations = []
        
        try:
            # Простые правила извлечения отношений
            entity_names = [e.name.lower() for e in entities]
            text_lower = text.lower()
            
            # Поиск паттернов отношений
            relation_patterns = {
                "IS_A": ["є", "являється", "is a", "is an"],
                "PART_OF": ["частина", "частиною", "part of", "belongs to"],
                "LOCATED_IN": ["знаходиться в", "розташований в", "located in", "in"],
                "WORKS_FOR": ["працює в", "працює для", "works for", "works at"],
                "RELATED_TO": ["пов'язаний з", "пов'язана з", "related to", "associated with"]
            }
            
            for relation_type, patterns in relation_patterns.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        # Ищем сущности вокруг паттерна
                        for i, source_entity in enumerate(entities):
                            for j, target_entity in enumerate(entities):
                                if i != j:
                                    relation = KnowledgeRelation(
                                        source=source_entity.name,
                                        target=target_entity.name,
                                        relation_type=relation_type,
                                        properties={
                                            "pattern": pattern,
                                            "extracted_by": "rule_based",
                                            "document_type": "general"
                                        },
                                        source_document="",
                                        confidence=0.6
                                    )
                                    relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []
    
    def create_document_node(self, document_path: str, document_type: str, metadata: Dict[str, Any]) -> str:
        """Создание узла документа"""
        try:
            document_id = os.path.basename(document_path)
            
            cypher_query = """
            MERGE (d:Document {id: $id})
            SET d.path = $path, d.type = $type, d.metadata = $metadata, d.created_at = $created_at
            RETURN d
            """
            
            result = self.neo4j_graph.query(
                cypher_query,
                {
                    "id": document_id,
                    "path": document_path,
                    "type": document_type,
                    "metadata": json.dumps(metadata),
                    "created_at": datetime.now().isoformat()
                }
            )
            
            return document_id
            
        except Exception as e:
            logger.error(f"Error creating document node: {e}")
            return ""
    
    def create_entity_node(self, entity: KnowledgeEntity) -> bool:
        """Создание узла сущности"""
        try:
            cypher_query = """
            MERGE (e:Entity {name: $name, type: $type})
            SET e.properties = $properties, e.confidence = $confidence, e.created_at = $created_at
            RETURN e
            """
            
            self.neo4j_graph.query(
                cypher_query,
                {
                    "name": entity.name,
                    "type": entity.type,
                    "properties": json.dumps(entity.properties),
                    "confidence": entity.confidence,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating entity node: {e}")
            return False
    
    def create_relation(self, relation: KnowledgeRelation) -> bool:
        """Создание отношения между сущностями"""
        try:
            cypher_query = """
            MATCH (source:Entity {name: $source})
            MATCH (target:Entity {name: $target})
            MERGE (source)-[r:RELATED_TO {type: $relation_type}]->(target)
            SET r.properties = $properties, r.confidence = $confidence, r.created_at = $created_at
            RETURN r
            """
            
            self.neo4j_graph.query(
                cypher_query,
                {
                    "source": relation.source,
                    "target": relation.target,
                    "relation_type": relation.relation_type,
                    "properties": json.dumps(relation.properties),
                    "confidence": relation.confidence,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating relation: {e}")
            return False
    
    def link_entities_to_document(self, document_id: str, entities: List[KnowledgeEntity]) -> bool:
        """Связывание сущностей с документом"""
        try:
            for entity in entities:
                cypher_query = """
                MATCH (d:Document {id: $document_id})
                MATCH (e:Entity {name: $entity_name, type: $entity_type})
                MERGE (d)-[r:CONTAINS]->(e)
                SET r.confidence = $confidence, r.created_at = $created_at
                """
                
                self.neo4j_graph.query(
                    cypher_query,
                    {
                        "document_id": document_id,
                        "entity_name": entity.name,
                        "entity_type": entity.type,
                        "confidence": entity.confidence,
                        "created_at": datetime.now().isoformat()
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error linking entities to document: {e}")
            return False
    
    def process_document_for_graph(self, document_path: str, content: str, document_type: str = "general") -> bool:
        """Обработка документа для добавления в knowledge graph"""
        try:
            logger.info(f"Processing document for graph: {document_path}")
            
            # Создание узла документа
            document_id = self.create_document_node(
                document_path, 
                document_type,
                {"content_length": len(content)}
            )
            
            if not document_id:
                return False
            
            # Извлечение сущностей
            entities = self.extract_entities_from_text(content, document_type)
            
            if not entities:
                logger.warning(f"No entities extracted from {document_path}")
                return True
            
            # Создание узлов сущностей
            for entity in entities:
                entity.source_document = document_id
                self.create_entity_node(entity)
            
            # Извлечение отношений
            relations = self.extract_relations_from_text(content, entities)
            
            # Создание отношений
            for relation in relations:
                relation.source_document = document_id
                self.create_relation(relation)
            
            # Связывание сущностей с документом
            self.link_entities_to_document(document_id, entities)
            
            logger.info(f"Added {len(entities)} entities and {len(relations)} relations to graph")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document for graph: {e}")
            return False
    
    def query_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        """Выполнение запроса к knowledge graph"""
        try:
            # Простой поиск сущностей
            cypher_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($query)
            OPTIONAL MATCH (e)-[r:RELATED_TO]-(related:Entity)
            RETURN e, r, related
            LIMIT 20
            """
            
            result = self.neo4j_graph.query(cypher_query, {"query": query})
            
            formatted_results = []
            for record in result:
                if record['e']:
                    entity_info = {
                        "entity": dict(record['e']),
                        "relation": dict(record['r']) if record['r'] else None,
                        "related_entity": dict(record['related']) if record['related'] else None
                    }
                    formatted_results.append(entity_info)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """Получение контекста из knowledge graph для запроса"""
        try:
            # Поиск релевантных сущностей и отношений
            results = self.query_knowledge_graph(query)
            
            context_parts = []
            current_length = 0
            
            for result in results:
                entity = result['entity']
                
                # Формирование контекстной информации
                context_part = f"Сутність: {entity.get('name', '')} (Тип: {entity.get('type', '')})"
                
                if result['relation'] and result['related_entity']:
                    relation = result['relation']
                    related = result['related_entity']
                    context_part += f"\
Пов'язано через: {relation.get('type', '')} з {related.get('name', '')}"
                
                # Проверка длины
                if current_length + len(context_part) > max_context_length:
                    break
                
                context_parts.append(context_part)
                current_length += len(context_part)
            
            return "\
\
".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context from graph: {e}")
            return ""
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Получение статистики knowledge graph"""
        try:
            stats = {}
            
            # Количество узлов
            node_count_query = "MATCH (n) RETURN count(n) as count"
            node_result = self.neo4j_graph.query(node_count_query)
            stats["total_nodes"] = node_result[0]["count"] if node_result else 0
            
            # Количество отношений
            rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
            rel_result = self.neo4j_graph.query(rel_count_query)
            stats["total_relationships"] = rel_result[0]["count"] if rel_result else 0
            
            # Типы сущностей
            entity_types_query = "MATCH (e:Entity) RETURN DISTINCT e.type as type, count(e) as count"
            entity_types_result = self.neo4j_graph.query(entity_types_query)
            stats["entity_types"] = {r["type"]: r["count"] for r in entity_types_result}
            
            # Типы отношений
            rel_types_query = "MATCH ()-[r:RELATED_TO]->() RETURN DISTINCT r.type as type, count(r) as count"
            rel_types_result = self.neo4j_graph.query(rel_types_query)
            stats["relationship_types"] = {r["type"]: r["count"] for r in rel_types_result}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def create_qa_chain(self, llm):
        """Создание QA цепочки для работы с knowledge graph"""
        try:
            qa_chain = GraphQAChain.from_llm(
                llm=llm,
                graph=self.neo4j_graph,
                verbose=True
            )
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            return None

# Функции для использования в других модулях
def create_knowledge_graph_manager() -> KnowledgeGraphManager:
    """Создание экземпляра менеджера knowledge graph"""
    return KnowledgeGraphManager()

def process_documents_for_graph(documents, content_getter):
    """Обработка документов для knowledge graph"""
    kg_manager = create_knowledge_graph_manager()
    
    processed_count = 0
    for doc in documents:
        content = content_getter(doc)
        if kg_manager.process_document_for_graph(doc.file_path, content, doc.document_type):
            processed_count += 1
    
    logger.info(f"Processed {processed_count} documents for knowledge graph")
    return processed_count