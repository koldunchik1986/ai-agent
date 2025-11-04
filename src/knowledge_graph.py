
"""
\u041c\u043e\u0434\u0443\u043b\u044c \u0440\u0430\u0431\u043e\u0442\u044b \u0441 Knowledge Graph \u043d\u0430 \u0431\u0430\u0437\u0435 Neo4j
\u0418\u043d\u0442\u0435\u0433\u0440\u0430\u0446\u0438\u044f \u0441 LangChain \u0434\u043b\u044f KAG (Knowledge Augmented Generation)
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

# NLP \u0434\u043b\u044f \u0438\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u044f \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439
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
    """\u041a\u043b\u0430\u0441\u0441 \u0434\u043b\u044f \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u044f \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0438 \u0432 knowledge graph"""
    name: str
    type: str
    properties: Dict[str, Any]
    source_document: str
    confidence: float = 0.8

@dataclass
class KnowledgeRelation:
    """\u041a\u043b\u0430\u0441\u0441 \u0434\u043b\u044f \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u044f \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u044f \u0432 knowledge graph"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]
    source_document: str
    confidence: float = 0.8

class KnowledgeGraphManager:
    """\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043a\u043b\u0430\u0441\u0441 \u0443\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0438\u044f Knowledge Graph"""
    
    def __init__(self):
        self.neo4j_graph = None
        self.py2neo_graph = None
        self.nlp_model = None
        self.ner_pipeline = None
        self.relation_extractor = None
        
        # \u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f
        self._init_neo4j()
        self._init_nlp_models()
        self._setup_constraints()
    
    def _init_neo4j(self):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u043f\u043e\u0434\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u044f \u043a Neo4j"""
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
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f NLP \u043c\u043e\u0434\u0435\u043b\u0435\u0439 \u0434\u043b\u044f \u0438\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u044f \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439"""
        try:
            # \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 spaCy \u043c\u043e\u0434\u0435\u043b\u0438 \u0434\u043b\u044f \u0443\u043a\u0440\u0430\u0438\u043d\u0441\u043a\u043e\u0433\u043e \u0438 \u0440\u0443\u0441\u0441\u043a\u043e\u0433\u043e \u044f\u0437\u044b\u043a\u043e\u0432
            try:
                self.nlp_model = spacy.load("xx_ent_wiki_sm")  # Multilingual model
            except OSError:
                logger.warning("spaCy multilingual model not found, using basic processing")
                self.nlp_model = None
            
            # Transformers \u0434\u043b\u044f NER
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
        """\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0430 \u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0435\u043d\u0438\u0439 \u0438 \u0438\u043d\u0434\u0435\u043a\u0441\u043e\u0432 \u0432 Neo4j"""
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
        """\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439 \u0438\u0437 \u0442\u0435\u043a\u0441\u0442\u0430"""
        entities = []
        
        try:
            if self.nlp_model:
                # \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u043c spaCy
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
                # \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u043c transformers NER
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
            
            # \u0424\u0438\u043b\u044c\u0442\u0440\u0430\u0446\u0438\u044f \u0434\u0443\u0431\u043b\u0438\u043a\u0430\u0442\u043e\u0432
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
        """\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0439 \u043c\u0435\u0436\u0434\u0443 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u044f\u043c\u0438"""
        relations = []
        
        try:
            # \u041f\u0440\u043e\u0441\u0442\u044b\u0435 \u043f\u0440\u0430\u0432\u0438\u043b\u0430 \u0438\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u044f \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0439
            entity_names = [e.name.lower() for e in entities]
            text_lower = text.lower()
            
            # \u041f\u043e\u0438\u0441\u043a \u043f\u0430\u0442\u0442\u0435\u0440\u043d\u043e\u0432 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0439
            relation_patterns = {
                "IS_A": ["\u0454", "\u044f\u0432\u043b\u044f\u0454\u0442\u044c\u0441\u044f", "is a", "is an"],
                "PART_OF": ["\u0447\u0430\u0441\u0442\u0438\u043d\u0430", "\u0447\u0430\u0441\u0442\u0438\u043d\u043e\u044e", "part of", "belongs to"],
                "LOCATED_IN": ["\u0437\u043d\u0430\u0445\u043e\u0434\u0438\u0442\u044c\u0441\u044f \u0432", "\u0440\u043e\u0437\u0442\u0430\u0448\u043e\u0432\u0430\u043d\u0438\u0439 \u0432", "located in", "in"],
                "WORKS_FOR": ["\u043f\u0440\u0430\u0446\u044e\u0454 \u0432", "\u043f\u0440\u0430\u0446\u044e\u0454 \u0434\u043b\u044f", "works for", "works at"],
                "RELATED_TO": ["\u043f\u043e\u0432'\u044f\u0437\u0430\u043d\u0438\u0439 \u0437", "\u043f\u043e\u0432'\u044f\u0437\u0430\u043d\u0430 \u0437", "related to", "associated with"]
            }
            
            for relation_type, patterns in relation_patterns.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        # \u0418\u0449\u0435\u043c \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0438 \u0432\u043e\u043a\u0440\u0443\u0433 \u043f\u0430\u0442\u0442\u0435\u0440\u043d\u0430
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
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0443\u0437\u043b\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430"""
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
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0443\u0437\u043b\u0430 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0438"""
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
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u044f \u043c\u0435\u0436\u0434\u0443 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u044f\u043c\u0438"""
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
        """\u0421\u0432\u044f\u0437\u044b\u0432\u0430\u043d\u0438\u0435 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439 \u0441 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u043c"""
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
        """\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430 \u0434\u043b\u044f \u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u044f \u0432 knowledge graph"""
        try:
            logger.info(f"Processing document for graph: {document_path}")
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0443\u0437\u043b\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430
            document_id = self.create_document_node(
                document_path, 
                document_type,
                {"content_length": len(content)}
            )
            
            if not document_id:
                return False
            
            # \u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439
            entities = self.extract_entities_from_text(content, document_type)
            
            if not entities:
                logger.warning(f"No entities extracted from {document_path}")
                return True
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0443\u0437\u043b\u043e\u0432 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439
            for entity in entities:
                entity.source_document = document_id
                self.create_entity_node(entity)
            
            # \u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0439
            relations = self.extract_relations_from_text(content, entities)
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0439
            for relation in relations:
                relation.source_document = document_id
                self.create_relation(relation)
            
            # \u0421\u0432\u044f\u0437\u044b\u0432\u0430\u043d\u0438\u0435 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439 \u0441 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u043c
            self.link_entities_to_document(document_id, entities)
            
            logger.info(f"Added {len(entities)} entities and {len(relations)} relations to graph")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document for graph: {e}")
            return False
    
    def query_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        """\u0412\u044b\u043f\u043e\u043b\u043d\u0435\u043d\u0438\u0435 \u0437\u0430\u043f\u0440\u043e\u0441\u0430 \u043a knowledge graph"""
        try:
            # \u041f\u0440\u043e\u0441\u0442\u043e\u0439 \u043f\u043e\u0438\u0441\u043a \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439
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
        """\u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430 \u0438\u0437 knowledge graph \u0434\u043b\u044f \u0437\u0430\u043f\u0440\u043e\u0441\u0430"""
        try:
            # \u041f\u043e\u0438\u0441\u043a \u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u044b\u0445 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439 \u0438 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0439
            results = self.query_knowledge_graph(query)
            
            context_parts = []
            current_length = 0
            
            for result in results:
                entity = result['entity']
                
                # \u0424\u043e\u0440\u043c\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u043d\u043e\u0439 \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u0438
                context_part = f"\u0421\u0443\u0442\u043d\u0456\u0441\u0442\u044c: {entity.get('name', '')} (\u0422\u0438\u043f: {entity.get('type', '')})"
                
                if result['relation'] and result['related_entity']:
                    relation = result['relation']
                    related = result['related_entity']
                    context_part += f"\
\u041f\u043e\u0432'\u044f\u0437\u0430\u043d\u043e \u0447\u0435\u0440\u0435\u0437: {relation.get('type', '')} \u0437 {related.get('name', '')}"
                
                # \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0434\u043b\u0438\u043d\u044b
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
        """\u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0441\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u043a\u0438 knowledge graph"""
        try:
            stats = {}
            
            # \u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u0443\u0437\u043b\u043e\u0432
            node_count_query = "MATCH (n) RETURN count(n) as count"
            node_result = self.neo4j_graph.query(node_count_query)
            stats["total_nodes"] = node_result[0]["count"] if node_result else 0
            
            # \u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0439
            rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
            rel_result = self.neo4j_graph.query(rel_count_query)
            stats["total_relationships"] = rel_result[0]["count"] if rel_result else 0
            
            # \u0422\u0438\u043f\u044b \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0435\u0439
            entity_types_query = "MATCH (e:Entity) RETURN DISTINCT e.type as type, count(e) as count"
            entity_types_result = self.neo4j_graph.query(entity_types_query)
            stats["entity_types"] = {r["type"]: r["count"] for r in entity_types_result}
            
            # \u0422\u0438\u043f\u044b \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0439
            rel_types_query = "MATCH ()-[r:RELATED_TO]->() RETURN DISTINCT r.type as type, count(r) as count"
            rel_types_result = self.neo4j_graph.query(rel_types_query)
            stats["relationship_types"] = {r["type"]: r["count"] for r in rel_types_result}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def create_qa_chain(self, llm):
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 QA \u0446\u0435\u043f\u043e\u0447\u043a\u0438 \u0434\u043b\u044f \u0440\u0430\u0431\u043e\u0442\u044b \u0441 knowledge graph"""
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

# \u0424\u0443\u043d\u043a\u0446\u0438\u0438 \u0434\u043b\u044f \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u044f \u0432 \u0434\u0440\u0443\u0433\u0438\u0445 \u043c\u043e\u0434\u0443\u043b\u044f\u0445
def create_knowledge_graph_manager() -> KnowledgeGraphManager:
    """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u044d\u043a\u0437\u0435\u043c\u043f\u043b\u044f\u0440\u0430 \u043c\u0435\u043d\u0435\u0434\u0436\u0435\u0440\u0430 knowledge graph"""
    return KnowledgeGraphManager()

def process_documents_for_graph(documents, content_getter):
    """\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u0434\u043b\u044f knowledge graph"""
    kg_manager = create_knowledge_graph_manager()
    
    processed_count = 0
    for doc in documents:
        content = content_getter(doc)
        if kg_manager.process_document_for_graph(doc.file_path, content, doc.document_type):
            processed_count += 1
    
    logger.info(f"Processed {processed_count} documents for knowledge graph")
    return processed_count
