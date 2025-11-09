"""
Основной AI-агент на базе Mistral AI 7B
Интеграция с document processing, model training, knowledge graph
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# LangChain
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Local imports
from config import config
from document_processor import DocumentProcessor
from model_training import ModelTrainer
from knowledge_graph import KnowledgeGraphManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Класс для представления ответа агента"""
    content: str
    sources: List[Dict[str, Any]]
    confidence: float
    response_time: float
    context_used: bool

class AIAgent:
    """Основной класс AI-агента"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.generation_pipeline = None
        self.llm = None
        
        # Компоненты
        self.document_processor = None
        self.knowledge_graph = None
        self.model_trainer = None
        
        # Память и контекст
        self.conversation_memory = ConversationBufferMemory()
        self.current_context = []
        
        # Инициализация
        self._init_model(model_path)
        self._init_components()
        self._init_generation_pipeline()
    
    def _init_model(self, model_path: Optional[str] = None):
        """Инициализация модели"""
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading fine-tuned model from: {model_path}")
                self.model_trainer = ModelTrainer()
                self.model_trainer.load_fine_tuned_model(model_path)
                self.model = self.model_trainer.model
                self.tokenizer = self.model_trainer.tokenizer
            else:
                logger.info(f"Loading base model: {config.model.model_name}")
                
                # Загрузка базовой модели
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model.model_name,
                    cache_dir=config.model.cache_dir,
                    torch_dtype=getattr(torch, config.model.torch_dtype),
                    device_map="sequential",
                    trust_remote_code=config.model.trust_remote_code,
                    use_cache=config.model.use_cache
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.model.model_name,
                    cache_dir=config.model.cache_dir,
                    trust_remote_code=config.model.trust_remote_code
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _init_components(self):
        """Инициализация компонентов агента"""
        try:
            # Document processor
            self.document_processor = DocumentProcessor()
            
            # Knowledge graph
            self.knowledge_graph = KnowledgeGraphManager()
            
            # Model trainer
            if self.model_trainer is None:
                self.model_trainer = ModelTrainer()
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _init_generation_pipeline(self):
        """Инициализация pipeline для генерации"""
        try:
            # Создание HuggingFace pipeline
            self.generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                top_p=config.model.top_p,
                top_k=config.model.top_k,
                repetition_penalty=config.model.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Создание LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=self.generation_pipeline)
            
            logger.info("Generation pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing generation pipeline: {e}")
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Добавление документов в базу знаний"""
        try:
            logger.info(f"Adding {len(file_paths)} documents to knowledge base")
            
            # Обработка документов
            processed_docs = []
            for file_path in file_paths:
                doc = self.document_processor.process_single_document(file_path)
                if doc:
                    processed_docs.append(doc)
            
            if not processed_docs:
                return {"success": False, "message": "No documents were processed successfully"}
            
            # Создание эмбеддингов
            embedding_success = self.document_processor.create_embeddings(processed_docs)
            
            # Добавление в knowledge graph
            graph_processed = 0
            for doc in processed_docs:
                content = " ".join([chunk.content for chunk in doc.chunks])
                if self.knowledge_graph.process_document_for_graph(
                    doc.file_path, content, doc.document_type
                ):
                    graph_processed += 1
            
            result = {
                "success": True,
                "processed_documents": len(processed_docs),
                "embeddings_created": embedding_success,
                "graph_processed": graph_processed,
                "message": f"Successfully added {len(processed_docs)} documents"
            }
            
            logger.info(f"Documents added: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"success": False, "message": str(e)}
    
    def train_on_documents(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Обучение модели на добавленных документах"""
        try:
            # Получение всех документов из vector store
            collection = self.document_processor.collection
            
            # Получение всех документов
            all_docs = collection.get()
            
            if not all_docs['documents']:
                return {"success": False, "message": "No documents found for training"}
            
            # Создание объектов ProcessedDocument из vector store
            processed_docs = []
            # Здесь нужно преобразовать данные из vector store в ProcessedDocument объекты
            # Упрощенная версия для демонстрации
            
            # Запуск обучения
            training_result = self.model_trainer.train_model(processed_docs, save_path)
            
            return {"success": True, "training_result": training_result}
            
        except Exception as e:
            logger.error(f"Error training on documents: {e}")
            return {"success": False, "message": str(e)}
    
    def retrieve_relevant_context(self, query: str, max_context_items: int = 5) -> List[Dict[str, Any]]:
        """Получение релевантного контекста из документов"""
        try:
            # Поиск в векторной базе данных
            vector_results = self.document_processor.search_similar_documents(query, max_context_items)
            
            # Поиск в knowledge graph
            graph_context = self.knowledge_graph.get_context_for_query(query)
            
            # Объединение результатов
            context_items = []
            
            # Добавление результатов из vector store
            for result in vector_results:
                context_items.append({
                    "type": "document",
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "source": "vector_store",
                    "relevance_score": 1 - result["distance"]
                })
            
            # Добавление контекста из knowledge graph
            if graph_context:
                context_items.append({
                    "type": "knowledge_graph",
                    "content": graph_context,
                    "source": "knowledge_graph",
                    "relevance_score": 0.8
                })
            
            # Сортировка по релевантности
            context_items.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return context_items[:max_context_items]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def generate_prompt_with_context(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """Генерация промпта с контекстом"""
        base_prompt = """Ти - AI-асистент, навчений на програмуванні та юридичних документах України. 
Твоя задача - давати точні, корисні та детальні відповіді на основі наданого контексту.

Контекст з бази знань:
"""
        
        # Добавление контекста
        if context_items:
            for i, item in enumerate(context_items, 1):
                base_prompt += f"\
{i}. {item['content']}\
"
                if item.get('metadata', {}).get('file_name'):
                    base_prompt += f"   Джерело: {item['metadata']['file_name']}\
"
            
            base_prompt += "\
"
        else:
            base_prompt += "Контекст не знайдено. Будь ласка, дай відповідь на основі своїх загальних знань.\
\
"
        
        base_prompt += f"Питання користувача: {query}\
\
"
        base_prompt += "Дай детальну відповідь українською мовою:"
        
        return base_prompt
    
    def generate_response(self, query: str, use_context: bool = True) -> AgentResponse:
        """Генерация ответа на запрос"""
        start_time = datetime.now()
        
        try:
            # Получение контекста
            context_items = []
            if use_context:
                context_items = self.retrieve_relevant_context(query)
            
            # Генерация промпта
            prompt = self.generate_prompt_with_context(query, context_items)
            
            # Генерация ответа
            response = self.generation_pipeline(
                prompt,
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Очистка ответа
            generated_text = response[0]['generated_text'] if response else ""
            generated_text = generated_text.strip()
            
            # Расчет времени ответа
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Форматирование источников
            sources = []
            for item in context_items:
                sources.append({
                    "content": item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"],
                    "metadata": item.get("metadata", {}),
                    "source": item["source"],
                    "relevance": item["relevance_score"]
                })
            
            # Расчет уверенности
            confidence = 0.8 if context_items else 0.6
            
            # Сохранение в память
            self.conversation_memory.save_context(
                {"input": query},
                {"output": generated_text}
            )
            
            return AgentResponse(
                content=generated_text,
                sources=sources,
                confidence=confidence,
                response_time=response_time,
                context_used=len(context_items) > 0
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            return AgentResponse(
                content=f"Вибачте, сталася помилка при генерації відповіді: {str(e)}",
                sources=[],
                confidence=0.0,
                response_time=(datetime.now() - start_time).total_seconds(),
                context_used=False
            )
    
    def handle_programming_query(self, query: str) -> AgentResponse:
        """Обработка программных запросов"""
        # Добавление специального контекста для программирования
        programming_context = """
Для програмування використовуй наступні підходи:
1. Аналізуй код на відповідність best practices
2. Пропонуй оптимізацію та рефакторинг
3. Вказуй на потенційні проблеми безпеки
4. Давай конкретні приклади коду
5. Рекомендуй відповідні бібліотеки та інструменти
"""
        
        # Получение релевантного контекста программирования
        context_items = self.retrieve_relevant_context(query)
        
        # Добавление программного контекста
        context_items.append({
            "type": "programming_guidelines",
            "content": programming_context,
            "source": "system",
            "relevance_score": 0.9
        })
        
        prompt = self.generate_prompt_with_context(query, context_items)
        
        # Генерация ответа с меньшей температурой для кода
        original_temp = config.model.temperature
        config.model.temperature = 0.3  # Более детерминированные ответы для кода
        
        try:
            response = self.generate_response(query, use_context=False)
            
            # Генерация с специальным промптом
            generation_result = self.generation_pipeline(
                prompt,
                max_new_tokens=config.model.max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            if generation_result:
                response.content = generation_result[0]['generated_text'].strip()
            
            return response
            
        finally:
            config.model.temperature = original_temp
    
    def handle_legal_query(self, query: str) -> AgentResponse:
        """Обработка юридических запросов"""
        legal_disclaimer = """
ВАЖЛИВО: Ця відповідь надається в інформаційних цілях і не є юридичною консультацією. 
Для отримання професійної юридичної допомоги зверніться до кваліфікованого юриста.

Аналізуючи юридичні питання, керуюся наступним:
1. Законодавство України
2. Судова практика
3. Формальні вимоги до документів
4. Процесуальні терміни та порядок
"""
        
        # Получение юридического контекста
        context_items = self.retrieve_relevant_context(query)
        
        # Добавление дисклеймера
        context_items.append({
            "type": "legal_disclaimer",
            "content": legal_disclaimer,
            "source": "system",
            "relevance_score": 1.0
        })
        
        # Генерация ответа
        prompt = self.generate_prompt_with_context(query, context_items)
        
        generation_result = self.generation_pipeline(
            prompt,
            max_new_tokens=config.model.max_new_tokens,
            temperature=0.5,  # Сбалансированная температура для юридических ответов
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = generation_result[0]['generated_text'].strip() if generation_result else ""
        
        # Форматирование источников
        sources = []
        for item in context_items:
            sources.append({
                "content": item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"],
                "metadata": item.get("metadata", {}),
                "source": item["source"],
                "relevance": item["relevance_score"]
            })
        
        return AgentResponse(
            content=generated_text,
            sources=sources,
            confidence=0.7,
            response_time=0.0,
            context_used=len(context_items) > 0
        )
    
    def detect_query_type(self, query: str) -> str:
        """Определение типа запроса"""
        query_lower = query.lower()
        
        programming_keywords = [
            "код", "програма", "function", "class", "def", "import", "bug", "error",
            "python", "java", "javascript", "debug", "compile", "алгоритм", "функція"
        ]
        
        legal_keywords = [
            "закон", "стаття", "кодекс", "позов", "скарга", "суд", "право", "законодавство",
            "договір", "угода", "власність", "відповідальність", "норма", "регулювання"
        ]
        
        programming_score = sum(1 for keyword in programming_keywords if keyword in query_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in query_lower)
        
        if programming_score > legal_score:
            return "programming"
        elif legal_score > programming_score:
            return "legal"
        else:
            return "general"
    
    def query(self, user_query: str) -> AgentResponse:
        """Основной метод обработки запроса"""
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Определение типа запроса
        query_type = self.detect_query_type(user_query)
        
        # Выбор соответствующего обработчика
        if query_type == "programming":
            return self.handle_programming_query(user_query)
        elif query_type == "legal":
            return self.handle_legal_query(user_query)
        else:
            return self.generate_response(user_query)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Получение статуса агента"""
        try:
            return {
                "model_loaded": self.model is not None,
                "components_initialized": all([
                    self.document_processor is not None,
                    self.knowledge_graph is not None,
                    self.model_trainer is not None
                ]),
                "vector_store_stats": self.document_processor.get_document_stats(),
                "knowledge_graph_stats": self.knowledge_graph.get_graph_statistics(),
                "conversation_memory_length": len(self.conversation_memory.buffer.split('\
')) if self.conversation_memory.buffer else 0,
                "model_name": config.model.model_name,
                "device": config.model.device
            }
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return {"error": str(e)}

# Функции для использования в других модулях
def create_ai_agent(model_path: Optional[str] = None) -> AIAgent:
    """Создание экземпляра AI-агента"""
    return AIAgent(model_path)

def quick_setup_agent(documents_path: str) -> AIAgent:
    """Быстрая настройка агента с документами"""
    agent = create_ai_agent()
    
    # Добавление документов
    document_files = []
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.endswith(('.pdf', '.docx', '.doc', '.txt', '.md')):
                document_files.append(os.path.join(root, file))
    
    if document_files:
        agent.add_documents(document_files)
        logger.info(f"Added {len(document_files)} documents to agent")
    
    return agent
