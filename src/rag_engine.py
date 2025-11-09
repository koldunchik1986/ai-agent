"""
RAG ENGINE (Retrieval-Augmented Generation)

Связывает векторный поиск с генерацией ответов
Оптимизировано для 8GB VRAM: ограниченный контекст, очистка памяти

Основной флоу:
1. Получить вопрос от пользователя
2. Найти релевантные чанки в ChromaDB
3. Сформировать промпт с контекстом
4. Сгенерировать ответ через модель
"""

import torch
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from config import config
from document_processor import DocumentProcessor

@dataclass
class RAGResponse:
    """Структура ответа RAG"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

class RAGEngine:
    """
    RAG ДВИГАТЕЛЬ ДЛЯ AI-АССИСТЕНТА
    
    Особенности для IDE:
    - Ограниченная память разговоров (3 последних сообщения)
    - Контекст проекта добавляется автоматически
    - Оптимизация промпта для кода
    """
    
    def __init__(self, llm_pipeline):
        # Компоненты
        self.doc_processor = DocumentProcessor()
        self.llm = llm_pipeline
        
        # Ограниченная память разговоров (3 сообщения)
        # Больше = больше контекста, но медленнее и больше VRAM
        self.memory = ConversationBufferWindowMemory(
            k=3,  # Только 3 последних сообщения
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )
        
        # ✅ ШАБЛОН ПРОМПТА ДЛЯ IDE
        # Оптимизирован для кода и документации
        self.prompt_template = """<s>[INST] Ти - AI-асистент для розробки. 
Використовуючи контекст проекту, дай точну відповідь.

КОНТЕКСТ ПРОЕКТУ:
{context}

ПИТАННЯ КОРИСТУВАЧА:
{question}

ІНСТРУКЦІЇ:
1. Відповідай українською мовою
2. Для коду: надавай готовий код з поясненнями
3. Для помилок: вказуй конкретне рішення
4. Якщо контекст не релевантний - відповідай "Немає відповідної інформації в проекті"

ВІДПОВІДЬ: [/INST]"""
        
        # Создание PromptTemplate
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # Конфигурация retriever (поиск в векторной базе)
        # search_kwargs настраивают параметры поиска
        self.retriever = self.doc_processor.vectorstore.as_retriever(
            search_kwargs={
                "k": config.vector.search_k,  # Количество результатов
                "score_threshold": config.vector.similarity_threshold  # Минимальная схожесть
            }
        )
        
        # ✅ СОЗДАНИЕ RAG ЦЕПОЧКИ
        # RetrievalQA: ищет релевантное, затем генерирует
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Простая стратегия (все результаты в один промпт)
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self.prompt,  # Наш промпт
                "memory": self.memory,  # Память разговоров
            },
            return_source_documents=True,  # Возвращать использованные источники
        )
    
    def add_project_context(self, project_path: str):
        """
        ДОБАВЛЕНИЕ КОНТЕКСТА ПРОЕКТА
        
        Сканирует директорию проекта и добавляет все файлы
        Используется при открытии проекта в IDE
        
        Args:
            project_path: Путь к корню проекта
        """
        print(f"📂 Сканирование проекта: {project_path}")
        
        # Проверка существования
        if not os.path.exists(project_path):
            print(f"❌ Проект не найден: {project_path}")
            return
        
        # Сканирование файлов
        processed_files = 0
        for root, dirs, files in os.walk(project_path):
            # Удаление игнорируемых директорий
            dirs[:] = [d for d in dirs if d not in config.ide.ignore_patterns]
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = Path(file).suffix.lower()
                
                # Проверка поддерживаемого формата
                if ext in config.documents.supported_formats:
                    # Обрабатываем файл
                    result = self.doc_processor.process_file(file_path)
                    if result:
                        processed_files += 1
        
        print(f"✅ Проект просканирован: {processed_files} файлов добавлено")
    
    def ask(self, question: str, project_context: Optional[str] = None) -> RAGResponse:
        """
        ЗАДАТЬ ВОПРОС С ИСПОЛЬЗОВАНИЕМ RAG
        
        Args:
            question: Вопрос пользователя
            project_context: Дополнительный контекст (необязательно)
            
        Returns:
            RAGResponse с ответом, источниками и метаданными
        """
        if not question or not question.strip():
            return RAGResponse(
                answer="",
                sources=[],
                confidence=0.0,
                metadata={"error": "Пустой вопрос"}
            )
        
        try:
            # ✅ ДОБАВЛЕНИЕ КОНТЕКСТА ПРОЕКТА
            # Если передан project_context, добавляем его в промпт
            if project_context:
                enhanced_question = f"{question}\n\nКонтекст файлу:\n{project_context[:500]}"
            else:
                enhanced_question = question
            
            # Запуск RAG цепочки
            result = self.qa_chain.invoke({"query": enhanced_question})
            
            # ✅ ОБРАБОТКА РЕЗУЛЬТАТА
            answer = result["result"]
            
            # Извлечение источников
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance": 1.0  # Возможно расчет через score
                })
            
            # Расчет уверенности
            # Если есть источники = высокая уверенность
            confidence = 0.85 if len(sources) > 0 else 0.5
            
            # ✅ ОЧИСТКА КЭША ПОСЛЕ ГЕНЕРАЦИИ
            # Освобождает память, занятую промежуточными тензорами
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                metadata={
                    "has_sources": len(sources) > 0,
                    "sources_count": len(sources),
                    "question_length": len(question)
                }
            )
        
        except Exception as e:
            print(f"❌ Ошибка в RAG: {e}")
            
            # Очистка памяти даже при ошибке
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return RAGResponse(
                answer="Вибачте, сталася помилка при обробці запиту.",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def get_stats(self) -> Dict:
        """Получить статистику RAG"""
        return {
            "memory_length": len(self.memory.chat_memory.messages),
            "documents_in_db": self.doc_processor.get_stats()["vectors_in_db"],
            "collection_name": config.vector.collection_name,
        }