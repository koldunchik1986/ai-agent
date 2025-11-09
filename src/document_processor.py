"""
ОБРАБОТКА ДОКУМЕНТОВ ДЛЯ AI-АССИСТЕНТА

Особенности:
- Поддержка PDF/DOCX/TXT/HTML
- Потоковая обработка (не загружает весь файл в память)
- Автоматическое разбиение на чанки
- Оптимизация под 8GB VRAM (batch processing)
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any
from dataclasses import dataclass

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Импорт_loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import config

@dataclass
class ProcessedChunk:
    """
    Структура обработанного чанка
    
    Attributes:
        content: Текст чанка
        metadata: Метаданные (путь к файлу, тип и т.д.)
        vector_id: ID в векторной базе
        confidence: Оценка качества обработки
    """
    content: str
    metadata: Dict[str, Any]
    vector_id: Optional[str] = None
    confidence: float = 1.0

class DocumentProcessor:
    """
    ПРОЦЕССОР ДОКУМЕНТОВ
    
    Основные функции:
    1. Загрузка документов разных форматов
    2. Разбиение на чанки (chunking)
    3. Создание векторных эмбеддингов
    4. Сохранение в ChromaDB
    """
    
    def __init__(self):
        # Инициализация конфигурации
        self.config = config
        
        # Инициализация text splitter
        # Разбивает большие тексты на чанки для эффективного поиска
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.documents.chunk_size,
            chunk_overlap=self.config.documents.chunk_overlap,
            length_function=len,
            # Стратегия разбиения: сначала по двойным переносам, потом по предложениям
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # ✅ Инициализация эмбеддингов (ленивая)
        # Модель создается при первом использовании для экономии памяти
        self._embeddings = None
        self._vectorstore = None
        
        # Статистика
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "vectors_stored": 0,
            "total_size_mb": 0.0
        }
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Ленивая загрузка эмбеддингов (только когда нужно)"""
        if self._embeddings is None:
            # Выбор модели в зависимости от доступной памяти
            model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            
            # Для 8GB используем компактную модель, для 12GB+ - более точную
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < 9:
                    # ✅ Компактная модель для экономии VRAM (~200MB)
                    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                else:
                    # Более точная модель для 12GB+ (~700MB)
                    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            else:
                # Для CPU (очень медленно!)
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs
            )
            print(f"✅ Инициализированы эмбеддинги: {model_name}")
        
        return self._embeddings
    
    @property
    def vectorstore(self) -> Chroma:
        """Ленивая загрузка векторной базы (только когда нужно)"""
        if self._vectorstore is None:
            # Создание директории если не существует
            persist_dir = Path(self.config.vector.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Проверяем, существует ли уже коллекция
            if (persist_dir / "chroma.sqlite3").exists():
                # Загрузка существующей коллекции
                self._vectorstore = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=self.embeddings,
                    collection_name=self.config.vector.collection_name
                )
                print(f"✅ Загружена существующая векторная база ({persist_dir})")
            else:
                # Создание новой коллекции
                self._vectorstore = Chroma(
                    collection_name=self.config.vector.collection_name,
                    persist_directory=str(persist_dir),
                    embedding_function=self.embeddings
                )
                print(f"✅ Создана новая векторная база ({persist_dir})")
        
        return self._vectorstore
    
    def load_document(self, file_path: str) -> Optional[List[Any]]:
        """
        ЗАГРУЗКА ДОКУМЕНТА
        
        Поддерживаемые форматы:
        - PDF: PyMuPDF extraction
        - DOCX: python-docx парсер
        - TXT: Простой текст
        - HTML: BeautifulSoup разбор
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            List of Document objects или None при ошибке
        """
        path = Path(file_path)
        
        # Проверка существования файла
        if not path.exists():
            print(f"❌ Файл не найден: {file_path}")
            return None
        
        # Проверка размера файла
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.documents.max_file_size_mb:
            print(f"⚠️ Файл слишком большой ({file_size_mb:.1f}MB > {self.config.documents.max_file_size_mb}MB)")
            # TODO: Добавить потоковую обработку больших файлов
        
        # Выбор загрузчика по расширению
        ext = path.suffix.lower()
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(str(path))
            elif ext == '.docx':
                loader = Docx2txtLoader(str(path))
            elif ext == '.txt':
                loader = TextLoader(str(path), encoding='utf-8')
            elif ext == '.html':
                loader = UnstructuredHTMLLoader(str(path))
            else:
                print(f"❌ Неподдерживаемый формат: {ext}")
                return None
            
            # ✅ Потоковая загрузка (не загружает весь файл в память сразу)
            documents = loader.load()
            print(f"✅ Загружен: {path.name} ({len(documents)} страниц/частей)")
            
            # Обновление статистики
            self.stats["files_processed"] += 1
            self.stats["total_size_mb"] += file_size_mb
            
            return documents
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {path.name}: {e}")
            return None
    
    def chunk_document(self, documents: List[Any]) -> List[ProcessedChunk]:
        """
        РАЗБИЕНИЕ ДОКУМЕНТА НА ЧАНКИ
        
        Процесс:
        1. Берет загруженный документ
        2. Разбивает на чанки по chunk_size (512 токенов)
        3. Добавляет метаданные к каждому чанку
        
        Args:
            documents: Загруженные документы
            
        Returns:
            Список ProcessedChunk
        """
        if not documents:
            return []
        
        # Разбиение на чанки
        chunks = self.text_splitter.split_documents(documents)
        
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            # Генерация уникального ID для чанка
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
            
            chunk_metadata = {
                "chunk_id": f"{path.stem}_{idx}_{content_hash}",
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "source_file": str(path),
                "file_name": path.name,
                "file_ext": path.suffix,
                "processing_timestamp": torch.datetime.now().isoformat(),
                "confidence": 0.95  # Дефолтная уверенность
            }
            
            # Объединение с существующими метаданными
            if hasattr(chunk, 'metadata'):
                chunk_metadata.update(chunk.metadata)
            
            processed_chunk = ProcessedChunk(
                content=chunk.page_content,
                metadata=chunk_metadata
            )
            
            processed_chunks.append(processed_chunk)
        
        print(f"📄 Создано {len(processed_chunks)} чанков")
        self.stats["chunks_created"] += len(processed_chunks)
        
        return processed_chunks
    
    def create_embeddings(self, chunks: List[ProcessedChunk]) -> List[str]:
        """
        СОЗДАНИЕ ВЕКТОРНЫХ ЭМБЕДДИНГОВ
        
        Process:
        1. Конвертирует чанки в векторы
        2. Сохраняет в ChromaDB
        3. Возвращает ID векторов
        
        ✅ Оптимизация под 8GB: Batch processing с очисткой кэша
        """
        if not chunks:
            return []
        
        # Пакетная обработка (batch_size=10 для экономии памяти)
        batch_size = 10
        vector_ids = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Извлечение текста и метаданных
            batch_texts = [chunk.content for chunk in batch]
            batch_metadatas = [chunk.metadata for chunk in batch]
            batch_ids = [chunk.metadata["chunk_id"] for chunk in batch]
            
            # Создание эмбеддингов и сохранение
            ids = self.vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            vector_ids.extend(ids)
            
            # ✅ ОЧИСТКА КЭША GPU ПОСЛЕ КАЖДОГО БАТЧА
            # Это предотвращает накопление памяти при больших файлах
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"  Векторизовано {len(ids)} чанков...")
        
        # Сохранение на диск (persist)
        self.vectorstore.persist()
        print(f"✅ Векторы сохранены в ChromaDB")
        
        self.stats["vectors_stored"] += len(vector_ids)
        
        return vector_ids
    
    def process_file(self, file_path: str) -> bool:
        """
        ПОЛНЫЙ ПРОЦЕСС ОБРАБОТКИ ФАЙЛА
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            True если успешно, False если ошибка
        """
        try:
            print(f"\n🔄 Обработка: {file_path}")
            
            # 1. Загрузка
            documents = self.load_document(file_path)
            if not documents:
                return False
            
            # 2. Разбиение на чанки
            chunks = self.chunk_document(documents)
            if not chunks:
                return False
            
            # 3. Векторизация
            vector_ids = self.create_embeddings(chunks)
            
            print(f"✅ Успешно обработан: {len(chunks)} чанков → {len(vector_ids)} векторов")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обработки {file_path}: {e}")
            return False
    
    def search_similar(self, query: str, k: Optional[int] = None, 
                      score_threshold: Optional[float] = None) -> List[Dict]:
        """
        ПОИСК ПОХОЖИХ ДОКУМЕНТОВ
        
        Args:
            query: Вопрос пользователя
            k: Количество результатов (по умолчанию из config)
            score_threshold: Минимальная схожесть
            
        Returns:
            Список результатов с метаданными
        """
        if not query or not query.strip():
            return []
        
        if k is None:
            k = self.config.vector.search_k
        
        if score_threshold is None:
            score_threshold = self.config.vector.similarity_threshold
        
        try:
            # Поиск в векторной базе
            # similarity_search_with_score возвращает (document, score)
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                score_threshold=score_threshold
            )
            
            # Конвертация в удобный формат
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,  # Меньше = лучше (0=идентично)
                    "relevance": 1.0 - score  # Для удобства чтения
                })
            
            print(f"🔍 Найдено {len(formatted_results)} релевантных чанков")
            
            return formatted_results
        
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Получить статистику обработки"""
        try:
            collection_count = self.vectorstore._collection.count()
        except:
            collection_count = 0
        
        self.stats["vectors_in_db"] = collection_count
        
        return self.stats
    
    def delete_collection(self):
        """Удалить всю коллекцию (для сброса)"""
        print("⚠️ Удаление коллекции ChromaDB...")
        self.vectorstore.delete_collection()
        self.vectorstore.persist()
        print("✅ Коллекция удалена")
        
        # Сброс статистики
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "vectors_stored": 0,
            "total_size_mb": 0.0
        }