"""
ОСНОВНОЙ КЛАСС AI-АССИСТЕНТА

Интегрирует все компоненты:
- DocumentProcessor (обработка документов)
- ModelTrainer (дообучение)
- RAGEngine (поиск + генерация)
- IDE integrations (VSCode, Android Studio)

Оптимизации для 8GB VRAM:
- Ленивая загрузка компонентов
- Очистка памяти после генерации
- Ограниченный контекст
"""

import os
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

# Импорт компонентов
from config import config
from document_processor import DocumentProcessor
from model_trainer import ModelTrainer
from rag_engine import RAGEngine, RAGResponse

class AIAssistant:
    """
    AI-АССИСТЕНТ ДЛЯ IDE И ДОКУМЕНТОВ
    
    Основные возможности:
    1. Добавление документов (PDF/DOCX/TXT/HTML)
    2. Чат с RAG (поиск по документам)
    3. Дообучение на проектах (LoRA)
    4. Анализ кода для IDE
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация ассистента
        
        Args:
            model_path: Путь к дообученной модели (опционально)
        """
        
        # Компоненты (ленивая инициализация)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.rag_engine = None
        self.doc_processor = None
        self.trainer = None
        
        # Текущий проект (для IDE)
        self.current_project = None
        
        # ✅ ЗАГРУЗКА МОДЕЛИ
        # Если передан model_path - загружаем дообученную версию
        self._load_model(model_path)
        
        print(f"✅ AI-Ассистент готов")
        print(f"   Модель: {config.model.model_name}")
        print(f"   Устройство: {config.model.device}")
        print(f"   VRAM: {self._get_vram_usage():.2f}/{VRAM_LIMIT_GB}GB")
    
    def _get_vram_usage(self) -> float:
        """Получить текущее использование VRAM"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _load_model(self, model_path: Optional[str] = None):
        """
        ЗАГРУЗКА МОДЕЛИ С 8-BIT КВАНТИЗАЦИЕЙ
        
        Важно: Модель загружается только один раз
        """
        print("\n🔄 Загрузка модели...")
        
        # Очистка памяти перед загрузкой
        torch.cuda.empty_cache()
        
        # 8-bit конфигурация
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,
        )
        
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or config.model.model_name,
            cache_dir=config.model.cache_dir,
            trust_remote_code=config.model.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Загрузка модели
        model_source = model_path or config.model.model_name
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            cache_dir=config.model.cache_dir,
            quantization_config=bnb_config,
            device_map={"": 0},  # Явно на GPU 0
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Создание генерационного pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        vram_used = self._get_vram_usage()
        print(f"✅ Модель загружена. VRAM: {vram_used:.2f}/{VRAM_LIMIT_GB}GB")
        
        # Инициализация RAG движка после загрузки модели
        self.rag_engine = RAGEngine(self.pipeline)
    
    # ============= ДОБАВЛЕНИЕ ДОКУМЕНТОВ =============
    
    def add_document(self, file_path: str) -> bool:
        """
        ДОБАВИТЬ ДОКУМЕНТ В БАЗУ ЗНАНИЙ
        
        Args:
            file_path: Путь к файлу (PDF/DOCX/TXT/HTML)
            
        Returns:
            True если успешно
        """
        
        # Ленивая инициализация процессора
        if self.doc_processor is None:
            self.doc_processor = DocumentProcessor()
        
        try:
            return self.doc_processor.process_file(file_path)
        except Exception as e:
            print(f"❌ Ошибка добавления документа: {e}")
            return False
    
    def add_project(self, project_path: str) -> Dict[str, Any]:
        """
        ДОБАВИТЬ ПРОЕКТ IDE (VSCode/Android Studio)
        
        Сканирует все поддерживаемые файлы в проекте
        
        Args:
            project_path: Путь к корню проекта
            
        Returns:
            Статистика обработки
        """
        print(f"\n📂 Сканирование проекта: {project_path}")
        
        if not os.path.exists(project_path):
            return {"success": False, "error": f"Проект не найден: {project_path}"}
        
        if self.doc_processor is None:
            self.doc_processor = DocumentProcessor()
        
        # Сканирование файлов
        processed_files = 0
        errors = []
        
        for root, dirs, files in os.walk(project_path):
            # Исключение служебных директорий
            dirs[:] = [d for d in dirs if d not in config.ide.ignore_patterns]
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = Path(file).suffix.lower()
                
                # Проверка поддерживаемого формата
                if ext in config.documents.supported_formats:
                    try:
                        if self.doc_processor.process_file(file_path):
                            processed_files += 1
                    except Exception as e:
                        errors.append(f"{file_path}: {e}")
        
        print(f"✅ Проект обработан: {processed_files} файлов")
        
        return {
            "success": True,
            "processed_files": processed_files,
            "errors_count": len(errors),
            "errors": errors
        }
    
    # ============= ЧАТ С RAG =============
    
    def chat(self, question: str) -> str:
        """
        ОБЩИЙ ЧАТ С ВОПРОСОМ
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Ответ от ассистента
        """
        if self.rag_engine is None:
            raise RuntimeError("RAG движок не инициализирован")
        
        print(f"\n💬 Вопрос: {question[:50]}...")
        
        # Генерация ответа
        response = self.rag_engine.ask(question)
        
        # Вывод источников (для отладки)
        if response.sources:
            print(f"📋 Использовано источников: {len(response.sources)}")
        
        return response.answer
    
    def analyze_code_file(self, file_path: str) -> str:
        """
        АНАЛИЗ ФАЙЛА С ИСХОДНЫМ КОДОМ
        
        Используется для интеграции с IDE:
        1. Читает файл
        2. Добавляет в контекст
        3. Анализирует
        
        Args:
            file_path: Путь к файлу кода
            
        Returns:
            Анализ и рекомендации
        """
        if not os.path.exists(file_path):
            return f"❌ Файл не найден: {file_path}"
        
        # Чтение файла
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except:
            return "❌ Не удалось прочитать файл (ошибка кодировки?)"
        
        # Ограничение размера
        max_tokens = config.model.max_file_tokens
        if len(code_content) > max_tokens * 4:  # Приблизительно
            code_content = code_content[:max_tokens * 4] + "\n... (обрезано из-за размера)"
        
        # Формирование вопроса
        file_name = Path(file_path).name
        question = f"Проаналізуй цей код з файлу {file_name}:\n\n{code_content}"
        
        # Генерация ответа с использованием контекста
        response = self.rag_engine.ask(question)
        
        return response.answer
    
    def debug_error(self, error_message: str, file_context: Optional[str] = None) -> str:
        """
        ОТЛАДКА ОШИБОК В КОДЕ
        
        Args:
            error_message: Сообщение об ошибке (стек трейс)
            file_context: Содержимое файла для контекста
            
        Returns:
            Рекомендации по исправлению
        """
        # Формирование промпта с ошибкой
        prompt = f"""[INST] Помоги виправити помилку в коді:

ПОМИЛКА:
{error_message}

"""
        if file_context:
            prompt += f"КОД ДЕ ПРОИЗОШЛА ОШИБКА:\n{file_context[:500]}\n\n"
        
        prompt += "ПОЯСНЕННЯ ПРИЧИНИ ТА ВИПРАВЛЕННЯ: [/INST]"
        
        # Генерация ответа
        response = self.pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.3,  # Более детерминированный ответ
        )
        
        return response[0]['generated_text'] if response else "Ошибка генерации"
    
    def suggest_code(self, description: str, language: str = "python") -> str:
        """
        ГЕНЕРАЦИЯ КОДА ПО ОПИСАНИЮ
        
        Args:
            description: Описание, что нужно сделать
            language: Язык программирования
            
        Returns:
            Сгенерированный код
        """
        prompt = f"""[INST] Напиши код на {language}:

ЗАПИТ:
{description}

ВИМОГИ:
1. Код має бути робочим та безпечним
2. Додай коментарі
3. Вкажи важливі моменти

КОД: [/INST]"""
        
        response = self.pipeline(
            prompt,
            max_new_tokens=config.model.max_new_tokens,
            temperature=0.7,
        )
        
        return response[0]['generated_text'] if response else ""
    
    # ============= ДООБУЧЕНИЕ =============
    
    def train_on_documents(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        ДООБУЧЕНИЕ НА ЗАГРУЖЕННЫХ ДОКУМЕНТАХ
        
        Запускает процесс LoRA fine-tuning на всех документах в векторной базе
        
        Args:
            output_dir: Директория для сохранения модели
            
        Returns:
            Метаданные обучения
        """
        
        # Ленивая инициализация тренера
        if self.trainer is None:
            self.trainer = ModelTrainer()
        
        # Получение всех чанков из векторной базы
        if self.doc_processor is None:
            self.doc_processor = DocumentProcessor()
        
        print("\n📚 Сбор чанков для обучения...")
        
        from document_processor import ProcessedChunk
        
        # Получение всех данных из ChromaDB
        collection = self.doc_processor.vectorstore._collection
        all_data = collection.get()
        
        if not all_data['documents']:
            return {"success": False, "error": "Нет документов для обучения"}
        
        # Конвертация в ProcessedChunk
        chunks = []
        for idx, doc_content in enumerate(all_data['documents']):
            metadata = all_data['metadatas'][idx]
            
            chunk = ProcessedChunk(
                content=doc_content,
                metadata=metadata,
                vector_id=all_data['ids'][idx] if 'ids' in all_data else None
            )
            chunks.append(chunk)
        
        print(f"✅ Найдено {len(chunks)} чанков для обучения")
        
        # Запуск обучения
        return self.trainer.train(chunks, output_dir)
    
    # ============= ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =============
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус ассистента"""
        return {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_used_gb": self._get_vram_usage(),
            "vram_total_gb": VRAM_LIMIT_GB,
            "model_loaded": self.model is not None,
            "rag_ready": self.rag_engine is not None,
            "documents_db": self.doc_processor.get_stats()["vectors_in_db"] if self.doc_processor else 0,
        }
    
    def clear_memory(self):
        """Принудительная очистка GPU памяти"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU память очищена. Использовано: {self._get_vram_usage():.2f}GB")