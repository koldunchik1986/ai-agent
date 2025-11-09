"""
AI-АССИСТЕНТ ДЛЯ P104-100 8GB VRAM (sm_61 Pascal)
=====================================================

Основной пакет AI-ассистента с оптимизацией под 8GB VRAM.

Компоненты:
- config: Конфигурация системы
- document_processor: Обработка PDF/DOCX/TXT/HTML
- model_trainer: LoRA дообучение
- rag_engine: Поиск + генерация
- agent: Главный класс ассистента
- vscode_integration: Интеграция с VSCode
- android_studio_integration: Интеграция с Android Studio
- cli: Командный интерфейс

Использование:
    from src.agent import AIAssistant
    assistant = AIAssistant()
    response = assistant.chat("Ваш вопрос?")
"""

__version__ = "1.0.0"
__author__ = "AI-Assistant Team"
__description__ = "Оптимизированный ИИ-ассистент для P104-100 8GB VRAM"

# Импорты для удобства использования
from .config import config
from .agent import AIAssistant
from .document_processor import DocumentProcessor, ProcessedChunk
from .model_trainer import ModelTrainer
from .rag_engine import RAGEngine, RAGResponse

__all__ = [
    "config",
    "AIAssistant", 
    "DocumentProcessor",
    "ProcessedChunk",
    "ModelTrainer",
    "RAGEngine",
    "RAGResponse"
]