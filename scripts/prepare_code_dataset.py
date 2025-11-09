#!/usr/bin/env python3

# ===================================================================
# ПОДГОТОВКА ДАТАСЕТА ИЗ КОДА ПРОЕКТА ДЛЯ ДООБУЧЕНИЯ
# ===================================================================
# Сканирует проект и создает обучающие примеры из кода
# Оптимизировано для Android Studio и VSCode проектов

import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import re

# Добавляем src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.document_processor import ProcessedChunk
from src.config import config

class CodeDatasetPreparer:
    """
    ПОДГОТОВКА ДАТАСЕТА ИЗ ИСХОДНИКОВ
    
    Поддерживает:
    - Python (Django, Flask, FastAPI)
    - Java (Spring, Android)
    - Kotlin (Android)
    - JavaScript (Node.js, React)
    - TypeScript (Angular, Vue)
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.training_examples = []
        
        # Паттерны для разных языков
        self.language_patterns = {
            'python': {
                'files': ['*.py'],
                'comment': '#',
                'class_pattern': r'^class\s+(\w+)',
                'function_pattern': r'^def\s+(\w+)'
            },
            'java': {
                'files': ['*.java'],
                'comment': '//',
                'class_pattern': r'^public\s+class\s+(\w+)',
                'function_pattern': r'public\s+\w+\s+(\w+)\s*\('
            },
            'kotlin': {
                'files': ['*.kt'],
                'comment': '//',
                'class_pattern': r'^class\s+(\w+)',
                'function_pattern': r'fun\s+(\w+)'
            },
            'javascript': {
                'files': ['*.js'],
                'comment': '//',
                'class_pattern': r'^class\s+(\w+)',
                'function_pattern': r'function\s+(\w+)'
            },
            'typescript': {
                'files': ['*.ts'],
                'comment': '//',
                'class_pattern': r'^class\s+(\w+)',
                'function_pattern': r'function\s+(\w+)'
            }
        }
        
        # Игнорируемые директории
        self.ignore_dirs = {
            'node_modules', '.git', '__pycache__', '.pytest_cache',
            'build', 'dist', 'target', 'out', '.gradle',
            '.idea', '.vscode', 'venv', 'env'
        }
        
        # Игнорируемые файлы
        self.ignore_files = {
            'min.js', 'min.css', 'bundle.js', 'vendor.js',
            '.min.js', '.min.css'
        }
    
    def scan_project(self) -> Dict[str, List[Path]]:
        """
        СКАНИРОВАНИЕ ПРОЕКТА НА ФАЙЛЫ
        
        Returns:
            Словарь {язык: список_файлов}
        """
        language_files = {}
        
        for language, patterns in self.language_patterns.items():
            files = []
            for pattern in patterns['files']:
                # Рекурсивный поиск с игнорированием
                for file_path in self.project_path.rglob(pattern):
                    # Проверка игнорируемых директорий
                    if any(ignored in file_path.parts for ignored in self.ignore_dirs):
                        continue
                    
                    # Проверка игнорируемых файлов
                    if any(ignored in file_path.name for ignored in self.ignore_files):
                        continue
                    
                    files.append(file_path)
            
            if files:
                language_files[language] = files
        
        return language_files
    
    def extract_code_insights(self, file_path: Path, language: str) -> List[Dict[str, Any]]:
        """
        ИЗВЛЕЧЕНИЕ ИНСАЙТОВ ИЗ КОДА
        
        Создает обучающие примеры вида:
        - "Что делает этот метод?"
        - "Как исправить эту ошибку?"
        - "Объясни этот алгоритм"
        """
        insights = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except:
                return insights
        
        # Ограничение размера (берем первые 2000 символов)
        if len(content) > 2000:
            content = content[:2000] + "\n... (обрезано)"
        
        # Получение имени файла без расширения
        file_name = file_path.stem
        
        # Базовые обучающие примеры для любого кода
        base_examples = [
            {
                "instruction": f"Проаналізуй код з файлу {file_path.name} та поясни його функціонал",
                "input": content,
                "output": f"Цей код з файлу {file_path.name} виконує наступні функції..."
            },
            {
                "instruction": f"Як працює цей метод у файлі {file_path.name}?",
                "input": content,
                "output": f"Основний метод у файлі {file_path.name} працює наступним чином..."
            },
            {
                "instruction": f"Знайди потенційні помилки у коді {file_path.name}",
                "input": content,
                "output": f"Аналізуючи код файлу {file_path.name}, я виявив наступні потенційні проблеми..."
            }
        ]
        
        # Специфичные примеры по языку
        if language == 'python':
            insights.extend(self._generate_python_examples(content, file_path))
        elif language == 'java':
            insights.extend(self._generate_java_examples(content, file_path))
        elif language == 'kotlin':
            insights.extend(self._generate_kotlin_examples(content, file_path))
        elif language in ['javascript', 'typescript']:
            insights.extend(self._generate_js_examples(content, file_path))
        
        # Добавляем базовые примеры
        insights.extend(base_examples)
        
        return insights
    
    def _generate_python_examples(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Генерация примеров для Python"""
        examples = []
        
        # Поиск функций
        func_matches = re.finditer(r'^def\s+(\w+)\s*\((.*?)\):', content, re.MULTILINE)
        for match in func_matches:
            func_name = match.group(1)
            params = match.group(2)
            
            # Пример для функции
            examples.append({
                "instruction": f"Поясни функцію {func_name} з параметрами ({params})",
                "input": content[match.start():match.start() + 500],
                "output": f"Функція {func_name} призначена для..."
            })
        
        # Поиск классов
        class_matches = re.finditer(r'^class\s+(\w+)\s*(?:\([^)]*\))?:', content, re.MULTILINE)
        for match in class_matches:
            class_name = match.group(1)
            
            examples.append({
                "instruction": f"Опиши клас {class_name} та його призначення",
                "input": content[match.start():match.start() + 800],
                "output": f"Клас {class_name} є..."
            })
        
        # Django-specific
        if 'django' in content.lower() or 'models.py' in str(file_path):
            examples.append({
                "instruction": f"Поясни Django моделі у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Цей файл містить Django моделі, які представляють..."
            })
        
        # Flask/FastAPI-specific
        if '@app.route' in content or 'FastAPI' in content:
            examples.append({
                "instruction": f"Опиши API endpoints у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Файл містить наступні API endpoints..."
            })
        
        return examples
    
    def _generate_java_examples(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Генерация примеров для Java"""
        examples = []
        
        # Android-specific
        if 'extends AppCompatActivity' in content or 'extends Fragment' in content:
            examples.append({
                "instruction": f"Опиши Android компонент у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Цей файл містить Android компонент, який..."
            })
        
        # Spring-specific
        if '@SpringBootApplication' in content or '@RestController' in content:
            examples.append({
                "instruction": f"Поясни Spring Boot компоненти у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Файл містить Spring Boot компоненти..."
            })
        
        # Поиск методов
        method_matches = re.finditer(r'public\s+(?:static\s+)?\w+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+)?\s*{', content)
        for match in method_matches:
            method_name = match.group(1)
            
            examples.append({
                "instruction": f"Опиши метод {method_name} та його логіку",
                "input": content[match.start():match.start() + 600],
                "output": f"Метод {method_name} виконує..."
            })
        
        return examples
    
    def _generate_kotlin_examples(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Генерация примеров для Kotlin (Android)"""
        examples = []
        
        # Android-specific
        if 'setContentView' in content or 'findViewById' in content:
            examples.append({
                "instruction": f"Поясни Android View логіку у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Файл містить логіку роботи з Android View..."
            })
        
        # ViewModel-specific
        if ': ViewModel()' in content or 'LiveData' in content:
            examples.append({
                "instruction": f"Опиши ViewModel та LiveData у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Файл містить ViewModel для управління даними..."
            })
        
        # Compose-specific
        if '@Composable' in content:
            examples.append({
                "instruction": f"Поясни Jetpack Compose функції у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Файл містить Jetpack Compose функції..."
            })
        
        return examples
    
    def _generate_js_examples(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Генерация примеров для JavaScript/TypeScript"""
        examples = []
        
        # React-specific
        if 'useState' in content or 'useEffect' in content:
            examples.append({
                "instruction": f"Опиши React hooks у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Файл використовує React hooks для..."
            })
        
        # Node.js-specific
        if 'require(' in content or 'import' in content:
            examples.append({
                "instruction": f"Поясни модулі та імпорти у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Файл містить наступні модулі та імпорти..."
            })
        
        # Express-specific
        if 'app.get(' in content or 'router.' in content:
            examples.append({
                "instruction": f"Опиши API endpoints у файлі {file_path.name}",
                "input": content[:1000],
                "output": "Файл містить наступні API endpoints..."
            })
        
        return examples
    
    def create_training_chunks(self) -> List[ProcessedChunk]:
        """
        СОЗДАНИЕ ЧАНКОВ ДЛЯ ОБУЧЕНИЯ
        
        Конвертирует примеры в ProcessedChunk для ModelTrainer
        """
        chunks = []
        
        # Сканирование проекта
        language_files = self.scan_project()
        
        for language, files in language_files.items():
            log(f"Обработка {len(files)} {language} файлов...")
            
            for file_path in files:
                # Генерация примеров для файла
                examples = self.extract_code_insights(file_path, language)
                
                for idx, example in enumerate(examples):
                    # Создание чанка
                    chunk = ProcessedChunk(
                        content=example["input"],
                        metadata={
                            "source_file": str(file_path),
                            "file_name": file_path.name,
                            "file_ext": file_path.suffix,
                            "language": language,
                            "example_type": example["instruction"][:50],
                            "example_index": idx,
                            "document_type": "code",
                            "processing_timestamp": "auto",
                            "confidence": 0.95
                        }
                    )
                    
                    # Добавление ответа в метаданные (для обучения)
                    chunk.metadata["expected_output"] = example["output"]
                    chunk.metadata["instruction"] = example["instruction"]
                    
                    chunks.append(chunk)
        
        return chunks
    
    def save_dataset(self, output_path: str):
        """
        СОХРАНЕНИЕ ДАТАСЕТА В ФАЙЛ
        
        Args:
            output_path: Путь для сохранения JSON файла
        """
        chunks = self.create_training_chunks()
        
        dataset = []
        for chunk in chunks:
            # Конвертация в формат для обучения
            dataset.append({
                "instruction": chunk.metadata["instruction"],
                "input": chunk.content,
                "output": chunk.metadata["expected_output"],
                "metadata": chunk.metadata
            })
        
        # Сохранение в JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        log(f"✅ Датасет сохранен: {output_path}")
        log(f"   Количество примеров: {len(dataset)}")
        log(f"   Языки: {set(chunk.metadata['language'] for chunk in chunks)}")

def main():
    """
    ГЛАВНАЯ ФУНКЦИЯ
    
    Использование:
        python prepare_code_dataset.py /path/to/project [output.json]
    """
    if len(sys.argv) < 2:
        print("Использование: python prepare_code_dataset.py /path/to/project [output.json]")
        sys.exit(1)
    
    project_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "code_dataset.json"
    
    # Создание preparer
    from src.agent import AIAssistant
    assistant = AIAssistant()  # Для доступа к ассистенту
    
    preparer = CodeDatasetPreparer(project_path)
    
    # Подготовка и сохранение датасета
    preparer.save_dataset(output_path)
    
    print(f"\n✅ Готово! Используйте для дообучения:")
    print(f"   1. Скопируйте {output_path} в data/documents/")
    print(f"   2. Запустите дообучение: /train")
    print(f"   3. Или добавьте в проект через /add {output_path}")

if __name__ == "__main__":
    main()