"""
МОДУЛЬ ДООБУЧЕНИЯ МОДЕЛИ ЧЕРЕЗ LoRA

Особенности для 8GB VRAM:
- 8-bit квантизация модели при загрузке
- Paged AdamW 8-bit оптимизатор
- Batch size = 1 с gradient accumulation
- LoRA без lm_head (экономия 500MB)
- Очистка кэша после каждого батча
"""

import os
import json
import torch
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from datasets import Dataset

from config import config
from document_processor import ProcessedChunk

# прогресс-бар для дообучения
from tqdm import tqdm
for epoch in tqdm(range(epochs), desc="Обучение"):

class ModelTrainer:
    """
    КЛАСС ДЛЯ ДООБУЧЕНИЯ MISTRAL-7B ЧЕРЕЗ LoRA
    
    Основные шаги:
    1. Загрузка модели с 8-bit квантизацией
    2. Настройка LoRA адаптера
    3. Подготовка обучающего датасета
    4. Запуск fine-tuning
    5. Сохранение модели и метаданных
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.peft_config = None
        
        # Инициализация токенизатора (легковесная операция)
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Инициализация токенизатора"""
        print("🔄 Загрузка токенизатора...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            cache_dir=config.model.cache_dir,
            trust_remote_code=config.model.trust_remote_code
        )
        
        # Установка pad_token если отсутствует
        # Необходимо для batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("✅ Токенизатор загружен")
    
    def load_model_for_training(self):
        """
        ЗАГРУЗКА МОДЕЛИ С 8-BIT КВАНТИЗАЦИЕЙ
        
        Этот метод критичен для 8GB VRAM:
        - Использует BitsAndBytesConfig для 8-bit загрузки
        - Явно указывает device_map на GPU 0
        - Подготавливает модель для k-bit обучения
        
        VRAM usage: ~6.5GB из 8GB
        """
        print("\n🔄 Загрузка модели с 8-bit квантизацией...")
        
        # Очистка памяти перед загрузкой (важно!)
        torch.cuda.empty_cache()
        
        # ✅ КОНФИГУРАЦИЯ 8-BIT КВАНТИЗАЦИИ
        # Это позволяет загрузить 14GB модель в 8GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,           # Включить 8-bit квантизацию
            load_in_4bit=False,          # Отключить 4-bit (качество хуже)
            bnb_4bit_compute_dtype=torch.float16,  # Для совместимости
        )
        
        # ✅ ЯВНОЕ НАЗНАЧЕНИЕ НА GPU 0
        # Предотвращает неправильное распределение памяти
        device_map = {"": 0}
        
        # Загрузка модели
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            cache_dir=config.model.cache_dir,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,  # Mixed precision
            trust_remote_code=config.model.trust_remote_code,
        )
        
        # ✅ ПОДГОТОВКА ДЛЯ K-BIT ОБУЧЕНИНГА
        # Обязательный шаг перед добавлением LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Вывод информации о памяти
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"✅ Модель загружена. Использовано VRAM: {vram_used:.2f}/{VRAM_LIMIT_GB}GB")
        
        if vram_used > SAFE_VRAM_USAGE_GB:
            print(f"⚠️ ВНИМАНИЕ: Использовано >{SAFE_VRAM_USAGE_GB}GB! ООМ возможен при обучении.")
    
    def setup_lora(self):
        """
        НАСТРОЙКА LoRA АДАПТЕРА
        
        LoRA (Low-Rank Adaptation) позволяет дообучать модель
        НЕ трогая все 7B параметров, а только небольшие адаптеры
        
        Экономия: Вместо 14GB для полного fine-tuning используем ~500MB
        
        Параметры:
        - lora_r=16: Ранг матрицы (чем больше, тем выше качество, больше памяти)
        - lora_alpha=32: Масштабирование (обычно lora_r * 2)
        - target_modules: Какие слои модели дообучать
        """
        print("\n🎯 Настройка LoRA адаптера...")
        
        # ✅ КОНФИГУРАЦИЯ LORA
        # task_type=CAUSAL_LM для генеративных моделей
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.training.lora_r,                    # 16 для 8GB
            lora_alpha=config.training.lora_alpha,       # 32
            lora_dropout=config.training.lora_dropout,   # 0.1
            target_modules=config.training.lora_target_modules,
            # Целевые модули без lm_head (экономия памяти)
            bias="none",  # Не трогаем bias параметры
        )
        
        # Применение LoRA к модели
        self.model = get_peft_model(self.model, self.peft_config)
        
        # Информация о обучаемых параметрах
        # trainable params: 16,777,216 || all params: 7,110,350,848
        self.model.print_trainable_parameters()
    
    def format_training_data(self, chunks: List[ProcessedChunk]) -> Dataset:
        """
        ФОРМАТИРОВАНИЕ ДАННЫХ ДЛЯ ОБУЧЕНИЯ
        
        Конвертирует ProcessedChunk в формат, понятный HuggingFace Trainer
        
        Args:
            chunks: Список обработанных чанков
            
        Returns:
            Dataset готовый к обучению
        """
        if not chunks:
            raise ValueError("Нет данных для обучения")
        
        print(f"\n📊 Форматирование {len(chunks)} чанков...")
        
        # Функция для создания training example
        def create_instruction(chunk: ProcessedChunk) -> str:
            """
            Создание инструктивного промпта в формате Mistral
            
            Формат: <s>[INST] Инструкция [/INST] Ответ</s>
            """
            # Определение типа файла для контекста
            file_ext = chunk.metadata.get("file_ext", "документ")
            file_name = chunk.metadata.get("file_name", "файл")
            
            # Создание инструкции в зависимости от типа файла
            if file_ext in ['.py', '.java', '.kt', '.js']:
                # Для кода
                instruction = (
                    f"Проаналізуй цей код з файлу {file_name} та поясни його функціонал:\n\n"
                    f"{chunk.content[:800]}"  # Ограничение длины
                )
            elif file_ext in ['.docx', '.pdf']:
                # Для документации
                instruction = (
                    f"Використовуючи інформацію з документа {file_name}, відповідай на питання:\n\n"
                    f"Зміст: {chunk.content[:500]}\n\n"
                    f"Питання:"
                )
            else:
                # Для общих файлов
                instruction = (
                    f"Використовуючи контекст з файлу {file_name}:\n\n"
                    f"{chunk.content[:600]}"
                )
            
            # Форматирование в стиль Mistral
            return f"<s>[INST] {instruction.strip()} [/INST]"
        
        # Форматирование всех чанков
        formatted_data = {"text": []}
        for chunk in chunks:
            formatted_data["text"].append(create_instruction(chunk))
        
        # Создание HuggingFace Dataset
        dataset = Dataset.from_dict(formatted_data)
        
        print(f"✅ Создано {len(dataset)} обучающих примеров")
        
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        ТОКЕНИЗАЦИЯ ДАТАСЕТА
        
        Конвертирует текст в токены (числовые ID)
        
        Args:
            dataset: Датасет с текстом
            
        Returns:
            Токенизированный датасет
        """
        print("\n🔄 Токенизация датасета...")
        
        def tokenize_function(examples):
            """
            Функция токенизации для batched processing
            
            Важно: truncation=True - отсекает длинные последовательности
                   padding=False - не добавляем padding (сделает collator)
            """
            return self.tokenizer(
                examples["text"],
                truncation=True,  # Обязательно! Иначе ошибка при длинных текстах
                padding=False,      # Не добавляем padding здесь
                max_length=config.model.max_new_tokens,
                return_overflowing_tokens=False,
            )
        
        # Применение токенизации (batched для скорости)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=10  # Небольшие батчи для экономии памяти
        )
        
        print(f"✅ Токенизировано {len(tokenized_dataset)} примеров")
        
        return tokenized_dataset
    
    def prepare_training_data(self, chunks: List[ProcessedChunk]) -> tuple[Dataset, Dataset]:
        """
        ПОДГОТОВКА ОБУЧАЮЩИХ ДАННЫХ
        
        Полный пайплайн:
        1. Форматирование
        2. Токенизация
        3. Разделение на train/validation
        
        Args:
            chunks: Список ProcessedChunk
            
        Returns:
            (train_dataset, eval_dataset)
        """
        if not chunks:
            raise ValueError("❌ Нет данных для обучения")
        
        print(f"\n📚 Подготовка {len(chunks)} чанков для обучения...")
        
        # 1. Форматирование
        dataset = self.format_training_data(chunks)
        
        # 2. Токенизация
        tokenized = self.tokenize_dataset(dataset)
        
        # 3. Разделение на train/validation (90/10)
        # stratify=False - нет категориальных меток
        split = tokenized.train_test_split(
            test_size=0.1,  # 10% на валидацию
            seed=42,        # Фиксируем random seed
            stratify=False
        )
        
        train_dataset = split["train"]
        eval_dataset = split["test"]
        
        print(f"✅ Датасет разделен: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        return train_dataset, eval_dataset
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Dataset) -> Trainer:
        """
        СОЗДАНИЕ ТРЕНЕРА
        
        Trainer управляет циклом обучения:
        - Forward pass
        - Loss calculation
        - Backward pass
        - Optimizer step
        - Logging & saving
        
        Args:
            train_dataset: Датасет для обучения
            eval_dataset: Датасет для валидации
            
        Returns:
            Trainer готовый к запуску
        """
        print("\n🎯 Создание Trainer...")
        
        # ✅ НАСТРОЙКИ ОБУЧЕНИЯ
        # Сохраняются в директорию training_output
        training_args = TrainingArguments(
            # Директория для чекпоинтов
            output_dir=f"{config.model.cache_dir}/training_output",
            
            # Количество эпох
            num_train_epochs=config.training.num_train_epochs,
            
            # ✅ ПАРАМЕТРЫ БАТЧЕЙ (КРИТИЧНО)
            per_device_train_batch_size=config.training.per_device_train_batch_size,  # 1 для 8GB
            per_device_eval_batch_size=config.training.per_device_eval_batch_size,    # 1
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,  # 8
            
            # Скорость обучения
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            
            # Оптимизатор (8-bit для экономии памяти)
            optim=config.training.optim,
            
            # Расписание скорости
            lr_scheduler_type=config.training.lr_scheduler_type,
            
            # Шаги разогрева
            warmup_steps=config.training.warmup_steps,
            
            # Mixed precision (ускорение на GPU)
            fp16=config.training.fp16,
            
            # Логирование и сохранение
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            evaluation_strategy="steps",
            eval_steps=config.training.eval_steps,
            save_total_limit=config.training.save_total_limit,
            
            # ✅ ОТКЛЮЧЕНИЕ ВЕЗБ/TensorBoard (нет доступа в Docker)
            report_to=None,
            
            # ✅ ЭКОНОМИЯ ПАМЯТИ
            dataloader_drop_last=True,      # Отбрасываем неполный батч
            dataloader_pin_memory=False,    # Отключаем pin_memory (экономия)
            remove_unused_columns=False,    # Не удаляем колонки
            load_best_model_at_end=True,    # Загрузить лучшую модель
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data Collator
        # Автоматически добавляет padding к батчам
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM (не masked)
        )
        
        # ✅ СОЗДАНИЕ TRAINER
        # Основной объект для обучения
        trainer = Trainer(
            model=self.model,              # Модель с LoRA
            args=training_args,            # Параметры обучения
            train_dataset=train_dataset,   # Тренировочные данные
            eval_dataset=eval_dataset,     # Валидационные данные
            data_collator=data_collator,   # Collator для padding
            tokenizer=self.tokenizer,      # Для сохранения конфига
        )
        
        print("✅ Trainer создан")
        
        return trainer
    
    def train(self, chunks: List[ProcessedChunk], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        ОСНОВНОЙ МЕТОД ОБУЧЕНИЯ
        
        Args:
            chunks: Список чанков для обучения
            output_dir: Директория для сохранения (автоматически если None)
            
        Returns:
            Словарь с метаданными обучения
        """
        if not chunks:
            raise ValueError("❌ Нет данных для обучения")
        
        print("\n" + "="*60)
        print("🚀 НАЧАЛО ОБУЧЕНИЯ LoRA")
        print("="*60)
        
        start_time = datetime.now()
        
        # ✅ 1. Подготовка данных
        train_dataset, eval_dataset = self.prepare_training_data(chunks)
        
        # ✅ 2. Загрузка модели
        self.load_model_for_training()
        
        # ✅ 3. Настройка LoRA
        self.setup_lora()
        
        # ✅ 4. Создание Trainer
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        # ✅ 5. ЗАПУСК ОБУЧЕНИЯ
        print("\n🎯 Начинается fine-tuning...")
        print(f"Примеров: {len(train_dataset)} train, {len(eval_dataset)} eval")
        print(f"Эпохи: {config.training.num_train_epochs}")
        print(f"Сохранение в: {output_dir or 'auto'}\n")
        
        # Запуск обучения
        train_result = trainer.train()
        
        # ✅ 6. ОЦЕНКА
        print("\n📊 Оценка модели...")
        eval_result = trainer.evaluate()
        
        # ✅ 7. СОХРАНЕНИЕ
        if output_dir is None:
            # Автоматическая генерация имени
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{config.model.cache_dir}/lora_finetuned_{timestamp}"
        
        print(f"\n💾 Сохранение модели в {output_dir}...")
        
        # Сохранение LoRA адаптера
        trainer.model.save_pretrained(output_dir)
        
        # Сохранение токенизатора
        self.tokenizer.save_pretrained(output_dir)
        
        # ✅ 8. МЕТАДАННЫЕ
        metadata = {
            "training_timestamp": datetime.now().isoformat(),
            "model_name": config.model.model_name,
            "gpu_used": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "training_duration": str(datetime.now() - start_time),
            "output_dir": output_dir,
            "parameters": {
                "epochs": config.training.num_train_epochs,
                "learning_rate": config.training.learning_rate,
                "batch_size": config.training.per_device_train_batch_size,
                "gradient_accumulation": config.training.gradient_accumulation_steps,
                "lora_r": config.training.lora_r,
            },
            "dataset": {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "total_chunks": len(chunks),
            },
            "results": {
                "train_loss": float(train_result.training_loss) if train_result.training_loss else None,
                "eval_loss": eval_result.get("eval_loss"),
                "train_runtime": train_result.training_time,
            }
        }
        
        # Сохранение метаданных в JSON
        metadata_path = os.path.join(output_dir, "training_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # ✅ 9. ОЧИСТКА ПАМЯТИ
        # Освобождение GPU памяти
        del self.model
        torch.cuda.empty_cache()
        
        duration = datetime.now() - start_time
        
        print("\n" + "="*60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("="*60)
        print(f"⏱️  Длительность: {duration}")
        print(f"📁 Сохранено в: {output_dir}")
        print(f"📉 Потери: train={metadata['results']['train_loss']:.4f}, eval={metadata['results']['eval_loss']:.4f}")
        
        return metadata
    
    def load_finetuned_model(self, model_path: str):
        """
        ЗАГРУЗКА ДООБУЧЕННОЙ МОДЕЛИ
        
        Args:
            model_path: Путь к директории с LoRA адаптером
            
        Returns:
            Модель с примененным LoRA
        """
        print(f"\n🔄 Загрузка дообученной модели из {model_path}...")
        
        # Очистка памяти
        torch.cuda.empty_cache()
        
        # Загрузка базовой модели
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            cache_dir=config.model.cache_dir,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
        
        # Загрузка LoRA адаптеров
        # Модель = base + LoRA weights
        self.model = get_peft_model(base_model, model_path)
        
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("✅ Дообученная модель загружена")
        
        return self.model