"""
Конфигурационный файл AI-агента
Настройки для Mistral AI 7B, GPU, путей и параметров обучения
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    """Конфигурация модели Mistral AI 7B"""
    model_name: str = "/home/ai-agent/models/Mistral-7B-Instruct-v0.3"
    cache_dir: str = "/home/ai-agent/models"
    device: str = "cuda"
    torch_dtype: str = "float16"
    load_in_8bit: bool = True  # Для 8GB GPU
    load_in_4bit: bool = False # Для 4GB GPU
    trust_remote_code: bool = True
    use_cache: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

@dataclass
class TrainingConfig:
    """Конфигурация обучения модели"""
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    fp16: bool = True  # Для GPU

    # LoRA параметры для эффективного обучения
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

@dataclass
class DataConfig:
    """Конфигурация данных"""
    documents_path: str = "/home/ai-agent/documents"
    cache_path: str = "/home/ai-agent/cache"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50

    # Поддерживаемые форматы
    supported_formats: List[str] = None

    # Специализация документов
    programming_extensions: List[str] = None
    legal_extensions: List[str] = None

@dataclass
class VectorConfig:
    """Конфигурация векторной базы данных"""
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    collection_name: str = "ai-agent-collection"
    persist_directory: str = "/home/ai-agent/chroma"

@dataclass
class KnowledgeGraphConfig:
    """Конфигурация Neo4j Knowledge Graph"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

@dataclass
class CLIConfig:
    """Конфигурация CLI интерфейса"""
    max_history: int = 100
    show_thinking: bool = False
    auto_save_context: bool = True
    context_file: str = "/home/ai-agent/cache/context.json"

class Config:
    """Основной класс конфигурации"""

    def __init__(self):
        # Установка путей из переменных окружения
        self.setup_paths()

        # Инициализация конфигураций
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.vector = VectorConfig()
        self.knowledge_graph = KnowledgeGraphConfig()
        self.cli = CLIConfig()

        # Настройка специфических параметров
        self.setup_specific_configs()

    def setup_paths(self):
        """Установка путей из переменных окружения"""
        base_path = os.getenv("AGENT_HOME", "/home/ai-agent")

        # Обновление путей в конфигурациях
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_path, "models")
        os.environ["HF_HOME"] = os.path.join(base_path, "models")
        os.environ["XDG_CACHE_HOME"] = os.path.join(base_path, "cache")

    def setup_specific_configs(self):
        """Настройка специфических параметров"""
        # Поддерживаемые форматы документов
        self.data.supported_formats = [".pdf", ".doc", ".docx", ".txt", ".md", ".py", ".java", ".kt", ".js", ".html", ".css"]

        # Программные файлы
        self.data.programming_extensions = [".py", ".java", ".kt", ".js", ".ts", ".cpp", ".c", ".h", ".cs", ".go", ".rs", ".php", ".rb", ".swift"]

        # Юридические документы
        self.data.legal_extensions = [".doc", ".docx", ".pdf", ".rtf"]

        # LoRA целевые модули для Mistral
        self.training.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
#            "lm_head"
        ]

    def update_from_env(self):
        """Обновление конфигурации из переменных окружения"""
        # Model config
        if os.getenv("MODEL_NAME"):
            self.model.model_name = os.getenv("MODEL_NAME")
        if os.getenv("DEVICE"):
            self.model.device = os.getenv("DEVICE")

        # Data paths
        if os.getenv("DOCUMENT_PATH"):
            self.data.documents_path = os.getenv("DOCUMENT_PATH")
        if os.getenv("CACHE_PATH"):
            self.data.cache_path = os.getenv("CACHE_PATH")

        # Neo4j config
        if os.getenv("NEO4J_URI"):
            self.knowledge_graph.uri = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USER"):
            self.knowledge_graph.user = os.getenv("NEO4J_USER")
        if os.getenv("NEO4J_PASSWORD"):
            self.knowledge_graph.password = os.getenv("NEO4J_PASSWORD")

        # Chroma config
        if os.getenv("CHROMA_HOST"):
            self.vector.chroma_host = os.getenv("CHROMA_HOST")
        if os.getenv("CHROMA_PORT"):
            self.vector.chroma_port = int(os.getenv("CHROMA_PORT"))

    def ensure_directories(self):
        """Создание необходимых директорий"""
        directories = [
            self.model.cache_dir,
            self.data.documents_path,
            self.data.cache_path,
            self.vector.persist_directory,
            Path(self.cli.context_file).parent
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def validate_gpu_config(self):
        """Валидация конфигурации GPU"""
        if self.model.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("Warning: CUDA not available, switching to CPU")
                    self.model.device = "cpu"
                    self.model.load_in_8bit = False
                else:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"GPU Memory: {gpu_memory:.1f}GB")
                    if gpu_memory < 4:
                        print("Warning: GPU memory less than 8GB, enabling 4-bit quantization")
                        self.model.load_in_4bit = True
                        self.model.load_in_8bit = False
            except ImportError:
                print("Warning: PyTorch not available, switching to CPU")
                self.model.device = "cpu"
                self.model.load_in_8bit = False

# Глобальный экземпляр конфигурации
config = Config()
config.update_from_env()
config.ensure_directories()
config.validate_gpu_config()
