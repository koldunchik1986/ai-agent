# ===================================================================
# DOCKERFILE ДЛЯ AI-АССИСТЕНТА (P104-100 8GB VRAM, sm_61 Pascal)
# ===================================================================
# Базовый образ: CUDA 11.8 Runtime (не devel для экономии места)
# Оптимизации:
# - Python 3.11 (Kali Linux 2025)
# - NVIDIA Container Toolkit (новый метод подписи)
# - sm_61 архитектура CUDA
# - Непривилегированный пользователь (безопасность)
# - Оптимизация памяти через переменные окружения

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Избегаем интерактивных запросов при установке
ENV DEBIAN_FRONTEND=noninteractive

# ===================================================================
# СИСТЕМНЫЕ ЗАВИСИМОСТИ
# ===================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11 (Kali 2025 использует 3.11)
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3-wheel \
    \
    # Системные инструменты
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    \
    # Для обработки документов
    poppler-utils \                    # PDF extraction
    tesseract-ocr \                    # OCR
    tesseract-ocr-ukr \                # Ukrainian language
    libxml2-dev \                      # XML/HTML парсинг
    libxslt1-dev \
    antiword \                         # DOC conversion
    \
    # Очистка кэша для уменьшения размера образа
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ===================================================================
# УСТАНОВКА PIP ДЛЯ PYTHON 3.11
# ===================================================================
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ===================================================================
# СОЗДАНИЕ ВИРТУАЛЬНОГО ОКРУЖЕНИЯ
# ===================================================================
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ===================================================================
# НАСТРОЙКИ CUDA И GPU
# ===================================================================
# ✅ АРХИТЕКТУРА sm_61 (Pascal) для PyTorch
# Включает современные архитектуры для совместимости
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# ✅ ЯВНОЕ УКАЗАНИЕ GPU (предотвращает ошибки)
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ===================================================================
# ОПТИМИЗАЦИИ ПАМЯТИ (КРИТИЧНО ДЛЯ 8GB)
# ===================================================================
# PyTorch CUDA allocator config
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6"

# OpenMP (запрет многопоточности для экономии памяти)
ENV OMP_NUM_THREADS=1

# Tokenizers (отключаем parallelism)
ENV TOKENIZERS_PARALLELISM=false

# Python
ENV PYTHONUNBUFFERED=1

# HuggingFace cache directory
ENV HF_HUB_CACHE=/app/data/models

# ===================================================================
# СОЗДАНИЕ ПОЛЬЗОВАТЕЛЯ (БЕЗОПАСНОСТЬ)
# ===================================================================
# Не запускаем от root
RUN groupadd -r aiuser && useradd -r -g aiuser -u 1000 aiuser

# ===================================================================
# УСТАНОВКА PYTHON ЗАВИСИМОСТЕЙ
# ===================================================================
WORKDIR /app

# Копирование requirements
COPY requirements.txt .

# ✅ Установка PyTorch с поддержкой sm_61 CUDA 11.8
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Установка PyTorch (сначала, т.к. самый тяжелый пакет)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118

# Установка остальных зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# ===================================================================
# КОПИРОВАНИЕ КОДА ПРИЛОЖЕНИЯ
# ===================================================================
COPY src/ ./src/

# ===================================================================
# СОЗДАНИЕ ДИРЕКТОРИЙ КОНТЕЙНЕРА
# ===================================================================
# С правильными правами для непривилегированного пользователя
RUN mkdir -p /app/data/{documents,models,chroma,cache,logs,vscode_projects} && \
    chown -R aiuser:aiuser /app/data && \
    chmod -R 755 /app/data

# ===================================================================
# НАСТРОЙКИ БЕЗОПАСНОСТИ
# ===================================================================
# Запуск от пользователя aiuser (не root)
USER aiuser

# Монтирование volume для данных (не кодируются в образ)
VOLUME ["/app/data"]

# Health check (проверка работоспособности)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3.11 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

# ===================================================================
# КОМАНДА ЗАПУСКА
# ===================================================================
CMD ["python3.11", "-m", "src.cli"]