FROM nvidia/cuda:11.8-devel-ubuntu22.04
 
# Установка системных зависимостей
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
 
# Создаем виртуальное окружение
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.10/site-packages"
 
# Копируем зависимости и устанавливаем их
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
 
# Устанавливаем переменные окружения для CUDA
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.1"  # P104-100 architecture
ENV PYTHONUNBUFFERED=1
 
# Создаем рабочую директорию
WORKDIR /workspace
 
# Копируем исходный код
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
 
# Создаем директории для данных с правильными правами
RUN mkdir -p /home/ai-agent/{models,documents,cache,logs} && \
    chmod 755 /home/ai-agent && \
    chown -R 1000:1000 /home/ai-agent
 
# Устанавливаем переменные для путей
ENV AGENT_HOME="/home/ai-agent"
ENV MODEL_CACHE_PATH="/home/ai-agent/models"
ENV DOCUMENT_PATH="/home/ai-agent/documents"
ENV CACHE_PATH="/home/ai-agent/cache"
 
# Открываем порт для potential web interface
EXPOSE 8000
 
# Команда запуска
CMD ["python", "-m", "src.cli_interface"]