#!/bin/bash

# ===================================================================
# УСТАНОВКА AI-АССИСТЕНТА НА KALI LINUX 2025
# ===================================================================
# Этот скрипт:
# 1. Проверяет GPU P104-100 и CUDA
# 2. Устанавливает Docker и NVIDIA Container Toolkit (НОВЫЙ МЕТОД)
# 3. Настраивает Python 3.11 и зависимости
# 4. Создает структуру проекта
# 5. Настраивает переменные окружения

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Логирование
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# ===================================================================
# НАЧАЛО УСТАНОВКИ
# ===================================================================
echo -e "${BLUE}"
echo "====================================================="
echo "  УСТАНОВКА AI-АССИСТЕНТА ДЛЯ P104-100 8GB VRAM"
echo "  Kali Linux 2025 | CUDA 11.8 | sm_61 Pascal"
echo "====================================================="
echo -e "${NC}"

log "Початок установки: $(date)"

# ===================================================================
# 1. ПРОВЕРКА ПРАВ
# ===================================================================
if [[ $EUID -eq 0 ]]; then
    error "НЕ запускайте від root! Використовуйте: ./setup.sh"
fi

# ===================================================================
# 2. ПРОВЕРКА GPU И CUDA
# ===================================================================
log "🔍 Перевірка GPU..."

if ! command -v nvidia-smi &> /dev/null; then
    error """NVIDIA драйвер не встановлено!
    
Установіть драйвери:
1. Зайдіть на сайт NVIDIA
2. Завантажте драйвер для P104-100
3. Встановіть: sudo apt install ./nvidia-driver-*.deb
4. Перезавантажте: sudo reboot
"""
fi

# Получение информации о GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*")

log "✅ GPU: ${GPU_NAME}"
log "✅ CUDA: ${CUDA_VERSION}"

# Проверка на P104-100
if [[ ! "$GPU_NAME" =~ "P104-100" ]]; then
    warn "Виявлено GPU: ${GPU_NAME}. Скрипт оптимізовано для P104-100."
    read -p "Продовжити? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ===================================================================
# 3. УСТАНОВКА PYTHON 3.11
# ===================================================================
log "🐍 Перевірка Python 3.11..."

if ! command -v python3.11 &> /dev/null; then
    log "Встановлення Python 3.11..."
    sudo apt update
    sudo apt install -y python3.11 python3.11-dev python3.11-venv python3-pip
else
    log "✅ Python 3.11 вже встановлено"
fi

PYTHON_VER=$(python3.11 --version)
log "✅ ${PYTHON_VER}"

# ===================================================================
# 4. УСТАНОВКА DOCKER (НОВЫЙ МЕТОД ДЛЯ KALI 2025)
# ===================================================================
log "🐳 Перевірка Docker..."

if ! command -v docker &> /dev/null; then
    log "Встановлення Docker (новий метод підпису репозиторіїв)..."
    
    # Видалення старих версій
    sudo apt purge -y docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc 2>/dev/null || true
    
    # Встановлення залежностей
    sudo apt update
    sudo apt install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # ✅ НОВЫЙ МЕТОД: Добавлення ключа через gpg --dearmor
    # (без застарілого apt-key)
    sudo rm -f /usr/share/keyrings/docker-archive-keyring.gpg
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # ✅ НОВЫЙ МЕТОД: Добавлення репозиторію з signed-by
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian bookworm stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Добавлення користувача до групи docker
    sudo usermod -aG docker $USER
    log "✅ Docker встановлено. Користувач додано до групи docker"
    log "⚠️  ПЕРЕЛОГІНТЕСЬ або виконайте: newgrp docker"
else
    log "✅ Docker вже встановлено"
fi

# ===================================================================
# 5. УСТАНОВКА NVIDIA CONTAINER TOOLKIT
# ===================================================================
log "🎮 Перевірка NVIDIA Container Toolkit..."

if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log "Встановлення NVIDIA Container Toolkit (новий метод)..."
    
    # Видалення старих пакетів
    sudo apt purge -y nvidia-docker2 nvidia-container-runtime nvidia-container-toolkit 2>/dev/null || true
    
    # ✅ НОВЫЙ МЕТОД: Добавлення репозиторію nvidia-container-toolkit
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    
    # Настройка Docker runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    log "✅ NVIDIA Container Toolkit встановлено"
else
    log "✅ NVIDIA Container Toolkit вже встановлено"
fi

# ===================================================================
# 6. ПРОВЕРКА GPU В DOCKER
# ===================================================================
log "🔍 Перевірка GPU у Docker..."
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log "✅ GPU доступний у Docker"
else
    error "❌ GPU не доступний у Docker! Перевірте драйвери та NVIDIA Container Toolkit."
fi

# ===================================================================
# 7. СОЗДАНИЕ СТРУКТУРЫ ПРОЕКТА
# ===================================================================
log "📁 Створення структури проекту..."
mkdir -p data/{documents,models,chroma,cache,logs,vscode_projects}
mkdir -p src

log "✅ Директорії створено:
    - data/documents (ваші документи)
    - data/models (кеш моделей)
    - data/chroma (векторна база)
    - data/cache (временні файли)"

# ===================================================================
# 8. НАСТРОЙКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ===================================================================
log "🔧 Створення .env файлу..."
cat > .env << EOF
# AI-Ассистент Конфігурація
AGENT_HOME=$(pwd)/data
MODEL_CACHE_PATH=$(pwd)/data/models
DOCUMENT_PATH=$(pwd)/data/documents
CACHE_PATH=$(pwd)/data/cache
VSCODE_PROJECTS_PATH=$(pwd)/data/vscode_projects

# GPU Налаштування
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6

# IDE Інтеграція
MAX_FILE_TOKENS=1500
CODE_TEMPERATURE=0.3
EOF
log "✅ .env файл створено"

# ===================================================================
# 9. УСТАНОВКА PYTHON ЗАВИСИМОСТЕЙ (ОПЦИОНАЛЬНО)
# ===================================================================
log "🐍 Створення Python venv для розробки..."
python3.11 -m venv venv

# Активация venv
source venv/bin/activate

# Установка PyTorch для sm_61
log "Встановлення PyTorch для CUDA 11.8 sm_61..."
pip install --upgrade pip wheel
pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118

# Установка остальных зависимостей
log "Встановлення залежностей..."
pip install --no-cache-dir -r requirements.txt

log "✅ Python середовище готове!"

# ===================================================================
# 10. НАСТРОЙКА ПРАВ ДОСТУПА
# ===================================================================
log "🔐 Налаштування прав доступу..."
sudo chown -R $USER:$USER data/
chmod -R 755 data/

# ===================================================================
# 11. ГЕНЕРАЦИЯ СКРИПТОВ
# ===================================================================
log "🔧 Створення виконуваних скриптів..."

# Делаем скрипты исполняемыми
chmod +x scripts/*.sh

log "✅ Скрипти готові:
    - ./scripts/run.sh (керування сервісами)
    - ./scripts/backup.sh (резервне копіювання)"

# ===================================================================
# ЗАВЕРШЕНИЕ УСТАНОВКИ
# ===================================================================
echo -e "${GREEN}"
echo "====================================================="
echo "  ✅ УСТАНОВКУ ЗАВЕРШЕНО!"
echo "====================================================="
echo -e "${NC}"

echo ""
log "📋 НАСТУПНІ КРОКИ:"

echo -e """
1. ${GREEN}Перелогінтесь${NC} або виконайте для застосування групи docker:
   ${BLUE}newgrp docker${NC}

2. ${GREEN}Запустіть ассистента${NC} (рекомендовано через Docker):
   ${BLUE}./scripts/run.sh start${NC}
   ${BLUE}./scripts/run.sh cli${NC}

3. ${GREEN}Альтернатива${NC} (локальний запуск):
   ${BLUE}source venv/bin/activate${NC}
   ${BLUE}python -m src.cli${NC}

4. ${GREEN}Додайте документи${NC}:
   - Скопіюйте файли до ${YELLOW}data/documents/${NC}
   - Або в CLI: ${BLUE}/add /app/data/documents/file.pdf${NC}

5. ${GREEN}Додайте проект${NC}:
   - В CLI: ${BLUE}/project /workspace/your-project${NC}

6. ${GREEN}Дообучення${NC}:
   - В CLI: ${BLUE}/train${NC}
   - Статус: ${BLUE}./scripts/run.sh status${NC}

7. ${GREEN}Перевірка GPU${NC}:
   ${BLUE}docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi${NC}
"""

echo -e "${BLUE}📁 Важливі директорії:${NC}"
echo "   📄 Документи: $(pwd)/data/documents"
echo "   🧠 Моделі: $(pwd)/data/models"
echo "   🔍 Векторна БД: $(pwd)/data/chroma"
echo "   💾 Кеш: $(pwd)/data/cache"

echo -e """
${YELLOW}⚠️  ВАЖЛИВІ НОТАТКИ:${NC}
- Проект оптимізовано для 8GB VRAM
- Використовуйте 8-bit квантизацію (вже налаштована)
- Для великих проектів (>1000 файлів) збільште RAM хоста до 32GB
- Перші запуски можуть бути повільними (кешування моделей)
"""