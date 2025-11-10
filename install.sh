#!/bin/bash

# ===================================================================
# УСТАНОВКА AI-АССИСТЕНТА В /home/ai-projects
# ===================================================================

set -e

# Цвета
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "====================================================="
echo "  УСТАНОВКА AI-АССИСТЕНТА В /home/ai-projects"
echo "  P104-100 8GB VRAM | Kali Linux 2025 | sm_61"
echo "====================================================="
echo -e "${NC}"

# Проверка что мы в правильной директории
INSTALL_DIR="/home/ai-projects/ai-assistant-p104"

if [[ "$PWD" != "$INSTALL_DIR" ]]; then
    echo -e "${RED}ОШИБКА: Запустите из директории $INSTALL_DIR${NC}"
    echo "Текущая директория: $PWD"
    echo "Ожидается: $INSTALL_DIR"
    exit 1
fi

# Проверка прав (НЕ root)
if [[ $EUID -eq 0 ]]; then
    echo -e "${RED}ОШИБКА: Не запускайте от root!${NC}"
    exit 1
fi

# ===================================================================
# ПРОВЕРКА СИСТЕМЫ
# ===================================================================
echo -e "${GREEN}[ПРОВЕРКА]${NC} GPU и система..."

# Проверка GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA драйвер не найден${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*")

echo "✅ GPU: $GPU_NAME"
echo "✅ CUDA: $CUDA_VERSION"

if [[ ! "$GPU_NAME" =~ "P104-100" ]]; then
    echo -e "${YELLOW}⚠️ Внимание: Обнаружен $GPU_NAME, а не P104-100${NC}"
fi

# ===================================================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ
# ===================================================================
echo -e "${GREEN}[УСТАНОВКА]${NC} Docker и зависимости..."

# Docker установка (если нет)
if ! command -v docker &> /dev/null; then
    echo "Установка Docker..."
    # Новый метод подписи для Kali 2025
    sudo apt update
    sudo apt install -y ca-certificates curl gnupg lsb-release
    
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian bookworm stable" | sudo tee /etc/apt/sources.list.d/docker.list
    
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    sudo usermod -aG docker $USER
    echo "✅ Docker установлен"
fi

# NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Установка NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# ===================================================================
# НАСТРОЙКА ПРОЕКТА
# ===================================================================
echo -e "${GREEN}[НАСТРОЙКА]${NC} Проект..."

# Сделать скрипты исполняемыми
chmod +x scripts/*.sh

# Создать структуру данных
mkdir -p data/{documents,models,chroma,cache,logs,vscode_projects}

# Создать .env если не существует
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Создан .env файл"
fi

# Настройка прав доступа
sudo chown -R $USER:$USER data/
chmod -R 755 data/

# ===================================================================
# ПРОВЕРКА GPU В DOCKER
# ===================================================================
echo -e "${GREEN}[ПРОВЕРКА]${NC} GPU в Docker..."
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✅ GPU доступен в Docker"
else
    echo -e "${RED}❌ GPU не доступен в Docker${NC}"
    exit 1
fi

# ===================================================================
# СБОРКА DOCKER ОБРАЗА
# ===================================================================
echo -e "${GREEN}[СБОРКА]${NC} Docker образ..."
docker compose -f docker/docker-compose.yml build

# ===================================================================
# ЗАВЕРШЕНИЕ
# ===================================================================
echo -e "${BLUE}"
echo "====================================================="
echo "  ✅ УСТАНОВКА ЗАВЕРШЕНА!"
echo "====================================================="
echo -e "${NC}"

echo -e """
📋 СЛЕДУЮЩИЕ ШАГИ:

1. ${GREEN}Перелогиньтесь${NC} или выполните:
   ${BLUE}newgrp docker${NC}

2. ${GREEN}Запустите ассистента:${NC}
   ${BLUE}./scripts/run.sh start${NC}
   
3. ${GREEN}Войдите в CLI:${NC}
   ${BLUE}./scripts/run.sh cli${NC}

4. ${GREEN}Добавьте документы:${NC}
   Поместите файлы в ${YELLOW}data/documents/${NC}
   Или используйте: ${BLUE}/add /app/data/documents/file.pdf${NC}

📁 ВАЖНЫЕ ПУТИ:
   Проект: $PWD
   Данные: $PWD/data
   Логи:   $PWD/data/logs

🔧 Управление:
   Статус: ./scripts/run.sh status
   Логи:   ./scripts/run.sh logs  
   Стоп:   ./scripts/run.sh stop
"""

# Сохранение информации о установке
cat > install_info.txt << EOF
Дата установки: $(date)
GPU: $GPU_NAME
CUDA: $CUDA_VERSION
Путь: $PWD
Данные: $PWD/data
EOF

echo "✅ Информация сохранена в install_info.txt"
