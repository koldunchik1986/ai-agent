#!/bin/bash

# Установочный скрипт для AI-агента на базе Mistral AI 7B
# Поддержка Kali Linux, CUDA 11.8, GPU P104-100

set -e

echo "🚀 Установка AI-агента на базе Mistral AI 7B"
echo "============================================"

# Проверка прав root
if [[ $EUID -ne 0 ]]; then
   echo "❌ Этот скрипт должен быть запущен с правами root"
   echo "   Используйте: sudo ./setup.sh"
   exit 1
fi

# Проверка ОС
if ! grep -q "Kali" /etc/os-release; then
    echo "⚠️  Предупреждение: Система не является Kali Linux"
    read -p "Продолжить установку? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Проверка CUDA
echo "🔍 Проверка CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA драйвер не найден. Установите NVIDIA драйверы:"
    echo "   sudo apt update"
#    echo "   sudo apt install nvidia-driver-535"
    echo "   sudo reboot"
    exit 1
fi

# Проверка версии CUDA
CUDA_VERSION=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | grep -o '[0-9]*\.[0-9]*')
echo "✅ Найдена CUDA версия: $CUDA_VERSION"

if [[ $(echo "$CUDA_VERSION < 11.8" | bc -l) -eq 1 ]]; then
    echo "⚠️  Рекомендуется CUDA 11.8 или выше"
fi

# Установка директорий
echo "📁 Создание директорий..."
BASE_DIR="/home/ai-agent"
mkdir -p $BASE_DIR/{models,documents,cache,logs,neo4j/{data,logs,import},chroma}
chown -R $SUDO_USER:$SUDO_USER $BASE_DIR
echo "✅ Директории созданы в $BASE_DIR"

# Установка Docker
echo "🐳 Проверка Docker..."
if ! command -v docker &> /dev/null; then
    echo "📦 Установка Docker..."
    apt-get update
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Добавление пользователя в группу docker
    usermod -aG docker $SUDO_USER
    echo "✅ Docker установлен. Пользователь $SUDO_USER добавлен в группу docker"
else
    echo "✅ Docker уже установлен"
fi

# Установка Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Установка Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose установлен"
else
    echo "✅ Docker Compose уже установлен"
fi

# Установка Docker GPU Support
echo "🎮 Настройка GPU поддержки..."
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "📦 Установка NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    # Используем современный способ без apt-key
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
    curl -s -L "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" | \
        sed 's#https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update
    apt-get install -y nvidia-docker2
    systemctl restart docker
    echo "✅ NVIDIA Container Toolkit установлен"
else
    echo "✅ GPU поддержка уже настроена"
fi

# Создание файла переменных окружения
echo "🔧 Создание переменных окружения..."
cat > /etc/environment.d/ai-agent.conf << EOF
AGENT_HOME="/home/ai-agent"
MODEL_CACHE_PATH="/home/ai-agent/models"
DOCUMENT_PATH="/home/ai-agent/documents"
CACHE_PATH="/home/ai-agent/cache"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="password"
CHROMA_HOST="localhost"
CHROMA_PORT="8001"
CUDA_VISIBLE_DEVICES="0"
EOF

echo "✅ Переменные окружения созданы"

# Настройка системных лимитов
echo "⚙️  Настройка системных лимитов..."
cat > /etc/security/limits.d/ai-agent.conf << EOF
$SUDO_USER soft nofile 65536
$SUDO_USER hard nofile 65536
$SUDO_USER soft nproc 32768
$SUDO_USER hard nproc 32768
EOF

echo "✅ Системные лимиты настроены"

# Создание сервисного файла
echo "🔧 Создание systemd сервиса..."
cat > /etc/systemd/system/ai-agent.service << EOF
[Unit]
Description=AI Agent Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PWD
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ai-agent
echo "✅ systemd сервис создан"

# Установка Python зависимостей для хоста (для разработки)
echo "🐍 Установка Python зависимостей..."
if ! command -v python3 &> /dev/null; then
    apt-get install -y python3 python3-pip python3-venv
fi

# Создание виртуального окружения для разработки
if [ ! -d "venv" ]; then
#    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
#    pip install -r requirements.txt
    echo "✅ Виртуальное окружение создано"
else
    echo "✅ Виртуальное окружение уже существует"
fi

echo ""
echo "🎉 Установка завершена!"
echo "======================"
echo ""
echo "📋 Следующие шаги:"
echo "1. Перезагрузите систему или выполните:"
echo "   source /etc/environment.d/ai-agent.conf"
echo ""
echo "2. Запустите сервисы:"
echo "   sudo systemctl start ai-agent"
echo "   или"
echo "   docker-compose up -d"
echo ""
echo "3. Проверьте статус:"
echo "   docker-compose ps"
echo ""
echo "4. Запустите CLI интерфейс:"
echo "   docker-compose exec ai-agent python -m src.cli_interface"
echo ""
echo "5. Для разработки используйте локальное окружение:"
echo "   source venv/bin/activate"
echo "   python -m src.cli_interface"
echo ""
echo "📁 Директории:"
echo "   Модели: $BASE_DIR/models"
echo "   Документы: $BASE_DIR/documents"
echo "   Кэш: $BASE_DIR/cache"
echo ""
echo "🌐 Web интерфейсы:"
echo "   Neo4j: http://localhost:7474 (neo4j/password)"
echo "   Chroma: http://localhost:8001"
echo ""
echo "📖 Документация: docs/README.md"