#!/bin/bash

# Минималистичный установочный скрипт для AI-агента на P104-100
# Предполагает, что NVIDIA драйверы и CUDA уже установлены

set -e

echo "🚀 Установка AI-агента (минималистичная версия)"
echo "=============================================="

# Проверка прав root
if [[ $EUID -ne 0 ]]; then
   echo "❌ Запустите с правами root: sudo ./setup.sh"
   exit 1
fi

# Создание директорий
echo "📁 Создание директорий..."
BASE_DIR="/home/ai-agent"
mkdir -p $BASE_DIR/{models,documents,cache,logs,neo4j/{data,logs,import},chroma}
chown -R $SUDO_USER:$SUDO_USER $BASE_DIR
echo "✅ Директории созданы"

# Установка Docker (только если не установлен)
if ! command -v docker &> /dev/null; then
    echo "🐳 Установка Docker..."
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

    # Ключ Docker (новый способ)
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Репозиторий Docker (для Kali используем Ubuntu 22.04)
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    usermod -aG docker $SUDO_USER
    echo "✅ Docker установлен"
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

# Установка NVIDIA Container Toolkit (только если GPU не работает в Docker)
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "🎮 Установка NVIDIA Container Toolkit..."

    # Ключ NVIDIA (новый способ)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

    # Репозиторий для Kali (используем Ubuntu 22.04)
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/ubuntu22.04/ $(dpkg --print-architecture) main" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    apt-get update
    apt-get install -y nvidia-container-toolkit
    systemctl restart docker
    echo "✅ NVIDIA Container Toolkit установлен"
else
    echo "✅ GPU поддержка в Docker уже настроена"
fi

# Переменные окружения
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

# Системные лимиты
echo "⚙️  Настройка системных лимитов..."
cat > /etc/security/limits.d/ai-agent.conf << EOF
$SUDO_USER soft nofile 65536
$SUDO_USER hard nofile 65536
EOF
echo "✅ Системные лимиты настроены"

# systemd сервис
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

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ai-agent
echo "✅ systemd сервис создан"

echo ""
echo "🎉 Установка завершена!"
echo "===================="
echo "📌 Следующие шаги:"
echo "1. Выйдите из root и перезагрузите сессию:"
echo "   exit"
echo "   newgrp docker"
echo ""
echo "2. Запустите сервисы:"
echo "   ./scripts/run.sh start"
echo ""
echo "3. Запустите CLI:"
echo "   ./scripts/run.sh cli"