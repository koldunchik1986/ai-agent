#!/bin/bash

# ===================================================================
# УПРАВЛЕНИЕ AI-АССИСТЕНТОМ (DOCKER-BASED)
# ===================================================================
# Универсальный скрипт для управления сервисами:
# - start/stop/restart
# - CLI вход
# - Логи и статус
# - Бэкап и восстановление
# - Различные режимы запуска

set -e

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Конфигурация
COMPOSE_FILE="docker/docker-compose.yml"
CONTAINER_NAME="ai-assistant-p104"
PROJECT_NAME="ai-assistant"

# Функции логирования
log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# ===================================================================
# ФУНКЦИИ УПРАВЛЕНИЯ
# ===================================================================

start_services() {
    log "🚀 Запуск сервисов..."
    
    # Проверка Docker
    if ! command -v docker &> /dev/null; then
        error "Docker не установлен! Запустите ./scripts/setup.sh"
    fi
    
    # Создание сети если не существует
    docker network create ai-network 2>/dev/null || true
    
    # Запуск с билдом (если первый раз)
    docker compose -f "$COMPOSE_FILE" up -d --build
    
    # Ожидание готовности
    log "Ожидание готовности сервисов..."
    for i in {1..30}; do
        if docker exec $CONTAINER_NAME python3.11 -c "import torch; print('OK')" 2>/dev/null; then
            log "✅ Сервисы готовы!"
            break
        fi
        if [ $i -eq 30 ]; then
            error "Сервисы не запустились за 30 секунд"
        fi
        sleep 2
    done
    
    # Информация о запуске
    show_status
}

stop_services() {
    log "⏹️ Остановка сервисов..."
    docker compose -f "$COMPOSE_FILE" down
    log "✅ Сервисы остановлены"
}

restart_services() {
    log "🔄 Перезапуск сервисов..."
    stop_services
    sleep 3
    start_services
}

show_status() {
    log "📊 Статус сервисов:"
    
    # Docker контейнеры
    echo -e "\n${BLUE}Docker Containers:${NC}"
    docker compose -f "$COMPOSE_FILE" ps
    
    # GPU статус
    echo -e "\n${BLUE}GPU Status:${NC}"
    if nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits | \
        while IFS=',' read -r name used total; do
            echo "  GPU: $name"
            echo "  VRAM: ${used}MB / ${total}MB ($((used*100/total))%)"
        done
    else
        echo "  ❌ nvidia-smi не доступен"
    fi
    
    # Проверка контейнера
    echo -e "\n${BLUE}AI Assistant Status:${NC}"
    if docker ps | grep -q $CONTAINER_NAME; then
        # Проверка VRAM в контейнере
        vram_output=$(docker exec $CONTAINER_NAME python3.11 -c "
import torch
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'{used:.2f}/{total:.2f}')
else:
    print('CPU mode')
" 2>/dev/null || echo "unknown")
        
        echo "  🟢 Контейнер запущен"
        echo "  VRAM: ${vram_output}GB"
        
        # Документы в базе
        doc_count=$(docker exec $CONTAINER_NAME python3.11 -c "
from src.document_processor import DocumentProcessor
dp = DocumentProcessor()
stats = dp.get_stats()
print(stats.get('vectors_in_db', 0))
" 2>/dev/null || echo "0")
        
        echo "  📄 Документов: $doc_count"
    else
        echo "  🔴 Контейнер остановлен"
    fi
}

show_logs() {
    log "📋 Логи сервисов:"
    echo -e "\n${BLUE}Последние 50 строк логов:${NC}"
    docker compose -f "$COMPOSE_FILE" logs --tail=50 -f
}

enter_cli() {
    log "🤖 Вход в CLI интерфейс..."
    
    if ! docker ps | grep -q $CONTAINER_NAME; then
        error "Контейнер не запущен! Используйте: $0 start"
    fi
    
    docker compose -f "$COMPOSE_FILE" exec $CONTAINER_NAME python3.11 -m src.cli
}

add_document() {
    local file_path="$1"
    
    if [ -z "$file_path" ]; then
        error "Укажите путь к файлу: $0 add /path/to/file.pdf"
    fi
    
    if [ ! -f "$file_path" ]; then
        error "Файл не найден: $file_path"
    fi
    
    # Конвертация в абсолютный путь
    file_path=$(realpath "$file_path")
    
    log "Добавление документа: $(basename "$file_path")"
    
    docker exec $CONTAINER_NAME python3.11 -c "
from src.agent import AIAssistant
assistant = AIAssistant()
result = assistant.add_document('$file_path')
print('✅ Успешно' if result else '❌ Ошибка')
"
}

train_model() {
    log "🎯 Запуск дообучения модели..."
    
    if ! docker ps | grep -q $CONTAINER_NAME; then
        error "Контейнер не запущен! Используйте: $0 start"
    fi
    
    read -p "Название модели (Enter для автоматического): " model_name
    if [ -z "$model_name" ]; then
        model_name="lora_$(date +%Y%m%d_%H%M%S)"
    fi
    
    output_dir="/app/data/models/$model_name"
    
    log "Дообучение начнется и займет несколько часов..."
    log "Модель будет сохранена в: $output_dir"
    
    docker exec -it $CONTAINER_NAME python3.11 -c "
from src.agent import AIAssistant
assistant = AIAssistant()
result = assistant.train_on_documents('$output_dir')
if result.get('success'):
    print(f'✅ Модель сохранена: {result[\"output_dir\"]}')
else:
    print(f'❌ Ошибка: {result.get(\"error\")}')
"
}

backup_data() {
    log "💾 Создание резервной копии..."
    ./scripts/backup.sh
}

clean_cache() {
    log "🧹 Очистка кэша..."
    
    docker exec $CONTAINER_NAME python3.11 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU кэш очищен')
else:
    print('ℹ️ Нет доступа к GPU')
"
    
    # Очистка Docker кэша
    docker system prune -f --volumes
    log "✅ Docker кэш очищен"
}

update_images() {
    log "🔄 Обновление Docker образов..."
    docker compose -f "$COMPOSE_FILE" pull
    docker compose -f "$COMPOSE_FILE" build --no-cache
    log "✅ Образы обновлены"
}

# ===================================================================
# ОБРАБОТКА КОМАНД
# ===================================================================

case "$1" in
    start)
        start_services
        ;;
    
    stop)
        stop_services
        ;;
    
    restart)
        restart_services
        ;;
    
    status)
        show_status
        ;;
    
    logs)
        show_logs
        ;;
    
    cli)
        enter_cli
        ;;
    
    add)
        add_document "$2"
        ;;
    
    train)
        train_model
        ;;
    
    backup)
        backup_data
        ;;
    
    clean)
        clean_cache
        ;;
    
    update)
        update_images
        ;;
    
    *)
        echo "Использование: $0 {start|stop|restart|status|logs|cli|add|train|backup|clean|update}"
        echo ""
        echo "Команды:"
        echo "  start   - запустить сервисы"
        echo "  stop    - остановить сервисы"
        echo "  restart - перезапустить сервисы"
        echo "  status  - показать статус"
        echo "  logs    - показать логи"
        echo "  cli     - войти в CLI"
        echo "  add     - добавить документ"
        echo "  train   - дообучить модель"
        echo "  backup  - создать бэкап"
        echo "  clean   - очистить кэш"
        echo "  update  - обновить образы"
        exit 1
        ;;
esac