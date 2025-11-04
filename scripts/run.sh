#!/bin/bash

# Скрипт быстрого запуска AI-агента

set -e

echo "🚀 Запуск AI-агента"

# Проверка директории
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml не найден. Запустите из корневой директории проекта."
    exit 1
fi

# Загрузка переменных окружения
if [ -f "/etc/environment.d/ai-agent.conf" ]; then
    source /etc/environment.d/ai-agent.conf
fi

# Функции
start_services() {
    echo "🐳 Запуск Docker сервисов..."
    docker-compose up -d
    
    echo "⏳ Ожидание запуска сервисов..."
    sleep 30
    
    # Проверка статуса
    echo "📊 Проверка статуса сервисов:"
    docker-compose ps
    
    echo ""
    echo "🌐 Доступные интерфейсы:"
    echo "   Neo4j: http://localhost:7474 (neo4j/password)"
    echo "   Chroma: http://localhost:8001"
}

stop_services() {
    echo "🛑 Остановка сервисов..."
    docker-compose down
}

restart_services() {
    stop_services
    start_services
}

show_logs() {
    docker-compose logs -f
}

run_cli() {
    echo "🖥️  Запуск CLI интерфейса..."
    docker-compose exec ai-agent python -m src.cli_interface
}

run_dev() {
    echo "🔧 Запуск в режиме разработки..."
    if [ ! -d "venv" ]; then
        echo "❌ Виртуальное окружение не найдено. Выполните ./setup.sh"
        exit 1
    fi
    
    source venv/bin/activate
    export AGENT_HOME="/home/sda3/ai-agent"
    export MODEL_CACHE_PATH="/home/sda3/ai-agent/models"
    export DOCUMENT_PATH="/home/sda3/ai-agent/documents"
    export CACHE_PATH="/home/sda3/ai-agent/cache"
    
    python -m src.cli_interface
}

add_documents() {
    if [ -z "$1" ]; then
        echo "❌ Укажите путь к документам: ./run.sh add <путь>"
        exit 1
    fi
    
    echo "📚 Добавление документов из: $1"
    
    # Копирование документов в директорию
    cp -r "$1"/* /home/sda3/ai-agent/documents/ 2>/dev/null || true
    
    # Запуск CLI с командой добавления
    docker-compose exec ai-agent python -c "
from src.cli_interface import AIAgentCLI
cli = AIAgentCLI()
cli._add_documents('$1')
"
}

backup_data() {
    echo "💾 Создание резервной копии..."
    
    BACKUP_DIR="/home/sda3/backups/ai-agent-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Копирование данных
    cp -r /home/sda3/ai-agent/models "$BACKUP_DIR/"
    cp -r /home/sda3/ai-agent/documents "$BACKUP_DIR/"
    cp -r /home/sda3/ai-agent/cache "$BACKUP_DIR/"
    
    # Экспорт данных из сервисов
    docker-compose exec -T neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j.dump
    docker cp $(docker-compose ps -q neo4j):/tmp/neo4j.dump "$BACKUP_DIR/"
    
    echo "✅ Резервная копия создана: $BACKUP_DIR"
}

restore_data() {
    if [ -z "$1" ]; then
        echo "❌ Укажите путь к резервной копии: ./run.sh restore <путь>"
        exit 1
    fi
    
    echo "🔄 Восстановление из резервной копии: $1"
    
    # Остановка сервисов
    docker-compose down
    
    # Восстановление файлов
    cp -r "$1"/models/* /home/sda3/ai-agent/models/ 2>/dev/null || true
    cp -r "$1"/documents/* /home/sda3/ai-agent/documents/ 2>/dev/null || true
    cp -r "$1"/cache/* /home/sda3/ai-agent/cache/ 2>/dev/null || true
    
    # Запуск сервисов
    docker-compose up -d
    sleep 30
    
    # Восстановление Neo4j
    if [ -f "$1/neo4j.dump" ]; then
        docker cp "$1/neo4j.dump" $(docker-compose ps -q neo4j):/tmp/
        docker-compose exec neo4j neo4j-admin load --from=/tmp/neo4j.dump --database=neo4j --overwrite-destination=true
        docker-compose restart neo4j
    fi
    
    echo "✅ Восстановление завершено"
}

update_model() {
    echo "🔄 Обновление модели..."
    
    # Скачивание последней версии Mistral 7B
    docker-compose exec ai-agent python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import config

print('Загрузка последней версии модели...')
tokenizer = AutoTokenizer.from_pretrained(
    config.model.model_name,
    cache_dir=config.model.cache_dir,
    trust_remote_code=config.model.trust_remote_code
)

model = AutoModelForCausalLM.from_pretrained(
    config.model.model_name,
    cache_dir=config.model.cache_dir,
    torch_dtype='auto',
    trust_remote_code=config.model.trust_remote_code
)

print('✅ Модель обновлена')
"
}

health_check() {
    echo "🏥 Проверка здоровья системы..."
    
    # Проверка Docker
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker не запущен"
        return 1
    fi
    
    # Проверка сервисов
    if ! docker-compose ps | grep -q "Up"; then
        echo "❌ Сервисы не запущены"
        echo "Выполните: ./run.sh start"
        return 1
    fi
    
    # Проверка GPU
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "⚠️  GPU недоступен"
    else
        echo "✅ GPU доступен"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    fi
    
    # Проверка сервисов
    echo ""
    echo "📊 Статус сервисов:"
    docker-compose ps
    
    # Проверка интерфейсов
    echo ""
    echo "🌐 Проверка интерфейсов:"
    if curl -s http://localhost:7474 >/dev/null; then
        echo "✅ Neo4j доступен"
    else
        echo "❌ Neo4j недоступен"
    fi
    
    if curl -s http://localhost:8001 >/dev/null; then
        echo "✅ Chroma доступен"
    else
        echo "❌ Chroma недоступен"
    fi
    
    echo ""
    echo "💾 Использование диска:"
    du -sh /home/sda3/ai-agent/* 2>/dev/null || echo "Директория не найдена"
}

# Основной обработчик команд
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
    logs)
        show_logs
        ;;
    cli)
        run_cli
        ;;
    dev)
        run_dev
        ;;
    add)
        add_documents "$2"
        ;;
    backup)
        backup_data
        ;;
    restore)
        restore_data "$2"
        ;;
    update)
        update_model
        ;;
    health)
        health_check
        ;;
    *)
        echo "AI Agent Management Script"
        echo "========================="
        echo ""
        echo "Использование: $0 {start|stop|restart|logs|cli|dev|add|backup|restore|update|health}"
        echo ""
        echo "Команды:"
        echo "  start     - Запустить все сервисы"
        echo "  stop      - Остановить все сервисы"
        echo "  restart   - Перезапустить сервисы"
        echo "  logs      - Показать логи"
        echo "  cli       - Запустить CLI интерфейс"
        echo "  dev       - Запустить в режиме разработки"
        echo "  add <path>- Добавить документы"
        echo "  backup    - Создать резервную копию"
        echo "  restore   - Восстановить из резервной копии"
        echo "  update    - Обновить модель"
        echo "  health    - Проверка здоровья системы"
        echo ""
        echo "Примеры:"
        echo "  ./run.sh start"
        echo "  ./run.sh add ~/documents"
        echo "  ./run.sh cli"
        exit 1
        ;;
esac