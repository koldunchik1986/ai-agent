#!/bin/bash

# Скрипт быстрого запуска AI-агента с подробным логированием

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Запуск AI-агента${NC}"
echo -e "${BLUE}==================${NC}"

# Проверка директории
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ docker-compose.yml не найден. Запустите из корневой директории проекта.${NC}"
    exit 1
fi

# Загрузка переменных окружения
if [ -f "/etc/environment.d/ai-agent.conf" ]; then
    source /etc/environment.d/ai-agent.conf
    echo -e "${GREEN}✅ Переменные окружения загружены${NC}"
else
    echo -e "${YELLOW}⚠️  Файл переменных окружения не найден, используем значения по умолчанию${NC}"
    export AGENT_HOME="/home/ai-agent"
    export MODEL_CACHE_PATH="/home/ai-agent/models"
    export DOCUMENT_PATH="/home/ai-agent/documents"
    export CACHE_PATH="/home/ai-agent/cache"
fi

# Функции
start_services() {
    echo -e "${BLUE}🐳 Запуск Docker сервисов...${NC}"

    # Логирование директорий на хосте
    echo -e "${YELLOW}📁 Содержимое директории AGENT_HOME (${AGENT_HOME}):${NC}"
    ls -la ${AGENT_HOME} 2>/dev/null || echo -e "${RED}❌ Директория не существует или недоступна${NC}"

    echo -e "${YELLOW}📚 Содержимое DOCUMENT_PATH (${DOCUMENT_PATH}):${NC}"
    ls -la ${DOCUMENT_PATH} 2>/dev/null || echo -e "${RED}❌ Директория не существует или недоступна${NC}"

    echo -e "${YELLOW}💾 Содержимое MODEL_CACHE_PATH (${MODEL_CACHE_PATH}):${NC}"
    ls -la ${MODEL_CACHE_PATH} 2>/dev/null || echo -e "${RED}❌ Директория не существует или недоступна${NC}"

    # Checking container status
    if docker-compose ps -q ai-agent >/dev/null 2>&1; then
       echo -e "${YELLOW} Container ai-agent already running. Stopping for restart...${NC}"
       docker-compose stop ai-agent
    fi

    # Running services
    docker-compose up -d

    echo -e "${YELLOW}⏳ Ожидание запуска сервисов...${NC}"
    sleep 20

    # Проверка статуса
    echo -e "${GREEN}📊 Проверка статуса сервисов:${NC}"
    docker-compose ps

    # Show list /workspace in container
    echo -e ""
    echo -e "${YELLOW} List /workspace at container 'ai-agent':${NC}"
    if docker-compose exec ai-agent ls -la /workspace/ 2>/dev/null; then
       echo -e "${GREEN} Succes list /workspace${NC}"
    else
       echo -e "${RED} Not succes list /workspace Container maybe not loaded or dont have acces to directory.${NC}"
    fi

    # Show list /workspace/src in container (if exist)
    echo -e ""
    echo -e "${YELLOW} List /workspace/src at container 'ai-agent':${NC}"
    if docker-compose exec ai-agent ls -la /workspace/src 2>/dev/null; then
       echo -e "${GREEN} Succes list /workspace/src${NC}"
    else
       echo -e "${RED} Not succes list /workspace/src Directory not finded or empty.${NC}"
    fi

    # Cheking cli_interface.py file
    echo -e ""
    echo -e "${YELLOW} List /workspace/src at container 'ai-agent':${NC}"
    if docker-compose exec ai-agent test -f "/workspace/cli_interface.py"; then
       echo -e "${GREEN} File /workspace/cli_interface.py finded${NC}"
    else
       echo -e "${RED} File /workspace/cli_interface.py not finded${NC}"
    fi

    if docker-compose exec ai-agent test -f "/workspace/src/cli_interface.py"; then
       echo -e "${GREEN} File /workspace/src/cli_interface.py finded${NC}"
    else
       echo -e "${RED} File /workspace/src/cli_interface.py not finded${NC}"
    fi


    echo -e ""
    echo -e "${GREEN}🌐 Доступные интерфейсы:${NC}"
    echo -e "   Neo4j: http://localhost:7474 (neo4j/password)"
    echo -e "   Chroma: http://localhost:8001"
}

stop_services() {
    echo -e "${RED}🛑 Остановка сервисов...${NC}"
    docker-compose down
}

restart_services() {
    stop_services
    start_services
}

show_logs() {
    echo -e "${BLUE}📋 Показать логи сервиса (введите имя, например ai-agent, neo4j, chroma):${NC}"
    read -p "Имя сервиса (или 'all' для всех): " service_name
    if [ "$service_name" == "all" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$service_name"
    fi
}

run_cli() {
    echo -e "${BLUE}🖥️  Запуск CLI интерфейса...${NC}"

    # Проверим, запущен ли контейнер
    if [ "$(docker-compose ps -q ai-agent | wc -l)" -eq 0 ] || [ "$(docker-compose ps -q ai-agent | xargs -r docker inspect -f '{{.State.Status}}')" != "running" ]; then
        echo -e "${YELLOW}⚠️  Контейнер ai-agent не запущен. Попробуйте сначала 'start'.${NC}"
        return 1
    fi

    # Проверим, есть ли модуль cli_interface в контейнере
    echo -e "${YELLOW}🔍 Проверка наличия модуля cli_interface в контейнере...${NC}"
    if docker-compose exec ai-agent python -c "import cli_interface; print('cli_interface OK')" 2>/dev/null; then
        echo -e "${GREEN}✅ Модуль cli_interface найден в контейнере${NC}"
    else
        echo -e "${RED}❌ Модуль cli_interface не найден в контейнере${NC}"
        echo -e "${YELLOW}Проверьте содержимое /workspace в контейнере:${NC}"
        docker-compose exec ai-agent ls -la /workspace/
        return 1
    fi

    docker-compose exec ai-agent python -m cli_interface
}

run_dev() {
    echo -e "${BLUE}🔧 Запуск в режиме разработки...${NC}"
    if [ ! -d "venv" ]; then
        echo -e "${RED}❌ Виртуальное окружение не найдено. Выполните ./setup.sh или создайте venv вручную.${NC}"
        exit 1
    fi

    source venv/bin/activate
    export AGENT_HOME="/home/ai-agent"
    export MODEL_CACHE_PATH="/home/ai-agent/models"
    export DOCUMENT_PATH="/home/ai-agent/documents"
    export CACHE_PATH="/home/ai-agent/cache"

    echo -e "${YELLOW}📁 Содержимое директории в режиме разработки:${NC}"
    pwd
    ls -la

    python -m cli_interface
}

add_documents() {
    if [ -z "$1" ]; then
        echo -e "${RED}❌ Укажите путь к документам: ./run.sh add <путь>${NC}"
        exit 1
    fi

    echo -e "${BLUE}📚 Добавление документов из: $1${NC}"

    # Копирование документов в директорию (опционально)
    # cp -r "$1"/* /home/ai-agent/documents/ 2>/dev/null || true

    # Запуск CLI с командой добавления
    docker-compose exec ai-agent python -c "
import sys
sys.path.insert(0, '/workspace')
from cli_interface import AIAgentCLI
cli = AIAgentCLI()
cli._add_documents('$1')
"
}

backup_data() {
    echo -e "${BLUE}💾 Создание резервной копии...${NC}"

    BACKUP_DIR="/home/backups/ai-agent-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # Копирование данных
    echo -e "${YELLOW}Копирование моделей...${NC}"
    cp -r /home/ai-agent/models "$BACKUP_DIR/" 2>/dev/null || echo -e "${YELLOW}Директория моделей пуста или не существует${NC}"

    echo -e "${YELLOW}Копирование документов...${NC}"
    cp -r /home/ai-agent/documents "$BACKUP_DIR/" 2>/dev/null || echo -e "${YELLOW}Директория документов пуста или не существует${NC}"

    echo -e "${YELLOW}Копирование кэша...${NC}"
    cp -r /home/ai-agent/cache "$BACKUP_DIR/" 2>/dev/null || echo -e "${YELLOW}Директория кэша пуста или не существует${NC}"

    # Экспорт данных из сервисов (опционально)
    # docker-compose exec -T neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j.dump
    # docker cp $(docker-compose ps -q neo4j):/tmp/neo4j.dump "$BACKUP_DIR/"

    echo -e "${GREEN}✅ Резервная копия создана: $BACKUP_DIR${NC}"
}

restore_data() {
    if [ -z "$1" ]; then
        echo -e "${RED}❌ Укажите путь к резервной копии: ./run.sh restore <путь>${NC}"
        exit 1
    fi

    echo -e "${BLUE}🔄 Восстановление из резервной копии: $1${NC}"

    # Остановка сервисов
    docker-compose down

    # Восстановление файлов
    if [ -d "$1/models" ]; then
        echo -e "${YELLOW}Восстановление моделей...${NC}"
        cp -r "$1/models/"* /home/ai-agent/models/ 2>/dev/null || echo -e "${YELLOW}Директория моделей в бэкапе пуста${NC}"
    fi

    if [ -d "$1/documents" ]; then
        echo -e "${YELLOW}Восстановление документов...${NC}"
        cp -r "$1/documents/"* /home/ai-agent/documents/ 2>/dev/null || echo -e "${YELLOW}Директория документов в бэкапе пуста${NC}"
    fi

    if [ -d "$1/cache" ]; then
        echo -e "${YELLOW}Восстановление кэша...${NC}"
        cp -r "$1/cache/"* /home/ai-agent/cache/ 2>/dev/null || echo -e "${YELLOW}Директория кэша в бэкапе пуста${NC}"
    fi

    # Запуск сервисов
    docker-compose up -d
    sleep 30

    # Восстановление Neo4j (опционально)
    # if [ -f "$1/neo4j.dump" ]; then
    #     docker cp "$1/neo4j.dump" $(docker-compose ps -q neo4j):/tmp/
    #     docker-compose exec neo4j neo4j-admin load --from=/tmp/neo4j.dump --database=neo4j --overwrite-destination=true
    #     docker-compose restart neo4j
    # fi

    echo -e "${GREEN}✅ Восстановление завершено${NC}"
}

update_model() {
    echo -e "${BLUE}🔄 Обновление модели...${NC}"

    # Скачивание последней версии Mistral 7B (пример)
    docker-compose exec ai-agent python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import config

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
    echo -e "${BLUE}🏥 Проверка здоровья системы...${NC}"

    # Проверка Docker
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker не запущен${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Docker запущен${NC}"
    fi

    # Проверка сервисов
    SERVICES_STATUS=$(docker-compose ps --format "table {{.Service}}\t{{.Status}}")
    echo -e "${GREEN}📊 Статус сервисов:${NC}"
    echo -e "$SERVICES_STATUS"

    if echo "$SERVICES_STATUS" | grep -q "Up"; then
        echo -e "${GREEN}✅ Сервисы запущены${NC}"
    else
        echo -e "${YELLOW}⚠️  Некоторые сервисы не запущены${NC}"
    fi

    # Проверка GPU
    if ! nvidia-smi >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  GPU недоступен${NC}"
    else
        echo -e "${GREEN}✅ GPU доступен${NC}"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    fi

    # Проверка интерфейсов
    echo -e ""
    echo -e "${GREEN}🌐 Проверка интерфейсов:${NC}"
    if curl -s http://localhost:7474 >/dev/null; then
        echo -e "${GREEN}✅ Neo4j доступен${NC}"
    else
        echo -e "${RED}❌ Neo4j недоступен${NC}"
    fi

    if curl -s http://localhost:8001 >/dev/null; then
        echo -e "${GREEN}✅ Chroma доступен${NC}"
    else
        echo -e "${RED}❌ Chroma недоступен${NC}"
    fi

    echo -e ""
    echo -e "${GREEN}💾 Использование диска:${NC}"
    df -sh /home/ai-agent/* 2>/dev/null || echo -e "${YELLOW}Директория /home/ai-agent не найдена${NC}"
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
        echo -e "${BLUE}AI Agent Management Script${NC}"
        echo -e "${BLUE}=========================${NC}"
        echo -e ""
        echo -e "Использование: $0 {start|stop|restart|logs|cli|dev|add|backup|restore|update|health}"
        echo -e ""
        echo -e "Команды:"
        echo -e "  ${GREEN}start${NC}     - Запустить все сервисы"
        echo -e "  ${RED}stop${NC}      - Остановить все сервисы"
        echo -e "  ${YELLOW}restart${NC}   - Перезапустить сервисы"
        echo -e "  ${BLUE}logs${NC}      - Показать логи (с выбором сервиса)"
        echo -e "  ${BLUE}cli${NC}       - Запустить CLI интерфейс"
        echo -e "  ${YELLOW}dev${NC}       - Запустить в режиме разработки"
        echo -e "  ${YELLOW}add${NC} <path> - Добавить документы"
        echo -e "  ${YELLOW}backup${NC}    - Создать резервную копию"
        echo -e "  ${YELLOW}restore${NC}   - Восстановить из резервной копии"
        echo -e "  ${YELLOW}update${NC}    - Обновить модель"
        echo -e "  ${GREEN}health${NC}    - Проверка здоровья системы"
        echo -e ""
        echo -e "Примеры:"
        echo -e "  ./run.sh start"
        echo -e "  ./run.sh add ~/documents"
        echo -e "  ./run.sh cli"
        echo -e "  ./run.sh logs ai-agent"
        exit 1
        ;;
esac
