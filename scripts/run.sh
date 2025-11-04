
#!/bin/bash

# \u0421\u043a\u0440\u0438\u043f\u0442 \u0431\u044b\u0441\u0442\u0440\u043e\u0433\u043e \u0437\u0430\u043f\u0443\u0441\u043a\u0430 AI-\u0430\u0433\u0435\u043d\u0442\u0430

set -e

echo "\ud83d\ude80 \u0417\u0430\u043f\u0443\u0441\u043a AI-\u0430\u0433\u0435\u043d\u0442\u0430"

# \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0438
if [ ! -f "docker-compose.yml" ]; then
    echo "\u274c docker-compose.yml \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d. \u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 \u0438\u0437 \u043a\u043e\u0440\u043d\u0435\u0432\u043e\u0439 \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0438 \u043f\u0440\u043e\u0435\u043a\u0442\u0430."
    exit 1
fi

# \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u0445 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f
if [ -f "/etc/environment.d/ai-agent.conf" ]; then
    source /etc/environment.d/ai-agent.conf
fi

# \u0424\u0443\u043d\u043a\u0446\u0438\u0438
start_services() {
    echo "\ud83d\udc33 \u0417\u0430\u043f\u0443\u0441\u043a Docker \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432..."
    docker-compose up -d
    
    echo "\u23f3 \u041e\u0436\u0438\u0434\u0430\u043d\u0438\u0435 \u0437\u0430\u043f\u0443\u0441\u043a\u0430 \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432..."
    sleep 30
    
    # \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0441\u0442\u0430\u0442\u0443\u0441\u0430
    echo "\ud83d\udcca \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0441\u0442\u0430\u0442\u0443\u0441\u0430 \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432:"
    docker-compose ps
    
    echo ""
    echo "\ud83c\udf10 \u0414\u043e\u0441\u0442\u0443\u043f\u043d\u044b\u0435 \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u044b:"
    echo "   Neo4j: http://localhost:7474 (neo4j/password)"
    echo "   Chroma: http://localhost:8001"
}

stop_services() {
    echo "\ud83d\uded1 \u041e\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432..."
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
    echo "\ud83d\udda5\ufe0f  \u0417\u0430\u043f\u0443\u0441\u043a CLI \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u0430..."
    docker-compose exec ai-agent python -m src.cli_interface
}

run_dev() {
    echo "\ud83d\udd27 \u0417\u0430\u043f\u0443\u0441\u043a \u0432 \u0440\u0435\u0436\u0438\u043c\u0435 \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u043a\u0438..."
    if [ ! -d "venv" ]; then
        echo "\u274c \u0412\u0438\u0440\u0442\u0443\u0430\u043b\u044c\u043d\u043e\u0435 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u0435 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u043e. \u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u0435 ./setup.sh"
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
        echo "\u274c \u0423\u043a\u0430\u0436\u0438\u0442\u0435 \u043f\u0443\u0442\u044c \u043a \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c: ./run.sh add <\u043f\u0443\u0442\u044c>"
        exit 1
    fi
    
    echo "\ud83d\udcda \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u0438\u0437: $1"
    
    # \u041a\u043e\u043f\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u0432 \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u044e
    cp -r "$1"/* /home/sda3/ai-agent/documents/ 2>/dev/null || true
    
    # \u0417\u0430\u043f\u0443\u0441\u043a CLI \u0441 \u043a\u043e\u043c\u0430\u043d\u0434\u043e\u0439 \u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u044f
    docker-compose exec ai-agent python -c "
from src.cli_interface import AIAgentCLI
cli = AIAgentCLI()
cli._add_documents('$1')
"
}

backup_data() {
    echo "\ud83d\udcbe \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0440\u0435\u0437\u0435\u0440\u0432\u043d\u043e\u0439 \u043a\u043e\u043f\u0438\u0438..."
    
    BACKUP_DIR="/home/sda3/backups/ai-agent-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # \u041a\u043e\u043f\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0434\u0430\u043d\u043d\u044b\u0445
    cp -r /home/sda3/ai-agent/models "$BACKUP_DIR/"
    cp -r /home/sda3/ai-agent/documents "$BACKUP_DIR/"
    cp -r /home/sda3/ai-agent/cache "$BACKUP_DIR/"
    
    # \u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u0434\u0430\u043d\u043d\u044b\u0445 \u0438\u0437 \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432
    docker-compose exec -T neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j.dump
    docker cp $(docker-compose ps -q neo4j):/tmp/neo4j.dump "$BACKUP_DIR/"
    
    echo "\u2705 \u0420\u0435\u0437\u0435\u0440\u0432\u043d\u0430\u044f \u043a\u043e\u043f\u0438\u044f \u0441\u043e\u0437\u0434\u0430\u043d\u0430: $BACKUP_DIR"
}

restore_data() {
    if [ -z "$1" ]; then
        echo "\u274c \u0423\u043a\u0430\u0436\u0438\u0442\u0435 \u043f\u0443\u0442\u044c \u043a \u0440\u0435\u0437\u0435\u0440\u0432\u043d\u043e\u0439 \u043a\u043e\u043f\u0438\u0438: ./run.sh restore <\u043f\u0443\u0442\u044c>"
        exit 1
    fi
    
    echo "\ud83d\udd04 \u0412\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 \u0438\u0437 \u0440\u0435\u0437\u0435\u0440\u0432\u043d\u043e\u0439 \u043a\u043e\u043f\u0438\u0438: $1"
    
    # \u041e\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432
    docker-compose down
    
    # \u0412\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 \u0444\u0430\u0439\u043b\u043e\u0432
    cp -r "$1"/models/* /home/sda3/ai-agent/models/ 2>/dev/null || true
    cp -r "$1"/documents/* /home/sda3/ai-agent/documents/ 2>/dev/null || true
    cp -r "$1"/cache/* /home/sda3/ai-agent/cache/ 2>/dev/null || true
    
    # \u0417\u0430\u043f\u0443\u0441\u043a \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432
    docker-compose up -d
    sleep 30
    
    # \u0412\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 Neo4j
    if [ -f "$1/neo4j.dump" ]; then
        docker cp "$1/neo4j.dump" $(docker-compose ps -q neo4j):/tmp/
        docker-compose exec neo4j neo4j-admin load --from=/tmp/neo4j.dump --database=neo4j --overwrite-destination=true
        docker-compose restart neo4j
    fi
    
    echo "\u2705 \u0412\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043e"
}

update_model() {
    echo "\ud83d\udd04 \u041e\u0431\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u0438..."
    
    # \u0421\u043a\u0430\u0447\u0438\u0432\u0430\u043d\u0438\u0435 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0435\u0439 \u0432\u0435\u0440\u0441\u0438\u0438 Mistral 7B
    docker-compose exec ai-agent python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import config

print('\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0435\u0439 \u0432\u0435\u0440\u0441\u0438\u0438 \u043c\u043e\u0434\u0435\u043b\u0438...')
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

print('\u2705 \u041c\u043e\u0434\u0435\u043b\u044c \u043e\u0431\u043d\u043e\u0432\u043b\u0435\u043d\u0430')
"
}

health_check() {
    echo "\ud83c\udfe5 \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0437\u0434\u043e\u0440\u043e\u0432\u044c\u044f \u0441\u0438\u0441\u0442\u0435\u043c\u044b..."
    
    # \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 Docker
    if ! docker info >/dev/null 2>&1; then
        echo "\u274c Docker \u043d\u0435 \u0437\u0430\u043f\u0443\u0449\u0435\u043d"
        return 1
    fi
    
    # \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432
    if ! docker-compose ps | grep -q "Up"; then
        echo "\u274c \u0421\u0435\u0440\u0432\u0438\u0441\u044b \u043d\u0435 \u0437\u0430\u043f\u0443\u0449\u0435\u043d\u044b"
        echo "\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u0435: ./run.sh start"
        return 1
    fi
    
    # \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 GPU
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "\u26a0\ufe0f  GPU \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d"
    else
        echo "\u2705 GPU \u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    fi
    
    # \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432
    echo ""
    echo "\ud83d\udcca \u0421\u0442\u0430\u0442\u0443\u0441 \u0441\u0435\u0440\u0432\u0438\u0441\u043e\u0432:"
    docker-compose ps
    
    # \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u043e\u0432
    echo ""
    echo "\ud83c\udf10 \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u043e\u0432:"
    if curl -s http://localhost:7474 >/dev/null; then
        echo "\u2705 Neo4j \u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d"
    else
        echo "\u274c Neo4j \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d"
    fi
    
    if curl -s http://localhost:8001 >/dev/null; then
        echo "\u2705 Chroma \u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d"
    else
        echo "\u274c Chroma \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d"
    fi
    
    echo ""
    echo "\ud83d\udcbe \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u0435 \u0434\u0438\u0441\u043a\u0430:"
    du -sh /home/sda3/ai-agent/* 2>/dev/null || echo "\u0414\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u044f \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430"
}

# \u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0447\u0438\u043a \u043a\u043e\u043c\u0430\u043d\u0434
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
        echo "\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u0435: $0 {start|stop|restart|logs|cli|dev|add|backup|restore|update|health}"
        echo ""
        echo "\u041a\u043e\u043c\u0430\u043d\u0434\u044b:"
        echo "  start     - \u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0432\u0441\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b"
        echo "  stop      - \u041e\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c \u0432\u0441\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b"
        echo "  restart   - \u041f\u0435\u0440\u0435\u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0441\u0435\u0440\u0432\u0438\u0441\u044b"
        echo "  logs      - \u041f\u043e\u043a\u0430\u0437\u0430\u0442\u044c \u043b\u043e\u0433\u0438"
        echo "  cli       - \u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c CLI \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441"
        echo "  dev       - \u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0432 \u0440\u0435\u0436\u0438\u043c\u0435 \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u043a\u0438"
        echo "  add <path>- \u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b"
        echo "  backup    - \u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0440\u0435\u0437\u0435\u0440\u0432\u043d\u0443\u044e \u043a\u043e\u043f\u0438\u044e"
        echo "  restore   - \u0412\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c \u0438\u0437 \u0440\u0435\u0437\u0435\u0440\u0432\u043d\u043e\u0439 \u043a\u043e\u043f\u0438\u0438"
        echo "  update    - \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c"
        echo "  health    - \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0437\u0434\u043e\u0440\u043e\u0432\u044c\u044f \u0441\u0438\u0441\u0442\u0435\u043c\u044b"
        echo ""
        echo "\u041f\u0440\u0438\u043c\u0435\u0440\u044b:"
        echo "  ./run.sh start"
        echo "  ./run.sh add ~/documents"
        echo "  ./run.sh cli"
        exit 1
        ;;
esac
