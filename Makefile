# Makefile для AI-агента на базе Mistral AI 7B
 
.PHONY: help install start stop restart logs clean test health backup restore status
 
# Цвета для вывода
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color
 
# Переменные
COMPOSE_FILE=docker-compose.yml
BACKUP_DIR=/home/sda3/backups
PROJECT_NAME=mistral-ai-agent
 
help: ## Показать справку
	@echo "$(BLUE)AI-агент на базе Mistral AI 7B$(NC)"
	@echo "$(YELLOW)Доступные команды:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\
", $$1, $$2}' $(MAKEFILE_LIST)
 
install: ## Установка проекта
	@echo "$(BLUE)🚀 Установка AI-агента...$(NC)"
	@sudo ./scripts/setup.sh
	@echo "$(GREEN)✅ Установка завершена!$(NC)"
 
start: ## Запустить все сервисы
	@echo "$(BLUE)🐳 Запуск сервисов...$(NC)"
	@./scripts/run.sh start
	@echo "$(GREEN)✅ Сервисы запущены$(NC)"
 
stop: ## Остановить все сервисы
	@echo "$(BLUE)🛑 Остановка сервисов...$(NC)"
	@./scripts/run.sh stop
	@echo "$(GREEN)✅ Сервисы остановлены$(NC)"
 
restart: stop start ## Перезапустить сервисы
 
logs: ## Показать логи сервисов
	@echo "$(BLUE)📋 Логи сервисов:$(NC)"
	@./scripts/run.sh logs
 
cli: ## Запустить CLI интерфейс
	@echo "$(BLUE)💬 Запуск CLI интерфейса...$(NC)"
	@./scripts/run.sh cli
 
dev: ## Запуск в режиме разработки
	@echo "$(BLUE)🔧 Запуск в режиме разработки...$(NC)"
	@./scripts/run.sh dev
 
health: ## Проверка здоровья системы
	@echo "$(BLUE)🏥 Проверка здоровья системы...$(NC)"
	@./scripts/run.sh health
 
status: ## Показать статус всех сервисов
	@echo "$(BLUE)📊 Статус сервисов:$(NC)"
	@docker-compose ps
	@echo ""
	@echo "$(BLUE)GPU статус:$(NC)"
	@nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "$(YELLOW)GPU не найден$(NC)"
 
test: ## Запустить тесты
	@echo "$(BLUE)🧪 Запуск тестов...$(NC)"
	@if [ ! -d "tests" ]; then \
		echo "$(YELLOW)Создание директории тестов...$(NC)"; \
		mkdir -p tests; \
	fi
	@python -m pytest tests/ -v || echo "$(YELLOW)Тесты не найдены или не прошли$(NC)"
 
clean: ## Очистка кэша и временных файлов
	@echo "$(BLUE)🧹 Очистка...$(NC)"
	@docker system prune -f
	@docker volume prune -f
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Очистка завершена$(NC)"
 
build: ## Пересобрать Docker образы
	@echo "$(BLUE)🔨 Пересборка Docker образов...$(NC)"
	@docker-compose build --no-cache
	@echo "$(GREEN)✅ Образы пересобраны$(NC)"
 
pull: ## Обновить образы
	@echo "$(BLUE)📥 Обновление Docker образов...$(NC)"
	@docker-compose pull
	@echo "$(GREEN)✅ Образы обновлены$(NC)
 
backup: ## Создать резервную копию
	@echo "$(BLUE)💾 Создание резервной копии...$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@./scripts/run.sh backup
	@echo "$(GREEN)✅ Резервная копия создана$(NC)"
 
restore: ## Восстановить из резервной копии (используйте: make restore BACKUP_DIR=/path/to/backup)
	@if [ -z "$(BACKUP_DIR)" ]; then \
		echo "$(RED)❌ Укажите путь к резервной копии: make restore BACKUP_DIR=/path/to/backup$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)🔄 Восстановление из $(BACKUP_DIR)...$(NC)"
	@./scripts/run.sh restore $(BACKUP_DIR)
	@echo "$(GREEN)✅ Восстановление завершено$(NC)"
 
add-docs: ## Добавить документы (используйте: make add-docs DOCS_PATH=/path/to/docs)
	@if [ -z "$(DOCS_PATH)" ]; then \
		echo "$(RED)❌ Укажите путь к документам: make add-docs DOCS_PATH=/path/to/docs$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)📚 Добавление документов из $(DOCS_PATH)...$(NC)"
	@./scripts/run.sh add $(DOCS_PATH)
	@echo "$(GREEN)✅ Документы добавлены$(NC)"
 
train: ## Обучить модель
	@echo "$(BLUE)🎮 Запуск обучения модели...$(NC)"
	@docker-compose exec ai-agent python -c "
from src.cli_interface import AIAgentCLI
cli = AIAgentCLI()
cli._train_model()
"
	@echo "$(GREEN)✅ Обучение завершено$(NC)"
 
monitor: ## Мониторинг ресурсов
	@echo "$(BLUE)📊 Мониторинг ресурсов:$(NC)"
	@echo "$(BLUE)Использование памяти:$(NC)"
	@free -h
	@echo ""
	@echo "$(BLUE)Использование диска:$(NC)"
	@df -h /home/sda3
	@echo ""
	@echo "$(BLUE)GPU статус:$(NC)"
	@nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "$(YELLOW)GPU не найден$(NC)"
 
update-model: ## Обновить базовую модель
	@echo "$(BLUE)🔄 Обновление базовой модели...$(NC)"
	@./scripts/run.sh update-model
	@echo "$(GREEN)✅ Модель обновлена$(NC)"
 
setup-dev: ## Настройка окружения для разработки
	@echo "$(BLUE)🔧 Настройка окружения разработки...$(NC)"
	@if [ ! -d "venv" ]; then \
		python3 -m venv venv; \
		echo "$(YELLOW)Виртуальное окружение создано$(NC)"; \
	fi
	@source venv/bin/activate && pip install -r requirements.txt
	@pre-commit install 2>/dev/null || echo "$(YELLOW)pre-commit не установлен$(NC)"
	@echo "$(GREEN)✅ Окружение разработки настроено$(NC)"
 
lint: ## Проверка кода
	@echo "$(BLUE)🔍 Проверка кода...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black --check src/; \
	else \
		echo "$(YELLOW)black не установлен$(NC)"; \
	fi
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/; \
	else \
		echo "$(YELLOW)flake8 не установлен$(NC)"; \
	fi
 
format: ## Форматирование кода
	@echo "$(BLUE)🎨 Форматирование кода...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black src/; \
		echo "$(GREEN)✅ Код отформатирован$(NC)"; \
	else \
		echo "$(YELLOW)black не установлен. Установите: pip install black$(NC)"; \
	fi
 
docs: ## Открыть документацию
	@echo "$(BLUE)📚 Документация:$(NC)"
	@echo "README: $(PWD)/docs/README.md"
	@echo "Examples: $(PWD)/docs/examples.md"
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open docs/README.md; \
	elif command -v open >/dev/null 2>&1; then \
		open docs/README.md; \
	else \
		echo "$(YELLOW)Откройте документацию вручную$(NC)"; \
	fi
 
# Команды для работы с моделями
list-models: ## Показать доступные модели
	@echo "$(BLUE)🤖 Доступные модели:$(NC)"
	@ls -la /home/sda3/ai-agent/models/ 2>/dev/null || echo "$(YELLOW)Директория моделей не найдена$(NC)"
 
save-model: ## Сохранить текущую модель
	@echo "$(BLUE)💾 Сохранение модели...$(NC)"
	@read -p "Введите имя модели: " model_name; \
	./scripts/run.sh save-model $$model_name
	@echo "$(GREEN)✅ Модель сохранена$(NC)"
 
# Полезные команды
info: ## Показать информацию о проекте
	@echo "$(BLUE)📋 Информация о проекте:$(NC)"
	@echo "Название: $(PROJECT_NAME)"
	@echo "Директория: $(PWD)"
	@echo "Docker Compose: $(COMPOSE_FILE)"
	@echo ""
	@echo "$(BLUE)Пути к данным:$(NC)"
	@echo "Модели: /home/sda3/ai-agent/models"
	@echo "Документы: /home/sda3/ai-agent/documents"
	@echo "Кэш: /home/sda3/ai-agent/cache"
	@echo "Бэкапы: $(BACKUP_DIR)"
	@echo ""
	@echo "$(BLUE)Сетевые порты:$(NC)"
	@echo "Neo4j: http://localhost:7474"
	@echo "Neo4j Bolt: localhost:7687"
	@echo "Chroma: http://localhost:8001"
	@echo "Agent (future): http://localhost:8000"
 
# Установка по умолчанию
.DEFAULT_GOAL := help