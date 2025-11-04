"""
CLI интерфейс для AI-агента
Интерактивная консоль с поддержкой команд, истории и удобного вывода
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

# CLI библиотеки
import click
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree

# Local imports
from config import config
from agent import create_ai_agent, AIAgent, AgentResponse
from document_processor import create_document_processor
from model_training import create_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAgentCLI:
    """CLI интерфейс для AI-агента"""
    
    def __init__(self):
        self.console = Console()
        self.agent: Optional[AIAgent] = None
        self.command_history = []
        self.session_start = datetime.now()
        
        # Пути и файлы
        self.history_file = Path(config.data.cache_path) / "cli_history.json"
        self.context_file = config.cli.context_file
        
        # Инициализация
        self._init_agent()
        self._load_history()
    
    def _init_agent(self):
        """Инициализация AI-агента"""
        try:
            self.console.print("[bold green]Иніціалізація AI-агента...[/bold green]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Завантаження моделі...", total=None)
                
                # Поиск кастомной модели
                custom_model_path = self._find_custom_model()
                
                self.agent = create_ai_agent(custom_model_path)
                
                progress.update(task, description="Агент готовий!")
            
            self.console.print("[bold green]✓[/bold green] AI-агент успішно ініціалізований")
            
        except Exception as e:
            self.console.print(f"[bold red]Помилка ініціалізації агента:[/bold red] {e}")
            self.console.print("[yellow]Продовження в обмеженому режимі...[/yellow]")
    
    def _find_custom_model(self) -> Optional[str]:
        """Поиск кастомной обученной модели"""
        try:
            models_dir = Path(config.model.cache_dir)
            if not models_dir.exists():
                return None
            
            # Поиск директорий с обученными моделями
            for item in models_dir.iterdir():
                if item.is_dir() and "fine_tuned" in item.name:
                    # Проверка наличия необходимых файлов
                    if (item / "config.json").exists() and (item / "pytorch_model.bin").exists():
                        self.console.print(f"[cyan]Знайдено кастомну модель: {item.name}[/cyan]")
                        return str(item)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding custom model: {e}")
            return None
    
    def _load_history(self):
        """Загрузка истории команд"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.command_history = data.get('commands', [])
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.command_history = []
    
    def _save_history(self):
        """Сохранение истории команд"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'commands': self.command_history[-100:],  # Сохраняем последние 100 команд
                    'last_session': self.session_start.isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def _format_response(self, response: AgentResponse) -> None:
        """Форматирование и вывод ответа"""
        # Основной ответ
        content = response.content.strip()
        
        if content:
            # Определяем тип контента для форматирования
            if self._is_code_content(content):
                # Вывод кода с подсветкой синтаксиса
                syntax = Syntax(content, "python", theme="monokai", line_numbers=True)
                self.console.print(Panel(syntax, title="💻 Код", border_style="blue"))
            else:
                # Обычный текст
                self.console.print(Panel(
                    Markdown(content),
                    title="🤖 Відповідь агента",
                    border_style="green"
                ))
        
        # Метаданные ответа
        metadata_table = Table(show_header=False, box=None)
        metadata_table.add_column("Параметр", style="cyan")
        metadata_table.add_column("Значення", style="white")
        
        metadata_table.add_row("Час відповіді", f"{response.response_time:.2f}с")
        metadata_table.add_row("Впевненість", f"{response.confidence:.2f}")
        metadata_table.add_row("Контекст використано", "✓" if response.context_used else "✗")
        
        self.console.print(metadata_table)
        
        # Источники
        if response.sources:
            self.console.print("\
[bold]📚 Джерела:[/bold]")
            for i, source in enumerate(response.sources, 1):
                source_title = source.get('metadata', {}).get('file_name', f'Джерело {i}')
                relevance = source.get('relevance', 0)
                
                self.console.print(f"{i}. {source_title} (релевантність: {relevance:.2f})")
                
                # Показываем часть контента
                content_preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                self.console.print(f"   [dim]{content_preview}[/dim]")
    
    def _is_code_content(self, content: str) -> bool:
        """Определение является ли контент кодом"""
        code_indicators = [
            "def ", "class ", "import ", "from ", "function", "var ", "let ", "const ",
            "if ", "for ", "while ", "try:", "except:", "catch", "{", "}", "=>"
        ]
        
        lines = content.split('\
')
        code_lines = sum(1 for line in lines if any(indicator in line for indicator in code_indicators))
        
        return code_lines > len(lines) * 0.3  # Если 30% строк содержат кодовые индикаторы
    
    def _show_help(self) -> None:
        """Показ справки"""
        help_text = """
[bold blue]Доступні команди:[/bold blue]

[green]Основні команди:[/green]
  • [cyan]help[/cyan] - показати цю довідку
  • [cyan]status[/cyan] - статус агента та системи
  • [cyan]clear[/cyan] - очистити екран
  • [cyan]history[/cyan] - показати історію команд
  • [cyan]exit[/cyan] або [cyan]quit[/cyan] - вихід

[green]Робота з документами:[/green]
  • [cyan]add <путь>[/cyan] - додати документ/директорію
  • [cyan]list-docs[/cyan] - показати додані документи
  • [cyan]train[/cyan] - навчити модель на документах
  • [cyan]search <запрос>[/cyan] - пошук по документах

[green]Навчання моделі:[/green]
  • [cyan]train-on <путь>[/cyan] - навчити на вказаних документах
  • [cyan]save-model <назва>[/cyan] - зберегти навчену модель

[green]Налаштування:[/green]
  • [cyan]set-temp <значение>[/cyan] - встановити температуру (0.0-1.0)
  • [cyan]set-tokens <число>[/cyan] - встановити макс. токенів

[green]Приклади використання:[/green]
  • Напиши функцію для сортування масиву на Python
  • Як скласти позовну заяву про стягнення боргу?
  • Проаналізуй цей код та знайди помилки
  • Які права виникають при укладенні договору оренди?
        """
        
        self.console.print(Panel(
            Markdown(help_text),
            title="📖 Довідка",
            border_style="blue"
        ))
    
    def _show_status(self) -> None:
        """Показ статуса агента"""
        try:
            if self.agent:
                status = self.agent.get_agent_status()
                
                # Основная таблица статуса
                status_table = Table(title="📊 Статус AI-агента", box=None)
                status_table.add_column("Параметр", style="cyan")
                status_table.add_column("Значення", style="white")
                
                status_table.add_row("Модель завантажена", "✓" if status.get('model_loaded') else "✗")
                status_table.add_row("Компоненти ініціалізовані", "✓" if status.get('components_initialized') else "✗")
                status_table.add_row("Назва моделі", status.get('model_name', 'Невідомо'))
                status_table.add_row("Пристрій", status.get('device', 'Невідомо'))
                
                self.console.print(status_table)
                
                # Статистика векторного хранилища
                vector_stats = status.get('vector_store_stats', {})
                if vector_stats:
                    self.console.print("\
[bold]📚 Векторне сховище:[/bold]")
                    vector_table = Table(show_header=True, box=None)
                    vector_table.add_column("Параметр", style="cyan")
                    vector_table.add_column("Значення", style="white")
                    
                    for key, value in vector_stats.items():
                        vector_table.add_row(key, str(value))
                    
                    self.console.print(vector_table)
                
                # Статистика knowledge graph
                graph_stats = status.get('knowledge_graph_stats', {})
                if graph_stats:
                    self.console.print("\
[bold]🕸️ Knowledge Graph:[/bold]")
                    graph_table = Table(show_header=True, box=None)
                    graph_table.add_column("Параметр", style="cyan")
                    graph_table.add_column("Значення", style="white")
                    
                    for key, value in graph_stats.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                graph_table.add_row(f"{key}.{sub_key}", str(sub_value))
                        else:
                            graph_table.add_row(key, str(value))
                    
                    self.console.print(graph_table)
            else:
                self.console.print("[red]Агент не ініціалізований[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Помилка отримання статусу:[/red] {e}")
    
    def _add_documents(self, path: str) -> None:
        """Добавление документов"""
        try:
            path_obj = Path(path)
            
            if not path_obj.exists():
                self.console.print(f"[red]Шлях не існує: {path}[/red]")
                return
            
            with Progress(console=self.console) as progress:
                task = progress.add_task("Додавання документів...", total=None)
                
                if path_obj.is_file():
                    document_files = [str(path_obj)]
                elif path_obj.is_dir():
                    document_files = []
                    for root, dirs, files in os.walk(path_obj):
                        for file in files:
                            if file.endswith(('.pdf', '.docx', '.doc', '.txt', '.md')):
                                document_files.append(os.path.join(root, file))
                else:
                    self.console.print(f"[red]Непідтримуваний тип об'єкта: {path}[/red]")
                    return
                
                if not document_files:
                    self.console.print("[yellow]Документів не знайдено[/yellow]")
                    return
                
                progress.update(task, description=f"Обробка {len(document_files)} файлів...")
                
                if self.agent:
                    result = self.agent.add_documents(document_files)
                    
                    if result['success']:
                        progress.update(task, description="✓ Документи додано успішно!")
                        self.console.print(f"[green]✓[/green] {result['message']}")
                    else:
                        progress.update(task, description="✗ Помилка додавання")
                        self.console.print(f"[red]✗[/red] {result['message']}")
                else:
                    self.console.print("[red]Агент не ініціалізований[/red]")
                    
        except Exception as e:
            self.console.print(f"[red]Помилка додавання документів:[/red] {e}")
    
    def _search_documents(self, query: str) -> None:
        """Поиск по документам"""
        try:
            if not self.agent:
                self.console.print("[red]Агент не ініціалізований[/red]")
                return
            
            with Progress(console=self.console) as progress:
                task = progress.add_task("Пошук по документах...", total=None)
                
                context_items = self.agent.retrieve_relevant_context(query)
                
                progress.update(task, description=f"Знайдено {len(context_items)} результатів")
            
            if context_items:
                self.console.print(f"\
[bold]Результати пошуку для:[/bold] '{query}'\
")
                
                for i, item in enumerate(context_items, 1):
                    source_name = item.get('metadata', {}).get('file_name', f'Джерело {i}')
                    relevance = item.get('relevance_score', 0)
                    
                    self.console.print(f"{i}. [cyan]{source_name}[/cyan] (релевантність: {relevance:.2f})")
                    
                    # Показываем часть контента
                    content = item['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    self.console.print(f"   [dim]{content}[/dim]\
")
            else:
                self.console.print("[yellow]Результатів не знайдено[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]Помилка пошуку:[/red] {e}")
    
    def _train_model(self) -> None:
        """Обучение модели"""
        try:
            if not self.agent:
                self.console.print("[red]Агент не ініціалізований[/red]")
                return
            
            self.console.print("[yellow]Навчання моделі може зайняти багато часу...[/yellow]")
            
            if Prompt.ask("Продовжити?", choices=["y", "n"], default="n") == "y":
                with Progress(console=self.console) as progress:
                    task = progress.add_task("Навчання моделі...", total=None)
                    
                    result = self.agent.train_on_documents()
                    
                    if result['success']:
                        progress.update(task, description="✓ Навчання завершено!")
                        self.console.print("[green]✓ Модель успішно навчена[/green]")
                        
                        # Показываем информацию о сохраненной модели
                        training_result = result.get('training_result', {})
                        if 'save_path' in training_result:
                            self.console.print(f"[cyan]Модель збережена в:[/cyan] {training_result['save_path']}")
                    else:
                        progress.update(task, description="✗ Помилка навчання")
                        self.console.print(f"[red]✗ Помилка навчання:[/red] {result['message']}")
            else:
                self.console.print("Навчання скасовано")
                
        except Exception as e:
            self.console.print(f"[red]Помилка навчання:[/red] {e}")
    
    def _set_parameter(self, param: str, value: str) -> None:
        """Установка параметров"""
        try:
            if param == "temp":
                temp_value = float(value)
                if 0.0 <= temp_value <= 1.0:
                    config.model.temperature = temp_value
                    self.console.print(f"[green]✓ Температура встановлена: {temp_value}[/green]")
                else:
                    self.console.print("[red]Температура повинна бути в діапазоні 0.0-1.0[/red]")
            
            elif param == "tokens":
                tokens_value = int(value)
                if tokens_value > 0 and tokens_value <= 4096:
                    config.model.max_new_tokens = tokens_value
                    self.console.print(f"[green]✓ Макс. токенів встановлено: {tokens_value}[/green]")
                else:
                    self.console.print("[red]Кількість токенів повинна бути в діапазоні 1-4096[/red]")
            
            else:
                self.console.print(f"[red]Невідомий параметр: {param}[/red]")
                
        except ValueError as e:
            self.console.print(f"[red]Помилка значення:[/red] {e}")
        except Exception as e:
            self.console.print(f"[red]Помилка встановлення параметра:[/red] {e}")
    
    def _process_command(self, user_input: str) -> bool:
        """Обработка команд"""
        user_input = user_input.strip()
        
        if not user_input:
            return True
        
        # Сохранение в историю
        self.command_history.append({
            'command': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Обработка команд
        if user_input.lower() in ['exit', 'quit']:
            return False
        
        elif user_input.lower() == 'help':
            self._show_help()
        
        elif user_input.lower() == 'status':
            self._show_status()
        
        elif user_input.lower() == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
        
        elif user_input.lower() == 'history':
            self._show_history()
        
        elif user_input.lower().startswith('add '):
            path = user_input[4:].strip()
            self._add_documents(path)
        
        elif user_input.lower().startswith('search '):
            query = user_input[7:].strip()
            self._search_documents(query)
        
        elif user_input.lower() == 'train':
            self._train_model()
        
        elif user_input.lower().startswith('set-'):
            parts = user_input[4:].split(' ', 1)
            if len(parts) == 2:
                self._set_parameter(parts[0], parts[1])
            else:
                self.console.print("[red]Неправильний формат команди. Використання: set-<param> <value>[/red]")
        
        elif user_input.lower().startswith('/'):
            # Системные команды
            self.console.print(f"[dim]Системна команда: {user_input}[/dim]")
        
        else:
            # Обычный запрос к AI
            self._process_ai_query(user_input)
        
        return True
    
    def _process_ai_query(self, query: str) -> None:
        """Обработка AI запроса"""
        try:
            if not self.agent:
                self.console.print("[red]Агент не ініціалізований. Неможливо обробити запит.[/red]")
                return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Генерація відповіді...", total=None)
                
                response = self.agent.query(query)
                
                progress.update(task, description="✓ Відповідь готова!")
            
            # Вывод ответа
            self._format_response(response)
            
        except Exception as e:
            self.console.print(f"[red]Помилка обробки запиту:[/red] {e}")
    
    def _show_history(self) -> None:
        """Показ истории команд"""
        if not self.command_history:
            self.console.print("[yellow]Історія порожня[/yellow]")
            return
        
        history_table = Table(title="📜 Історія команд", show_header=True)
        history_table.add_column("№", style="cyan", width=4)
        history_table.add_column("Команда", style="white")
        history_table.add_column("Час", style="dim")
        
        for i, cmd in enumerate(self.command_history[-20:], 1):  # Показываем последние 20
            timestamp = cmd.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp[:8]
            else:
                time_str = 'Невідомо'
            
            command = cmd.get('command', '')
            if len(command) > 60:
                command = command[:57] + "..."
            
            history_table.add_row(str(i), command, time_str)
        
        self.console.print(history_table)
    
    def run(self) -> None:
        """Основной цикл CLI"""
        # Показ приветствия
        welcome_text = """
# 🤖 AI-Агент на базі Mistral AI 7B

Ласкаво просимо до інтерактивного CLI інтерфейсу!

Агент спеціалізується на:
• 💻 Програмуванні та аналізі коду
• ⚖️ Юридичних питаннях (законодавство України)
• 📚 Роботі з документами

Введіть [cyan]help[/cyan] для списку команд або почніть ставити запитання!
        """
        
        self.console.print(Panel(
            Markdown(welcome_text),
            title="AI-Агент",
            border_style="green"
        ))
        
        # Основной цикл
        while True:
            try:
                # Ввод пользователя
                user_input = Prompt.ask(
                    "\
[bold blue]🔍 Ваш запит[/bold blue]",
                    default="",
                    show_default=False
                )
                
                if not user_input.strip():
                    continue
                
                # Обработка команды
                should_continue = self._process_command(user_input)
                
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\
[yellow]Завершення роботи...[/yellow]")
                break
            except EOFError:
                self.console.print("\
[yellow]Завершення роботи...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"\
[red]Виникла помилка:[/red] {e}")
        
        # Сохранение истории и выход
        self._save_history()
        self.console.print("[bold green]Дякуємо за використання AI-агента![/bold green]")

# Основная функция для запуска CLI
def main():
    """Главная функция"""
    try:
        cli = AIAgentCLI()
        cli.run()
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Критична помилка запуску CLI:[/bold red] {e}")
        logger.error(f"Critical CLI error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()