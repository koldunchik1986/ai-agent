"""
CLI ИНТЕРФЕЙС ДЛЯ AI-АССИСТЕНТА

Реализует интерактивный терминальный интерфейс с:
- Командами для управления
- Подсветкой синтаксиса через Rich
- Индикаторами прогресса
- Поддержкой истории команд
"""

import sys
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown

# Инициализация Rich console
console = Console()

class CLI:
    """
    КОМАНДНЫЙ ИНТЕРФЕЙС АССИСТЕНТА
    
    Команды:
    - /help - показать справку
    - /status - статус системы
    - /add <файл> - добавить документ
    - /project <путь> - добавить проект
    - /train - дообучить модель
    - /clear - очистить экран
    - /code <файл> - анализировать код
    - exit/quit - выход
    """
    
    def __init__(self):
        self.assistant = None
        self._init_assistant()
    
    def _init_assistant(self):
        """Инициализация ассистента с индикатором прогресса"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Загрузка AI-Ассистента...", total=None)
            
            try:
                from agent import AIAssistant
                self.assistant = AIAssistant()
                progress.update(task, description="✅ Ассистент готов!")
            except Exception as e:
                progress.update(task, description=f"❌ Ошибка: {str(e)[:50]}...")
                console.print(f"\n[red]Детали ошибки:[/red] {e}")
                sys.exit(1)
    
    def print_welcome(self):
        """Вывод приветственного сообщения"""
        welcome_text = """
🤖 [bold blue]AI-Ассистент для розробки[/bold blue]
[dim]Оптимізовано для P104-100 8GB VRAM (sm_61)[/dim]

📋 [bold]Доступні команди:[/bold]
- /help - показати цю довідку
- /status - статус системи
- /add <файл> - додати документ
- /project <путь> - додати проект
- /train - дообучити модель
- /clear - очистити екран

💡 [bold]Просто пишіть запитання[/bold] для чату з RAG
        """
        
        console.print(Panel(
            welcome_text,
            title="🚀 Ласкаво просимо!",
            border_style="blue"
        ))
    
    def print_help(self):
        """Вывод справки"""
        table = Table(title="📚 Справка по командам", show_header=True, header_style="bold cyan")
        table.add_column("Команда", style="cyan")
        table.add_column("Опис", style="white")
        table.add_column("Приклад", style="dim")
        
        table.add_row("/help", "Показати цю довідку", "/help")
        table.add_row("/status", "Статус GPU і моделі", "/status")
        table.add_row("/add <файл>", "Додати документ", "/add /app/data/docs/file.pdf")
        table.add_row("/project <путь>", "Додати проект IDE", "/project /workspace/myapp")
        table.add_row("/train", "Дообучити модель", "/train")
        table.add_row("/clear", "Очистити екран", "/clear")
        table.add_row("/code <файл>", "Аналізувати код", "/code /app/src/main.py")
        table.add_row("exit/quit", "Вийти", "exit")
        
        console.print(table)
        
        console.print("\n💬 [bold]Чат:[/bold] просто пишіть запитання, наприклад:")
        console.print("    Як працює цей метод?")
        console.print("    Знайди помилку в коді")
    
    def print_status(self):
        """Вывод статуса системы"""
        status = self.assistant.get_status()
        
        # GPU Status Table
        gpu_table = Table(title="🎮 Статус GPU")
        gpu_table.add_column("Параметр", style="cyan")
        gpu_table.add_column("Значення", style="white")
        
        gpu_table.add_row("Модель", str(status["gpu"]))
        gpu_table.add_row("VRAM використано", f"{status['vram_used_gb']:.2f}GB")
        gpu_table.add_row("VRAM всього", f"{status['vram_total_gb']}GB")
        gpu_table.add_row("Використання", f"{(status['vram_used_gb']/status['vram_total_gb']*100):.1f}%")
        
        console.print(gpu_table)
        
        # Assistant Status Table
        assistant_table = Table(title="🤖 Статус Ассистента")
        assistant_table.add_column("Компонент", style="cyan")
        assistant_table.add_column("Статус", style="white")
        
        assistant_table.add_row("Модель завантажена", "✅" if status["model_loaded"] else "❌")
        assistant_table.add_row("RAG готовий", "✅" if status["rag_ready"] else "❌")
        assistant_table.add_row("Документів у БД", str(status["documents_db"]))
        
        console.print(assistant_table)
    
    def run(self):
        """Главный цикл CLI"""
        self.print_welcome()
        
        while True:
            try:
                # Запрос ввода
                user_input = Prompt.ask("\n[bold cyan]Ви[/bold cyan]")
                
                # Обработка команд
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("[dim]👋 До побачення![/dim]")
                    break
                
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # Обычный чат
                if user_input.strip():
                    self._handle_chat(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[dim]Перервано[/dim]")
                continue
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]❌ Неочікувана помилка: {e}[/red]")
    
    def _handle_command(self, cmd: str):
        """Обработка команд"""
        parts = cmd.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == '/help':
            self.print_help()
        
        elif command == '/status':
            self.print_status()
        
        elif command == '/add':
            self._add_document(args)
        
        elif command == '/project':
            self._add_project(args)
        
        elif command == '/train':
            self._train_model()
        
        elif command == '/clear':
            os.system('clear')
        
        elif command == '/code':
            self._analyze_code(args)
        
        else:
            console.print(f"[red]❌ Невідома команда: {command}[/red]")
            console.print("[dim]Використовуйте /help для списку команд[/dim]")
    
    def _handle_chat(self, question: str):
        """Обработка чат-запроса"""
        try:
            with console.status("[bold yellow]🤖 Думаю...[/bold yellow]"):
                response = self.assistant.chat(question)
            
            # Форматирование ответа
            # Попытка определить, это код или текст
            if "```" in response:
                # Кодовые блоки через Rich Syntax
                console.print("\n[bold green]🤖 Ассистент:[/bold green]")
                console.print(Markdown(response))
            else:
                # Обычный текст в панели
                console.print(Panel(
                    response,
                    title="🤖 Ассистент",
                    border_style="green"
                ))
            
            # Показать краткую статистику
            status = self.assistant.get_status()
            console.print(
                f"[dim]VRAM: {status['vram_used_gb']:.2f}GB | "
                f"Документів: {status['documents_db']}[/dim]"
            )
        
        except Exception as e:
            console.print(f"[red]❌ Помилка генерації: {e}[/red]")
    
    def _add_document(self, file_path: str):
        """Добавить документ"""
        if not file_path:
            console.print("[red]❌ Вкажіть шлях до файлу: /add /path/to/file.pdf[/red]")
            return
        
        # Проверка существования файла
        if not os.path.exists(file_path):
            console.print(f"[red]❌ Файл не знайден: {file_path}[/red]")
            return
        
        with console.status(f"[yellow]📄 Обробка {file_path}...[/yellow]"):
            success = self.assistant.add_document(file_path)
        
        if success:
            console.print(f"[green]✅ Документ додано:[/green] {os.path.basename(file_path)}")
        else:
            console.print(f"[red]❌ Помилка обробки[/red]")
    
    def _add_project(self, project_path: str):
        """Добавить проект"""
        if not project_path:
            console.print("[red]❌ Вкажіть шлях до проекту: /project /path/to/project[/red]")
            return
        
        if not os.path.exists(project_path):
            console.print(f"[red]❌ Проект не знайден: {project_path}[/red]")
            return
        
        with console.status(f"[yellow]📂 Сканування проекту...[/yellow]"):
            result = self.assistant.add_project(project_path)
        
        if result["success"]:
            console.print(
                f"[green]✅ Проект додано:[/green] {result['processed_files']} файлів"
            )
        else:
            console.print(f"[red]❌ Помилка: {result.get('error', 'невідома')}[/red]")
    
    def _train_model(self):
        """Дообучить модель"""
        confirm = Prompt.ask(
            "\n[yellow]⚠️  Дообучення займе 2-4 години на 1000 документів. Продовжити? (y/n)[/yellow]"
        )
        
        if confirm.lower() != 'y':
            return
        
        output_name = Prompt.ask(
            "\nНазва збереженої моделі (Enter для автоматичної)",
            default=f"lora_{os.path.basename(os.getcwd())}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        output_dir = f"/app/data/models/{output_name}"
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                
                task = progress.add_task(
                    "[yellow]🎯 Дообучення моделі... (це може зайняти кілька годин)[/yellow]",
                    total=None
                )
                
                result = self.assistant.train_on_documents(output_dir)
                
                progress.update(task, description="✅ Дообучення завершено!")
            
            console.print(f"\n[green]✅ Модель збережена:[/green] {result['output_dir']}")
            console.print(f"[dim]Потреби часу: {result['training_duration']}[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ Помилка дообучення: {e}[/red]")
    
    def _analyze_code(self, file_path: str):
        """Анализ кода"""
        if not file_path:
            console.print("[red]❌ Вкажіть файл: /code /path/to/file.py[/red]")
            return
        
        if not os.path.exists(file_path):
            console.print(f"[red]❌ Файл не знайден: {file_path}[/red]")
            return
        
        console.print(f"\n[blue]📄 Аналіз коду:[/blue] {file_path}")
        
        try:
            with console.status("[yellow]🔍 Аналізую код...[/yellow]"):
                analysis = self.assistant.analyze_code_file(file_path)
            
            # Попытка определить язык по расширению
            ext = Path(file_path).suffix.lower()
            lexer_map = {
                '.py': 'python',
                '.java': 'java',
                '.kt': 'kotlin',
                '.js': 'javascript',
                '.html': 'html',
                '.xml': 'xml',
            }
            lexer = lexer_map.get(ext, 'text')
            
            # Вывод анализа
            if "```" in analysis:
                # Если в ответе есть кодовые блоки
                console.print("\n[bold green]🤖 Аналіз:[/bold green]")
                console.print(Markdown(analysis))
            else:
                console.print(Panel(
                    analysis,
                    title="🤖 Аналіз коду",
                    border_style="green"
                ))
        
        except Exception as e:
            console.print(f"[red]❌ Помилка аналізу: {e}[/red]")

def main():
    """Точка входа в CLI"""
    cli = CLI()
    cli.run()

if __name__ == "__main__":
    main()