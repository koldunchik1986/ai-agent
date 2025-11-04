
"""
CLI \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441 \u0434\u043b\u044f AI-\u0430\u0433\u0435\u043d\u0442\u0430
\u0418\u043d\u0442\u0435\u0440\u0430\u043a\u0442\u0438\u0432\u043d\u0430\u044f \u043a\u043e\u043d\u0441\u043e\u043b\u044c \u0441 \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u043e\u0439 \u043a\u043e\u043c\u0430\u043d\u0434, \u0438\u0441\u0442\u043e\u0440\u0438\u0438 \u0438 \u0443\u0434\u043e\u0431\u043d\u043e\u0433\u043e \u0432\u044b\u0432\u043e\u0434\u0430
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

# CLI \u0431\u0438\u0431\u043b\u0438\u043e\u0442\u0435\u043a\u0438
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
    """CLI \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441 \u0434\u043b\u044f AI-\u0430\u0433\u0435\u043d\u0442\u0430"""
    
    def __init__(self):
        self.console = Console()
        self.agent: Optional[AIAgent] = None
        self.command_history = []
        self.session_start = datetime.now()
        
        # \u041f\u0443\u0442\u0438 \u0438 \u0444\u0430\u0439\u043b\u044b
        self.history_file = Path(config.data.cache_path) / "cli_history.json"
        self.context_file = config.cli.context_file
        
        # \u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f
        self._init_agent()
        self._load_history()
    
    def _init_agent(self):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f AI-\u0430\u0433\u0435\u043d\u0442\u0430"""
        try:
            self.console.print("[bold green]\u0418\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u0430\u0446\u0456\u044f AI-\u0430\u0433\u0435\u043d\u0442\u0430...[/bold green]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("\u0417\u0430\u0432\u0430\u043d\u0442\u0430\u0436\u0435\u043d\u043d\u044f \u043c\u043e\u0434\u0435\u043b\u0456...", total=None)
                
                # \u041f\u043e\u0438\u0441\u043a \u043a\u0430\u0441\u0442\u043e\u043c\u043d\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438
                custom_model_path = self._find_custom_model()
                
                self.agent = create_ai_agent(custom_model_path)
                
                progress.update(task, description="\u0410\u0433\u0435\u043d\u0442 \u0433\u043e\u0442\u043e\u0432\u0438\u0439!")
            
            self.console.print("[bold green]\u2713[/bold green] AI-\u0430\u0433\u0435\u043d\u0442 \u0443\u0441\u043f\u0456\u0448\u043d\u043e \u0456\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u043e\u0432\u0430\u043d\u0438\u0439")
            
        except Exception as e:
            self.console.print(f"[bold red]\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u0456\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u0430\u0446\u0456\u0457 \u0430\u0433\u0435\u043d\u0442\u0430:[/bold red] {e}")
            self.console.print("[yellow]\u041f\u0440\u043e\u0434\u043e\u0432\u0436\u0435\u043d\u043d\u044f \u0432 \u043e\u0431\u043c\u0435\u0436\u0435\u043d\u043e\u043c\u0443 \u0440\u0435\u0436\u0438\u043c\u0456...[/yellow]")
    
    def _find_custom_model(self) -> Optional[str]:
        """\u041f\u043e\u0438\u0441\u043a \u043a\u0430\u0441\u0442\u043e\u043c\u043d\u043e\u0439 \u043e\u0431\u0443\u0447\u0435\u043d\u043d\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438"""
        try:
            models_dir = Path(config.model.cache_dir)
            if not models_dir.exists():
                return None
            
            # \u041f\u043e\u0438\u0441\u043a \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0439 \u0441 \u043e\u0431\u0443\u0447\u0435\u043d\u043d\u044b\u043c\u0438 \u043c\u043e\u0434\u0435\u043b\u044f\u043c\u0438
            for item in models_dir.iterdir():
                if item.is_dir() and "fine_tuned" in item.name:
                    # \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043d\u0430\u043b\u0438\u0447\u0438\u044f \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u044b\u0445 \u0444\u0430\u0439\u043b\u043e\u0432
                    if (item / "config.json").exists() and (item / "pytorch_model.bin").exists():
                        self.console.print(f"[cyan]\u0417\u043d\u0430\u0439\u0434\u0435\u043d\u043e \u043a\u0430\u0441\u0442\u043e\u043c\u043d\u0443 \u043c\u043e\u0434\u0435\u043b\u044c: {item.name}[/cyan]")
                        return str(item)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding custom model: {e}")
            return None
    
    def _load_history(self):
        """\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0438\u0441\u0442\u043e\u0440\u0438\u0438 \u043a\u043e\u043c\u0430\u043d\u0434"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.command_history = data.get('commands', [])
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.command_history = []
    
    def _save_history(self):
        """\u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0438\u0441\u0442\u043e\u0440\u0438\u0438 \u043a\u043e\u043c\u0430\u043d\u0434"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'commands': self.command_history[-100:],  # \u0421\u043e\u0445\u0440\u0430\u043d\u044f\u0435\u043c \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 100 \u043a\u043e\u043c\u0430\u043d\u0434
                    'last_session': self.session_start.isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def _format_response(self, response: AgentResponse) -> None:
        """\u0424\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0438 \u0432\u044b\u0432\u043e\u0434 \u043e\u0442\u0432\u0435\u0442\u0430"""
        # \u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043e\u0442\u0432\u0435\u0442
        content = response.content.strip()
        
        if content:
            # \u041e\u043f\u0440\u0435\u0434\u0435\u043b\u044f\u0435\u043c \u0442\u0438\u043f \u043a\u043e\u043d\u0442\u0435\u043d\u0442\u0430 \u0434\u043b\u044f \u0444\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f
            if self._is_code_content(content):
                # \u0412\u044b\u0432\u043e\u0434 \u043a\u043e\u0434\u0430 \u0441 \u043f\u043e\u0434\u0441\u0432\u0435\u0442\u043a\u043e\u0439 \u0441\u0438\u043d\u0442\u0430\u043a\u0441\u0438\u0441\u0430
                syntax = Syntax(content, "python", theme="monokai", line_numbers=True)
                self.console.print(Panel(syntax, title="\ud83d\udcbb \u041a\u043e\u0434", border_style="blue"))
            else:
                # \u041e\u0431\u044b\u0447\u043d\u044b\u0439 \u0442\u0435\u043a\u0441\u0442
                self.console.print(Panel(
                    Markdown(content),
                    title="\ud83e\udd16 \u0412\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u044c \u0430\u0433\u0435\u043d\u0442\u0430",
                    border_style="green"
                ))
        
        # \u041c\u0435\u0442\u0430\u0434\u0430\u043d\u043d\u044b\u0435 \u043e\u0442\u0432\u0435\u0442\u0430
        metadata_table = Table(show_header=False, box=None)
        metadata_table.add_column("\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440", style="cyan")
        metadata_table.add_column("\u0417\u043d\u0430\u0447\u0435\u043d\u043d\u044f", style="white")
        
        metadata_table.add_row("\u0427\u0430\u0441 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u0456", f"{response.response_time:.2f}\u0441")
        metadata_table.add_row("\u0412\u043f\u0435\u0432\u043d\u0435\u043d\u0456\u0441\u0442\u044c", f"{response.confidence:.2f}")
        metadata_table.add_row("\u041a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 \u0432\u0438\u043a\u043e\u0440\u0438\u0441\u0442\u0430\u043d\u043e", "\u2713" if response.context_used else "\u2717")
        
        self.console.print(metadata_table)
        
        # \u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438
        if response.sources:
            self.console.print("\
[bold]\ud83d\udcda \u0414\u0436\u0435\u0440\u0435\u043b\u0430:[/bold]")
            for i, source in enumerate(response.sources, 1):
                source_title = source.get('metadata', {}).get('file_name', f'\u0414\u0436\u0435\u0440\u0435\u043b\u043e {i}')
                relevance = source.get('relevance', 0)
                
                self.console.print(f"{i}. {source_title} (\u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u0456\u0441\u0442\u044c: {relevance:.2f})")
                
                # \u041f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u043c \u0447\u0430\u0441\u0442\u044c \u043a\u043e\u043d\u0442\u0435\u043d\u0442\u0430
                content_preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                self.console.print(f"   [dim]{content_preview}[/dim]")
    
    def _is_code_content(self, content: str) -> bool:
        """\u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u044f\u0432\u043b\u044f\u0435\u0442\u0441\u044f \u043b\u0438 \u043a\u043e\u043d\u0442\u0435\u043d\u0442 \u043a\u043e\u0434\u043e\u043c"""
        code_indicators = [
            "def ", "class ", "import ", "from ", "function", "var ", "let ", "const ",
            "if ", "for ", "while ", "try:", "except:", "catch", "{", "}", "=>"
        ]
        
        lines = content.split('\
')
        code_lines = sum(1 for line in lines if any(indicator in line for indicator in code_indicators))
        
        return code_lines > len(lines) * 0.3  # \u0415\u0441\u043b\u0438 30% \u0441\u0442\u0440\u043e\u043a \u0441\u043e\u0434\u0435\u0440\u0436\u0430\u0442 \u043a\u043e\u0434\u043e\u0432\u044b\u0435 \u0438\u043d\u0434\u0438\u043a\u0430\u0442\u043e\u0440\u044b
    
    def _show_help(self) -> None:
        """\u041f\u043e\u043a\u0430\u0437 \u0441\u043f\u0440\u0430\u0432\u043a\u0438"""
        help_text = """
[bold blue]\u0414\u043e\u0441\u0442\u0443\u043f\u043d\u0456 \u043a\u043e\u043c\u0430\u043d\u0434\u0438:[/bold blue]

[green]\u041e\u0441\u043d\u043e\u0432\u043d\u0456 \u043a\u043e\u043c\u0430\u043d\u0434\u0438:[/green]
  \u2022 [cyan]help[/cyan] - \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0438 \u0446\u044e \u0434\u043e\u0432\u0456\u0434\u043a\u0443
  \u2022 [cyan]status[/cyan] - \u0441\u0442\u0430\u0442\u0443\u0441 \u0430\u0433\u0435\u043d\u0442\u0430 \u0442\u0430 \u0441\u0438\u0441\u0442\u0435\u043c\u0438
  \u2022 [cyan]clear[/cyan] - \u043e\u0447\u0438\u0441\u0442\u0438\u0442\u0438 \u0435\u043a\u0440\u0430\u043d
  \u2022 [cyan]history[/cyan] - \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0438 \u0456\u0441\u0442\u043e\u0440\u0456\u044e \u043a\u043e\u043c\u0430\u043d\u0434
  \u2022 [cyan]exit[/cyan] \u0430\u0431\u043e [cyan]quit[/cyan] - \u0432\u0438\u0445\u0456\u0434

[green]\u0420\u043e\u0431\u043e\u0442\u0430 \u0437 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c\u0438:[/green]
  \u2022 [cyan]add <\u043f\u0443\u0442\u044c>[/cyan] - \u0434\u043e\u0434\u0430\u0442\u0438 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442/\u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0456\u044e
  \u2022 [cyan]list-docs[/cyan] - \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0438 \u0434\u043e\u0434\u0430\u043d\u0456 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0438
  \u2022 [cyan]train[/cyan] - \u043d\u0430\u0432\u0447\u0438\u0442\u0438 \u043c\u043e\u0434\u0435\u043b\u044c \u043d\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445
  \u2022 [cyan]search <\u0437\u0430\u043f\u0440\u043e\u0441>[/cyan] - \u043f\u043e\u0448\u0443\u043a \u043f\u043e \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445

[green]\u041d\u0430\u0432\u0447\u0430\u043d\u043d\u044f \u043c\u043e\u0434\u0435\u043b\u0456:[/green]
  \u2022 [cyan]train-on <\u043f\u0443\u0442\u044c>[/cyan] - \u043d\u0430\u0432\u0447\u0438\u0442\u0438 \u043d\u0430 \u0432\u043a\u0430\u0437\u0430\u043d\u0438\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445
  \u2022 [cyan]save-model <\u043d\u0430\u0437\u0432\u0430>[/cyan] - \u0437\u0431\u0435\u0440\u0435\u0433\u0442\u0438 \u043d\u0430\u0432\u0447\u0435\u043d\u0443 \u043c\u043e\u0434\u0435\u043b\u044c

[green]\u041d\u0430\u043b\u0430\u0448\u0442\u0443\u0432\u0430\u043d\u043d\u044f:[/green]
  \u2022 [cyan]set-temp <\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435>[/cyan] - \u0432\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u0438 \u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0443 (0.0-1.0)
  \u2022 [cyan]set-tokens <\u0447\u0438\u0441\u043b\u043e>[/cyan] - \u0432\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u0438 \u043c\u0430\u043a\u0441. \u0442\u043e\u043a\u0435\u043d\u0456\u0432

[green]\u041f\u0440\u0438\u043a\u043b\u0430\u0434\u0438 \u0432\u0438\u043a\u043e\u0440\u0438\u0441\u0442\u0430\u043d\u043d\u044f:[/green]
  \u2022 \u041d\u0430\u043f\u0438\u0448\u0438 \u0444\u0443\u043d\u043a\u0446\u0456\u044e \u0434\u043b\u044f \u0441\u043e\u0440\u0442\u0443\u0432\u0430\u043d\u043d\u044f \u043c\u0430\u0441\u0438\u0432\u0443 \u043d\u0430 Python
  \u2022 \u042f\u043a \u0441\u043a\u043b\u0430\u0441\u0442\u0438 \u043f\u043e\u0437\u043e\u0432\u043d\u0443 \u0437\u0430\u044f\u0432\u0443 \u043f\u0440\u043e \u0441\u0442\u044f\u0433\u043d\u0435\u043d\u043d\u044f \u0431\u043e\u0440\u0433\u0443?
  \u2022 \u041f\u0440\u043e\u0430\u043d\u0430\u043b\u0456\u0437\u0443\u0439 \u0446\u0435\u0439 \u043a\u043e\u0434 \u0442\u0430 \u0437\u043d\u0430\u0439\u0434\u0438 \u043f\u043e\u043c\u0438\u043b\u043a\u0438
  \u2022 \u042f\u043a\u0456 \u043f\u0440\u0430\u0432\u0430 \u0432\u0438\u043d\u0438\u043a\u0430\u044e\u0442\u044c \u043f\u0440\u0438 \u0443\u043a\u043b\u0430\u0434\u0435\u043d\u043d\u0456 \u0434\u043e\u0433\u043e\u0432\u043e\u0440\u0443 \u043e\u0440\u0435\u043d\u0434\u0438?
        """
        
        self.console.print(Panel(
            Markdown(help_text),
            title="\ud83d\udcd6 \u0414\u043e\u0432\u0456\u0434\u043a\u0430",
            border_style="blue"
        ))
    
    def _show_status(self) -> None:
        """\u041f\u043e\u043a\u0430\u0437 \u0441\u0442\u0430\u0442\u0443\u0441\u0430 \u0430\u0433\u0435\u043d\u0442\u0430"""
        try:
            if self.agent:
                status = self.agent.get_agent_status()
                
                # \u041e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u0442\u0430\u0431\u043b\u0438\u0446\u0430 \u0441\u0442\u0430\u0442\u0443\u0441\u0430
                status_table = Table(title="\ud83d\udcca \u0421\u0442\u0430\u0442\u0443\u0441 AI-\u0430\u0433\u0435\u043d\u0442\u0430", box=None)
                status_table.add_column("\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440", style="cyan")
                status_table.add_column("\u0417\u043d\u0430\u0447\u0435\u043d\u043d\u044f", style="white")
                
                status_table.add_row("\u041c\u043e\u0434\u0435\u043b\u044c \u0437\u0430\u0432\u0430\u043d\u0442\u0430\u0436\u0435\u043d\u0430", "\u2713" if status.get('model_loaded') else "\u2717")
                status_table.add_row("\u041a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u0438 \u0456\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u043e\u0432\u0430\u043d\u0456", "\u2713" if status.get('components_initialized') else "\u2717")
                status_table.add_row("\u041d\u0430\u0437\u0432\u0430 \u043c\u043e\u0434\u0435\u043b\u0456", status.get('model_name', '\u041d\u0435\u0432\u0456\u0434\u043e\u043c\u043e'))
                status_table.add_row("\u041f\u0440\u0438\u0441\u0442\u0440\u0456\u0439", status.get('device', '\u041d\u0435\u0432\u0456\u0434\u043e\u043c\u043e'))
                
                self.console.print(status_table)
                
                # \u0421\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u043a\u0430 \u0432\u0435\u043a\u0442\u043e\u0440\u043d\u043e\u0433\u043e \u0445\u0440\u0430\u043d\u0438\u043b\u0438\u0449\u0430
                vector_stats = status.get('vector_store_stats', {})
                if vector_stats:
                    self.console.print("\
[bold]\ud83d\udcda \u0412\u0435\u043a\u0442\u043e\u0440\u043d\u0435 \u0441\u0445\u043e\u0432\u0438\u0449\u0435:[/bold]")
                    vector_table = Table(show_header=True, box=None)
                    vector_table.add_column("\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440", style="cyan")
                    vector_table.add_column("\u0417\u043d\u0430\u0447\u0435\u043d\u043d\u044f", style="white")
                    
                    for key, value in vector_stats.items():
                        vector_table.add_row(key, str(value))
                    
                    self.console.print(vector_table)
                
                # \u0421\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u043a\u0430 knowledge graph
                graph_stats = status.get('knowledge_graph_stats', {})
                if graph_stats:
                    self.console.print("\
[bold]\ud83d\udd78\ufe0f Knowledge Graph:[/bold]")
                    graph_table = Table(show_header=True, box=None)
                    graph_table.add_column("\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440", style="cyan")
                    graph_table.add_column("\u0417\u043d\u0430\u0447\u0435\u043d\u043d\u044f", style="white")
                    
                    for key, value in graph_stats.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                graph_table.add_row(f"{key}.{sub_key}", str(sub_value))
                        else:
                            graph_table.add_row(key, str(value))
                    
                    self.console.print(graph_table)
            else:
                self.console.print("[red]\u0410\u0433\u0435\u043d\u0442 \u043d\u0435 \u0456\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u043e\u0432\u0430\u043d\u0438\u0439[/red]")
                
        except Exception as e:
            self.console.print(f"[red]\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u043e\u0442\u0440\u0438\u043c\u0430\u043d\u043d\u044f \u0441\u0442\u0430\u0442\u0443\u0441\u0443:[/red] {e}")
    
    def _add_documents(self, path: str) -> None:
        """\u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432"""
        try:
            path_obj = Path(path)
            
            if not path_obj.exists():
                self.console.print(f"[red]\u0428\u043b\u044f\u0445 \u043d\u0435 \u0456\u0441\u043d\u0443\u0454: {path}[/red]")
                return
            
            with Progress(console=self.console) as progress:
                task = progress.add_task("\u0414\u043e\u0434\u0430\u0432\u0430\u043d\u043d\u044f \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0456\u0432...", total=None)
                
                if path_obj.is_file():
                    document_files = [str(path_obj)]
                elif path_obj.is_dir():
                    document_files = []
                    for root, dirs, files in os.walk(path_obj):
                        for file in files:
                            if file.endswith(('.pdf', '.docx', '.doc', '.txt', '.md')):
                                document_files.append(os.path.join(root, file))
                else:
                    self.console.print(f"[red]\u041d\u0435\u043f\u0456\u0434\u0442\u0440\u0438\u043c\u0443\u0432\u0430\u043d\u0438\u0439 \u0442\u0438\u043f \u043e\u0431'\u0454\u043a\u0442\u0430: {path}[/red]")
                    return
                
                if not document_files:
                    self.console.print("[yellow]\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0456\u0432 \u043d\u0435 \u0437\u043d\u0430\u0439\u0434\u0435\u043d\u043e[/yellow]")
                    return
                
                progress.update(task, description=f"\u041e\u0431\u0440\u043e\u0431\u043a\u0430 {len(document_files)} \u0444\u0430\u0439\u043b\u0456\u0432...")
                
                if self.agent:
                    result = self.agent.add_documents(document_files)
                    
                    if result['success']:
                        progress.update(task, description="\u2713 \u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0438 \u0434\u043e\u0434\u0430\u043d\u043e \u0443\u0441\u043f\u0456\u0448\u043d\u043e!")
                        self.console.print(f"[green]\u2713[/green] {result['message']}")
                    else:
                        progress.update(task, description="\u2717 \u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u0434\u043e\u0434\u0430\u0432\u0430\u043d\u043d\u044f")
                        self.console.print(f"[red]\u2717[/red] {result['message']}")
                else:
                    self.console.print("[red]\u0410\u0433\u0435\u043d\u0442 \u043d\u0435 \u0456\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u043e\u0432\u0430\u043d\u0438\u0439[/red]")
                    
        except Exception as e:
            self.console.print(f"[red]\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u0434\u043e\u0434\u0430\u0432\u0430\u043d\u043d\u044f \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0456\u0432:[/red] {e}")
    
    def _search_documents(self, query: str) -> None:
        """\u041f\u043e\u0438\u0441\u043a \u043f\u043e \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c"""
        try:
            if not self.agent:
                self.console.print("[red]\u0410\u0433\u0435\u043d\u0442 \u043d\u0435 \u0456\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u043e\u0432\u0430\u043d\u0438\u0439[/red]")
                return
            
            with Progress(console=self.console) as progress:
                task = progress.add_task("\u041f\u043e\u0448\u0443\u043a \u043f\u043e \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445...", total=None)
                
                context_items = self.agent.retrieve_relevant_context(query)
                
                progress.update(task, description=f"\u0417\u043d\u0430\u0439\u0434\u0435\u043d\u043e {len(context_items)} \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0456\u0432")
            
            if context_items:
                self.console.print(f"\
[bold]\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0438 \u043f\u043e\u0448\u0443\u043a\u0443 \u0434\u043b\u044f:[/bold] '{query}'\
")
                
                for i, item in enumerate(context_items, 1):
                    source_name = item.get('metadata', {}).get('file_name', f'\u0414\u0436\u0435\u0440\u0435\u043b\u043e {i}')
                    relevance = item.get('relevance_score', 0)
                    
                    self.console.print(f"{i}. [cyan]{source_name}[/cyan] (\u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u0456\u0441\u0442\u044c: {relevance:.2f})")
                    
                    # \u041f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u043c \u0447\u0430\u0441\u0442\u044c \u043a\u043e\u043d\u0442\u0435\u043d\u0442\u0430
                    content = item['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    self.console.print(f"   [dim]{content}[/dim]\
")
            else:
                self.console.print("[yellow]\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0456\u0432 \u043d\u0435 \u0437\u043d\u0430\u0439\u0434\u0435\u043d\u043e[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u043f\u043e\u0448\u0443\u043a\u0443:[/red] {e}")
    
    def _train_model(self) -> None:
        """\u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u0438"""
        try:
            if not self.agent:
                self.console.print("[red]\u0410\u0433\u0435\u043d\u0442 \u043d\u0435 \u0456\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u043e\u0432\u0430\u043d\u0438\u0439[/red]")
                return
            
            self.console.print("[yellow]\u041d\u0430\u0432\u0447\u0430\u043d\u043d\u044f \u043c\u043e\u0434\u0435\u043b\u0456 \u043c\u043e\u0436\u0435 \u0437\u0430\u0439\u043d\u044f\u0442\u0438 \u0431\u0430\u0433\u0430\u0442\u043e \u0447\u0430\u0441\u0443...[/yellow]")
            
            if Prompt.ask("\u041f\u0440\u043e\u0434\u043e\u0432\u0436\u0438\u0442\u0438?", choices=["y", "n"], default="n") == "y":
                with Progress(console=self.console) as progress:
                    task = progress.add_task("\u041d\u0430\u0432\u0447\u0430\u043d\u043d\u044f \u043c\u043e\u0434\u0435\u043b\u0456...", total=None)
                    
                    result = self.agent.train_on_documents()
                    
                    if result['success']:
                        progress.update(task, description="\u2713 \u041d\u0430\u0432\u0447\u0430\u043d\u043d\u044f \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043e!")
                        self.console.print("[green]\u2713 \u041c\u043e\u0434\u0435\u043b\u044c \u0443\u0441\u043f\u0456\u0448\u043d\u043e \u043d\u0430\u0432\u0447\u0435\u043d\u0430[/green]")
                        
                        # \u041f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u043c \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u044e \u043e \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u043d\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438
                        training_result = result.get('training_result', {})
                        if 'save_path' in training_result:
                            self.console.print(f"[cyan]\u041c\u043e\u0434\u0435\u043b\u044c \u0437\u0431\u0435\u0440\u0435\u0436\u0435\u043d\u0430 \u0432:[/cyan] {training_result['save_path']}")
                    else:
                        progress.update(task, description="\u2717 \u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u043d\u0430\u0432\u0447\u0430\u043d\u043d\u044f")
                        self.console.print(f"[red]\u2717 \u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u043d\u0430\u0432\u0447\u0430\u043d\u043d\u044f:[/red] {result['message']}")
            else:
                self.console.print("\u041d\u0430\u0432\u0447\u0430\u043d\u043d\u044f \u0441\u043a\u0430\u0441\u043e\u0432\u0430\u043d\u043e")
                
        except Exception as e:
            self.console.print(f"[red]\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u043d\u0430\u0432\u0447\u0430\u043d\u043d\u044f:[/red] {e}")
    
    def _set_parameter(self, param: str, value: str) -> None:
        """\u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432"""
        try:
            if param == "temp":
                temp_value = float(value)
                if 0.0 <= temp_value <= 1.0:
                    config.model.temperature = temp_value
                    self.console.print(f"[green]\u2713 \u0422\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430 \u0432\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0430: {temp_value}[/green]")
                else:
                    self.console.print("[red]\u0422\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430 \u043f\u043e\u0432\u0438\u043d\u043d\u0430 \u0431\u0443\u0442\u0438 \u0432 \u0434\u0456\u0430\u043f\u0430\u0437\u043e\u043d\u0456 0.0-1.0[/red]")
            
            elif param == "tokens":
                tokens_value = int(value)
                if tokens_value > 0 and tokens_value <= 4096:
                    config.model.max_new_tokens = tokens_value
                    self.console.print(f"[green]\u2713 \u041c\u0430\u043a\u0441. \u0442\u043e\u043a\u0435\u043d\u0456\u0432 \u0432\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u043e: {tokens_value}[/green]")
                else:
                    self.console.print("[red]\u041a\u0456\u043b\u044c\u043a\u0456\u0441\u0442\u044c \u0442\u043e\u043a\u0435\u043d\u0456\u0432 \u043f\u043e\u0432\u0438\u043d\u043d\u0430 \u0431\u0443\u0442\u0438 \u0432 \u0434\u0456\u0430\u043f\u0430\u0437\u043e\u043d\u0456 1-4096[/red]")
            
            else:
                self.console.print(f"[red]\u041d\u0435\u0432\u0456\u0434\u043e\u043c\u0438\u0439 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440: {param}[/red]")
                
        except ValueError as e:
            self.console.print(f"[red]\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u0437\u043d\u0430\u0447\u0435\u043d\u043d\u044f:[/red] {e}")
        except Exception as e:
            self.console.print(f"[red]\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u0432\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u043d\u044f \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u0430:[/red] {e}")
    
    def _process_command(self, user_input: str) -> bool:
        """\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u043a\u043e\u043c\u0430\u043d\u0434"""
        user_input = user_input.strip()
        
        if not user_input:
            return True
        
        # \u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0432 \u0438\u0441\u0442\u043e\u0440\u0438\u044e
        self.command_history.append({
            'command': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # \u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u043a\u043e\u043c\u0430\u043d\u0434
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
                self.console.print("[red]\u041d\u0435\u043f\u0440\u0430\u0432\u0438\u043b\u044c\u043d\u0438\u0439 \u0444\u043e\u0440\u043c\u0430\u0442 \u043a\u043e\u043c\u0430\u043d\u0434\u0438. \u0412\u0438\u043a\u043e\u0440\u0438\u0441\u0442\u0430\u043d\u043d\u044f: set-<param> <value>[/red]")
        
        elif user_input.lower().startswith('/'):
            # \u0421\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0435 \u043a\u043e\u043c\u0430\u043d\u0434\u044b
            self.console.print(f"[dim]\u0421\u0438\u0441\u0442\u0435\u043c\u043d\u0430 \u043a\u043e\u043c\u0430\u043d\u0434\u0430: {user_input}[/dim]")
        
        else:
            # \u041e\u0431\u044b\u0447\u043d\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441 \u043a AI
            self._process_ai_query(user_input)
        
        return True
    
    def _process_ai_query(self, query: str) -> None:
        """\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 AI \u0437\u0430\u043f\u0440\u043e\u0441\u0430"""
        try:
            if not self.agent:
                self.console.print("[red]\u0410\u0433\u0435\u043d\u0442 \u043d\u0435 \u0456\u043d\u0456\u0446\u0456\u0430\u043b\u0456\u0437\u043e\u0432\u0430\u043d\u0438\u0439. \u041d\u0435\u043c\u043e\u0436\u043b\u0438\u0432\u043e \u043e\u0431\u0440\u043e\u0431\u0438\u0442\u0438 \u0437\u0430\u043f\u0438\u0442.[/red]")
                return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0456\u044f \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u0456...", total=None)
                
                response = self.agent.query(query)
                
                progress.update(task, description="\u2713 \u0412\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u044c \u0433\u043e\u0442\u043e\u0432\u0430!")
            
            # \u0412\u044b\u0432\u043e\u0434 \u043e\u0442\u0432\u0435\u0442\u0430
            self._format_response(response)
            
        except Exception as e:
            self.console.print(f"[red]\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u043e\u0431\u0440\u043e\u0431\u043a\u0438 \u0437\u0430\u043f\u0438\u0442\u0443:[/red] {e}")
    
    def _show_history(self) -> None:
        """\u041f\u043e\u043a\u0430\u0437 \u0438\u0441\u0442\u043e\u0440\u0438\u0438 \u043a\u043e\u043c\u0430\u043d\u0434"""
        if not self.command_history:
            self.console.print("[yellow]\u0406\u0441\u0442\u043e\u0440\u0456\u044f \u043f\u043e\u0440\u043e\u0436\u043d\u044f[/yellow]")
            return
        
        history_table = Table(title="\ud83d\udcdc \u0406\u0441\u0442\u043e\u0440\u0456\u044f \u043a\u043e\u043c\u0430\u043d\u0434", show_header=True)
        history_table.add_column("\u2116", style="cyan", width=4)
        history_table.add_column("\u041a\u043e\u043c\u0430\u043d\u0434\u0430", style="white")
        history_table.add_column("\u0427\u0430\u0441", style="dim")
        
        for i, cmd in enumerate(self.command_history[-20:], 1):  # \u041f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u043c \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 20
            timestamp = cmd.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp[:8]
            else:
                time_str = '\u041d\u0435\u0432\u0456\u0434\u043e\u043c\u043e'
            
            command = cmd.get('command', '')
            if len(command) > 60:
                command = command[:57] + "..."
            
            history_table.add_row(str(i), command, time_str)
        
        self.console.print(history_table)
    
    def run(self) -> None:
        """\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0446\u0438\u043a\u043b CLI"""
        # \u041f\u043e\u043a\u0430\u0437 \u043f\u0440\u0438\u0432\u0435\u0442\u0441\u0442\u0432\u0438\u044f
        welcome_text = """
# \ud83e\udd16 AI-\u0410\u0433\u0435\u043d\u0442 \u043d\u0430 \u0431\u0430\u0437\u0456 Mistral AI 7B

\u041b\u0430\u0441\u043a\u0430\u0432\u043e \u043f\u0440\u043e\u0441\u0438\u043c\u043e \u0434\u043e \u0456\u043d\u0442\u0435\u0440\u0430\u043a\u0442\u0438\u0432\u043d\u043e\u0433\u043e CLI \u0456\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u0443!

\u0410\u0433\u0435\u043d\u0442 \u0441\u043f\u0435\u0446\u0456\u0430\u043b\u0456\u0437\u0443\u0454\u0442\u044c\u0441\u044f \u043d\u0430:
\u2022 \ud83d\udcbb \u041f\u0440\u043e\u0433\u0440\u0430\u043c\u0443\u0432\u0430\u043d\u043d\u0456 \u0442\u0430 \u0430\u043d\u0430\u043b\u0456\u0437\u0456 \u043a\u043e\u0434\u0443
\u2022 \u2696\ufe0f \u042e\u0440\u0438\u0434\u0438\u0447\u043d\u0438\u0445 \u043f\u0438\u0442\u0430\u043d\u043d\u044f\u0445 (\u0437\u0430\u043a\u043e\u043d\u043e\u0434\u0430\u0432\u0441\u0442\u0432\u043e \u0423\u043a\u0440\u0430\u0457\u043d\u0438)
\u2022 \ud83d\udcda \u0420\u043e\u0431\u043e\u0442\u0456 \u0437 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c\u0438

\u0412\u0432\u0435\u0434\u0456\u0442\u044c [cyan]help[/cyan] \u0434\u043b\u044f \u0441\u043f\u0438\u0441\u043a\u0443 \u043a\u043e\u043c\u0430\u043d\u0434 \u0430\u0431\u043e \u043f\u043e\u0447\u043d\u0456\u0442\u044c \u0441\u0442\u0430\u0432\u0438\u0442\u0438 \u0437\u0430\u043f\u0438\u0442\u0430\u043d\u043d\u044f!
        """
        
        self.console.print(Panel(
            Markdown(welcome_text),
            title="AI-\u0410\u0433\u0435\u043d\u0442",
            border_style="green"
        ))
        
        # \u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0446\u0438\u043a\u043b
        while True:
            try:
                # \u0412\u0432\u043e\u0434 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044f
                user_input = Prompt.ask(
                    "\
[bold blue]\ud83d\udd0d \u0412\u0430\u0448 \u0437\u0430\u043f\u0438\u0442[/bold blue]",
                    default="",
                    show_default=False
                )
                
                if not user_input.strip():
                    continue
                
                # \u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u043a\u043e\u043c\u0430\u043d\u0434\u044b
                should_continue = self._process_command(user_input)
                
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\
[yellow]\u0417\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043d\u044f \u0440\u043e\u0431\u043e\u0442\u0438...[/yellow]")
                break
            except EOFError:
                self.console.print("\
[yellow]\u0417\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043d\u044f \u0440\u043e\u0431\u043e\u0442\u0438...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"\
[red]\u0412\u0438\u043d\u0438\u043a\u043b\u0430 \u043f\u043e\u043c\u0438\u043b\u043a\u0430:[/red] {e}")
        
        # \u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0438\u0441\u0442\u043e\u0440\u0438\u0438 \u0438 \u0432\u044b\u0445\u043e\u0434
        self._save_history()
        self.console.print("[bold green]\u0414\u044f\u043a\u0443\u0454\u043c\u043e \u0437\u0430 \u0432\u0438\u043a\u043e\u0440\u0438\u0441\u0442\u0430\u043d\u043d\u044f AI-\u0430\u0433\u0435\u043d\u0442\u0430![/bold green]")

# \u041e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u0444\u0443\u043d\u043a\u0446\u0438\u044f \u0434\u043b\u044f \u0437\u0430\u043f\u0443\u0441\u043a\u0430 CLI
def main():
    """\u0413\u043b\u0430\u0432\u043d\u0430\u044f \u0444\u0443\u043d\u043a\u0446\u0438\u044f"""
    try:
        cli = AIAgentCLI()
        cli.run()
    except Exception as e:
        console = Console()
        console.print(f"[bold red]\u041a\u0440\u0438\u0442\u0438\u0447\u043d\u0430 \u043f\u043e\u043c\u0438\u043b\u043a\u0430 \u0437\u0430\u043f\u0443\u0441\u043a\u0443 CLI:[/bold red] {e}")
        logger.error(f"Critical CLI error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
