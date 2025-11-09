# src/vscode_integration.py (дополнительный файл)
import os
import asyncio
from typing import Dict, Any

class VSCodeIntegration:
    """
    ИНТЕГРАЦИЯ С VS CODE ЧЕРЕЗ LANGUAGE SERVER PROTOCOL
    
    Позволяет использовать ассистента прямо в редакторе:
    - Hover documentation
    - Code completion
    - Error diagnostics
    """
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.active_project = None
    
    async def on_file_open(self, file_path: str):
        """Обработка открытия файла в VSCode"""
        if file_path.endswith(('.py', '.java', '.kt', '.js')):
            # Добавляем файл в контекст
            await asyncio.to_thread(
                self.assistant.add_document, file_path
            )
    
    async def on_hover(self, file_path: str, line: int) -> Dict[str, Any]:
        """Обработка hover запроса"""
        # Читаем файл
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Получаем контекст (±5 строк)
        context = "".join(lines[max(0, line-5):min(len(lines), line+6)])
        
        # Формируем вопрос
        question = f"Поясни цей код (рядок {line}):\n\n{context}"
        
        # Генерируем ответ
        response = await asyncio.to_thread(
            self.assistant.chat, question
        )
        
        return {
            "contents": response,
            "range": {
                "startLineNumber": line,
                "endLineNumber": line
            }
        }