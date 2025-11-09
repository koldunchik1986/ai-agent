"""
ИНТЕГРАЦИЯ С ANDROID STUDIO

Позволяет использовать AI-ассистента прямо в Android Studio через:
- Plugin API
- File watchers
- Custom inspections
- Code completion
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class AndroidProject:
    """Структура Android проекта"""
    path: str
    package_name: str
    modules: List[str]
    gradle_version: str
    compile_sdk: int
    min_sdk: int
    target_sdk: int

@dataclass
class CodeInspection:
    """Результат инспекции кода"""
    severity: str  # ERROR, WARNING, INFO
    message: str
    line: int
    column: int
    quick_fix: Optional[str] = None

class AndroidStudioIntegration:
    """
    ИНТЕГРАЦИЯ С ANDROID STUDIO
    
    Основные функции:
    1. Сканирование структуры Android проекта
    2. Анализ Gradle файлов
    3. Инспекция XML разметок
    4. Проверка Kotlin/Java кода
    5. Генерация boilerplate кода
    """
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.current_project = None
        self.inspection_cache = {}
        
        # Android-специфичные паттерны
        self.android_patterns = {
            'activity_pattern': r'class\s+(\w+)Activity\s*:\s*AppCompatActivity',
            'fragment_pattern': r'class\s+(\w+)Fragment\s*:\s*Fragment',
            'viewmodel_pattern': r'class\s+(\w+)ViewModel\s*:\s*ViewModel',
            'repository_pattern': r'class\s+(\w+)Repository',
        }
        
        # Стандартные Android файлы
        self.standard_files = [
            'AndroidManifest.xml',
            'build.gradle',
            'settings.gradle',
            'gradle.properties',
            'local.properties'
        ]
    
    def scan_android_project(self, project_path: str) -> AndroidProject:
        """
        СКАНИРОВАНИЕ ANDROID ПРОЕКТА
        
        Определяет:
        - Структуру модулей
        - Версии Gradle и SDK
        - Пакет приложения
        - Зависимости
        """
        project_path = Path(project_path)
        
        if not project_path.exists():
            raise FileNotFoundError(f"Проект не найден: {project_path}")
        
        # Чтение build.gradle (app module)
        app_gradle = project_path / "app" / "build.gradle"
        if not app_gradle.exists():
            raise FileNotFoundError("Не найден app/build.gradle")
        
        # Парсинг build.gradle
        with open(app_gradle, 'r') as f:
            gradle_content = f.read()
        
        # Извлечение информации из Gradle
        package_name = self._extract_package_name(gradle_content)
        compile_sdk = self._extract_sdk_version(gradle_content, 'compileSdk')
        min_sdk = self._extract_sdk_version(gradle_content, 'minSdk')
        target_sdk = self._extract_sdk_version(gradle_content, 'targetSdk')
        
        # Определение версии Gradle
        gradle_wrapper = project_path / "gradle" / "wrapper" / "gradle-wrapper.properties"
        gradle_version = "unknown"
        if gradle_wrapper.exists():
            with open(gradle_wrapper, 'r') as f:
                for line in f:
                    if 'gradle-' in line and '.zip' in line:
                        gradle_version = line.split('gradle-')[1].split('-')[0]
                        break
        
        # Находим все модули
        modules = []
        settings_gradle = project_path / "settings.gradle"
        if settings_gradle.exists():
            with open(settings_gradle, 'r') as f:
                for line in f:
                    if 'include' in line and ':' in line:
                        module = line.split(':')[1].strip().strip("'\"")
                        modules.append(module)
        
        project = AndroidProject(
            path=str(project_path),
            package_name=package_name,
            modules=modules or ['app'],
            gradle_version=gradle_version,
            compile_sdk=compile_sdk,
            min_sdk=min_sdk,
            target_sdk=target_sdk
        )
        
        self.current_project = project
        return project
    
    def _extract_package_name(self, gradle_content: str) -> str:
        """Извлечение package name из build.gradle"""
        import re
        match = re.search(r'applicationId\s+["\']([^"\']+)["\']', gradle_content)
        return match.group(1) if match else "com.example.app"
    
    def _extract_sdk_version(self, gradle_content: str, sdk_type: str) -> int:
        """Извлечение SDK версии из build.gradle"""
        import re
        match = re.search(rf'{sdk_type}\s+(\d+)', gradle_content)
        return int(match.group(1)) if match else 21
    
    def inspect_xml_layout(self, xml_content: str) -> List[CodeInspection]:
        """
        ИНСПЕКЦИЯ XML РАЗМЕТКИ
        
        Проверяет:
        - Отсутствующие ID
        - Неиспользуемые атрибуты
        - Ошибки вьюх
        - Производительность
        """
        inspections = []
        
        # Базовые проверки
        if 'android:id' not in xml_content:
            inspections.append(CodeInspection(
                severity="WARNING",
                message="Отсутствует android:id у корневого элемента",
                line=1,
                column=1
            ))
        
        # Проверка вложенности
        if xml_content.count('<') > 50:
            inspections.append(CodeInspection(
                severity="WARNING",
                message="Слишком глубокая вложенность вьюх (может влиять на производительность)",
                line=1,
                column=1,
                quick_fix="Рассмотрите использование include или merge"
            ))
        
        # Проверка ConstraintLayout
        if 'ConstraintLayout' in xml_content and 'app:layout_constraint' not in xml_content:
            inspections.append(CodeInspection(
                severity="ERROR",
                message="ConstraintLayout без ограничений",
                line=1,
                column=1,
                quick_fix="Добавьте app:layout_constraint* атрибуты"
            ))
        
        return inspections
    
    def generate_view_binding_code(self, layout_name: str, package_name: str) -> str:
        """
        ГЕНЕРАЦИЯ КОДА ДЛЯ VIEW BINDING
        
        Args:
            layout_name: Имя layout файла (без .xml)
            package_name: Пакет приложения
            
        Returns:
            Kotlin код для использования View Binding
        """
        
        class_name = f"{layout_name.replace('_', ' ').title().replace(' ', '')}Binding"
        
        code_template = f"""package {package_name}.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import {package_name}.databinding.{class_name}

class {layout_name.replace('_', ' ').title().replace(' ', '')}Fragment : Fragment() {{
    
    private var _binding: {class_name}? = null
    private val binding get() = _binding!!
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {{
        _binding = {class_name}.inflate(inflater, container, false)
        return binding.root
    }}
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {{
        super.onViewCreated(view, savedInstanceState)
        
        // TODO: Инициализация UI
        setupUI()
        observeData()
    }}
    
    private fun setupUI() {{
        // Настройка UI элементов
        // binding.button.setOnClickListener {{ }}
    }}
    
    private fun observeData() {{
        // Наблюдение за данными
        // viewModel.data.observe(viewLifecycleOwner) {{ data ->
        //     binding.textView.text = data
        // }}
    }}
    
    override fun onDestroyView() {{
        super.onDestroyView()
        _binding = null
    }}
}}"""
        
        return code_template
    
    def analyze_build_gradle(self, gradle_content: str) -> List[CodeInspection]:
        """
        АНАЛИЗ BUILD.GRADLE ФАЙЛОВ
        
        Проверяет:
        - Устаревшие зависимости
        - Неиспользуемые библиотеки
        - Версии SDK
        - Производительность настройки
        """
        inspections = []
        
        # Проверка версий
        if 'compileSdk 21' in gradle_content or 'targetSdk 21' in gradle_content:
            inspections.append(CodeInspection(
                severity="WARNING",
                message="Используется старая версия SDK (21). Рекомендуется 30+",
                line=1,
                column=1,
                quick_fix="Обновите до targetSdk 30+"
            ))
        
        # Проверка устаревших зависимостей
        deprecated_libs = [
            'compile "com.android.support:',
            'androidTestCompile'
        ]
        
        for deprecated in deprecated_libs:
            if deprecated in gradle_content:
                inspections.append(CodeInspection(
                    severity="ERROR",
                    message=f"Используется устаревшая зависимость: {deprecated}",
                    line=1,
                    column=1,
                    quick_fix="Замените на implementation и AndroidX"
                ))
        
        # Проверка ProGuard
        if 'minifyEnabled false' in gradle_content and 'buildTypes.release' in gradle_content:
            inspections.append(CodeInspection(
                severity="INFO",
                message="ProGuard отключен для релизной сборки",
                line=1,
                column=1,
                quick_fix="Включите minifyEnabled true для оптимизации"
            ))
        
        return inspections
    
    def generate_repository_pattern(self, entity_name: str, package_name: str) -> Dict[str, str]:
        """
        ГЕНЕРАЦИЯ REPOSITORY PATTERN
        
        Создает:
        - Repository interface
        - Repository implementation
        - Data source interface
        - Entity class
        
        Args:
            entity_name: Название сущности (User, Product и т.д.)
            package_name: Пакет приложения
            
        Returns:
            Словарь с сгенерированными файлами
        """
        entity_lower = entity_name.lower()
        entity_upper = entity_name
        
        # Entity class
        entity_code = f"""package {package_name}.data.model

data class {entity_upper}(
    val id: String,
    val name: String,
    val createdAt: Long = System.currentTimeMillis()
)"""
        
        # Repository interface
        repository_interface = f"""package {package_name}.domain.repository

interface {entity_upper}Repository {{
    suspend fun getAll(): List<{entity_upper}>
    suspend fun getById(id: String): {entity_upper}?
    suspend fun create(item: {entity_upper}): {entity_upper}
    suspend fun update(item: {entity_upper})
    suspend fun delete(id: String)
}}"""
        
        # Repository implementation
        repository_impl = f"""package {package_name}.data.repository

import {package_name}.data.model.{entity_upper}
import {package_name}.domain.repository.{entity_upper}Repository
import {package_name}.data.datasource.{entity_upper}DataSource

class {entity_upper}RepositoryImpl(
    private val dataSource: {entity_upper}DataSource
) : {entity_upper}Repository {{
    
    override suspend fun getAll(): List<{entity_upper}> {{
        return dataSource.getAll()
    }}
    
    override suspend fun getById(id: String): {entity_upper}? {{
        return dataSource.getById(id)
    }}
    
    override suspend fun create(item: {entity_upper}): {entity_upper} {{
        return dataSource.create(item)
    }}
    
    override suspend fun update(item: {entity_upper}) {{
        dataSource.update(item)
    }}
    
    override suspend fun delete(id: String) {{
        dataSource.delete(id)
    }}
}}"""
        
        # Data source interface
        data_source_interface = f"""package {package_name}.data.datasource

import {package_name}.data.model.{entity_upper}

interface {entity_upper}DataSource {{
    suspend fun getAll(): List<{entity_upper}>
    suspend fun getById(id: String): {entity_upper}?
    suspend fun create(item: {entity_upper}): {entity_upper}
    suspend fun update(item: {entity_upper})
    suspend fun delete(id: String)
}}"""
        
        return {
            "entity": entity_code,
            "repository_interface": repository_interface,
            "repository_impl": repository_impl,
            "data_source_interface": data_source_interface
        }
    
    async def analyze_project_structure(self) -> Dict[str, Any]:
        """
        АНАЛИЗ СТРУКТУРЫ ПРОЕКТА
        
        Полный анализ:
        - Архитектура (MVP, MVVM, Clean)
        - Зависимости
        - Производительность
        - Безопасность
        """
        if not self.current_project:
            return {"error": "Проект не загружен"}
        
        project_path = Path(self.current_project.path)
        
        # Сбор статистики
        stats = {
            "total_files": 0,
            "kotlin_files": 0,
            "java_files": 0,
            "xml_files": 0,
            "gradle_files": 0,
            "architecture": "unknown",
            "issues": []
        }
        
        # Сканирование файлов
        for root, dirs, files in os.walk(project_path):
            for file in files:
                stats["total_files"] += 1
                
                if file.endswith('.kt'):
                    stats["kotlin_files"] += 1
                elif file.endswith('.java'):
                    stats["java_files"] += 1
                elif file.endswith('.xml'):
                    stats["xml_files"] += 1
                elif file.endswith('.gradle'):
                    stats["gradle_files"] += 1
        
        # Определение архитектуры
        if (project_path / "app" / "src" / "main" / "java" / "presentation").exists():
            stats["architecture"] = "MVVM + Clean Architecture"
        elif (project_path / "app" / "src" / "main" / "java" / "mvp").exists():
            stats["architecture"] = "MVP"
        else:
            stats["architecture"] = "Standard Android"
        
        # Использование AI для анализа
        project_summary = f"""
Android проект с:
- {stats['kotlin_files']} Kotlin файлов
- {stats['java_files']} Java файлов  
- {stats['xml_files']} XML файлов
- Архитектура: {stats['architecture']}
- Gradle: {self.current_project.gradle_version}
- SDK: compile={self.current_project.compile_sdk}, target={self.current_project.target_sdk}
"""
        
        # Получение рекомендаций от AI
        question = f"Проаналізуй цей Android проект та дай рекомендації:\n{project_summary}"
        
        recommendations = await asyncio.to_thread(
            self.assistant.chat, question
        )
        
        return {
            "statistics": stats,
            "architecture": stats["architecture"],
            "recommendations": recommendations,
            "project_info": {
                "package": self.current_project.package_name,
                "modules": self.current_project.modules,
                "gradle_version": self.current_project.gradle_version
            }
        }