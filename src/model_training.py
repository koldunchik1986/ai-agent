
"""
Модуль обучения модели Mistral AI 7B
Поддержка LoRA,量化训练 для эффективного использования GPU
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import torch
from datetime import datetime

# Transformers \u0438 PEFT \u0434\u043b\u044f \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    TaskType, PeftModel
)
from datasets import Dataset, DatasetDict
import numpy as np

# Local imports
from config import config
from document_processor import DocumentProcessor, ProcessedDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """\u041a\u043b\u0430\u0441\u0441 \u0434\u043b\u044f \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u044f \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0435\u0433\u043e \u043f\u0440\u0438\u043c\u0435\u0440\u0430"""
    instruction: str
    input_text: str = ""
    output_text: str = ""
    source_file: str = ""
    document_type: str = ""

class ModelTrainer:
    """\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043a\u043b\u0430\u0441\u0441 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f \u043c\u043e\u0434\u0435\u043b\u0438"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.training_state = {}
        
        # \u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f
        self._init_model()
        self._init_tokenizer()
    
    def _init_model(self):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u043c\u043e\u0434\u0435\u043b\u0438 Mistral 7B"""
        try:
            logger.info(f"Loading model: {config.model.model_name}")
            
            # \u041a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446\u0438\u044f \u043a\u0432\u0430\u043d\u0442\u0438\u0437\u0430\u0446\u0438\u0438 \u0434\u043b\u044f 8GB GPU
            if config.model.load_in_8bit or config.model.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=config.model.load_in_8bit,
                    load_in_4bit=config.model.load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                bnb_config = None
            
            # \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u043c\u043e\u0434\u0435\u043b\u0438
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name,
                cache_dir=config.model.cache_dir,
                torch_dtype=getattr(torch, config.model.torch_dtype),
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=config.model.trust_remote_code,
                use_cache=config.model.use_cache
            )
            
            # \u041f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u043a\u0430 \u043c\u043e\u0434\u0435\u043b\u0438 \u0434\u043b\u044f k-bit training
            if config.model.load_in_8bit or config.model.load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info(f"Model loaded successfully on {config.model.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _init_tokenizer(self):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u0442\u043e\u043a\u0435\u043d\u0438\u0437\u0430\u0442\u043e\u0440\u0430"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model.model_name,
                cache_dir=config.model.cache_dir,
                trust_remote_code=config.model.trust_remote_code
            )
            
            # \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 pad_token \u0435\u0441\u043b\u0438 \u043e\u0442\u0441\u0443\u0442\u0441\u0442\u0432\u0443\u0435\u0442
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def setup_lora(self):
        """\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0430 LoRA \u0434\u043b\u044f \u044d\u0444\u0444\u0435\u043a\u0442\u0438\u0432\u043d\u043e\u0433\u043e \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f"""
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.training.lora_r,
                lora_alpha=config.training.lora_alpha,
                lora_dropout=config.training.lora_dropout,
                target_modules=config.training.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # \u0412\u044b\u0432\u043e\u0434 \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u0438 \u043e \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u0430\u0445
            self.peft_model.print_trainable_parameters()
            
            logger.info("LoRA setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up LoRA: {e}")
            raise
    
    def format_training_examples(self, documents: List[ProcessedDocument]) -> List[TrainingExample]:
        """\u0424\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u0432 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u044b"""
        examples = []
        
        for doc in documents:
            for chunk in doc.chunks:
                # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u043e\u0432 \u0432 \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u0438 \u043e\u0442 \u0442\u0438\u043f\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430
                if doc.document_type == "programming":
                    examples.extend(self._create_programming_examples(chunk))
                elif doc.document_type == "legal":
                    examples.extend(self._create_legal_examples(chunk))
                else:
                    examples.extend(self._create_general_examples(chunk))
        
        logger.info(f"Created {len(examples)} training examples")
        return examples
    
    def _create_programming_examples(self, chunk) -> List[TrainingExample]:
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u043e\u0432 \u0434\u043b\u044f \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f"""
        examples = []
        content = chunk.content.strip()
        
        if not content:
            return examples
        
        # \u0420\u0430\u0437\u043d\u044b\u0435 \u0442\u0438\u043f\u044b \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u043d\u044b\u0445 \u0437\u0430\u0434\u0430\u0447
        if "def " in content or "function" in content or "class " in content:
            examples.append(TrainingExample(
                instruction="Explain this code and suggest improvements:",
                input_text=content,
                output_text="This code contains functions/classes that implement specific functionality. The code could be improved by adding error handling, documentation, and following best practices.",
                source_file=chunk.source_file,
                document_type="programming"
            ))
        
        if "import " in content or "package " in content:
            examples.append(TrainingExample(
                instruction="What are the dependencies in this code?",
                input_text=content,
                output_text="This code imports/uses the following libraries and packages for its functionality.",
                source_file=chunk.source_file,
                document_type="programming"
            ))
        
        # \u041e\u0431\u0449\u0438\u0439 \u043f\u0440\u0438\u043c\u0435\u0440
        examples.append(TrainingExample(
            instruction="Analyze this programming code:",
            input_text=content,
            output_text="This is a programming code snippet that implements specific functionality using standard programming patterns and practices.",
            source_file=chunk.source_file,
            document_type="programming"
        ))
        
        return examples
    
    def _create_legal_examples(self, chunk) -> List[TrainingExample]:
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u043e\u0432 \u0434\u043b\u044f \u044e\u0440\u0438\u0434\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432"""
        examples = []
        content = chunk.content.strip()
        
        if not content:
            return examples
        
        # \u042e\u0440\u0438\u0434\u0438\u0447\u0435\u0441\u043a\u0438\u0435 \u0438\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u0438
        if "\u0437\u0430\u043a\u043e\u043d" in content.lower() or "\u0441\u0442\u0430\u0442\u0442\u044f" in content.lower() or "\u043a\u043e\u0434\u0435\u043a\u0441" in content.lower():
            examples.append(TrainingExample(
                instruction="\u041f\u0440\u043e\u0430\u043d\u0430\u043b\u0456\u0437\u0443\u0439\u0442\u0435 \u0446\u0435\u0439 \u044e\u0440\u0438\u0434\u0438\u0447\u043d\u0438\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442:",
                input_text=content,
                output_text="\u0426\u0435\u0439 \u044e\u0440\u0438\u0434\u0438\u0447\u043d\u0438\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u043c\u0456\u0441\u0442\u0438\u0442\u044c \u043d\u043e\u0440\u043c\u0438 \u043f\u0440\u0430\u0432\u0430, \u0449\u043e \u0440\u0435\u0433\u0443\u043b\u044e\u044e\u0442\u044c \u043a\u043e\u043d\u043a\u0440\u0435\u0442\u043d\u0456 \u043f\u0440\u0430\u0432\u043e\u0432\u0456\u0434\u043d\u043e\u0441\u0438\u043d\u0438 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u043d\u043e \u0434\u043e \u0437\u0430\u043a\u043e\u043d\u043e\u0434\u0430\u0432\u0441\u0442\u0432\u0430 \u0423\u043a\u0440\u0430\u0457\u043d\u0438.",
                source_file=chunk.source_file,
                document_type="legal"
            ))
        
        if "\u043f\u043e\u0437\u043e\u0432" in content.lower() or "\u0441\u043a\u0430\u0440\u0433\u0430" in content.lower():
            examples.append(TrainingExample(
                instruction="\u0414\u0430\u0439\u0442\u0435 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0456\u0457 \u0449\u043e\u0434\u043e \u0446\u044c\u043e\u0433\u043e \u043f\u043e\u0437\u043e\u0432\u0443/\u0441\u043a\u0430\u0440\u0433\u0438:",
                input_text=content,
                output_text="\u0426\u0435\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u0454 \u043f\u0440\u043e\u0446\u0435\u0441\u0443\u0430\u043b\u044c\u043d\u0438\u043c \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u043c \u0441\u0443\u0434\u043e\u0432\u043e\u0433\u043e \u0440\u043e\u0437\u0433\u043b\u044f\u0434\u0443. \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0454\u0442\u044c\u0441\u044f \u043f\u0435\u0440\u0435\u0432\u0456\u0440\u0438\u0442\u0438 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u043d\u0456\u0441\u0442\u044c \u0444\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u0438\u043c \u0432\u0438\u043c\u043e\u0433\u0430\u043c \u0442\u0430 \u043d\u0430\u044f\u0432\u043d\u0456\u0441\u0442\u044c \u043d\u0435\u043e\u0431\u0445\u0456\u0434\u043d\u0438\u0445 \u0434\u043e\u043a\u0430\u0437\u0456\u0432.",
                source_file=chunk.source_file,
                document_type="legal"
            ))
        
        # \u0417\u0430\u0433\u0430\u043b\u044c\u043d\u0438\u0439 \u044e\u0440\u0438\u0434\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u043f\u0440\u0438\u043c\u0435\u0440
        examples.append(TrainingExample(
            instruction="\u041f\u0440\u043e\u0430\u043d\u0430\u043b\u0456\u0437\u0443\u0439\u0442\u0435 \u044e\u0440\u0438\u0434\u0438\u0447\u043d\u0438\u0439 \u0437\u043c\u0456\u0441\u0442 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430:",
            input_text=content,
            output_text="\u0426\u0435\u0439 \u044e\u0440\u0438\u0434\u0438\u0447\u043d\u0438\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u0440\u0435\u0433\u0443\u043b\u044e\u0454 \u043f\u0440\u0430\u0432\u043e\u0432\u0456 \u0432\u0456\u0434\u043d\u043e\u0441\u0438\u043d\u0438 \u043c\u0456\u0436 \u0441\u0442\u043e\u0440\u043e\u043d\u0430\u043c\u0438 \u0442\u0430 \u0432\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u044e\u0454 \u0457\u0445\u043d\u0456 \u043f\u0440\u0430\u0432\u0430 \u0442\u0430 \u043e\u0431\u043e\u0432'\u044f\u0437\u043a\u0438.",
            source_file=chunk.source_file,
            document_type="legal"
        ))
        
        return examples
    
    def _create_general_examples(self, chunk) -> List[TrainingExample]:
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043e\u0431\u0449\u0438\u0445 \u043f\u0440\u0438\u043c\u0435\u0440\u043e\u0432"""
        examples = []
        content = chunk.content.strip()
        
        if not content:
            return examples
        
        examples.append(TrainingExample(
            instruction="\u041f\u0440\u043e\u0430\u043d\u0430\u043b\u0456\u0437\u0443\u0439\u0442\u0435 \u0446\u0435\u0439 \u0442\u0435\u043a\u0441\u0442:",
            input_text=content,
            output_text="\u0426\u0435\u0439 \u0442\u0435\u043a\u0441\u0442 \u043c\u0456\u0441\u0442\u0438\u0442\u044c \u0456\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0456\u044e \u043d\u0430 \u043f\u0435\u0432\u043d\u0443 \u0442\u0435\u043c\u0443, \u044f\u043a\u0443 \u043c\u043e\u0436\u043d\u0430 \u0432\u0438\u043a\u043e\u0440\u0438\u0441\u0442\u043e\u0432\u0443\u0432\u0430\u0442\u0438 \u0434\u043b\u044f \u043e\u0442\u0440\u0438\u043c\u0430\u043d\u043d\u044f \u0437\u043d\u0430\u043d\u044c \u0442\u0430 \u0440\u043e\u0437\u0443\u043c\u0456\u043d\u043d\u044f \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0443.",
            source_file=chunk.source_file,
            document_type="general"
        ))
        
        return examples
    
    def format_for_training(self, examples: List[TrainingExample]) -> Dataset:
        """\u0424\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u043e\u0432 \u0434\u043b\u044f \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f"""
        formatted_data = []
        
        for example in examples:
            # \u0424\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0432 Mistral instruction format
            if example.input_text:
                formatted_text = f"<s>[INST] {example.instruction}\
\
{example.input_text} [/INST] {example.output_text}</s>"
            else:
                formatted_text = f"<s>[INST] {example.instruction} [/INST] {example.output_text}</s>"
            
            formatted_data.append({
                "text": formatted_text,
                "source_file": example.source_file,
                "document_type": example.document_type
            })
        
        return Dataset.from_list(formatted_data)
    
    def tokenize_function(self, examples):
        """\u0424\u0443\u043d\u043a\u0446\u0438\u044f \u0442\u043e\u043a\u0435\u043d\u0438\u0437\u0430\u0446\u0438\u0438"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config.model.max_new_tokens,
            return_overflowing_tokens=False,
        )
    
    def prepare_training_data(self, documents: List[ProcessedDocument]) -> Tuple[Dataset, Dataset]:
        """\u041f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u043a\u0430 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445"""
        # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0445 \u043f\u0440\u0438\u043c\u0435\u0440\u043e\u0432
        examples = self.format_training_examples(documents)
        
        if not examples:
            raise ValueError("No training examples created")
        
        # \u0424\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0434\u043b\u044f \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f
        dataset = self.format_for_training(examples)
        
        # \u0422\u043e\u043a\u0435\u043d\u0438\u0437\u0430\u0446\u0438\u044f
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # \u0420\u0430\u0437\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u043d\u0430 train/validation (80/20)
        train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        
        return train_test_split["train"], train_test_split["test"]
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Dataset) -> Trainer:
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0442\u0440\u0435\u043d\u0435\u0440\u0430"""
        # \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0430 LoRA \u0435\u0441\u043b\u0438 \u043d\u0443\u0436\u043d\u043e
        if config.training.use_lora and self.peft_model is None:
            self.setup_lora()
        
        model_to_train = self.peft_model if config.training.use_lora else self.model
        
        # \u0410\u0440\u0433\u0443\u043c\u0435\u043d\u0442\u044b \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f
        training_args = TrainingArguments(
            output_dir=f"{config.model.cache_dir}/training_output",
            num_train_epochs=config.training.num_train_epochs,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            per_device_eval_batch_size=config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            eval_steps=config.training.eval_steps,
            save_total_limit=config.training.save_total_limit,
            learning_rate=config.training.learning_rate,
            lr_scheduler_type=config.training.lr_scheduler_type,
            optim=config.training.optim,
            fp16=config.training.fp16,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # \u041e\u0442\u043a\u043b\u044e\u0447\u0430\u0435\u043c wandb/tensorboard \u0434\u043b\u044f \u043e\u0444\u043b\u0430\u0439\u043d \u0440\u0435\u0436\u0438\u043c\u0430
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0442\u0440\u0435\u043d\u0435\u0440\u0430
        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        return trainer
    
    def train_model(self, documents: List[ProcessedDocument], save_path: Optional[str] = None) -> Dict[str, Any]:
        """\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043c\u0435\u0442\u043e\u0434 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f \u043c\u043e\u0434\u0435\u043b\u0438"""
        try:
            logger.info("Starting model training...")
            
            # \u041f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u043a\u0430 \u0434\u0430\u043d\u043d\u044b\u0445
            train_dataset, eval_dataset = self.prepare_training_data(documents)
            
            logger.info(f"Training dataset size: {len(train_dataset)}")
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0442\u0440\u0435\u043d\u0435\u0440\u0430
            trainer = self.create_trainer(train_dataset, eval_dataset)
            
            # \u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435
            logger.info("Starting training process...")
            training_result = trainer.train()
            
            # \u041e\u0446\u0435\u043d\u043a\u0430
            logger.info("Evaluating model...")
            eval_result = trainer.evaluate()
            
            # \u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u0438
            if save_path is None:
                save_path = f"{config.model.cache_dir}/fine_tuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Saving model to: {save_path}")
            
            if config.training.use_lora:
                self.peft_model.save_pretrained(save_path)
            else:
                trainer.save_model(save_path)
            
            self.tokenizer.save_pretrained(save_path)
            
            # \u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u043c\u0435\u0442\u0430\u0434\u0430\u043d\u043d\u044b\u0445 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f
            metadata = {
                "training_time": datetime.now().isoformat(),
                "model_name": config.model.model_name,
                "training_config": asdict(config.training),
                "training_result": {
                    "train_loss": training_result.training_loss,
                    "train_runtime": training_result.training_time,
                    "train_samples_per_second": training_result.train_samples_per_second,
                },
                "eval_result": eval_result,
                "dataset_info": {
                    "train_size": len(train_dataset),
                    "eval_size": len(eval_dataset),
                },
                "save_path": save_path
            }
            
            metadata_path = os.path.join(save_path, "training_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info("Training completed successfully!")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def load_fine_tuned_model(self, model_path: str):
        """\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0434\u043e\u043e\u0431\u0443\u0447\u0435\u043d\u043d\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438"""
        try:
            logger.info(f"Loading fine-tuned model from: {model_path}")
            
            if config.training.use_lora:
                # \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0431\u0430\u0437\u043e\u0432\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.model.model_name,
                    cache_dir=config.model.cache_dir,
                    torch_dtype=getattr(torch, config.model.torch_dtype),
                    device_map="auto",
                    trust_remote_code=config.model.trust_remote_code
                )
                
                # \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 LoRA \u0430\u0434\u0430\u043f\u0442\u0435\u0440\u0430
                self.peft_model = PeftModel.from_pretrained(base_model, model_path)
                self.model = self.peft_model
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    cache_dir=config.model.cache_dir,
                    torch_dtype=getattr(torch, config.model.torch_dtype),
                    device_map="auto",
                    trust_remote_code=config.model.trust_remote_code
                )
            
            # \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0442\u043e\u043a\u0435\u043d\u0438\u0437\u0430\u0442\u043e\u0440\u0430
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise
    
    def get_training_status(self) -> Dict[str, Any]:
        """\u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0441\u0442\u0430\u0442\u0443\u0441\u0430 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f"""
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "peft_model_loaded": self.peft_model is not None,
            "lora_enabled": config.training.use_lora,
            "device": config.model.device,
            "model_name": config.model.model_name
        }

# \u0424\u0443\u043d\u043a\u0446\u0438\u0438 \u0434\u043b\u044f \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u044f \u0432 \u0434\u0440\u0443\u0433\u0438\u0445 \u043c\u043e\u0434\u0443\u043b\u044f\u0445
def create_trainer() -> ModelTrainer:
    """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u044d\u043a\u0437\u0435\u043c\u043f\u043b\u044f\u0440\u0430 \u0442\u0440\u0435\u043d\u0435\u0440\u0430"""
    return ModelTrainer()

def train_on_documents(documents: List[ProcessedDocument], save_path: Optional[str] = None) -> Dict[str, Any]:
    """\u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u0438 \u043d\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445"""
    trainer = create_trainer()
    return trainer.train_model(documents, save_path)

def load_custom_model(model_path: str) -> ModelTrainer:
    """\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u043a\u0430\u0441\u0442\u043e\u043c\u043d\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438"""
    trainer = create_trainer()
    trainer.load_fine_tuned_model(model_path)
    return trainer
