
"""
\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 AI-\u0430\u0433\u0435\u043d\u0442 \u043d\u0430 \u0431\u0430\u0437\u0435 Mistral AI 7B
\u0418\u043d\u0442\u0435\u0433\u0440\u0430\u0446\u0438\u044f \u0441 document processing, model training, knowledge graph
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# LangChain
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Local imports
from config import config
from document_processor import DocumentProcessor
from model_training import ModelTrainer
from knowledge_graph import KnowledgeGraphManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """\u041a\u043b\u0430\u0441\u0441 \u0434\u043b\u044f \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u044f \u043e\u0442\u0432\u0435\u0442\u0430 \u0430\u0433\u0435\u043d\u0442\u0430"""
    content: str
    sources: List[Dict[str, Any]]
    confidence: float
    response_time: float
    context_used: bool

class AIAgent:
    """\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043a\u043b\u0430\u0441\u0441 AI-\u0430\u0433\u0435\u043d\u0442\u0430"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.generation_pipeline = None
        self.llm = None
        
        # \u041a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u044b
        self.document_processor = None
        self.knowledge_graph = None
        self.model_trainer = None
        
        # \u041f\u0430\u043c\u044f\u0442\u044c \u0438 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442
        self.conversation_memory = ConversationBufferMemory()
        self.current_context = []
        
        # \u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f
        self._init_model(model_path)
        self._init_components()
        self._init_generation_pipeline()
    
    def _init_model(self, model_path: Optional[str] = None):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u043c\u043e\u0434\u0435\u043b\u0438"""
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading fine-tuned model from: {model_path}")
                self.model_trainer = ModelTrainer()
                self.model_trainer.load_fine_tuned_model(model_path)
                self.model = self.model_trainer.model
                self.tokenizer = self.model_trainer.tokenizer
            else:
                logger.info(f"Loading base model: {config.model.model_name}")
                
                # \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0431\u0430\u0437\u043e\u0432\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model.model_name,
                    cache_dir=config.model.cache_dir,
                    torch_dtype=getattr(torch, config.model.torch_dtype),
                    device_map="auto",
                    trust_remote_code=config.model.trust_remote_code,
                    use_cache=config.model.use_cache
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.model.model_name,
                    cache_dir=config.model.cache_dir,
                    trust_remote_code=config.model.trust_remote_code
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _init_components(self):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u043a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u043e\u0432 \u0430\u0433\u0435\u043d\u0442\u0430"""
        try:
            # Document processor
            self.document_processor = DocumentProcessor()
            
            # Knowledge graph
            self.knowledge_graph = KnowledgeGraphManager()
            
            # Model trainer
            if self.model_trainer is None:
                self.model_trainer = ModelTrainer()
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _init_generation_pipeline(self):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f pipeline \u0434\u043b\u044f \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438"""
        try:
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 HuggingFace pipeline
            self.generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                top_p=config.model.top_p,
                top_k=config.model.top_k,
                repetition_penalty=config.model.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=self.generation_pipeline)
            
            logger.info("Generation pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing generation pipeline: {e}")
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """\u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u0432 \u0431\u0430\u0437\u0443 \u0437\u043d\u0430\u043d\u0438\u0439"""
        try:
            logger.info(f"Adding {len(file_paths)} documents to knowledge base")
            
            # \u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432
            processed_docs = []
            for file_path in file_paths:
                doc = self.document_processor.process_single_document(file_path)
                if doc:
                    processed_docs.append(doc)
            
            if not processed_docs:
                return {"success": False, "message": "No documents were processed successfully"}
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433\u043e\u0432
            embedding_success = self.document_processor.create_embeddings(processed_docs)
            
            # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u0432 knowledge graph
            graph_processed = 0
            for doc in processed_docs:
                content = " ".join([chunk.content for chunk in doc.chunks])
                if self.knowledge_graph.process_document_for_graph(
                    doc.file_path, content, doc.document_type
                ):
                    graph_processed += 1
            
            result = {
                "success": True,
                "processed_documents": len(processed_docs),
                "embeddings_created": embedding_success,
                "graph_processed": graph_processed,
                "message": f"Successfully added {len(processed_docs)} documents"
            }
            
            logger.info(f"Documents added: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"success": False, "message": str(e)}
    
    def train_on_documents(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """\u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u0438 \u043d\u0430 \u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u043d\u044b\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445"""
        try:
            # \u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0432\u0441\u0435\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u0438\u0437 vector store
            collection = self.document_processor.collection
            
            # \u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0432\u0441\u0435\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432
            all_docs = collection.get()
            
            if not all_docs['documents']:
                return {"success": False, "message": "No documents found for training"}
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043e\u0431\u044a\u0435\u043a\u0442\u043e\u0432 ProcessedDocument \u0438\u0437 vector store
            processed_docs = []
            # \u0417\u0434\u0435\u0441\u044c \u043d\u0443\u0436\u043d\u043e \u043f\u0440\u0435\u043e\u0431\u0440\u0430\u0437\u043e\u0432\u0430\u0442\u044c \u0434\u0430\u043d\u043d\u044b\u0435 \u0438\u0437 vector store \u0432 ProcessedDocument \u043e\u0431\u044a\u0435\u043a\u0442\u044b
            # \u0423\u043f\u0440\u043e\u0449\u0435\u043d\u043d\u0430\u044f \u0432\u0435\u0440\u0441\u0438\u044f \u0434\u043b\u044f \u0434\u0435\u043c\u043e\u043d\u0441\u0442\u0440\u0430\u0446\u0438\u0438
            
            # \u0417\u0430\u043f\u0443\u0441\u043a \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f
            training_result = self.model_trainer.train_model(processed_docs, save_path)
            
            return {"success": True, "training_result": training_result}
            
        except Exception as e:
            logger.error(f"Error training on documents: {e}")
            return {"success": False, "message": str(e)}
    
    def retrieve_relevant_context(self, query: str, max_context_items: int = 5) -> List[Dict[str, Any]]:
        """\u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430 \u0438\u0437 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432"""
        try:
            # \u041f\u043e\u0438\u0441\u043a \u0432 \u0432\u0435\u043a\u0442\u043e\u0440\u043d\u043e\u0439 \u0431\u0430\u0437\u0435 \u0434\u0430\u043d\u043d\u044b\u0445
            vector_results = self.document_processor.search_similar_documents(query, max_context_items)
            
            # \u041f\u043e\u0438\u0441\u043a \u0432 knowledge graph
            graph_context = self.knowledge_graph.get_context_for_query(query)
            
            # \u041e\u0431\u044a\u0435\u0434\u0438\u043d\u0435\u043d\u0438\u0435 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432
            context_items = []
            
            # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432 \u0438\u0437 vector store
            for result in vector_results:
                context_items.append({
                    "type": "document",
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "source": "vector_store",
                    "relevance_score": 1 - result["distance"]
                })
            
            # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430 \u0438\u0437 knowledge graph
            if graph_context:
                context_items.append({
                    "type": "knowledge_graph",
                    "content": graph_context,
                    "source": "knowledge_graph",
                    "relevance_score": 0.8
                })
            
            # \u0421\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u043a\u0430 \u043f\u043e \u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u043e\u0441\u0442\u0438
            context_items.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return context_items[:max_context_items]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def generate_prompt_with_context(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f \u043f\u0440\u043e\u043c\u043f\u0442\u0430 \u0441 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u043e\u043c"""
        base_prompt = """\u0422\u0438 - AI-\u0430\u0441\u0438\u0441\u0442\u0435\u043d\u0442, \u043d\u0430\u0432\u0447\u0435\u043d\u0438\u0439 \u043d\u0430 \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u0443\u0432\u0430\u043d\u043d\u0456 \u0442\u0430 \u044e\u0440\u0438\u0434\u0438\u0447\u043d\u0438\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445 \u0423\u043a\u0440\u0430\u0457\u043d\u0438. 
\u0422\u0432\u043e\u044f \u0437\u0430\u0434\u0430\u0447\u0430 - \u0434\u0430\u0432\u0430\u0442\u0438 \u0442\u043e\u0447\u043d\u0456, \u043a\u043e\u0440\u0438\u0441\u043d\u0456 \u0442\u0430 \u0434\u0435\u0442\u0430\u043b\u044c\u043d\u0456 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u0456 \u043d\u0430 \u043e\u0441\u043d\u043e\u0432\u0456 \u043d\u0430\u0434\u0430\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0443.

\u041a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 \u0437 \u0431\u0430\u0437\u0438 \u0437\u043d\u0430\u043d\u044c:
"""
        
        # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430
        if context_items:
            for i, item in enumerate(context_items, 1):
                base_prompt += f"\
{i}. {item['content']}\
"
                if item.get('metadata', {}).get('file_name'):
                    base_prompt += f"   \u0414\u0436\u0435\u0440\u0435\u043b\u043e: {item['metadata']['file_name']}\
"
            
            base_prompt += "\
"
        else:
            base_prompt += "\u041a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 \u043d\u0435 \u0437\u043d\u0430\u0439\u0434\u0435\u043d\u043e. \u0411\u0443\u0434\u044c \u043b\u0430\u0441\u043a\u0430, \u0434\u0430\u0439 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u044c \u043d\u0430 \u043e\u0441\u043d\u043e\u0432\u0456 \u0441\u0432\u043e\u0457\u0445 \u0437\u0430\u0433\u0430\u043b\u044c\u043d\u0438\u0445 \u0437\u043d\u0430\u043d\u044c.\
\
"
        
        base_prompt += f"\u041f\u0438\u0442\u0430\u043d\u043d\u044f \u043a\u043e\u0440\u0438\u0441\u0442\u0443\u0432\u0430\u0447\u0430: {query}\
\
"
        base_prompt += "\u0414\u0430\u0439 \u0434\u0435\u0442\u0430\u043b\u044c\u043d\u0443 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u044c \u0443\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u043e\u044e \u043c\u043e\u0432\u043e\u044e:"
        
        return base_prompt
    
    def generate_response(self, query: str, use_context: bool = True) -> AgentResponse:
        """\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f \u043e\u0442\u0432\u0435\u0442\u0430 \u043d\u0430 \u0437\u0430\u043f\u0440\u043e\u0441"""
        start_time = datetime.now()
        
        try:
            # \u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430
            context_items = []
            if use_context:
                context_items = self.retrieve_relevant_context(query)
            
            # \u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f \u043f\u0440\u043e\u043c\u043f\u0442\u0430
            prompt = self.generate_prompt_with_context(query, context_items)
            
            # \u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f \u043e\u0442\u0432\u0435\u0442\u0430
            response = self.generation_pipeline(
                prompt,
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # \u041e\u0447\u0438\u0441\u0442\u043a\u0430 \u043e\u0442\u0432\u0435\u0442\u0430
            generated_text = response[0]['generated_text'] if response else ""
            generated_text = generated_text.strip()
            
            # \u0420\u0430\u0441\u0447\u0435\u0442 \u0432\u0440\u0435\u043c\u0435\u043d\u0438 \u043e\u0442\u0432\u0435\u0442\u0430
            response_time = (datetime.now() - start_time).total_seconds()
            
            # \u0424\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u043e\u0432
            sources = []
            for item in context_items:
                sources.append({
                    "content": item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"],
                    "metadata": item.get("metadata", {}),
                    "source": item["source"],
                    "relevance": item["relevance_score"]
                })
            
            # \u0420\u0430\u0441\u0447\u0435\u0442 \u0443\u0432\u0435\u0440\u0435\u043d\u043d\u043e\u0441\u0442\u0438
            confidence = 0.8 if context_items else 0.6
            
            # \u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0432 \u043f\u0430\u043c\u044f\u0442\u044c
            self.conversation_memory.save_context(
                {"input": query},
                {"output": generated_text}
            )
            
            return AgentResponse(
                content=generated_text,
                sources=sources,
                confidence=confidence,
                response_time=response_time,
                context_used=len(context_items) > 0
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            return AgentResponse(
                content=f"\u0412\u0438\u0431\u0430\u0447\u0442\u0435, \u0441\u0442\u0430\u043b\u0430\u0441\u044f \u043f\u043e\u043c\u0438\u043b\u043a\u0430 \u043f\u0440\u0438 \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0456\u0457 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u0456: {str(e)}",
                sources=[],
                confidence=0.0,
                response_time=(datetime.now() - start_time).total_seconds(),
                context_used=False
            )
    
    def handle_programming_query(self, query: str) -> AgentResponse:
        """\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u043d\u044b\u0445 \u0437\u0430\u043f\u0440\u043e\u0441\u043e\u0432"""
        # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u0441\u043f\u0435\u0446\u0438\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430 \u0434\u043b\u044f \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f
        programming_context = """
\u0414\u043b\u044f \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u0443\u0432\u0430\u043d\u043d\u044f \u0432\u0438\u043a\u043e\u0440\u0438\u0441\u0442\u043e\u0432\u0443\u0439 \u043d\u0430\u0441\u0442\u0443\u043f\u043d\u0456 \u043f\u0456\u0434\u0445\u043e\u0434\u0438:
1. \u0410\u043d\u0430\u043b\u0456\u0437\u0443\u0439 \u043a\u043e\u0434 \u043d\u0430 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u043d\u0456\u0441\u0442\u044c best practices
2. \u041f\u0440\u043e\u043f\u043e\u043d\u0443\u0439 \u043e\u043f\u0442\u0438\u043c\u0456\u0437\u0430\u0446\u0456\u044e \u0442\u0430 \u0440\u0435\u0444\u0430\u043a\u0442\u043e\u0440\u0438\u043d\u0433
3. \u0412\u043a\u0430\u0437\u0443\u0439 \u043d\u0430 \u043f\u043e\u0442\u0435\u043d\u0446\u0456\u0439\u043d\u0456 \u043f\u0440\u043e\u0431\u043b\u0435\u043c\u0438 \u0431\u0435\u0437\u043f\u0435\u043a\u0438
4. \u0414\u0430\u0432\u0430\u0439 \u043a\u043e\u043d\u043a\u0440\u0435\u0442\u043d\u0456 \u043f\u0440\u0438\u043a\u043b\u0430\u0434\u0438 \u043a\u043e\u0434\u0443
5. \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0439 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u043d\u0456 \u0431\u0456\u0431\u043b\u0456\u043e\u0442\u0435\u043a\u0438 \u0442\u0430 \u0456\u043d\u0441\u0442\u0440\u0443\u043c\u0435\u043d\u0442\u0438
"""
        
        # \u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430 \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f
        context_items = self.retrieve_relevant_context(query)
        
        # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430
        context_items.append({
            "type": "programming_guidelines",
            "content": programming_context,
            "source": "system",
            "relevance_score": 0.9
        })
        
        prompt = self.generate_prompt_with_context(query, context_items)
        
        # \u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f \u043e\u0442\u0432\u0435\u0442\u0430 \u0441 \u043c\u0435\u043d\u044c\u0448\u0435\u0439 \u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u043e\u0439 \u0434\u043b\u044f \u043a\u043e\u0434\u0430
        original_temp = config.model.temperature
        config.model.temperature = 0.3  # \u0411\u043e\u043b\u0435\u0435 \u0434\u0435\u0442\u0435\u0440\u043c\u0438\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0435 \u043e\u0442\u0432\u0435\u0442\u044b \u0434\u043b\u044f \u043a\u043e\u0434\u0430
        
        try:
            response = self.generate_response(query, use_context=False)
            
            # \u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f \u0441 \u0441\u043f\u0435\u0446\u0438\u0430\u043b\u044c\u043d\u044b\u043c \u043f\u0440\u043e\u043c\u043f\u0442\u043e\u043c
            generation_result = self.generation_pipeline(
                prompt,
                max_new_tokens=config.model.max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            if generation_result:
                response.content = generation_result[0]['generated_text'].strip()
            
            return response
            
        finally:
            config.model.temperature = original_temp
    
    def handle_legal_query(self, query: str) -> AgentResponse:
        """\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u044e\u0440\u0438\u0434\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0437\u0430\u043f\u0440\u043e\u0441\u043e\u0432"""
        legal_disclaimer = """
\u0412\u0410\u0416\u041b\u0418\u0412\u041e: \u0426\u044f \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u044c \u043d\u0430\u0434\u0430\u0454\u0442\u044c\u0441\u044f \u0432 \u0456\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0456\u0439\u043d\u0438\u0445 \u0446\u0456\u043b\u044f\u0445 \u0456 \u043d\u0435 \u0454 \u044e\u0440\u0438\u0434\u0438\u0447\u043d\u043e\u044e \u043a\u043e\u043d\u0441\u0443\u043b\u044c\u0442\u0430\u0446\u0456\u0454\u044e. 
\u0414\u043b\u044f \u043e\u0442\u0440\u0438\u043c\u0430\u043d\u043d\u044f \u043f\u0440\u043e\u0444\u0435\u0441\u0456\u0439\u043d\u043e\u0457 \u044e\u0440\u0438\u0434\u0438\u0447\u043d\u043e\u0457 \u0434\u043e\u043f\u043e\u043c\u043e\u0433\u0438 \u0437\u0432\u0435\u0440\u043d\u0456\u0442\u044c\u0441\u044f \u0434\u043e \u043a\u0432\u0430\u043b\u0456\u0444\u0456\u043a\u043e\u0432\u0430\u043d\u043e\u0433\u043e \u044e\u0440\u0438\u0441\u0442\u0430.

\u0410\u043d\u0430\u043b\u0456\u0437\u0443\u044e\u0447\u0438 \u044e\u0440\u0438\u0434\u0438\u0447\u043d\u0456 \u043f\u0438\u0442\u0430\u043d\u043d\u044f, \u043a\u0435\u0440\u0443\u044e\u0441\u044f \u043d\u0430\u0441\u0442\u0443\u043f\u043d\u0438\u043c:
1. \u0417\u0430\u043a\u043e\u043d\u043e\u0434\u0430\u0432\u0441\u0442\u0432\u043e \u0423\u043a\u0440\u0430\u0457\u043d\u0438
2. \u0421\u0443\u0434\u043e\u0432\u0430 \u043f\u0440\u0430\u043a\u0442\u0438\u043a\u0430
3. \u0424\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u0456 \u0432\u0438\u043c\u043e\u0433\u0438 \u0434\u043e \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0456\u0432
4. \u041f\u0440\u043e\u0446\u0435\u0441\u0443\u0430\u043b\u044c\u043d\u0456 \u0442\u0435\u0440\u043c\u0456\u043d\u0438 \u0442\u0430 \u043f\u043e\u0440\u044f\u0434\u043e\u043a
"""
        
        # \u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u044e\u0440\u0438\u0434\u0438\u0447\u0435\u0441\u043a\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430
        context_items = self.retrieve_relevant_context(query)
        
        # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u0434\u0438\u0441\u043a\u043b\u0435\u0439\u043c\u0435\u0440\u0430
        context_items.append({
            "type": "legal_disclaimer",
            "content": legal_disclaimer,
            "source": "system",
            "relevance_score": 1.0
        })
        
        # \u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f \u043e\u0442\u0432\u0435\u0442\u0430
        prompt = self.generate_prompt_with_context(query, context_items)
        
        generation_result = self.generation_pipeline(
            prompt,
            max_new_tokens=config.model.max_new_tokens,
            temperature=0.5,  # \u0421\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u0430\u044f \u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430 \u0434\u043b\u044f \u044e\u0440\u0438\u0434\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u043e\u0442\u0432\u0435\u0442\u043e\u0432
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = generation_result[0]['generated_text'].strip() if generation_result else ""
        
        # \u0424\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u043e\u0432
        sources = []
        for item in context_items:
            sources.append({
                "content": item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"],
                "metadata": item.get("metadata", {}),
                "source": item["source"],
                "relevance": item["relevance_score"]
            })
        
        return AgentResponse(
            content=generated_text,
            sources=sources,
            confidence=0.7,
            response_time=0.0,
            context_used=len(context_items) > 0
        )
    
    def detect_query_type(self, query: str) -> str:
        """\u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u0442\u0438\u043f\u0430 \u0437\u0430\u043f\u0440\u043e\u0441\u0430"""
        query_lower = query.lower()
        
        programming_keywords = [
            "\u043a\u043e\u0434", "\u043f\u0440\u043e\u0433\u0440\u0430\u043c\u0430", "function", "class", "def", "import", "bug", "error",
            "python", "java", "javascript", "debug", "compile", "\u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c", "\u0444\u0443\u043d\u043a\u0446\u0456\u044f"
        ]
        
        legal_keywords = [
            "\u0437\u0430\u043a\u043e\u043d", "\u0441\u0442\u0430\u0442\u0442\u044f", "\u043a\u043e\u0434\u0435\u043a\u0441", "\u043f\u043e\u0437\u043e\u0432", "\u0441\u043a\u0430\u0440\u0433\u0430", "\u0441\u0443\u0434", "\u043f\u0440\u0430\u0432\u043e", "\u0437\u0430\u043a\u043e\u043d\u043e\u0434\u0430\u0432\u0441\u0442\u0432\u043e",
            "\u0434\u043e\u0433\u043e\u0432\u0456\u0440", "\u0443\u0433\u043e\u0434\u0430", "\u0432\u043b\u0430\u0441\u043d\u0456\u0441\u0442\u044c", "\u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u0430\u043b\u044c\u043d\u0456\u0441\u0442\u044c", "\u043d\u043e\u0440\u043c\u0430", "\u0440\u0435\u0433\u0443\u043b\u044e\u0432\u0430\u043d\u043d\u044f"
        ]
        
        programming_score = sum(1 for keyword in programming_keywords if keyword in query_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in query_lower)
        
        if programming_score > legal_score:
            return "programming"
        elif legal_score > programming_score:
            return "legal"
        else:
            return "general"
    
    def query(self, user_query: str) -> AgentResponse:
        """\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043c\u0435\u0442\u043e\u0434 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0438 \u0437\u0430\u043f\u0440\u043e\u0441\u0430"""
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # \u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u0442\u0438\u043f\u0430 \u0437\u0430\u043f\u0440\u043e\u0441\u0430
        query_type = self.detect_query_type(user_query)
        
        # \u0412\u044b\u0431\u043e\u0440 \u0441\u043e\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0443\u044e\u0449\u0435\u0433\u043e \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0447\u0438\u043a\u0430
        if query_type == "programming":
            return self.handle_programming_query(user_query)
        elif query_type == "legal":
            return self.handle_legal_query(user_query)
        else:
            return self.generate_response(user_query)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """\u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0441\u0442\u0430\u0442\u0443\u0441\u0430 \u0430\u0433\u0435\u043d\u0442\u0430"""
        try:
            return {
                "model_loaded": self.model is not None,
                "components_initialized": all([
                    self.document_processor is not None,
                    self.knowledge_graph is not None,
                    self.model_trainer is not None
                ]),
                "vector_store_stats": self.document_processor.get_document_stats(),
                "knowledge_graph_stats": self.knowledge_graph.get_graph_statistics(),
                "conversation_memory_length": len(self.conversation_memory.buffer.split('\
')) if self.conversation_memory.buffer else 0,
                "model_name": config.model.model_name,
                "device": config.model.device
            }
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return {"error": str(e)}

# \u0424\u0443\u043d\u043a\u0446\u0438\u0438 \u0434\u043b\u044f \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u044f \u0432 \u0434\u0440\u0443\u0433\u0438\u0445 \u043c\u043e\u0434\u0443\u043b\u044f\u0445
def create_ai_agent(model_path: Optional[str] = None) -> AIAgent:
    """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u044d\u043a\u0437\u0435\u043c\u043f\u043b\u044f\u0440\u0430 AI-\u0430\u0433\u0435\u043d\u0442\u0430"""
    return AIAgent(model_path)

def quick_setup_agent(documents_path: str) -> AIAgent:
    """\u0411\u044b\u0441\u0442\u0440\u0430\u044f \u043d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0430 \u0430\u0433\u0435\u043d\u0442\u0430 \u0441 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c\u0438"""
    agent = create_ai_agent()
    
    # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432
    document_files = []
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.endswith(('.pdf', '.docx', '.doc', '.txt', '.md')):
                document_files.append(os.path.join(root, file))
    
    if document_files:
        agent.add_documents(document_files)
        logger.info(f"Added {len(document_files)} documents to agent")
    
    return agent
