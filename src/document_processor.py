
"""
Модуль обработки документов
Поддержка PDF, DOC, DOCX файлов с анализом содержания
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib

# Document processing libraries
import PyPDF2
import docx
from docx2txt import process as docx2txt_process
import pymupdf as fitz
from unstructured.partition.auto import partition

# ML embeddings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Local imports
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Класс для представления чанка документа"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    chunk_type: str = "text"

@dataclass
class ProcessedDocument:
    """\u041a\u043b\u0430\u0441\u0441 \u0434\u043b\u044f \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u044f \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u043d\u043e\u0433\u043e \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430"""
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    chunks: List[DocumentChunk]
    embedding_vector: Optional[List[float]] = None
    document_hash: str = ""
    
class DocumentProcessor:
    """\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043a\u043b\u0430\u0441\u0441 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0438 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432"""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.processed_documents = []
        
        # \u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433\u043e\u0432
        self._init_embedding_model()
        
        # \u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u0432\u0435\u043a\u0442\u043e\u0440\u043d\u043e\u0433\u043e \u0445\u0440\u0430\u043d\u0438\u043b\u0438\u0449\u0430
        self._init_vector_store()
    
    def _init_embedding_model(self):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u043c\u043e\u0434\u0435\u043b\u0438 \u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433\u043e\u0432"""
        try:
            self.embedding_model = SentenceTransformer(config.vector.embedding_model)
            logger.info(f"Embedding model loaded: {config.vector.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _init_vector_store(self):
        """\u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f ChromaDB"""
        try:
            # \u0421\u043e\u0437\u0434\u0430\u0435\u043c \u043a\u043b\u0438\u0435\u043d\u0442 Chroma
            self.vector_store = chromadb.PersistentClient(
                path=config.vector.persist_directory
            )
            
            # \u041f\u043e\u043b\u0443\u0447\u0430\u0435\u043c \u0438\u043b\u0438 \u0441\u043e\u0437\u0434\u0430\u0435\u043c \u043a\u043e\u043b\u043b\u0435\u043a\u0446\u0438\u044e
            self.collection = self.vector_store.get_or_create_collection(
                name=config.vector.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Vector store initialized: {config.vector.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def get_file_hash(self, file_path: str) -> str:
        """\u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0445\u0435\u0448\u0430 \u0444\u0430\u0439\u043b\u0430 \u0434\u043b\u044f \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0439"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_supported_format(self, file_path: str) -> bool:
        """\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u0438\u0432\u0430\u0435\u043c\u043e\u0433\u043e \u0444\u043e\u0440\u043c\u0430\u0442\u0430 \u0444\u0430\u0439\u043b\u0430"""
        return Path(file_path).suffix.lower() in config.data.supported_formats
    
    def detect_document_type(self, file_path: str) -> str:
        """\u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u0442\u0438\u043f\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430 (programming, legal, general)"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in config.data.programming_extensions:
            return "programming"
        elif file_ext in config.data.legal_extensions:
            return "legal"
        else:
            return "general"
    
    def extract_text_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0442\u0435\u043a\u0441\u0442\u0430 \u0438\u0437 PDF \u0441 \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u0435\u0439 \u043e \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430\u0445"""
        try:
            # \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u043c pymupdf \u0434\u043b\u044f \u043b\u0443\u0447\u0448\u0435\u0433\u043e \u0438\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u044f
            doc = fitz.open(file_path)
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # \u041e\u0447\u0438\u0441\u0442\u043a\u0430 \u0442\u0435\u043a\u0441\u0442\u0430
                text = self._clean_text(text)
                
                if text.strip():
                    pages_text.append((text, page_num + 1))
            
            doc.close()
            return pages_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            # Fallback \u043d\u0430 PyPDF2
            return self._extract_pdf_fallback(file_path)
    
    def _extract_pdf_fallback(self, file_path: str) -> List[Tuple[str, int]]:
        """\u0417\u0430\u043f\u0430\u0441\u043d\u043e\u0439 \u043c\u0435\u0442\u043e\u0434 \u0438\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u044f \u0442\u0435\u043a\u0441\u0442\u0430 \u0438\u0437 PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages_text = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    text = self._clean_text(text)
                    
                    if text.strip():
                        pages_text.append((text, page_num + 1))
                
                return pages_text
        except Exception as e:
            logger.error(f"Fallback PDF extraction failed: {e}")
            return []
    
    def extract_text_from_docx(self, file_path: str) -> List[Tuple[str, int]]:
        """\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0442\u0435\u043a\u0441\u0442\u0430 \u0438\u0437 DOCX"""
        try:
            # \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u043c unstructured \u0434\u043b\u044f \u043b\u0443\u0447\u0448\u0435\u0433\u043e \u0438\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u044f
            elements = partition(filename=file_path)
            
            text_content = []
            for element in elements:
                if element.text.strip():
                    text_content.append((element.text, None))
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            # Fallback \u043d\u0430 docx2txt
            try:
                text = docx2txt_process(file_path)
                text = self._clean_text(text)
                return [(text, None)] if text.strip() else []
            except Exception as fallback_error:
                logger.error(f"Fallback DOCX extraction failed: {fallback_error}")
                return []
    
    def extract_text_from_txt(self, file_path: str) -> List[Tuple[str, int]]:
        """\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0442\u0435\u043a\u0441\u0442\u0430 \u0438\u0437 TXT \u0444\u0430\u0439\u043b\u043e\u0432"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text = self._clean_text(text)
                return [(text, None)] if text.strip() else []
        except UnicodeDecodeError:
            # \u041f\u0440\u043e\u0431\u0443\u0435\u043c \u0434\u0440\u0443\u0433\u0438\u0435 \u043a\u043e\u0434\u0438\u0440\u043e\u0432\u043a\u0438
            for encoding in ['cp1251', 'latin1', 'ascii']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                        text = self._clean_text(text)
                        return [(text, None)] if text.strip() else []
                except:
                    continue
            return []
    
    def _clean_text(self, text: str) -> str:
        """\u041e\u0447\u0438\u0441\u0442\u043a\u0430 \u0442\u0435\u043a\u0441\u0442\u0430 \u043e\u0442 \u043b\u0438\u0448\u043d\u0438\u0445 \u0441\u0438\u043c\u0432\u043e\u043b\u043e\u0432 \u0438 \u043f\u0440\u043e\u0431\u0435\u043b\u043e\u0432"""
        # \u0423\u0434\u0430\u043b\u0435\u043d\u0438\u0435 \u043b\u0438\u0448\u043d\u0438\u0445 \u043f\u0440\u043e\u0431\u0435\u043b\u043e\u0432 \u0438 \u043f\u0435\u0440\u0435\u043d\u043e\u0441\u043e\u0432 \u0441\u0442\u0440\u043e\u043a
        text = re.sub(r'\s+', ' ', text)
        
        # \u0423\u0434\u0430\u043b\u0435\u043d\u0438\u0435 \u0441\u043f\u0435\u0446\u0438\u0430\u043b\u044c\u043d\u044b\u0445 \u0441\u0438\u043c\u0432\u043e\u043b\u043e\u0432, \u043d\u043e \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u043f\u0443\u043d\u043a\u0442\u0443\u0430\u0446\u0438\u0438
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"'\/\\\@#\$\%\^\&\*\+\=\|\~\`]', ' ', text)
        
        # \u0423\u0434\u0430\u043b\u0435\u043d\u0438\u0435 \u043c\u043d\u043e\u0436\u0435\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0445 \u043f\u0440\u043e\u0431\u0435\u043b\u043e\u0432
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def split_text_into_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """\u0420\u0430\u0437\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u0442\u0435\u043a\u0441\u0442\u0430 \u043d\u0430 \u0447\u0430\u043d\u043a\u0438 \u0441 \u043f\u0435\u0440\u0435\u043a\u0440\u044b\u0442\u0438\u0435\u043c"""
        if not text.strip():
            return []
        
        chunks = []
        text_length = len(text)
        
        for start in range(0, text_length, config.data.chunk_size - config.data.chunk_overlap):
            end = start + config.data.chunk_size
            
            if start >= text_length:
                break
            
            chunk_text = text[start:end]
            
            # \u0421\u043e\u0437\u0434\u0430\u0435\u043c \u043c\u0435\u0442\u0430\u0434\u0430\u043d\u043d\u044b\u0435 \u0434\u043b\u044f \u0447\u0430\u043d\u043a\u0430
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_start": start,
                "chunk_end": end,
                "chunk_length": len(chunk_text)
            })
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=f"chunk_{start}_{end}",
                source_file=metadata.get("file_path", ""),
                page_number=metadata.get("page_number"),
                chunk_type=metadata.get("document_type", "text")
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def process_single_document(self, file_path: str) -> Optional[ProcessedDocument]:
        """\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u043e\u0434\u043d\u043e\u0433\u043e \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        if not self.is_supported_format(file_path):
            logger.warning(f"Unsupported file format: {file_path}")
            return None
        
        file_size = os.path.getsize(file_path)
        if file_size > config.data.max_file_size_mb * 1024 * 1024:
            logger.warning(f"File too large: {file_path} ({file_size} bytes)")
            return None
        
        file_ext = Path(file_path).suffix.lower()
        document_type = self.detect_document_type(file_path)
        file_hash = self.get_file_hash(file_path)
        
        logger.info(f"Processing document: {file_path} ({document_type})")
        
        # \u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0442\u0435\u043a\u0441\u0442\u0430
        pages_text = []
        
        try:
            if file_ext == '.pdf':
                pages_text = self.extract_text_from_pdf(file_path)
            elif file_ext in ['.doc', '.docx']:
                pages_text = self.extract_text_from_docx(file_path)
            elif file_ext in ['.txt', '.md']:
                pages_text = self.extract_text_from_txt(file_path)
            else:
                logger.warning(f"Unsupported file extension: {file_ext}")
                return None
            
            if not pages_text:
                logger.warning(f"No text extracted from: {file_path}")
                return None
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0447\u0430\u043d\u043a\u043e\u0432
            all_chunks = []
            for text, page_num in pages_text:
                metadata = {
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "file_type": file_ext,
                    "document_type": document_type,
                    "file_size": file_size,
                    "page_number": page_num,
                    "document_hash": file_hash
                }
                
                chunks = self.split_text_into_chunks(text, metadata)
                all_chunks.extend(chunks)
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u043d\u043e\u0433\u043e \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430
            processed_doc = ProcessedDocument(
                file_path=file_path,
                file_name=Path(file_path).name,
                file_type=file_ext,
                file_size=file_size,
                chunks=all_chunks,
                document_hash=file_hash
            )
            
            logger.info(f"Processed {file_path}: {len(all_chunks)} chunks")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return None
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> List[ProcessedDocument]:
        """\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0432\u0441\u0435\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u0432 \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0438"""
        processed_docs = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return processed_docs
        
        # \u041f\u043e\u0438\u0441\u043a \u0444\u0430\u0439\u043b\u043e\u0432
        if recursive:
            file_pattern = "**/*"
        else:
            file_pattern = "*"
        
        files = []
        for ext in config.data.supported_formats:
            files.extend(directory.glob(f"{file_pattern}{ext}"))
        
        logger.info(f"Found {len(files)} documents to process")
        
        # \u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u043a\u0430\u0436\u0434\u043e\u0433\u043e \u0444\u0430\u0439\u043b\u0430
        for file_path in files:
            processed_doc = self.process_single_document(str(file_path))
            if processed_doc:
                processed_docs.append(processed_doc)
        
        logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs
    
    def create_embeddings(self, documents: List[ProcessedDocument]) -> bool:
        """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433\u043e\u0432 \u0434\u043b\u044f \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u0438 \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0432 \u0432\u0435\u043a\u0442\u043e\u0440\u043d\u043e\u0435 \u0445\u0440\u0430\u043d\u0438\u043b\u0438\u0449\u0435"""
        try:
            all_texts = []
            all_metadatas = []
            all_ids = []
            
            for doc in documents:
                for chunk in doc.chunks:
                    all_texts.append(chunk.content)
                    
                    metadata = {
                        "chunk_id": chunk.chunk_id,
                        "source_file": chunk.source_file,
                        "file_name": chunk.metadata.get("file_name", ""),
                        "file_type": chunk.metadata.get("file_type", ""),
                        "document_type": chunk.metadata.get("document_type", ""),
                        "page_number": chunk.page_number,
                        "chunk_start": chunk.metadata.get("chunk_start", 0),
                        "chunk_end": chunk.metadata.get("chunk_end", 0),
                        "document_hash": doc.document_hash
                    }
                    all_metadatas.append(metadata)
                    all_ids.append(chunk.chunk_id)
            
            if not all_texts:
                logger.warning("No text chunks to embed")
                return False
            
            logger.info(f"Creating embeddings for {len(all_texts)} chunks")
            
            # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433\u043e\u0432
            embeddings = self.embedding_model.encode(
                all_texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=False
            )
            
            # \u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0432 ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=all_texts,
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            logger.info(f"Successfully embedded and stored {len(all_texts)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return False
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """\u041f\u043e\u0438\u0441\u043a \u043f\u043e\u0445\u043e\u0436\u0438\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432 \u043f\u043e \u0437\u0430\u043f\u0440\u043e\u0441\u0443"""
        try:
            query_embedding = self.embedding_model.encode([query])
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            # \u0424\u043e\u0440\u043c\u0430\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "id": results['ids'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_stats(self) -> Dict[str, Any]:
        """\u041f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u0435 \u0441\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u043a\u0438 \u043f\u043e \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u043d\u044b\u043c \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c"""
        try:
            collection_count = self.collection.count()
            
            return {
                "total_chunks": collection_count,
                "collection_name": config.vector.collection_name,
                "persist_directory": config.vector.persist_directory,
                "embedding_model": config.vector.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

# \u0424\u0443\u043d\u043a\u0446\u0438\u0438 \u0434\u043b\u044f \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u044f \u0432 \u0434\u0440\u0443\u0433\u0438\u0445 \u043c\u043e\u0434\u0443\u043b\u044f\u0445
def create_document_processor() -> DocumentProcessor:
    """\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u044d\u043a\u0437\u0435\u043c\u043f\u043b\u044f\u0440\u0430 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0447\u0438\u043a\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432"""
    return DocumentProcessor()

def process_document_batch(file_paths: List[str]) -> List[ProcessedDocument]:
    """\u041f\u0430\u043a\u0435\u0442\u043d\u0430\u044f \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432"""
    processor = create_document_processor()
    processed_docs = []
    
    for file_path in file_paths:
        doc = processor.process_single_document(file_path)
        if doc:
            processed_docs.append(doc)
    
    # \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433\u043e\u0432
    if processed_docs:
        processor.create_embeddings(processed_docs)
    
    return processed_docs
