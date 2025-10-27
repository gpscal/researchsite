"""
LangChain-based service for PDF processing, indexing, and retrieval.
Replaces pdf_processor, pdf_service, and parts of rag_service.
"""

import os
import tempfile
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Iterator, List
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables at module level
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Set up HuggingFace token for model downloads
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    logger.info("HuggingFace token configured for model downloads")
else:
    logger.warning("HF_TOKEN not found in environment - some models may fail to download")

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
    logger.warning("Using legacy Chroma import")

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_anthropic import ChatAnthropic
import chromadb
from chromadb.config import Settings

# Re-use config from the existing RAG service for consistency
try:
    from rag_service import get_rag_service
    RAG_SERVICE_AVAILABLE = True
except ImportError:
    RAG_SERVICE_AVAILABLE = False
    logger.warning("RAG service not available for config import")

# Set up configuration
_persist_dir = "data/vector_store"
_collection_name = "research_collection"

# Try to get config from RAG service if available
if RAG_SERVICE_AVAILABLE:
    try:
        _rag_service = get_rag_service()
        _persist_dir = _rag_service.config.get("PERSIST_DIR", _persist_dir)
        _collection_name = _rag_service.config.get("COLLECTION", _collection_name)
    except Exception as e:
        logger.warning(f"Could not get RAG service configuration: {e}")
        logger.info(f"Using default configuration: {_persist_dir}, {_collection_name}")

# Use a LangChain-compatible embedder
_model_name = "all-MiniLM-L6-v2"  # Default model
try:
    if RAG_SERVICE_AVAILABLE and '_rag_service' in globals() and _rag_service:
        _model_name = _rag_service.config.get("EMBED_MODEL_LOCAL", _model_name)
except Exception:
    pass

logger.info(f"Initializing embeddings with model: {_model_name}")

# Set up cache directory for Hugging Face models
import tempfile
import os

# Use temp directory for cache to avoid disk space issues
cache_dir = "/tmp/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

# Set environment variables for Hugging Face
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir

try:
    # Initialize embeddings - cache directory is set via environment variables
    _embedder = SentenceTransformerEmbeddings(model_name=_model_name)
    logger.info("Embeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    raise

# LangChain-compatible wrapper for QwenVL
class QwenVLWrapper:
    """Wrapper to make QwenVL compatible with LangChain interface."""
    
    def __init__(self, model_name: str):
        from research_llm import QwenVLLL
        self.model = QwenVLLL(model_name=model_name)
        logger.info(f"QwenVL wrapper initialized with model: {model_name}")
    
    def _extract_prompt(self, messages):
        """Extract prompt text from various message formats."""
        if isinstance(messages, str):
            return messages
        elif isinstance(messages, list):
            # Handle list of messages
            prompt_parts = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    prompt_parts.append(msg.content)
                elif isinstance(msg, dict):
                    prompt_parts.append(msg.get('content', ''))
                else:
                    prompt_parts.append(str(msg))
            return '\n'.join(prompt_parts)
        elif hasattr(messages, 'to_string'):
            return messages.to_string()
        elif hasattr(messages, 'content'):
            return messages.content
        else:
            return str(messages)
    
    def invoke(self, messages):
        """LangChain-compatible invoke method."""
        prompt = self._extract_prompt(messages)
        # Use smaller max_tokens for faster responses to simple queries
        response = self.model.generate(prompt, max_tokens=256)
        # Return as a string (LangChain expects strings from LLMs)
        return str(response) if response else ""
    
    def stream(self, messages):
        """LangChain-compatible stream method."""
        prompt = self._extract_prompt(messages)
        # Use smaller max_tokens for faster responses
        for chunk in self.model.generate_stream(prompt, max_tokens=256):
            # Yield strings directly (LangChain expects string chunks)
            yield str(chunk) if chunk else ""
    
    def __call__(self, messages):
        """Make the wrapper callable."""
        return self.invoke(messages)

# Initialize LLM providers
_anthropic_llm = None
_qwenvl_llm = None

# Initialize Anthropic
anthropic_model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if anthropic_api_key:
    try:
        logger.info(f"Initializing Anthropic LLM with model: {anthropic_model_name}")
        _anthropic_llm = ChatAnthropic(model=anthropic_model_name, api_key=anthropic_api_key)
        logger.info("Anthropic LLM initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic LLM: {e}")
        logger.error("Make sure ANTHROPIC_API_KEY is valid in .env file")
else:
    logger.warning("ANTHROPIC_API_KEY not set - Anthropic LLM not available")

# Initialize QwenVL (lazy load on first use to save memory)
def get_qwenvl():
    """Lazy load QwenVL model."""
    global _qwenvl_llm
    if _qwenvl_llm is None:
        try:
            qwenvl_model_name = os.getenv("QWENVL_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
            logger.info(f"Initializing QwenVL with model: {qwenvl_model_name}")
            _qwenvl_llm = QwenVLWrapper(qwenvl_model_name)
            logger.info("QwenVL initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QwenVL: {e}")
            raise RuntimeError(f"QwenVL initialization failed: {e}")
    return _qwenvl_llm


class LangChainService:
    def __init__(self):
        """Initializes the LangChain service, setting up the vector store and retrieval chain."""
        # Ensure the persist directory exists
        Path(_persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize document tracking system
        self.doc_tracking_path = Path(_persist_dir) / "document_tracking.json"
        self._load_document_tracking()
        
        # Use persistent Chroma client for data persistence
        chroma_client = chromadb.PersistentClient(
            path=_persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        
        self.vector_store = Chroma(
            client=chroma_client,
            collection_name=_collection_name,
            embedding_function=_embedder,
        )
        
        # Initialize the collection properly (this ensures it exists even when empty)
        try:
            # Get or create the collection
            collection = self.vector_store._collection
            doc_count = collection.count()
            logger.info(f"Vector store initialized with {doc_count} documents")
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")
        
        # Check if at least one LLM is available
        if _anthropic_llm is None:
            logger.warning("Anthropic LLM not initialized. QwenVL will be used if requested.")
        
        # Store retrieval chains for different providers
        self.retrieval_chains = {}
        if _anthropic_llm:
            self.retrieval_chains['anthropic'] = self._create_retrieval_chain(_anthropic_llm)
        
        logger.info("LangChain service initialized successfully")

    def _create_retrieval_chain(self, llm):
        """Creates the LangChain retrieval chain for question-answering."""
        from operator import itemgetter
        
        # Create retriever with fallback for empty store
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 8}  # Return top 8 most relevant docs for comprehensive answers
        )
        
        system_prompt = (
            "You are a helpful research assistant. The user has uploaded PDF documents that are indexed. "
            "Below is the context retrieved from those indexed documents.\n"
            "\n"
            "CRITICAL RULES:\n"
            "1. The context below IS from indexed PDFs - never say 'no documents are indexed' if you see text below\n"
            "2. If you see document text in the context, acknowledge what PDFs ARE available\n"
            "3. Answer based on what's in the context, citing source and page numbers\n"
            "4. If the context doesn't answer the specific question, say what information IS available instead\n"
            "5. Only say 'no documents indexed' if the context section below is completely empty\n"
            "\n"
            "Retrieved Context:\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create a function to format documents into a single context string WITH metadata
        def format_docs(docs):
            if not docs:
                return "No documents have been indexed yet. Please upload PDF documents to enable retrieval."
            
            formatted = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', 'unknown')
                formatted.append(f"[Document {i+1}] From: {source}, Page: {page}\n{doc.page_content}")
            
            return "\n\n---\n\n".join(formatted)
        
        # Wrap QwenVL with RunnableLambda if it's not already a LangChain model
        if isinstance(llm, QwenVLWrapper):
            # Wrap the QwenVL wrapper with RunnableLambda to make it chain-compatible
            llm_runnable = RunnableLambda(llm.invoke)
        else:
            # Use the LLM directly (e.g., ChatAnthropic already implements Runnable)
            llm_runnable = llm
        
        # Create RAG chain with proper error handling
        rag_chain = (
            {
                "context": itemgetter("input") | retriever | format_docs, 
                "input": itemgetter("input")
            }
            | prompt
            | llm_runnable
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _get_retrieval_chain(self, provider='anthropic'):
        """Get or create retrieval chain for the specified provider."""
        if provider not in self.retrieval_chains:
            if provider == 'qwenvl':
                llm = get_qwenvl()
                self.retrieval_chains['qwenvl'] = self._create_retrieval_chain(llm)
            elif provider == 'anthropic':
                if _anthropic_llm is None:
                    raise RuntimeError("Anthropic LLM not initialized")
                self.retrieval_chains['anthropic'] = self._create_retrieval_chain(_anthropic_llm)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        return self.retrieval_chains[provider]

    def _load_document_tracking(self) -> None:
        """Load document tracking data from disk."""
        if self.doc_tracking_path.exists():
            try:
                with open(self.doc_tracking_path, 'r') as f:
                    self.document_tracking = json.load(f)
                logger.info(f"Loaded tracking data for {len(self.document_tracking)} documents")
            except Exception as e:
                logger.warning(f"Could not load document tracking: {e}")
                self.document_tracking = {}
        else:
            self.document_tracking = {}
    
    def _save_document_tracking(self) -> None:
        """Save document tracking data to disk."""
        try:
            with open(self.doc_tracking_path, 'w') as f:
                json.dump(self.document_tracking, f, indent=2)
            logger.info("Document tracking data saved")
        except Exception as e:
            logger.error(f"Failed to save document tracking: {e}")
    
    def _generate_document_id(self, file_bytes: bytes, filename: str) -> str:
        """Generates a unique ID for a document based on its content and name."""
        content_hash = hashlib.md5(file_bytes).hexdigest()
        name_hash = hashlib.md5(filename.encode()).hexdigest()
        return f"{name_hash[:8]}_{content_hash[:16]}"
    
    def _generate_chunk_hash(self, text: str) -> str:
        """Generate a hash for a text chunk for deduplication."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_existing_pages(self, document_id: str) -> set:
        """Get set of already processed page numbers for a document."""
        if document_id in self.document_tracking:
            return set(self.document_tracking[document_id].get("processed_pages", []))
        return set()
    
    def _get_existing_chunk_hashes(self, document_id: str) -> set:
        """Get set of existing chunk hashes for a document to prevent duplicates."""
        try:
            collection = self.vector_store._collection
            results = collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            chunk_hashes = set()
            if results and results.get("metadatas"):
                for metadata in results["metadatas"]:
                    if "chunk_hash" in metadata:
                        chunk_hashes.add(metadata["chunk_hash"])
            
            logger.info(f"Found {len(chunk_hashes)} existing chunks for document {document_id}")
            return chunk_hashes
        except Exception as e:
            logger.warning(f"Could not retrieve existing chunks: {e}")
            return set()
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get information about a processed document."""
        if document_id not in self.document_tracking:
            return {
                "success": False,
                "error": "Document not found",
                "document_id": document_id
            }
        
        doc_info = self.document_tracking[document_id]
        return {
            "success": True,
            "document_id": document_id,
            "filename": doc_info.get("filename", "unknown"),
            "total_pages": doc_info.get("total_pages", 0),
            "processed_pages": sorted(doc_info.get("processed_pages", [])),
            "total_chunks": doc_info.get("total_chunks", 0),
            "last_updated": doc_info.get("last_updated", "unknown"),
            "upload_count": doc_info.get("upload_count", 1)
        }

    def index_pdf(self, file_bytes: bytes, filename: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Loads a PDF from bytes, splits it into chunks, and indexes it in the vector store.
        Implements incremental processing - only processes new pages that haven't been indexed yet.
        
        Args:
            file_bytes: PDF file content as bytes
            filename: Original filename
            force_reindex: If True, reprocess all pages even if already indexed
            
        Returns:
            Dict with processing results including which pages were processed
        """
        document_id = self._generate_document_id(file_bytes, filename)
        
        # Get existing processing info
        existing_pages = self._get_existing_pages(document_id) if not force_reindex else set()
        existing_chunk_hashes = self._get_existing_chunk_hashes(document_id) if not force_reindex else set()
        
        is_new_document = document_id not in self.document_tracking
        
        logger.info(f"Processing PDF: {filename} (document_id: {document_id})")
        if not is_new_document and not force_reindex:
            logger.info(f"Document already exists. Processed pages: {sorted(existing_pages)}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()
            total_pages = len(pages)
            
            logger.info(f"PDF has {total_pages} pages total")
            
            # Filter to only new pages
            new_pages = []
            new_page_numbers = []
            skipped_pages = []
            
            for page_doc in pages:
                # Get page number (0-indexed in metadata, but we'll store 1-indexed)
                page_num = page_doc.metadata.get("page", 0) + 1
                
                if force_reindex or page_num not in existing_pages:
                    new_pages.append(page_doc)
                    new_page_numbers.append(page_num)
                else:
                    skipped_pages.append(page_num)
            
            if not new_pages:
                logger.info(f"All {total_pages} pages already processed. No new content to index.")
                return {
                    "success": True,
                    "filename": filename,
                    "document_id": document_id,
                    "page_count": total_pages,
                    "new_pages": 0,
                    "skipped_pages": len(skipped_pages),
                    "chunks": 0,
                    "new_chunks": 0,
                    "duplicate_chunks": 0,
                    "message": "Document already fully indexed. No new pages to process."
                }
            
            logger.info(f"Processing {len(new_pages)} new pages: {sorted(new_page_numbers)}")
            if skipped_pages:
                logger.info(f"Skipping {len(skipped_pages)} already-processed pages: {sorted(skipped_pages)}")
            
            # Split new pages into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(new_pages)
            
            # Deduplicate chunks and add metadata
            new_chunks = []
            duplicate_count = 0
            
            for split in splits:
                # Generate hash for deduplication
                chunk_hash = self._generate_chunk_hash(split.page_content)
                
                # Skip if this exact chunk already exists
                if chunk_hash in existing_chunk_hashes:
                    duplicate_count += 1
                    logger.debug(f"Skipping duplicate chunk with hash {chunk_hash}")
                    continue
                
                # Add metadata
                split.metadata["document_id"] = document_id
                split.metadata["source"] = filename
                split.metadata["chunk_hash"] = chunk_hash
                
                new_chunks.append(split)
                existing_chunk_hashes.add(chunk_hash)  # Track to prevent duplicates within this batch
            
            # Add new chunks to vector store
            if new_chunks:
                self.vector_store.add_documents(documents=new_chunks)
                logger.info(f"Added {len(new_chunks)} new chunks to vector store")
            
            # Update tracking data
            if document_id not in self.document_tracking:
                self.document_tracking[document_id] = {
                    "filename": filename,
                    "total_pages": total_pages,
                    "processed_pages": [],
                    "total_chunks": 0,
                    "upload_count": 0,
                    "created_at": datetime.now().isoformat()
                }
            
            # Update processed pages and chunk count
            doc_info = self.document_tracking[document_id]
            all_processed_pages = set(doc_info.get("processed_pages", [])) | set(new_page_numbers)
            doc_info["processed_pages"] = sorted(list(all_processed_pages))
            doc_info["total_pages"] = total_pages
            doc_info["total_chunks"] = doc_info.get("total_chunks", 0) + len(new_chunks)
            doc_info["upload_count"] = doc_info.get("upload_count", 0) + 1
            doc_info["last_updated"] = datetime.now().isoformat()
            
            # Save tracking data
            self._save_document_tracking()
            
            result = {
                "success": True,
                "filename": filename,
                "document_id": document_id,
                "page_count": total_pages,
                "new_pages": len(new_pages),
                "skipped_pages": len(skipped_pages),
                "chunks": len(splits),
                "new_chunks": len(new_chunks),
                "duplicate_chunks": duplicate_count,
                "processed_page_numbers": sorted(new_page_numbers),
                "all_processed_pages": sorted(list(all_processed_pages)),
                "is_complete": len(all_processed_pages) == total_pages
            }
            
            if not is_new_document:
                result["message"] = f"Incremental update: processed {len(new_pages)} new pages, skipped {len(skipped_pages)} existing pages"
            
            logger.info(f"Processing complete: {len(new_chunks)} new chunks added, {duplicate_count} duplicates skipped")
            
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing PDF: {e}")
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def query_stream(self, question: str, provider: str = 'anthropic', use_training_data: bool = True, use_web: bool = False, top_k: int = 8) -> Iterator[str]:
        """Queries the retrieval chain and streams the response."""
        try:
            # If not using training data, just answer directly without context
            if not use_training_data:
                if provider == 'qwenvl':
                    llm = get_qwenvl()
                    # Answer without context
                    for chunk in llm.stream(question):
                        yield json.dumps({"content": chunk})
                else:
                    # Use Anthropic without context
                    if _anthropic_llm is None:
                        yield json.dumps({"error": "Anthropic LLM not available"})
                        yield json.dumps({"done": True})
                        return
                    
                    # Simple prompt without RAG context
                    for chunk in _anthropic_llm.stream(question):
                        if hasattr(chunk, 'content'):
                            yield json.dumps({"content": chunk.content})
                        else:
                            yield json.dumps({"content": str(chunk)})
                
                yield json.dumps({"sources": []})
                yield json.dumps({"done": True})
                return
            
            # Check if the collection has any documents
            collection = self.vector_store._collection
            doc_count = collection.count()
            
            if doc_count == 0:
                # No documents indexed yet - answer without context
                if provider == 'qwenvl':
                    llm = get_qwenvl()
                    for chunk in llm.stream(question):
                        yield json.dumps({"content": chunk})
                else:
                    if _anthropic_llm is None:
                        yield json.dumps({"error": "Anthropic LLM not available"})
                        yield json.dumps({"done": True})
                        return
                    for chunk in _anthropic_llm.stream(question):
                        if hasattr(chunk, 'content'):
                            yield json.dumps({"content": chunk.content})
                        else:
                            yield json.dumps({"content": str(chunk)})
                
                yield json.dumps({"sources": []})
                yield json.dumps({"done": True})
                return
            
            # Get documents from retriever directly for sources
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(question)
            
            # Prepare sources (will send at the END of streaming)
            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "unknown"),
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                for doc in docs
            ]
            
            # Format context from retrieved documents
            def format_docs(docs):
                if not docs:
                    return "No documents have been indexed yet. Please upload PDF documents to enable retrieval."
                formatted = []
                for i, doc in enumerate(docs):
                    source = doc.metadata.get('source', 'unknown')
                    page = doc.metadata.get('page', 'unknown')
                    formatted.append(f"[Document {i+1}] From: {source}, Page: {page}\n{doc.page_content}")
                return "\n\n---\n\n".join(formatted)
            
            context = format_docs(docs)
            
            # Build the full prompt
            system_prompt = (
                "You are a helpful research assistant. The user has uploaded PDF documents that are indexed. "
                "Below is the context retrieved from those indexed documents.\n"
                "\n"
                "CRITICAL RULES:\n"
                "1. The context below IS from indexed PDFs - never say 'no documents are indexed' if you see text below\n"
                "2. If you see document text in the context, acknowledge what PDFs ARE available\n"
                "3. Answer based on what's in the context, citing source and page numbers\n"
                "4. If the context doesn't answer the specific question, say what information IS available instead\n"
                "5. Only say 'no documents indexed' if the context section below is completely empty\n"
                "\n"
                f"Retrieved Context:\n{context}"
            )
            
            full_prompt = f"{system_prompt}\n\nHuman: {question}\n\nAssistant:"
            
            # Handle streaming differently for QwenVL vs Anthropic
            if provider == 'qwenvl':
                llm = get_qwenvl()
                # Stream directly from QwenVL
                for chunk in llm.stream(full_prompt):
                    yield json.dumps({"content": chunk})
            else:
                # Use the retrieval chain for Anthropic
                retrieval_chain = self._get_retrieval_chain(provider)
                for chunk in retrieval_chain.stream({"input": question}):
                    yield json.dumps({"content": chunk})
            
            # Send sources AFTER content is complete
            if sources:
                yield json.dumps({"sources": sources})
            
            yield json.dumps({"done": True})
        except Exception as e:
            logger.error(f"Error in query_stream: {e}", exc_info=True)
            yield json.dumps({"error": f"Query failed: {str(e)}"})
            yield json.dumps({"done": True})

    def query(self, question: str, provider: str = 'anthropic', use_training_data: bool = True, use_web: bool = False, top_k: int = 8) -> Dict[str, Any]:
        """Queries the retrieval chain and returns the final response."""
        try:
            # If not using training data, just answer directly without context
            if not use_training_data:
                if provider == 'qwenvl':
                    llm = get_qwenvl()
                    answer = llm.invoke(question)
                else:
                    # Use Anthropic without context
                    if _anthropic_llm is None:
                        return {
                            "success": False,
                            "error": "Anthropic LLM not available",
                            "answer": "Sorry, the LLM is not available."
                        }
                    response = _anthropic_llm.invoke(question)
                    answer = response.content if hasattr(response, 'content') else str(response)
                
                return {
                    "success": True,
                    "answer": answer,
                    "sources": []
                }
            
            # Check if the collection has any documents
            collection = self.vector_store._collection
            doc_count = collection.count()
            
            if doc_count == 0:
                # No documents indexed yet - answer without context
                if provider == 'qwenvl':
                    llm = get_qwenvl()
                    answer = llm.invoke(question)
                else:
                    if _anthropic_llm is None:
                        return {
                            "success": False,
                            "error": "Anthropic LLM not available",
                            "answer": "Sorry, the LLM is not available."
                        }
                    response = _anthropic_llm.invoke(question)
                    answer = response.content if hasattr(response, 'content') else str(response)
                
                return {
                    "success": True,
                    "answer": answer,
                    "sources": []
                }
            
            # Get documents from retriever directly for sources
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(question)
            
            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "unknown"),
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                for doc in docs
            ]
            
            # Handle QwenVL differently from Anthropic
            if provider == 'qwenvl':
                # Format context from retrieved documents
                def format_docs(docs):
                    if not docs:
                        return "No documents have been indexed yet."
                    formatted = []
                    for i, doc in enumerate(docs):
                        source = doc.metadata.get('source', 'unknown')
                        page = doc.metadata.get('page', 'unknown')
                        formatted.append(f"[Document {i+1}] From: {source}, Page: {page}\n{doc.page_content}")
                    return "\n\n---\n\n".join(formatted)
                
                context = format_docs(docs)
                
                # Build the full prompt
                system_prompt = (
                    "You are a helpful research assistant. The user has uploaded PDF documents that are indexed. "
                    "Below is the context retrieved from those indexed documents.\n"
                    "\n"
                    "CRITICAL RULES:\n"
                    "1. The context below IS from indexed PDFs - never say 'no documents are indexed' if you see text below\n"
                    "2. If you see document text in the context, acknowledge what PDFs ARE available\n"
                    "3. Answer based on what's in the context, citing source and page numbers\n"
                    "4. If the context doesn't answer the specific question, say what information IS available instead\n"
                    "5. Only say 'no documents indexed' if the context section below is completely empty\n"
                    "\n"
                    f"Retrieved Context:\n{context}"
                )
                
                full_prompt = f"{system_prompt}\n\nHuman: {question}\n\nAssistant:"
                
                llm = get_qwenvl()
                answer = llm.invoke(full_prompt)
            else:
                # Use the retrieval chain for Anthropic
                retrieval_chain = self._get_retrieval_chain(provider)
                answer = retrieval_chain.invoke({"input": question})
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error in query: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "answer": "Sorry, I encountered an error processing your query. Please try again."
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Returns statistics about the vector store collection."""
        try:
            collection = self.vector_store._collection
            doc_count = collection.count()
            
            # Get unique document sources
            if doc_count > 0:
                results = collection.get(include=["metadatas"])
                metadatas = results.get("metadatas", [])
                sources = set(m.get("source", "unknown") for m in metadatas if m)
            else:
                sources = set()
            
            return {
                "success": True,
                "document_count": doc_count,
                "unique_sources": len(sources),
                "sources": list(sources)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "document_count": 0,
                "unique_sources": 0,
                "sources": []
            }

_langchain_service_instance = None

def get_langchain_service() -> LangChainService:
    """Returns a singleton instance of the LangChainService."""
    global _langchain_service_instance
    if _langchain_service_instance is None:
        _langchain_service_instance = LangChainService()
    return _langchain_service_instance

