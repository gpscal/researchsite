"""
LangChain-based service for PDF processing, indexing, and retrieval.
Replaces pdf_processor, pdf_service, and parts of rag_service.
"""

import os
import tempfile
import hashlib
import json
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
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
    # Initialize embeddings with proper cache directory
    _embedder = SentenceTransformerEmbeddings(
        model_name=_model_name,
        model_kwargs={"cache_folder": cache_dir}
    )
    logger.info("Embeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    # Fallback: try without cache folder
    try:
        _embedder = SentenceTransformerEmbeddings(model_name=_model_name)
        logger.info("Embeddings initialized without cache folder")
    except Exception as final_error:
        logger.error(f"All embedding initialization attempts failed: {final_error}")
        raise

# Use LangChain's Anthropic LLM with valid model name
anthropic_model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    logger.error("ANTHROPIC_API_KEY not set in environment")
    logger.error("Please create a .env file with ANTHROPIC_API_KEY=your-key-here")
    _llm = None
else:
    try:
        logger.info(f"Initializing Anthropic LLM with model: {anthropic_model_name}")
        _llm = ChatAnthropic(model=anthropic_model_name, api_key=anthropic_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic LLM: {e}")
        logger.error("Make sure ANTHROPIC_API_KEY is valid in .env file")
        _llm = None


class LangChainService:
    def __init__(self):
        """Initializes the LangChain service, setting up the vector store and retrieval chain."""
        # Ensure the persist directory exists
        Path(_persist_dir).mkdir(parents=True, exist_ok=True)
        
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
        
        # Check if LLM is available
        if _llm is None:
            logger.error("Cannot create LangChain service: Anthropic LLM not initialized")
            logger.error("Please set ANTHROPIC_API_KEY in .env file")
            raise RuntimeError(
                "Anthropic LLM not initialized. "
                "Please create a .env file with ANTHROPIC_API_KEY=your-api-key"
            )
        
        self.retrieval_chain = self._create_retrieval_chain()
        logger.info("LangChain service initialized successfully")

    def _create_retrieval_chain(self):
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
        
        # Create RAG chain with proper error handling
        rag_chain = (
            {
                "context": itemgetter("input") | retriever | format_docs, 
                "input": itemgetter("input")
            }
            | prompt
            | _llm
            | StrOutputParser()
        )
        
        return rag_chain

    def _generate_document_id(self, file_bytes: bytes, filename: str) -> str:
        """Generates a unique ID for a document based on its content and name."""
        content_hash = hashlib.md5(file_bytes).hexdigest()
        name_hash = hashlib.md5(filename.encode()).hexdigest()
        return f"{name_hash[:8]}_{content_hash[:16]}"

    def index_pdf(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Loads a PDF from bytes, splits it into chunks, and indexes it in the vector store."""
        document_id = self._generate_document_id(file_bytes, filename)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(pages)
            
            # Add document_id and original filename to each chunk's metadata
            for split in splits:
                split.metadata["document_id"] = document_id
                split.metadata["source"] = filename

            self.vector_store.add_documents(documents=splits)
            
            return {
                "success": True,
                "filename": filename,
                "document_id": document_id,
                "page_count": len(pages),
                "chunks": len(splits)
            }
        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def query_stream(self, question: str) -> Iterator[str]:
        """Queries the retrieval chain and streams the response."""
        try:
            # Get documents from retriever directly for sources
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
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
            
            # Stream the answer FIRST
            for chunk in self.retrieval_chain.stream({"input": question}):
                yield json.dumps({"content": chunk})
            
            # Send sources AFTER content is complete
            if sources:
                yield json.dumps({"sources": sources})
            
            yield json.dumps({"done": True})
        except Exception as e:
            logger.error(f"Error in query_stream: {e}", exc_info=True)
            yield json.dumps({"error": f"Query failed: {str(e)}"})
            yield json.dumps({"done": True})

    def query(self, question: str) -> Dict[str, Any]:
        """Queries the retrieval chain and returns the final response."""
        try:
            # Get documents from retriever directly for sources
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
            docs = retriever.invoke(question)
            
            # Get the answer
            answer = self.retrieval_chain.invoke({"input": question})
            
            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "unknown"),
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                for doc in docs
            ]
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

_langchain_service_instance = None

def get_langchain_service() -> LangChainService:
    """Returns a singleton instance of the LangChainService."""
    global _langchain_service_instance
    if _langchain_service_instance is None:
        _langchain_service_instance = LangChainService()
    return _langchain_service_instance
