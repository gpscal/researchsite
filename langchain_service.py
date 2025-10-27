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

# LangChain-compatible wrapper for WizardLM
class WizardLMWrapper:
    """Wrapper to make WizardLM compatible with LangChain interface."""
    
    def __init__(self, model_name: str):
        from research_llm import WizardLMLLM
        self.model = WizardLMLLM(model_name=model_name)
        logger.info(f"WizardLM wrapper initialized with model: {model_name}")
    
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
        response = self.model.generate(prompt, max_tokens=1024)
        # Return as a string (LangChain expects strings from LLMs)
        return str(response) if response else ""
    
    def stream(self, messages):
        """LangChain-compatible stream method."""
        prompt = self._extract_prompt(messages)
        for chunk in self.model.generate_stream(prompt, max_tokens=1024):
            # Yield strings directly (LangChain expects string chunks)
            yield str(chunk) if chunk else ""
    
    def __call__(self, messages):
        """Make the wrapper callable."""
        return self.invoke(messages)

# Initialize LLM providers
_anthropic_llm = None
_wizardlm_llm = None

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

# Initialize WizardLM (lazy load on first use to save memory)
def get_wizardlm():
    """Lazy load WizardLM model."""
    global _wizardlm_llm
    if _wizardlm_llm is None:
        try:
            wizardlm_model_name = os.getenv("WIZARDLM_MODEL", "QuixiAI/WizardLM-13B-Uncensored")
            logger.info(f"Initializing WizardLM with model: {wizardlm_model_name}")
            _wizardlm_llm = WizardLMWrapper(wizardlm_model_name)
            logger.info("WizardLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WizardLM: {e}")
            raise RuntimeError(f"WizardLM initialization failed: {e}")
    return _wizardlm_llm


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
            logger.warning("Anthropic LLM not initialized. WizardLM will be used if requested.")
        
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
        
        # Wrap WizardLM with RunnableLambda if it's not already a LangChain model
        if isinstance(llm, WizardLMWrapper):
            # Wrap the WizardLM wrapper with RunnableLambda to make it chain-compatible
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
            if provider == 'wizardlm':
                llm = get_wizardlm()
                self.retrieval_chains['wizardlm'] = self._create_retrieval_chain(llm)
            elif provider == 'anthropic':
                if _anthropic_llm is None:
                    raise RuntimeError("Anthropic LLM not initialized")
                self.retrieval_chains['anthropic'] = self._create_retrieval_chain(_anthropic_llm)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        return self.retrieval_chains[provider]

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

    def query_stream(self, question: str, provider: str = 'anthropic') -> Iterator[str]:
        """Queries the retrieval chain and streams the response."""
        try:
            # Check if the collection has any documents
            collection = self.vector_store._collection
            doc_count = collection.count()
            
            if doc_count == 0:
                # No documents indexed yet
                yield json.dumps({"content": "No documents have been indexed yet. Please upload PDF documents first to enable question-answering."})
                yield json.dumps({"done": True})
                return
            
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
            
            # Handle streaming differently for WizardLM vs Anthropic
            if provider == 'wizardlm':
                llm = get_wizardlm()
                # Stream directly from WizardLM
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

    def query(self, question: str, provider: str = 'anthropic') -> Dict[str, Any]:
        """Queries the retrieval chain and returns the final response."""
        try:
            # Check if the collection has any documents
            collection = self.vector_store._collection
            doc_count = collection.count()
            
            if doc_count == 0:
                # No documents indexed yet
                return {
                    "success": True,
                    "answer": "No documents have been indexed yet. Please upload PDF documents first to enable question-answering.",
                    "sources": []
                }
            
            # Get documents from retriever directly for sources
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
            docs = retriever.invoke(question)
            
            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "unknown"),
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                for doc in docs
            ]
            
            # Handle WizardLM differently from Anthropic
            if provider == 'wizardlm':
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
                
                llm = get_wizardlm()
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
