"""
RAG Research LLM Service
Wrapper for integrating the RAG-ResearchLLM functionality into the Flask app
"""

import os
import sys
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings

# Add the RAG-ResearchLLM directory to the path
# RAG_DIR = Path(__file__).parent.parent.parent / "RAG-ResearchLLM" # This is no longer needed as research_llm.py is in the same directory
# sys.path.insert(0, str(RAG_DIR)) # This is no longer needed as research_llm.py is in the same directory

# Import from research_llm module
try:
    from research_llm import (
        load_config,
        LocalEmbedder,
        AnthropicLLM,
        chunk_text,
        DEFAULTS
    )
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: RAG-ResearchLLM not available: {e}")
    RAG_AVAILABLE = False

try:
    from autonomous_web_search import (
        AutonomousWebSearch,
        WebSearchRAG,
        SearchResult,
        WebCrawler,
        WebIndex
    )
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("WARNING: Autonomous web search not available")

try:
    # Add current directory to path for PDF processor import
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from pdf_processor import PDFProcessor, process_pdf_for_training
    PDF_PROCESSING_AVAILABLE = True
    print("INFO: PDF processing available")
except ImportError as e:
    PDF_PROCESSING_AVAILABLE = False
    print(f"WARNING: PDF processing not available: {e}")

# Import search engine service
try:
    from search_engine import get_search_service, SearchEngineService
    SEARCH_ENGINE_AVAILABLE = True
except ImportError:
    SEARCH_ENGINE_AVAILABLE = False
    print("WARNING: Search engine service not available")


# The RAG service is now simplified and will mainly be used for non-PDF related RAG,
# such as web search, and for providing configuration to the new langchain_service.
# PDF-specific ingestion and querying is handled by langchain_service.
class RAGService:
    """Service wrapper for RAG Research LLM functionality"""
    
    # Class-level process tracking
    _active_processes = {}
    
    def __init__(self, config: Optional[Dict] = None, collection_name: str = None):
        """Initialize the RAG service"""
        if not RAG_AVAILABLE:
            raise RuntimeError("RAG-ResearchLLM not available. Check imports.")
        
        # Load configuration
        self.config = config or load_config()
        
        # Set collection name (support multi-collection RAG)
        if collection_name:
            self.config["COLLECTION"] = collection_name
        elif "COLLECTION" not in self.config:
            self.config["COLLECTION"] = "research_collection"
        
        # Set paths relative to Portfolio-Website-main/data
        base_dir = Path(__file__).parent.parent / "data"
        base_dir.mkdir(exist_ok=True)
        
        # RAG-ResearchLLM directory for training data (where scripts save it)
        rag_llm_dir = Path(__file__).parent.parent.parent / "RAG-ResearchLLM"
        
        self.config["BASE_DIR"] = str(base_dir)
        self.config["PERSIST_DIR"] = str(base_dir / "vector_store")
        self.config["WEB_INDEX_PATH"] = str(base_dir / "web_index.db")
        # Use RAG-ResearchLLM training_data directory
        self.config["TRAINING_DATA_DIR"] = str(rag_llm_dir / "training_data")
        self.config["CUSTOM_MODEL_PATH"] = str(base_dir / "custom_model")
        self.config["CONVERSATIONS_DIR"] = str(base_dir / "training_data" / "conversations")
        self.config["VOIP_TRAINING_DIR"] = str(base_dir / "training_data" / "VoIPLLM")
        self.config["VOIP_PCAP_DIR"] = str(base_dir / "voip_pcaps")
        
        # Create directories
        Path(self.config["PERSIST_DIR"]).mkdir(exist_ok=True)
        Path(self.config["TRAINING_DATA_DIR"]).mkdir(exist_ok=True, parents=True)
        Path(self.config["CUSTOM_MODEL_PATH"]).mkdir(exist_ok=True, parents=True)
        self.custom_model_dir = Path(self.config["CUSTOM_MODEL_PATH"]).resolve()
        self.active_custom_model_id: Optional[str] = None
        if any(self.custom_model_dir.glob("checkpoint-*")) or (self.custom_model_dir / "model.safetensors").exists():
            self.active_custom_model_id = "base"
        
        # Initialize components lazily
        self._embedder = None
        self._llm = None
        self._collection = None
        self._web_search = None
        
        print("✓ RAG Service initialized")
    
    def _get_embedder(self):
        """Get or create embedder (local only)"""
        if self._embedder is None:
            # Prefer a stronger model on high-VRAM GPUs
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > (40 * 1024 * 1024 * 1024):
                    model_name = self.config.get("EMBED_MODEL_LOCAL", "sentence-transformers/all-mpnet-base-v2")
                else:
                    model_name = self.config.get("EMBED_MODEL_LOCAL", "all-MiniLM-L6-v2")
            except Exception:
                model_name = self.config.get("EMBED_MODEL_LOCAL", "all-MiniLM-L6-v2")

            self._embedder = LocalEmbedder(model_name)
        return self._embedder
    
    def _get_llm(self, override_provider: Optional[str] = None):
        """Always return Anthropic LLM"""
        if self._llm is not None:
            return self._llm

        model_name = os.getenv("ANTHROPIC_MODEL", DEFAULTS.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"))
        self._llm = AnthropicLLM(model_name=model_name)
        print(f"INFO: Using Anthropic LLM: {model_name}")
        return self._llm

    
    def _get_collection(self):
        """Get or create ChromaDB collection"""
        if self._collection is None:
            client = chromadb.PersistentClient(
                path=self.config["PERSIST_DIR"],
                settings=Settings(anonymized_telemetry=False)
            )
            try:
                self._collection = client.get_collection(self.config["COLLECTION"])
            except:
                self._collection = client.create_collection(self.config["COLLECTION"])
        return self._collection
    
    # The query and query_stream methods below are now primarily for non-PDF RAG,
    # like web search. PDF queries are now handled by langchain_service.
    # I am keeping them for now to avoid breaking other functionalities,
    # but they could be refactored further to separate web RAG from PDF RAG more cleanly.

    def _get_web_search(self):
        """Get or create web search (thread-safe)"""
        # Don't cache web search - create fresh for each request to avoid threading issues
        if WEB_SEARCH_AVAILABLE:
            return AutonomousWebSearch(
                index_path=self.config["WEB_INDEX_PATH"],
                crawler_config={
                    "max_pages": self.config["WEB_CRAWLER_MAX_PAGES"],
                    "max_depth": self.config["WEB_CRAWLER_MAX_DEPTH"],
                    "delay": self.config["WEB_CRAWLER_DELAY"]
                }
            )
        return None
    
    def _search_conversation_history(self, question: str, top_k: int = 3) -> Tuple[List[str], List[dict]]:
        """
        Search through conversation logs for relevant past interactions
        
        Args:
            question: The query to search for
            top_k: Number of conversations to return
            
        Returns:
            Tuple of (context_parts, conversation_sources)
        """
        try:
            conversations_dir = Path(self.config["CONVERSATIONS_DIR"])
            
            if not conversations_dir.exists():
                print(f"WARNING: Conversations directory not found: {conversations_dir}")
                return [], []
            
            # Load all conversations
            all_conversations = []
            for jsonl_file in conversations_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                conv = json.loads(line)
                                all_conversations.append(conv)
                except Exception as e:
                    print(f"WARNING: Could not read {jsonl_file}: {e}")
                    continue
            
            if not all_conversations:
                print("INFO: No conversation history found")
                return [], []
            
            print(f"INFO: Found {len(all_conversations)} conversations to search")
            
            # Get embedder with error handling for CUDA issues
            try:
                embedder = self._get_embedder()
                query_embedding = embedder.embed_query(question)
            except Exception as embed_error:
                print(f"ERROR: Failed to generate query embedding: {embed_error}")
                # Try to clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("INFO: Cleared CUDA cache, retrying embedding...")
                        embedder = self._get_embedder()
                        query_embedding = embedder.embed_query(question)
                    else:
                        raise
                except Exception as retry_error:
                    print(f"ERROR: Retry failed: {retry_error}")
                    print("WARNING: Conversation history search disabled due to embedding errors")
                    return [], []
            
            # Create searchable text from each conversation
            conv_data = []
            for conv in all_conversations:
                messages = conv.get('messages', [])
                metadata = conv.get('metadata', {})
                
                # Extract user and assistant messages
                user_msgs = []
                assistant_msgs = []
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == 'user':
                        user_msgs.append(content)
                    elif role == 'assistant':
                        assistant_msgs.append(content)
                
                # Combine for embedding
                combined_text = " ".join(user_msgs + assistant_msgs)
                
                conv_data.append({
                    'text': combined_text,
                    'user_messages': user_msgs,
                    'assistant_messages': assistant_msgs,
                    'metadata': metadata,
                    'full_conv': conv
                })
            
            # Embed all conversation texts with error handling
            try:
                conv_texts = [c['text'] for c in conv_data]
                conv_embeddings = embedder.embed_documents(conv_texts)
            except Exception as embed_error:
                print(f"ERROR: Failed to embed conversation texts: {embed_error}")
                # Try to clear CUDA cache and retry
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("INFO: Cleared CUDA cache, retrying conversation embeddings...")
                        conv_embeddings = embedder.embed_documents(conv_texts)
                    else:
                        raise
                except Exception as retry_error:
                    print(f"ERROR: Retry failed: {retry_error}")
                    print("WARNING: Conversation history search disabled due to embedding errors")
                    return [], []
            
            # Calculate cosine similarity
            import numpy as np
            similarities = []
            for i, conv_emb in enumerate(conv_embeddings):
                similarity = np.dot(query_embedding, conv_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(conv_emb)
                )
                similarities.append((i, float(similarity), conv_data[i]))
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_conversations = similarities[:top_k]
            
            # Format context parts and sources
            context_parts = []
            sources = []
            
            for idx, (conv_idx, score, conv_info) in enumerate(top_conversations):
                # Only include conversations with reasonable similarity
                if score < 0.3:  # Threshold for relevance
                    continue
                    
                metadata = conv_info['metadata']
                domain = metadata.get('domain', 'unknown')
                timestamp = metadata.get('timestamp', 'unknown')
                model = metadata.get('model', 'unknown')
                
                # Format the conversation for context
                user_q = conv_info['user_messages'][0] if conv_info['user_messages'] else "No question"
                assistant_a = conv_info['assistant_messages'][0] if conv_info['assistant_messages'] else "No answer"
                
                context_part = f"""[ConversationHistory-{idx+1}] Previous Conversation (Similarity: {score:.2f}):
Question: {user_q}
Answer: {assistant_a}
Domain: {domain}, Timestamp: {timestamp}"""
                
                context_parts.append(context_part)
                
                sources.append({
                    "source": f"conversation_history_{domain}",
                    "chunk": conv_idx,
                    "text": f"Q: {user_q}\nA: {assistant_a}",
                    "similarity": score,
                    "domain": domain,
                    "timestamp": timestamp,
                    "model": model
                })
            
            print(f"INFO: Found {len(context_parts)} relevant conversations (threshold: 0.3)")
            return context_parts, sources
            
        except Exception as e:
            print(f"ERROR: Failed to search conversation history: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    # The ingest_* methods below are kept for backwards compatibility or for ingesting non-PDF data.
    # PDF ingestion is now handled by langchain_service.index_pdf.
    def ingest_text(self, text: str, source: str = "uploaded_text") -> Dict:
        """
        Ingest text content into the vector store
        
        Args:
            text: Text content to ingest
            source: Source identifier
            
        Returns:
            Dict with ingestion results
        """
        try:
            embedder = self._get_embedder()
            collection = self._get_collection()
            
            # Chunk the text
            chunks = chunk_text(
                text,
                self.config["CHUNK_SIZE"],
                self.config["CHUNK_OVERLAP"]
            )
            
            if not chunks:
                return {"success": False, "error": "No chunks created from text"}
            
            # Generate embeddings
            embeddings = embedder.embed_documents(chunks)
            
            # Create metadata and IDs
            metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
            ids = [f"{source}_{i}" for i in range(len(chunks))]
            
            # Add to collection
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                "success": True,
                "chunks_added": len(chunks),
                "source": source
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ingest_file(self, file_path: str) -> Dict:
        """
        Ingest a file into the vector store
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict with ingestion results
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"success": False, "error": "File not found"}
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Ingest the content
            result = self.ingest_text(content, str(file_path))
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ingest_image_bytes(self, image_bytes: bytes, filename: str) -> Dict:
        """
        Placeholder for ingesting image bytes.
        A real implementation would process the image (e.g., OCR, image captioning)
        and then ingest the extracted text/features into the vector store.
        """
        print(f"Dummy ingesting image bytes for: {filename}")
        # Simulate image processing and ingestion
        dummy_text = f"Image content from {filename}: This is a dummy representation of the image content."
        
        # Use the existing ingest_text method to add dummy content to vector store
        result = self.ingest_text(dummy_text, source=f"image_upload_{filename}")
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Image {filename} processed and ingested (dummy)",
                "chunks_added": result.get("chunks_added", 0),
                "source": result.get("source")
            }
        else:
            return {"success": False, "error": f"Failed to ingest dummy image content: {result.get('error')}"}

    def query_stream(self, question: str, top_k: int = 3, use_web: bool = False, model_type: Optional[str] = None, use_training_data: bool = True, use_conversation_history: bool = False, messages: List[Dict] = None):
        """
        Query the RAG system with streaming response
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            use_web: Whether to use web search
            model_type: Override LLM provider
            use_training_data: Whether to use training data
            use_conversation_history: Whether to search conversation history (stored past conversations)
            messages: Current chat session messages for context (conversation memory)
            
        Yields:
            Dict with streaming content and metadata
        """
        try:
            # Search conversation history if enabled
            conversation_context = []
            conversation_sources = []
            if use_conversation_history:
                print("INFO: Searching conversation history...")
                conversation_context, conversation_sources = self._search_conversation_history(question, top_k=top_k)
                if conversation_context:
                    print(f"INFO: Found {len(conversation_context)} relevant past conversations")
            
            if use_web:
                # Use enhanced search engine with auto-indexing for web-enhanced RAG
                if SEARCH_ENGINE_AVAILABLE:
                    # Use the new search engine that automatically indexes websites
                    search_service = get_search_service()
                    search_result = search_service.search_with_auto_indexing(
                        query=question,
                        num_results=top_k,
                        auto_index=True,
                        max_index_pages=20,
                        use_google=True  # Enable Google search for auto-indexing
                    )
                    
                    if search_result.get('success'):
                        # Get local training data context
                        embedder = self._get_embedder()
                        llm = self._get_llm(model_type)
                        collection = self._get_collection()
                        
                        local_sources = []
                        local_context = ""
                        
                        if use_training_data:
                            try:
                                query_embedding = embedder.embed_query(question)
                                results = collection.query(
                                    query_embeddings=[query_embedding],
                                    n_results=top_k
                                )
                                
                                if results["documents"][0]:
                                    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                                        local_context += f"[Local-{i+1}] {doc}\n\n"
                                        local_sources.append({
                                            "source": meta.get("source", "unknown"),
                                            "chunk": meta.get("chunk", i),
                                            "text": doc
                                        })
                            except Exception as e:
                                print(f"WARNING: Could not retrieve local training data: {e}")
                        
                        # Prepare enhanced context from search engine
                        combined_context = ""
                        web_results = []
                        
                        if local_context:
                            combined_context += f"Local Training Data:\n{local_context}\n"
                        
                        # Add Google search results
                        google_results = search_result.get('google_results', [])
                        if google_results:
                            combined_context += f"Current Web Search Results:\n\n"
                            for i, result in enumerate(google_results[:top_k]):
                                combined_context += f"[Google-{i+1}] {result['title']}\n"
                                combined_context += f"URL: {result['url']}\n"
                                combined_context += f"Summary: {result['snippet']}\n\n"
                                
                                web_results.append({
                                    "url": result['url'],
                                    "title": result['title'],
                                    "snippet": result['snippet'],
                                    "score": result.get('relevance_score', 0.8),
                                    "content_type": result.get('content_type', 'web'),
                                    "source": "google"
                                })
                        
                        # Add local indexed results
                        local_search_results = search_result.get('local_results', [])
                        if local_search_results:
                            combined_context += f"Previously Indexed Content:\n\n"
                            for i, result in enumerate(local_search_results[:top_k]):
                                combined_context += f"[Indexed-{i+1}] {result['title']}\n"
                                combined_context += f"URL: {result['url']}\n"
                                combined_context += f"Content: {result['snippet']}\n\n"
                                
                                web_results.append({
                                    "url": result['url'],
                                    "title": result['title'],
                                    "snippet": result['snippet'],
                                    "score": result.get('score', 0.7),
                                    "content_type": "indexed",
                                    "source": "local_index"
                                })
                        
                        # Add conversation history context if available
                        if conversation_context:
                            conversation_context_str = "\n\n".join(conversation_context)
                            combined_context += f"Relevant Past Conversations:\n{conversation_context_str}\n\n"
                        
                        # Generate streaming answer using enhanced context
                        system_prompt = """You are a helpful research assistant with access to both current web search results, previously indexed content, and past conversation history. Answer questions based on the provided context from local training data, current web search results, previously indexed websites, and relevant past conversations.
                        
                        When using information from sources:
                        - Cite local sources as [Local-N]
                        - Cite current web search results as [Google-N]
                        - Cite previously indexed content as [Indexed-N]
                        - Cite past conversations as [ConversationHistory-N]
                        - Provide detailed, comprehensive answers
                        - Include relevant details and examples
                        - Be accurate and informative
                        - Prioritize the most recent and relevant information"""
                        
                        user_prompt = f"""Context:
{combined_context}

Question: {question}

Please provide a comprehensive answer using the information from local training data, current web search results, previously indexed content, and past conversations. Cite your sources appropriately."""
                        
                        # Stream the response
                        if hasattr(llm, 'generate_stream'):
                            for chunk in llm.generate_stream(user_prompt, system=system_prompt, messages=messages):
                                yield {"content": chunk}
                        else:
                            # Fallback to non-streaming
                            answer_text = llm.generate(user_prompt, system=system_prompt, messages=messages)
                            for word in answer_text.split():
                                yield {"content": word + " "}
                        
                        # Send sources and indexing info
                        # Combine all sources
                        all_sources = local_sources + conversation_sources
                        
                        yield {
                            "sources": all_sources, 
                            "web_results": web_results,
                            "conversation_sources": conversation_sources,
                            "indexing_info": {
                                "newly_indexed_domains": search_result.get('newly_indexed_domains', []),
                                "total_indexed_domains": search_result.get('total_indexed_domains', 0),
                                "total_indexed_pages": search_result.get('total_indexed_pages', 0)
                            },
                            "conversation_history_count": len(conversation_sources)
                        }
                        yield {"done": True}
                        
                    else:
                        print(f"Enhanced search failed: {search_result.get('error', 'Unknown error')}")
                        # Fall back to regular Google search
                        use_web = False
                else:
                    # Search engine not available, fall back to regular RAG
                    print("Search engine not available, using regular RAG")
                    use_web = False
            
            else:
                # Fallback to regular RAG without web search
                llm = self._get_llm(model_type)

                if use_training_data:
                    embedder = self._get_embedder()
                    collection = self._get_collection()

                    query_embedding = embedder.embed_query(question)
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k
                    )

                    context = ""
                    sources = []
                    if results["documents"][0]:
                        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                            context += f"[{i+1}] {doc}\n\n"
                            sources.append({
                                "source": meta.get("source", "unknown"),
                                "chunk": i,
                                "text": doc
                            })

                    if conversation_context:
                        conversation_context_str = "\n\n".join(conversation_context)
                        context += f"\nRelevant Past Conversations:\n{conversation_context_str}\n\n"

                    system_prompt = "You are a helpful research assistant. Answer questions based on the provided context from training data and past conversations. Cite sources as [N] for training data or [ConversationHistory-N] for past conversations."
                    user_prompt = f"""Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

                    if hasattr(llm, 'generate_stream'):
                        for chunk in llm.generate_stream(user_prompt, system=system_prompt, messages=messages):
                            yield {"content": chunk}
                    else:
                        answer_text = llm.generate(user_prompt, system=system_prompt, messages=messages)
                        for word in answer_text.split():
                            yield {"content": word + " "}

                    all_sources = sources + conversation_sources
                    yield {"sources": all_sources, "conversation_sources": conversation_sources, "conversation_history_count": len(conversation_sources)}
                    yield {"done": True}
                else:
                    if conversation_context:
                        conversation_context_str = "\n\n".join(conversation_context)
                        context = f"Relevant Past Conversations:\n{conversation_context_str}\n\n"
                        system_prompt = "You are a helpful research assistant. Answer questions using your knowledge and relevant past conversations. Cite past conversations as [ConversationHistory-N] when referencing them."
                        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a comprehensive answer."""
                    else:
                        system_prompt = "You are a helpful research assistant. Answer questions directly using your knowledge."
                        user_prompt = question

                    if hasattr(llm, 'generate_stream'):
                        for chunk in llm.generate_stream(user_prompt, system=system_prompt, messages=messages):
                            yield {"content": chunk}
                    else:
                        answer_text = llm.generate(user_prompt, system=system_prompt, messages=messages)
                        for word in answer_text.split():
                            yield {"content": word + " "}

                    yield {"sources": conversation_sources, "conversation_sources": conversation_sources, "conversation_history_count": len(conversation_sources)}
                    yield {"done": True}

        except Exception as e:
            import traceback
            yield {"error": str(e), "traceback": traceback.format_exc()}
    
    def query(self, question: str, top_k: int = 3, use_web: bool = False, model_type: Optional[str] = None, use_training_data: bool = True, use_conversation_history: bool = False, stream: bool = False, messages: List[Dict] = None) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            use_web: Whether to use web search
            use_conversation_history: Whether to search conversation history (stored past conversations)
            messages: Current chat session messages for context (conversation memory)
            
        Returns:
            Dict with answer and sources
        """
        try:
            # Search conversation history if enabled
            conversation_context = []
            conversation_sources = []
            if use_conversation_history:
                print("INFO: Searching conversation history...")
                conversation_context, conversation_sources = self._search_conversation_history(question, top_k=top_k)
                if conversation_context:
                    print(f"INFO: Found {len(conversation_context)} relevant past conversations")
            
            if use_web:
                # Use enhanced search engine with auto-indexing for web-enhanced RAG
                if SEARCH_ENGINE_AVAILABLE:
                    # Use the new search engine that automatically indexes websites
                    search_service = get_search_service()
                    search_result = search_service.search_with_auto_indexing(
                        query=question,
                        num_results=top_k,
                        auto_index=True,
                        max_index_pages=20,
                        use_google=True  # Enable Google search for auto-indexing
                    )

                    if search_result.get('success'):
                        # Get local training data context
                        embedder = self._get_embedder()
                        llm = self._get_llm(model_type)
                        collection = self._get_collection()

                        local_sources = []
                        local_context = ""

                        if use_training_data:
                            try:
                                query_embedding = embedder.embed_query(question)
                                results = collection.query(
                                    query_embeddings=[query_embedding],
                                    n_results=top_k
                                )

                                if results["documents"][0]:
                                    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                                        local_context += f"[Local-{i+1}] {doc}\n\n"
                                        local_sources.append({
                                            "source": meta.get("source", "unknown"),
                                            "chunk": meta.get("chunk", i),
                                            "text": doc
                                        })
                            except Exception as e:
                                print(f"WARNING: Could not retrieve local training data: {e}")

                        # Prepare enhanced context from search engine
                        combined_context = ""
                        web_results = []

                        if local_context:
                            combined_context += f"Local Training Data:\n{local_context}\n"

                        # Add Google search results
                        google_results = search_result.get('google_results', [])
                        if google_results:
                            combined_context += f"Current Web Search Results:\n\n"
                            for i, result in enumerate(google_results[:top_k]):
                                combined_context += f"[Google-{i+1}] {result['title']}\n"
                                combined_context += f"URL: {result['url']}\n"
                                combined_context += f"Summary: {result['snippet']}\n\n"

                                web_results.append({
                                    "url": result['url'],
                                    "title": result['title'],
                                    "snippet": result['snippet'],
                                    "score": result.get('relevance_score', 0.8),
                                    "content_type": result.get('content_type', 'web'),
                                    "source": "google"
                                })

                        # Add local indexed results
                        local_search_results = search_result.get('local_results', [])
                        if local_search_results:
                            combined_context += f"Previously Indexed Content:\n\n"
                            for i, result in enumerate(local_search_results[:top_k]):
                                combined_context += f"[Indexed-{i+1}] {result['title']}\n"
                                combined_context += f"URL: {result['url']}\n"
                                combined_context += f"Content: {result['snippet']}\n\n"

                                web_results.append({
                                    "url": result['url'],
                                    "title": result['title'],
                                    "snippet": result['snippet'],
                                    "score": result.get('score', 0.7),
                                    "content_type": "indexed",
                                    "source": "local_index"
                                })

                        # Add conversation history context if available
                        if conversation_context:
                            conversation_context_str = "\n\n".join(conversation_context)
                            combined_context += f"Relevant Past Conversations:\n{conversation_context_str}\n\n"

                        # Generate answer using enhanced context
                        system_prompt = """You are a helpful research assistant with access to both current web search results, previously indexed content, and past conversation history. Answer questions based on the provided context from local training data, current web search results, previously indexed websites, and relevant past conversations.

                        When using information from sources:
                        - Cite local sources as [Local-N]
                        - Cite current web search results as [Google-N]
                        - Cite previously indexed content as [Indexed-N]
                        - Cite past conversations as [ConversationHistory-N]
                        - Provide detailed, comprehensive answers
                        - Include relevant details and examples
                        - Be accurate and informative
                        - Prioritize the most recent and relevant information"""

                        user_prompt = f"""Context:
{combined_context}

Question: {question}

Please provide a comprehensive answer using the information from local training data, current web search results, previously indexed content, and past conversations. Cite your sources appropriately."""

                        answer_text = llm.generate(user_prompt, system=system_prompt, messages=messages)

                        # Combine all sources
                        all_sources = local_sources + conversation_sources

                        return {
                            "success": True,
                            "answer": answer_text,
                            "sources": all_sources,
                            "local_sources": local_sources,
                            "conversation_sources": conversation_sources,
                            "web_results": web_results,
                            "indexing_info": {
                                "newly_indexed_domains": search_result.get('newly_indexed_domains', []),
                                "total_indexed_domains": search_result.get('total_indexed_domains', 0),
                                "total_indexed_pages": search_result.get('total_indexed_pages', 0)
                            },
                            "conversation_history_count": len(conversation_sources)
                        }
                    else:
                        print(f"Enhanced search failed: {search_result.get('error', 'Unknown error')}")
                        # Fall back to regular RAG
                        use_web = False
                else:
                    # Search engine not available, fall back to regular RAG
                    print("Search engine not available, using regular RAG")
                    use_web = False
            else:
                # Fallback to regular RAG without web search
                llm = self._get_llm(model_type)

                if use_training_data:
                    embedder = self._get_embedder()
                    collection = self._get_collection()

                    query_embedding = embedder.embed_query(question)
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k
                    )

                    context = ""
                    sources = []
                    if results["documents"][0]:
                        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                            context += f"[{i+1}] {doc}\n\n"
                            sources.append({
                                "source": meta.get("source", "unknown"),
                                "chunk": i,
                                "text": doc
                            })

                    if conversation_context:
                        conversation_context_str = "\n\n".join(conversation_context)
                        context += f"\nRelevant Past Conversations:\n{conversation_context_str}\n\n"

                    system_prompt = "You are a helpful research assistant. Answer questions based on the provided context from training data and past conversations. Cite sources as [N] for training data or [ConversationHistory-N] for past conversations."
                    user_prompt = f"""Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

                    answer_text = llm.generate(user_prompt, system=system_prompt, messages=messages)

                    all_sources = sources + conversation_sources
                    return {
                        "success": True,
                        "answer": answer_text,
                        "sources": all_sources,
                        "web_results": [],
                        "conversation_sources": conversation_sources,
                        "conversation_history_count": len(conversation_sources)
                    }
                else:
                    if conversation_context:
                        conversation_context_str = "\n\n".join(conversation_context)
                        context = f"Relevant Past Conversations:\n{conversation_context_str}\n\n"
                        system_prompt = "You are a helpful research assistant. Answer questions using your knowledge and relevant past conversations. Cite past conversations as [ConversationHistory-N] when referencing them."
                        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a comprehensive answer."""
                    else:
                        system_prompt = "You are a helpful research assistant. Answer questions directly using your knowledge."
                        user_prompt = question

                    answer_text = llm.generate(user_prompt, system=system_prompt, messages=messages)

                    return {
                        "success": True,
                        "answer": answer_text,
                        "sources": conversation_sources,
                        "web_results": [],
                        "conversation_sources": conversation_sources,
                        "conversation_history_count": len(conversation_sources)
                    }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    # ------------------------------------------------------------------
    # PCAP packet windows indexing and retrieval (per-file collections)
    # ------------------------------------------------------------------
    def _get_chroma_client(self):
        return chromadb.PersistentClient(
            path=self.config["PERSIST_DIR"],
            settings=Settings(anonymized_telemetry=False)
        )

    def _get_pcap_collection(self, file_id: str):
        client = self._get_chroma_client()
        coll_name = f"pcap_packets_{file_id}"
        try:
            return client.get_collection(coll_name)
        except Exception:
            return client.create_collection(coll_name)

    def index_pcap_packet_windows(self, file_id: str, detailed_packets: list, window_size: int = 20, stride: int = 20) -> Dict:
        """Index per-file packet windows for retrieval without token overflow.

        detailed_packets: list of packet dicts as produced by parse_pcap_for_voip_context
        """
        try:
            if not detailed_packets:
                return {"success": False, "error": "No packets to index"}

            collection = self._get_pcap_collection(file_id)
            embedder = self._get_embedder()

            documents: list = []
            metadatas: list = []
            ids: list = []

            total = len(detailed_packets)
            idx = 0
            while idx < total:
                start = idx
                end = min(idx + window_size, total)
                window = detailed_packets[start:end]
                # Build concise window text
                lines = [f"Packets {window[0]['number']}..{window[-1]['number']}"]
                for pkt in window:
                    line = f"#{pkt['number']} t={pkt['timestamp']:.6f}s len={pkt['length']}"
                    if 'ip' in pkt:
                        ip = pkt['ip']
                        line += f" IP {ip['src']}->{ip['dst']} ttl={ip.get('ttl')}"
                    if 'udp' in pkt:
                        u = pkt['udp']
                        line += f" UDP {u['sport']}->{u['dport']} len={u['length']}"
                    if 'tcp' in pkt:
                        t = pkt['tcp']
                        line += f" TCP {t['sport']}->{t['dport']} flags={t['flags']}"
                    if 'sip' in pkt:
                        line += " SIP"
                    if 'rtp' in pkt:
                        line += " RTP"
                    lines.append(line)
                doc_text = "\n".join(lines)
                documents.append(doc_text)
                metadatas.append({
                    "file_id": file_id,
                    "start_packet": window[0]['number'],
                    "end_packet": window[-1]['number']
                })
                ids.append(f"{file_id}_{window[0]['number']}_{window[-1]['number']}")
                idx += stride

            # Embed and add
            embeddings = embedder.embed_documents(documents)
            collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
            return {"success": True, "indexed_windows": len(documents)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def query_pcap_packet_windows(self, file_id: str, question: str, top_k: int = 5) -> Dict:
        """Retrieve top-K packet windows for a question."""
        try:
            collection = self._get_pcap_collection(file_id)
            embedder = self._get_embedder()
            query_embedding = embedder.embed_query(question)
            results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

            contexts = []
            sources = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                contexts.append(f"[PCAP-{i+1}] {doc}")
                sources.append({
                    "source": f"pcap_packets_{file_id}",
                    "start_packet": meta.get("start_packet"),
                    "end_packet": meta.get("end_packet"),
                    "text": doc
                })
            return {"success": True, "context": "\n\n".join(contexts), "sources": sources}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global service instances (support multiple collections)
_services = {}

def get_rag_service(collection_name: str = "research_collection") -> RAGService:
    """Get or create RAG service instance for a specific collection"""
    global _services
    if collection_name not in _services:
        _services[collection_name] = RAGService(collection_name=collection_name)
    return _services[collection_name]
