"""
LangChain-based service for PDF processing, indexing, and retrieval.
Replaces pdf_processor, pdf_service, and parts of rag_service.
"""

import os
import tempfile
import hashlib
import json
from typing import Dict, Any, Iterator

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_anthropic import ChatAnthropic

# Re-use config from the existing RAG service for consistency
from rag_service import get_rag_service

_rag_service = get_rag_service()
_persist_dir = _rag_service.config.get("PERSIST_DIR", "data/vector_store")
_collection_name = _rag_service.config.get("COLLECTION", "research_collection")

# Use a LangChain-compatible embedder with the same model
_model_name = _rag_service.config.get("EMBED_MODEL_LOCAL", "all-MiniLM-L6-v2")
_embedder = SentenceTransformerEmbeddings(model_name=_model_name)

# Use LangChain's Anthropic LLM
anthropic_model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
_llm = ChatAnthropic(model=anthropic_model_name)


class LangChainService:
    def __init__(self):
        """Initializes the LangChain service, setting up the vector store and retrieval chain."""
        self.vector_store = Chroma(
            collection_name=_collection_name,
            embedding_function=_embedder,
            persist_directory=_persist_dir,
        )
        self.retrieval_chain = self._create_retrieval_chain()

    def _create_retrieval_chain(self):
        """Creates the LangChain retrieval chain for question-answering."""
        retriever = self.vector_store.as_retriever()
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(_llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        return chain

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
        context_sent = False
        for chunk in self.retrieval_chain.stream({"input": question}):
            if "answer" in chunk and chunk["answer"]:
                yield json.dumps({"content": chunk["answer"]})
            
            if "context" in chunk and not context_sent:
                sources = [
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", "unknown"),
                        "text": doc.page_content,
                    }
                    for doc in chunk["context"]
                ]
                if sources:
                    yield json.dumps({"sources": sources})
                    context_sent = True
        
        yield json.dumps({"done": True})

    def query(self, question: str) -> Dict[str, Any]:
        """Queries the retrieval chain and returns the final response."""
        result = self.retrieval_chain.invoke({"input": question})
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "text": doc.page_content,
            }
            for doc in result.get("context", [])
        ]
        return {
            "success": True,
            "answer": result.get("answer", ""),
            "sources": sources
        }

_langchain_service_instance = None

def get_langchain_service() -> LangChainService:
    """Returns a singleton instance of the LangChainService."""
    global _langchain_service_instance
    if _langchain_service_instance is None:
        _langchain_service_instance = LangChainService()
    return _langchain_service_instance
