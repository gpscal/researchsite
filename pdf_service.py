"""
PDF Service for managing PDF documents and providing advanced querying capabilities
"""

from __future__ import annotations

import os
import json
import hashlib
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from chromadb.config import Settings

from pdf_processor import PDFProcessor, PDFPage, PDFChunk
from embedding_manager import batch_encode_texts, encode_text, get_gpu_info

# Constants
UPLOAD_DIR = Path("data/uploads")
PDF_DB_PATH = Path("data/pdf_index.sqlite")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PDFService:
    """
    Service for managing PDF documents with enhanced retrieval capabilities
    """
    
    def __init__(self, enable_ocr: bool = True):
        """Initialize the PDF service"""
        # Initialize with memory-safe settings
        self.enable_ocr = enable_ocr
        self.pdf_processor = PDFProcessor(
            use_gpu=(DEVICE == "cuda"),
            memory_safe=True,
            max_image_size=2000,  # Limit image size for OCR to prevent memory issues
            enable_ocr=self.enable_ocr
        )
        self.upload_dir = UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database for PDF metadata and page tracking
        self._init_db()
        
    def _init_db(self):
        """Initialize the SQLite database for PDF tracking"""
        conn = sqlite3.connect(PDF_DB_PATH)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT,
            author TEXT,
            page_count INTEGER NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_path TEXT NOT NULL,
            metadata TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_pages (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            text TEXT,
            has_ocr BOOLEAN,
            width REAL,
            height REAL,
            FOREIGN KEY (document_id) REFERENCES pdf_documents(id),
            UNIQUE(document_id, page_number)
        )
        ''')
        
        # Create indices for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pdf_pages_doc_id ON pdf_pages(document_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pdf_pages_page_num ON pdf_pages(page_number)')
        
        conn.commit()
        conn.close()
    
    def process_and_store_pdf(self, file_bytes: bytes, filename: str, batch_size: int = 16) -> Dict:
        """
        Process a PDF file and store its contents and metadata
        
        Args:
            file_bytes: The PDF file bytes
            filename: The original filename
            
        Returns:
            Dict with processing results
        """
        try:
            # Generate a unique ID for this document
            pdf_id = self._generate_document_id(file_bytes, filename)
            
            # Save the file to disk
            file_path = self.upload_dir / f"{pdf_id}.pdf"
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            # Extract document metadata
            doc_metadata = self.pdf_processor.get_document_metadata(file_bytes)
            
            # Extract text from all pages
            pages = self.pdf_processor.extract_text_from_bytes(file_bytes, filename)
            
            # Store document record
            self._store_document_record(pdf_id, filename, file_path, doc_metadata, len(pages))
            
            # Store individual pages
            self._store_page_records(pdf_id, pages)
            
            # Generate chunks for vector storage
            # Check file size and adjust processing accordingly
            file_size_mb = len(file_bytes) / (1024 * 1024)
            
            # For very large PDFs, use more aggressive memory optimization
            if file_size_mb > 20:  # PDFs larger than 20MB
                print(f"Large PDF detected ({file_size_mb:.1f}MB), using memory optimization")
                # Process in smaller chunks with garbage collection between pages
                import gc
                processed_pages = []
                for i, page in enumerate(pages):
                    # Process one page at a time
                    if i > 0 and i % 5 == 0:  # GC every 5 pages
                        gc.collect()
                    processed_pages.append(page)
                
                chunks = self.pdf_processor.smart_chunk_pdf(processed_pages)
                # Final garbage collection after processing
                gc.collect()
            else:
                chunks = self.pdf_processor.smart_chunk_pdf(pages)
            
            return {
                "success": True,
                "document_id": pdf_id,
                "filename": filename,
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "chunks": chunks,
                "metadata": doc_metadata
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _generate_document_id(self, file_bytes: bytes, filename: str) -> str:
        """Generate a unique ID for a document"""
        # Create a hash based on file content and name
        content_hash = hashlib.md5(file_bytes).hexdigest()
        name_hash = hashlib.md5(filename.encode()).hexdigest()
        return f"{name_hash[:8]}_{content_hash[:16]}"
    
    def _store_document_record(self, doc_id: str, filename: str, file_path: Path, 
                              metadata: Dict, page_count: int):
        """Store document record in database"""
        conn = sqlite3.connect(PDF_DB_PATH)
        cursor = conn.cursor()
        
        # Extract title and author from metadata if available
        title = metadata.get('title', filename)
        author = metadata.get('author', 'Unknown')
        
        cursor.execute(
            '''INSERT OR REPLACE INTO pdf_documents 
               (id, filename, title, author, page_count, file_path, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (doc_id, filename, title, author, page_count, str(file_path), json.dumps(metadata))
        )
        
        conn.commit()
        conn.close()
    
    def _store_page_records(self, doc_id: str, pages: List[PDFPage]):
        """Store page records in database"""
        conn = sqlite3.connect(PDF_DB_PATH)
        cursor = conn.cursor()
        
        for page in pages:
            page_id = f"{doc_id}_p{page.page_number}"
            cursor.execute(
                '''INSERT OR REPLACE INTO pdf_pages
                   (id, document_id, page_number, text, has_ocr, width, height)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (page_id, doc_id, page.page_number, page.text, 
                 page.has_ocr, page.width, page.height)
            )
        
        conn.commit()
        conn.close()
    
    def get_page_text(self, document_id: str, page_number: int) -> Dict:
        """
        Retrieve the full text of a specific page
        
        Args:
            document_id: The document ID
            page_number: The page number to retrieve
            
        Returns:
            Dict with page text and metadata
        """
        conn = sqlite3.connect(PDF_DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT p.*, d.filename, d.title
               FROM pdf_pages p
               JOIN pdf_documents d ON p.document_id = d.id
               WHERE p.document_id = ? AND p.page_number = ?''',
            (document_id, page_number)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {
                "success": False,
                "error": f"Page {page_number} not found for document {document_id}"
            }
        
        return {
            "success": True,
            "document_id": document_id,
            "page_number": page_number,
            "text": row["text"],
            "filename": row["filename"],
            "title": row["title"],
            "has_ocr": bool(row["has_ocr"]),
            "width": row["width"],
            "height": row["height"]
        }
    
    def get_document_info(self, document_id: str) -> Dict:
        """
        Get information about a document
        
        Args:
            document_id: The document ID
            
        Returns:
            Dict with document information
        """
        conn = sqlite3.connect(PDF_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM pdf_documents WHERE id = ?",
            (document_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {
                "success": False,
                "error": f"Document {document_id} not found"
            }
        
        return {
            "success": True,
            "document_id": row["id"],
            "filename": row["filename"],
            "title": row["title"],
            "author": row["author"],
            "page_count": row["page_count"],
            "upload_time": row["upload_time"],
            "metadata": json.loads(row["metadata"] or "{}")
        }
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> Dict:
        """
        List all documents with pagination
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            Dict with list of documents
        """
        conn = sqlite3.connect(PDF_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT id, filename, title, author, page_count, upload_time
               FROM pdf_documents
               ORDER BY upload_time DESC
               LIMIT ? OFFSET ?''',
            (limit, offset)
        )
        
        rows = cursor.fetchall()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as count FROM pdf_documents")
        total = cursor.fetchone()["count"]
        
        conn.close()
        
        documents = []
        for row in rows:
            documents.append({
                "document_id": row["id"],
                "filename": row["filename"],
                "title": row["title"],
                "author": row["author"],
                "page_count": row["page_count"],
                "upload_time": row["upload_time"]
            })
        
        return {
            "success": True,
            "total": total,
            "documents": documents
        }
    
    def query_document_pages(self, document_id: str, query: str, top_k: int = 3) -> Dict:
        """
        Query a specific document to find the most relevant pages
        
        Args:
            document_id: The document ID
            query: The query text
            top_k: Maximum number of results to return
            
        Returns:
            Dict with relevant pages and scores
        """
        conn = sqlite3.connect(PDF_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get document info
        cursor.execute(
            "SELECT filename, title FROM pdf_documents WHERE id = ?",
            (document_id,)
        )
        doc_row = cursor.fetchone()
        
        if not doc_row:
            conn.close()
            return {
                "success": False,
                "error": f"Document {document_id} not found"
            }
        
        # Get all pages for this document
        cursor.execute(
            "SELECT id, page_number, text FROM pdf_pages WHERE document_id = ?",
            (document_id,)
        )
        
        page_rows = cursor.fetchall()
        conn.close()
        
        if not page_rows:
            return {
                "success": False,
                "error": f"No pages found for document {document_id}"
            }
        
        # Create embeddings for pages and query
        pages = [(row["id"], row["page_number"], row["text"]) for row in page_rows]
        page_texts = [p[2] for p in pages]
        
        try:
            # Encode query and pages
            query_embedding = encode_text(query).cpu().numpy()
            page_embeddings = batch_encode_texts(page_texts).astype(np.float32)
            
            # Calculate cosine similarity
            similarities = np.dot(page_embeddings, query_embedding) / (
                np.linalg.norm(page_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                page_id, page_number, text = pages[idx]
                score = float(similarities[idx])
                
                # Create a preview with context around query terms
                preview = self._create_context_preview(text, query)
                
                results.append({
                    "page_id": page_id,
                    "page_number": page_number,
                    "score": score,
                    "preview": preview,
                    "document_id": document_id
                })
            
            return {
                "success": True,
                "document_id": document_id,
                "filename": doc_row["filename"],
                "title": doc_row["title"],
                "results": results
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _create_context_preview(self, text: str, query: str, max_length: int = 300) -> str:
        """Create a preview with context around query terms"""
        # Simple implementation: find the first occurrence of query terms
        text = text.lower()
        query_terms = query.lower().split()
        
        # Find best match position
        best_pos = 0
        best_score = 0
        
        for i in range(len(text) - 20):
            window = text[i:i+200]
            score = sum(1 for term in query_terms if term in window)
            if score > best_score:
                best_score = score
                best_pos = i
        
        # Extract context
        start = max(0, best_pos - 50)
        end = min(len(text), best_pos + max_length - 50)
        
        # Truncate to word boundaries
        if start > 0:
            while start < len(text) and text[start] != ' ':
                start += 1
        
        if end < len(text):
            while end > 0 and text[end] != ' ':
                end -= 1
        
        preview = text[start:end].strip()
        
        # Add ellipsis if truncated
        if start > 0:
            preview = "..." + preview
        if end < len(text):
            preview = preview + "..."
        
        return preview


# Global singleton instance
_pdf_service = None

def get_pdf_service(enable_ocr: bool = True) -> PDFService:
    """Get the PDF service singleton"""
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService(enable_ocr=enable_ocr)
    return _pdf_service
