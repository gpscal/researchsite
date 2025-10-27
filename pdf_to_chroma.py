#!/usr/bin/env python3
"""
Standalone PDF → Chroma Vector Store CLI

Extract text from PDFs and ingest directly into Chroma vector store.
No service wrappers - just pure PDF processor → embeddings → Chroma.

Usage:
    python3 pdf_to_chroma.py --pdf /path/to/file.pdf [options]
    python3 pdf_to_chroma.py --dir /path/to/folder [options]
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import required modules
from pdf_processor import PDFProcessor, PDFChunk
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_enable_ocr() -> bool:
    """Get OCR setting from environment"""
    return os.getenv("ENABLE_PDF_OCR", "true").lower() not in {"0", "false", "no"}

def generate_document_id(file_bytes: bytes, filename: str) -> str:
    """Generate unique document ID"""
    content_hash = hashlib.md5(file_bytes).hexdigest()
    name_hash = hashlib.md5(filename.encode()).hexdigest()
    return f"{name_hash[:8]}_{content_hash[:16]}"

def process_pdf_file(pdf_path: Path, processor: PDFProcessor) -> tuple[List[PDFChunk], str]:
    """Process a single PDF file"""
    print(f"Processing: {pdf_path.name}")

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    doc_id = generate_document_id(file_bytes, pdf_path.name)

    # Extract pages
    pages = processor.extract_text_from_bytes(file_bytes, pdf_path.name)
    print(f"  Extracted {len(pages)} pages")

    # Create chunks
    chunks = processor.smart_chunk_pdf(pages)
    print(f"  Created {len(chunks)} chunks")

    return chunks, doc_id

def ingest_to_chroma(chunks: List[PDFChunk], doc_id: str, filename: str,
                    collection: chromadb.Collection, embedder: SentenceTransformer,
                    batch_size: int = 16) -> None:
    """Ingest chunks into Chroma vector store"""
    if not chunks:
        return

    print(f"Ingesting {len(chunks)} chunks to Chroma...")

    # Prepare data
    texts = [chunk.text for chunk in chunks]
    metadatas = [{
        "source": filename,
        "page": chunk.page_number,
        "document_id": doc_id,
        "chunk_index": i
    } for i, chunk in enumerate(chunks)]

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedder.encode(batch_texts, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
        print(f"  Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Add to Chroma
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

    print(f"✓ Ingested {len(chunks)} chunks for document {doc_id}")

def find_pdf_files(directory: Path) -> List[Path]:
    """Find all PDF files in directory recursively"""
    return list(directory.rglob("*.pdf"))

def main():
    parser = argparse.ArgumentParser(description="Extract PDF text and ingest to Chroma vector store")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf", type=str, help="Path to single PDF file")
    input_group.add_argument("--dir", type=str, help="Directory to scan for PDFs recursively")

    # Processing options
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size (default: 1500)")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap (default: 200)")

    # Vector store options
    parser.add_argument("--persist-dir", type=str, default="data/vector_store",
                       help="Chroma persist directory (default: data/vector_store)")
    parser.add_argument("--collection", type=str, default="research_collection",
                       help="Chroma collection name (default: research_collection)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence transformer model (default: all-MiniLM-L6-v2)")

    # OCR options
    parser.add_argument("--no-ocr", action="store_true",
                       help="Disable OCR (overrides ENABLE_PDF_OCR env)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Embedding batch size (default: 16)")

    args = parser.parse_args()

    # Determine OCR setting
    enable_ocr = get_enable_ocr() and not args.no_ocr
    print(f"OCR enabled: {enable_ocr}")

    # Initialize PDF processor
    processor = PDFProcessor(
        enable_ocr=enable_ocr,
        memory_safe=True,
        max_image_size=2000
    )

    # Initialize embedder
    print(f"Loading model: {args.model}")
    embedder = SentenceTransformer(args.model)

    # Initialize Chroma
    print(f"Connecting to Chroma at: {args.persist_dir}")
    client = chromadb.PersistentClient(
        path=args.persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        collection = client.get_collection(args.collection)
        print(f"Using existing collection: {args.collection}")
    except:
        collection = client.create_collection(args.collection)
        print(f"Created new collection: {args.collection}")

    # Process input
    if args.pdf:
        # Single PDF
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            sys.exit(1)

        chunks, doc_id = process_pdf_file(pdf_path, processor)
        ingest_to_chroma(chunks, doc_id, pdf_path.name, collection, embedder, args.batch_size)

    elif args.dir:
        # Directory of PDFs
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)

        pdf_files = find_pdf_files(dir_path)
        if not pdf_files:
            print(f"No PDF files found in: {dir_path}")
            sys.exit(1)

        print(f"Found {len(pdf_files)} PDF files")

        for pdf_path in pdf_files:
            try:
                chunks, doc_id = process_pdf_file(pdf_path, processor)
                ingest_to_chroma(chunks, doc_id, pdf_path.name, collection, embedder, args.batch_size)
                print(f"✓ Completed: {pdf_path.name}\n")
            except Exception as e:
                print(f"✗ Failed: {pdf_path.name} - {e}\n")
                continue

    print("Done!")

if __name__ == "__main__":
    main()

