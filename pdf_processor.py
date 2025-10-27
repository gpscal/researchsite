"""Enhanced PDF text extraction and chunking for the research assistant."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pdfplumber
import torch
from PIL import Image
from PyPDF2 import PdfReader

# Import OCR capabilities
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Import DeepSeek OCR capability (Hugging Face model wrapper)
try:
    from deepseek_ocr import DeepSeekOCRClient
    DEEPSEEK_AVAILABLE = True
except Exception:
    DEEPSEEK_AVAILABLE = False

# Determine which OCR is available
OCR_AVAILABLE = TESSERACT_AVAILABLE or DEEPSEEK_AVAILABLE

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class PDFChunk:
    text: str
    page_number: int
    metadata: Optional[Dict] = None


@dataclass
class PDFPage:
    text: str
    page_number: int
    width: float
    height: float
    images: List[Dict] = None
    tables: List[Dict] = None
    has_ocr: bool = False


class PDFProcessor:
    """Enhanced PDF processor with OCR capabilities and metadata extraction"""
    
    def __init__(self, ocr_language: str = "eng", use_gpu: bool = True, 
                 memory_safe: bool = True, max_image_size: int = 4000,
                 use_deepseek: bool = True, enable_ocr: bool = True):
        """Initialize the PDF processor
        
        Args:
            ocr_language: Language for OCR (default: English)
            use_gpu: Whether to use GPU for processing if available
            memory_safe: Use memory-efficient processing for systems with limited RAM
            max_image_size: Maximum image dimension to process with OCR (prevents memory issues)
            use_deepseek: Whether to use DeepSeek OCR API (if available) instead of Tesseract
        """
        self.ocr_language = ocr_language
        self.use_gpu = use_gpu and DEVICE == "cuda"
        self.memory_safe = memory_safe
        self.max_image_size = max_image_size
        self.enable_ocr = enable_ocr
        
        # Check system memory and GPU to tune settings for high-VRAM GPUs (e.g., A40 48GB)
        try:
            import psutil
            system_ram_gb = psutil.virtual_memory().total / (1024**3)

            # Default GC behavior
            self.force_gc = self.memory_safe

            # If high VRAM GPU detected, relax memory safety
            try:
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if vram_gb >= 40:  # A40 class
                        self.memory_safe = False
                        self.max_image_size = max(self.max_image_size, 8000)
                        self.force_gc = False
                        print(f"High-VRAM GPU detected ({vram_gb:.1f}GB): using optimized OCR settings")
            except Exception:
                pass

            # On lower RAM hosts, keep conservative settings
            if system_ram_gb < 24 and self.memory_safe:
                print(f"Running in memory-safe mode (System RAM: {system_ram_gb:.1f}GB)")
                if system_ram_gb < 16:
                    self.max_image_size = min(self.max_image_size, 2000)
                    print(f"Reducing max image size to {self.max_image_size}px for memory efficiency")
                self.force_gc = True
        except ImportError:
            print("psutil not available, using default memory settings")
            self.force_gc = self.memory_safe
        
        # Use DeepSeek OCR if available and requested
        self.use_deepseek = self.enable_ocr and use_deepseek and DEEPSEEK_AVAILABLE
        
        if not self.enable_ocr:
            self.ocr_available = False
        else:
            # Check OCR availability
            if self.use_deepseek:
                try:
                    # Initialize DeepSeek OCR client (Hugging Face model)
                    self.deepseek_client = DeepSeekOCRClient()
                    print(f"Using DeepSeek OCR API for text extraction")
                    self.ocr_available = True
                except Exception as e:
                    print(f"Warning: DeepSeek OCR is not properly configured: {e}")
                    self.use_deepseek = False
                    self.ocr_available = TESSERACT_AVAILABLE
            else:
                # Fall back to Tesseract if DeepSeek is not available or not requested
                self.ocr_available = TESSERACT_AVAILABLE
                if self.ocr_available:
                    try:
                        # Test if tesseract is properly installed
                        pytesseract.get_tesseract_version()
                        print(f"Using Tesseract OCR version {pytesseract.get_tesseract_version()}")
                    except Exception as e:
                        print(f"Warning: Tesseract OCR is not properly configured: {e}")
                        self.ocr_available = False
    
    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> List[PDFPage]:
        """Extract text from PDF bytes with enhanced extraction"""
        try:
            pages = self._extract_with_pdfplumber(file_bytes)
            # If pages are empty or have very little text, try OCR
            if not pages or all(len(page.text.strip()) < 100 for page in pages):
                if self.enable_ocr and self.ocr_available:
                    print(f"PDF {filename} has little text, attempting OCR extraction")
                    return self._extract_with_ocr(file_bytes, filename)
            return pages
        except Exception as e:
            print(f"Error with pdfplumber: {e}, falling back to PyPDF2")
            try:
                return self._extract_with_pypdf(file_bytes)
            except Exception as e2:
                print(f"Error with PyPDF2: {e2}, trying OCR as last resort")
                if self.enable_ocr and self.ocr_available:
                    return self._extract_with_ocr(file_bytes, filename)
                else:
                    print("OCR not available, extraction failed")
                    return []

    def _extract_with_pdfplumber(self, file_bytes: bytes) -> List[PDFPage]:
        """Extract text with pdfplumber with enhanced metadata"""
        pages: List[PDFPage] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                
                # Extract tables if available
                tables = []
                try:
                    for table in page.extract_tables():
                        if table:
                            table_text = "\n".join([" | ".join([str(cell or "") for cell in row]) for row in table])
                            tables.append({"content": table_text})
                except Exception as e:
                    print(f"Error extracting tables from page {idx}: {e}")
                
                # Extract images for potential OCR
                images = []
                try:
                    for img in page.images:
                        images.append({
                            "bbox": (img["x0"], img["y0"], img["x1"], img["y1"]),
                            "width": img["width"],
                            "height": img["height"]
                        })
                except Exception as e:
                    print(f"Error extracting images from page {idx}: {e}")
                
                # Combine tables with text if tables exist but text is limited
                if tables and len(text) < 200:
                    table_texts = [t["content"] for t in tables]
                    text = text + "\n\n" + "\n\n".join(table_texts)
                
                # Create PDF page with metadata
                page_obj = PDFPage(
                    text=text,
                    page_number=idx,
                    width=float(page.width),
                    height=float(page.height),
                    images=images,
                    tables=tables
                )
                
                pages.append(page_obj)
                
                # If page has little text but has images, prepare for OCR
                if len(text) < 100 and images and self.enable_ocr and self.ocr_available:
                    page_obj = self._enhance_page_with_ocr(page, page_obj)
        
        return pages

    def _enhance_page_with_ocr(self, plumber_page, page_obj: PDFPage) -> PDFPage:
        """Enhance page with OCR if it has images but little text"""
        if not (self.enable_ocr and self.ocr_available):
            return page_obj
            
        try:
            # If using DeepSeek OCR, we'll use a different approach to reduce memory usage
            if self.use_deepseek:
                return self._enhance_with_deepseek(plumber_page, page_obj)
            
            # Tesseract OCR approach (original implementation)
            # Extract page as image
            img = plumber_page.to_image()
            pil_img = img.original
            
            # Check image dimensions and resize if too large to avoid memory issues
            width, height = pil_img.size
            if max(width, height) > self.max_image_size:
                # Calculate new dimensions while preserving aspect ratio
                scale = self.max_image_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                print(f"Resizing large image from {width}x{height} to {new_width}x{new_height}")
                pil_img = pil_img.resize((new_width, new_height))
            
            # Use memory-efficient processing
            if self.memory_safe:
                # Perform OCR with settings for memory efficiency
                ocr_text = pytesseract.image_to_string(
                    pil_img, 
                    lang=self.ocr_language,
                    config='--psm 1 --oem 1'  # Use Tesseract LSTM engine only
                )
            else:
                # Full quality OCR
                ocr_text = pytesseract.image_to_string(
                    pil_img, 
                    lang=self.ocr_language,
                    config='--psm 1'
                )
            
            # If OCR found substantial text, append it
            if len(ocr_text.strip()) > len(page_obj.text.strip()):
                page_obj.text += "\n\n" + ocr_text.strip()
                page_obj.has_ocr = True
            
            # Force garbage collection to free memory if needed
            if self.force_gc:
                import gc
                del pil_img
                gc.collect()
                
            return page_obj
        except Exception as e:
            print(f"OCR enhancement failed for page {page_obj.page_number}: {e}")
            return page_obj
            
    def _enhance_with_deepseek(self, plumber_page, page_obj: PDFPage) -> PDFPage:
        """Enhance page with DeepSeek OCR to minimize memory usage"""
        try:
            # Extract page as image and run DeepSeek-OCR locally (HF model)
            img = plumber_page.to_image()
            img_bytes = io.BytesIO()
            img.original.convert("RGB").save(img_bytes, format="PNG")
            img_bytes.seek(0)

            result = self.deepseek_client.extract_text_from_image_bytes(img_bytes.getvalue(), prompt="<image>\n<|grounding|>Convert the document to markdown. ")
            
            if result.get("success", False) and result.get("pages"):
                # Get the OCR text from the first page (since we sent a single image)
                page_data = result.get("pages")[0]
                ocr_text = page_data if isinstance(page_data, str) else str(page_data)
                
                # If OCR found substantial text, append it
                if len(ocr_text.strip()) > len(page_obj.text.strip()):
                    page_obj.text += "\n\n" + ocr_text.strip()
                    page_obj.has_ocr = True
            
            # Force garbage collection
            if self.force_gc:
                import gc
                del img
                del img_bytes
                gc.collect()
            
            return page_obj
        except Exception as e:
            print(f"DeepSeek OCR enhancement failed for page {page_obj.page_number}: {e}")
            return page_obj

    def _extract_with_ocr(self, file_bytes: bytes, filename: str) -> List[PDFPage]:
        """Extract text using OCR as a fallback for scanned documents"""
        if not (self.enable_ocr and self.ocr_available):
            return []
            
        # Use DeepSeek OCR if available (much less RAM usage)
        if self.use_deepseek:
            return self._extract_with_deepseek(file_bytes, filename)
            
        # Fall back to Tesseract OCR
        pages = []
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            # Use PyPDF2 to get page count
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
            
            # Process each page with OCR
            for idx in range(page_count):
                try:
                    # Convert PDF page to image using pdf2image (via pytesseract)
                    ocr_data = pytesseract.image_to_data(
                        f"{temp_path}[{idx}]", 
                        lang=self.ocr_language,
                        output_type=Output.DICT,
                        config='--psm 1'
                    )
                    
                    # Extract text from OCR data
                    ocr_text = " ".join([word for word in ocr_data['text'] if word.strip()])
                    
                    # Create PDF page with OCR data
                    page_obj = PDFPage(
                        text=ocr_text,
                        page_number=idx+1,
                        width=0,  # Cannot determine without original
                        height=0,
                        images=[],
                        tables=[],
                        has_ocr=True
                    )
                    
                    pages.append(page_obj)
                except Exception as e:
                    print(f"OCR processing failed for page {idx+1}: {e}")
            
            # Clean up temp file
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
        
        return pages
        
    def _extract_with_deepseek(self, file_bytes: bytes, filename: str) -> List[PDFPage]:
        """Extract text using DeepSeek OCR API for minimal RAM usage"""
        try:
            print(f"Using DeepSeek-OCR (HF) for PDF: {filename}")
            # Use local DeepSeek-OCR model to process the PDF per page
            result = self.deepseek_client.extract_text_from_pdf(file_bytes)
            
            if not result.get("success", False):
                print(f"DeepSeek OCR extraction failed: {result.get('error', 'Unknown error')}")
                return []
            
            # Convert DeepSeek OCR results to our PDFPage format
            pages = []
            for idx, page_data in enumerate(result.get("pages", []), start=1):
                # DeepSeek-OCR returns text strings per page in our wrapper
                text = page_data if isinstance(page_data, str) else str(page_data)
                tables = []
                
                # Create page object
                page_obj = PDFPage(
                    text=text,
                    page_number=idx,
                    width=page_data.get("width", 0) if isinstance(page_data, dict) else 0,
                    height=page_data.get("height", 0) if isinstance(page_data, dict) else 0,
                    tables=tables,
                    images=[],
                    has_ocr=True
                )
                
                pages.append(page_obj)
            
            return pages
            
        except Exception as e:
            print(f"DeepSeek OCR extraction failed: {e}")
            return []

    def _extract_with_pypdf(self, file_bytes: bytes) -> List[PDFPage]:
        """Extract text with PyPDF2 as fallback"""
        reader = PdfReader(io.BytesIO(file_bytes))
        pages: List[PDFPage] = []
        
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            
            # Get page dimensions if available
            width = 0
            height = 0
            try:
                media_box = page.mediabox
                width = float(media_box.width)
                height = float(media_box.height)
            except:
                pass
                
            pages.append(PDFPage(
                text=text,
                page_number=idx,
                width=width,
                height=height,
                images=[],
                tables=[]
            ))
            
        return pages

    def smart_chunk_pdf(self, pages: Iterable[PDFPage], chunk_size: int = 1500, overlap: int = 200) -> List[PDFChunk]:
        """Create semantic chunks from PDF pages with page context preservation"""
        chunks: List[PDFChunk] = []
        
        # First pass: Convert pages to basic chunks
        for page in pages:
            # Store page metadata
            metadata = {
                "page_number": page.page_number,
                "has_ocr": page.has_ocr,
                "page_width": page.width,
                "page_height": page.height,
                "table_count": len(page.tables) if page.tables else 0,
                "image_count": len(page.images) if page.images else 0
            }
            
            # Add page number context at the start
            text = f"Page {page.page_number}:\n" + page.text
            
            # Chunk by sentences/paragraphs to preserve context
            if len(text) <= chunk_size:
                # Page fits in a single chunk
                chunks.append(PDFChunk(text=text, page_number=page.page_number, metadata=metadata))
            else:
                # Split into chunks with overlap
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    
                    # Try to find paragraph or sentence break for clean cutting
                    if end < len(text):
                        # Look for paragraph break
                        para_break = text.rfind('\n\n', start, end)
                        if para_break > start + 200:  # Ensure minimum chunk size
                            end = para_break + 2
                        else:
                            # Look for line break
                            line_break = text.rfind('\n', start, end)
                            if line_break > start + 200:
                                end = line_break + 1
                            else:
                                # Look for sentence break
                                sentence_break = text.rfind('. ', start, end)
                                if sentence_break > start + 200:
                                    end = sentence_break + 2
                    
                    # Ensure we don't exceed text length
                    end = min(end, len(text))
                    
                    chunk_text = text[start:end].strip()
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_start"] = start
                    chunk_metadata["chunk_end"] = end
                    
                    chunks.append(PDFChunk(text=chunk_text, page_number=page.page_number, metadata=chunk_metadata))
                    
                    # Move to next chunk with overlap
                    start = end - overlap
        
        return chunks
        
    def get_page_text(self, pages: List[PDFPage], page_number: int) -> str:
        """Get the complete text of a specific page"""
        for page in pages:
            if page.page_number == page_number:
                return page.text
        return ""
        
    def get_document_metadata(self, file_bytes: bytes) -> Dict:
        """Extract document-level metadata"""
        try:
            from datetime import datetime
            reader = PdfReader(io.BytesIO(file_bytes))
            info = reader.metadata

            metadata = {
                "page_count": len(reader.pages),
                "title": info.title if info and hasattr(info, "title") else None,
                "author": info.author if info and hasattr(info, "author") else None,
                "subject": info.subject if info and hasattr(info, "subject") else None,
                "creator": info.creator if info and hasattr(info, "creator") else None,
            }

            # Handle creation_date - convert datetime to ISO string for JSON serialization
            if info and hasattr(info, "creation_date") and info.creation_date:
                if isinstance(info.creation_date, datetime):
                    metadata["creation_date"] = info.creation_date.isoformat()
                else:
                    metadata["creation_date"] = str(info.creation_date)

            # Handle modification_date if available
            if info and hasattr(info, "modification_date") and info.modification_date:
                if isinstance(info.modification_date, datetime):
                    metadata["modification_date"] = info.modification_date.isoformat()
                else:
                    metadata["modification_date"] = str(info.modification_date)

            return {k: v for k, v in metadata.items() if v is not None}
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}

def process_pdf_for_training(pdf_path: str, questions_per_page: int = 3, chunk_size: int = 1000) -> dict:
    """
    Process PDF for training - extracts text and prepares it for training data generation.
    """
    try:
        processor = PDFProcessor()
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()
            
        # Extract full text with enhanced processor
        pages = processor.extract_text_from_bytes(file_bytes, Path(pdf_path).name)
        
        # Get document metadata
        metadata = processor.get_document_metadata(file_bytes)
        
        # Create chunks for training
        chunks = processor.smart_chunk_pdf(pages, chunk_size=chunk_size)
        
        # Save to training data format
        output_file = f"{Path(pdf_path).stem}_training_data.jsonl"
        
        return {
            "success": True,
            "message": f"PDF {pdf_path} processed with {len(chunks)} chunks",
            "output_file": output_file,
            "pages_processed": len(pages),
            "chunks_generated": len(chunks),
            "metadata": metadata
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error processing PDF: {e}",
            "error": str(e),
            "traceback": traceback.format_exc()
        }