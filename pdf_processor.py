"""
Simple PDF processor module for compatibility.
This is a placeholder module to prevent import errors.
"""

import os
from typing import Dict, Any, List
from pathlib import Path

class PDFProcessor:
    """Simple PDF processor for compatibility."""
    
    def __init__(self, memory_safe: bool = True, max_image_size: int = 2000, 
                 use_deepseek: bool = True, enable_ocr: bool = True):
        self.memory_safe = memory_safe
        self.max_image_size = max_image_size
        self.use_deepseek = use_deepseek
        self.enable_ocr = enable_ocr
        print("INFO: PDF processor initialized (placeholder)")

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file - placeholder implementation."""
        return {
            "success": False,
            "error": "PDF processing not fully implemented - using LangChain service instead"
        }

def process_pdf_for_training(file_path: str) -> Dict[str, Any]:
    """Process PDF for training - placeholder implementation."""
    return {
        "success": False,
        "error": "PDF processing not fully implemented - using LangChain service instead"
    }