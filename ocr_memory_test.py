"""
Memory usage test script to compare Tesseract OCR and DeepSeek OCR.
This script processes a PDF file with both OCR engines and measures memory usage.
"""

import os
import sys
import time
import tracemalloc
import gc
from pathlib import Path

from pdf_processor import PDFProcessor

# Set the PDF file to process
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
# Use first PDF file in uploads directory if no argument provided
test_file = sys.argv[1] if len(sys.argv) > 1 else sorted(UPLOAD_DIR.glob("*.pdf"))[0]

def test_memory_usage(use_deepseek=False):
    """Test memory usage with either Tesseract or DeepSeek OCR."""
    print(f"\n{'=' * 50}")
    print(f"Testing {'DeepSeek' if use_deepseek else 'Tesseract'} OCR")
    print(f"{'=' * 50}")
    
    # Clear memory before test
    gc.collect()
    
    # Initialize PDF processor with appropriate settings
    processor = PDFProcessor(
        memory_safe=True,
        max_image_size=2000,
        use_deepseek=use_deepseek
    )
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    
    # Process PDF file
    print(f"Processing file: {test_file}")
    with open(test_file, "rb") as f:
        file_bytes = f.read()
        pages = processor.extract_text_from_bytes(file_bytes, os.path.basename(test_file))
    
    # Measure memory usage
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.time()
    tracemalloc.stop()
    
    # Print results
    print(f"Pages processed: {len(pages)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    # Return memory metrics
    return {
        "pages": len(pages),
        "time": end_time - start_time,
        "current_memory_mb": current / 1024 / 1024,
        "peak_memory_mb": peak / 1024 / 1024
    }

def main():
    print(f"Testing memory usage with PDF: {test_file}")
    
    # Test with Tesseract
    tesseract_metrics = test_memory_usage(use_deepseek=False)
    
    # Test with DeepSeek
    deepseek_metrics = test_memory_usage(use_deepseek=True)
    
    # Compare results
    print("\n" + "=" * 50)
    print("Memory Usage Comparison")
    print("=" * 50)
    print(f"{'Metric':<20} {'Tesseract':<15} {'DeepSeek':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Peak memory comparison
    tesseract_peak = tesseract_metrics["peak_memory_mb"]
    deepseek_peak = deepseek_metrics["peak_memory_mb"]
    improvement = ((tesseract_peak - deepseek_peak) / tesseract_peak * 100) if tesseract_peak > 0 else 0
    
    print(f"{'Peak memory (MB)':<20} {tesseract_peak:<15.2f} {deepseek_peak:<15.2f} {improvement:<14.1f}%")
    
    # Processing time comparison
    tesseract_time = tesseract_metrics["time"]
    deepseek_time = deepseek_metrics["time"]
    time_diff = ((tesseract_time - deepseek_time) / tesseract_time * 100) if tesseract_time > 0 else 0
    
    print(f"{'Processing time (s)':<20} {tesseract_time:<15.2f} {deepseek_time:<15.2f} {time_diff:<14.1f}%")
    
    # Output conclusion
    print("\nConclusion:")
    if improvement > 10:
        print(f"DeepSeek OCR uses {improvement:.1f}% less peak memory than Tesseract OCR.")
    else:
        print(f"Memory usage difference is minimal: {improvement:.1f}%")
    
    if deepseek_time < tesseract_time:
        print(f"DeepSeek OCR is {abs(time_diff):.1f}% faster than Tesseract OCR.")
    else:
        print(f"DeepSeek OCR is {abs(time_diff):.1f}% slower than Tesseract OCR (network latency).")

if __name__ == "__main__":
    main()



