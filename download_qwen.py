#!/usr/bin/env python3
"""
Script to download and cache the Qwen2.5-VL-32B-Instruct model.
This will download the model to the local cache for faster loading.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_qwenvl_model():
    """Download and cache the Qwen2.5-VL model."""
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        
        model_name = "Qwen/Qwen2.5-VL-32B-Instruct"
        hf_token = os.getenv("HF_TOKEN")
        
        print(f"INFO: Downloading Qwen2.5-VL model: {model_name}")
        print("INFO: This may take a while depending on your internet connection...")
        
        # Download processor (handles both tokenizer and image processing)
        print("INFO: Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Download model (this will cache it locally)
        print("INFO: Downloading model (this will cache it locally)...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu"  # Load to CPU first to avoid memory issues
        )
        
        print("INFO: Model downloaded and cached successfully!")
        print(f"INFO: Model cache location: ~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
        
        # Clean up memory
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download Qwen2.5-VL model: {e}")
        return False

if __name__ == "__main__":
    print("Qwen2.5-VL-32B-Instruct Model Downloader")
    print("=" * 40)
    
    if not os.getenv("HF_TOKEN"):
        print("ERROR: HF_TOKEN not found in environment variables")
        print("Please set HF_TOKEN in your .env file")
        sys.exit(1)
    
    success = download_qwenvl_model()
    
    if success:
        print("\n✅ Model download completed successfully!")
        print("You can now use Qwen2.5-VL by setting LLM_PROVIDER=qwenvl in your .env file")
    else:
        print("\n❌ Model download failed!")
        sys.exit(1)
