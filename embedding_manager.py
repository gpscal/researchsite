"""Enhanced embedding utilities for the research assistant with GPU optimization."""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
_DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Enhanced device detection with GPU model and properties
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_GPU_INFO = {}

if _DEVICE == "cuda":
    try:
        cuda_id = torch.cuda.current_device()
        _GPU_INFO = {
            "name": torch.cuda.get_device_name(cuda_id),
            "total_memory": torch.cuda.get_device_properties(cuda_id).total_memory / (1024**3),  # GB
            "cuda_version": torch.version.cuda,
        }
        logger.info(f"Using GPU: {_GPU_INFO['name']} with {_GPU_INFO['total_memory']:.2f}GB memory")

        # Favor higher matmul precision on capable GPUs
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")

# Batch size optimization based on GPU memory
def get_optimal_batch_size() -> int:
    """Determine optimal batch size based on available GPU memory and system RAM"""
    import psutil
    
    # Check system RAM first - more conservative with limited system memory
    system_ram_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"System RAM: {system_ram_gb:.2f}GB")
    
    # Much more conservative batch sizes when system RAM is limited
    if system_ram_gb < 8:  # Very limited RAM
        logger.warning("Low system RAM detected, using minimal batch size")
        return 4
    elif system_ram_gb < 16:  # Limited RAM (8-16GB)
        logger.warning("Limited system RAM detected, using conservative batch size")
        return 8
    
    # For GPU, adjust based on available VRAM
    if _DEVICE != "cuda" or not _GPU_INFO:
        # CPU mode or unknown GPU - conservative default based on system RAM
        return min(16, max(8, int(system_ram_gb / 2)))
    
    # Estimate based on GPU memory (simple heuristic)
    gpu_memory_gb = _GPU_INFO.get("total_memory", 0)

    # Prefer large batches on A40-class GPUs (>= 40GB VRAM)
    if gpu_memory_gb >= 40:
        return 64
    # High-end GPU
    if gpu_memory_gb > 10:
        return min(32, max(8, int(system_ram_gb / 2)))
    # Mid-range GPU
    if gpu_memory_gb > 6:
        return min(24, max(8, int(system_ram_gb / 2)))
    # Entry GPU
    if gpu_memory_gb > 4:
        return min(16, max(4, int(system_ram_gb / 4)))
    # Low memory
    return min(8, max(4, int(system_ram_gb / 4)))

# Optimal batch size for current device
_OPTIMAL_BATCH_SIZE = get_optimal_batch_size()

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = _DEFAULT_MODEL) -> SentenceTransformer:
    """Load and cache the sentence transformer model with device optimization."""
    try:
        logger.info(f"Loading embedding model '{model_name}' on {_DEVICE}")
        model = SentenceTransformer(model_name, device=_DEVICE)
        
        # Print model size information
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        logger.info(f"Model loaded, size: {model_size_mb:.2f}MB")
        
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        # Fallback to CPU if GPU loading fails
        if _DEVICE == "cuda":
            logger.info("Falling back to CPU for embedding model")
            return SentenceTransformer(model_name, device="cpu")
        raise

def embedding_device() -> str:
    """Return the current embedding device"""
    return _DEVICE

def get_gpu_info() -> Dict:
    """Return information about the GPU if available"""
    return _GPU_INFO

def encode_texts(texts: Iterable[str], batch_size: Optional[int] = None) -> torch.Tensor:
    """Encode multiple texts with optimized batch processing
    
    Args:
        texts: Iterable of texts to encode
        batch_size: Override default batch size (otherwise uses optimal size for device)
        
    Returns:
        Tensor of embeddings
    """
    model = get_embedding_model()
    # Convert to list to support generators used multiple times
    text_list: List[str] = list(texts)
    if not text_list:
        return torch.empty((0, model.get_sentence_embedding_dimension()), device=_DEVICE)
    
    # Use optimal batch size for device if not specified
    actual_batch_size = batch_size or _OPTIMAL_BATCH_SIZE
    
    # Log processing information for large batches
    if len(text_list) > 100:
        logger.info(f"Encoding {len(text_list)} texts with batch size {actual_batch_size}")
    
    try:
        with torch.no_grad():
            embeddings = model.encode(
                text_list,
                batch_size=actual_batch_size,
                convert_to_tensor=True,
                device=_DEVICE,
                show_progress_bar=len(text_list) > 100,  # Show progress for large batches
                normalize_embeddings=True,
            )
        return embeddings
    except RuntimeError as e:
        # Handle out of memory errors gracefully
        if "CUDA out of memory" in str(e) and batch_size is None:
            # Try again with smaller batch size
            reduced_batch = max(1, _OPTIMAL_BATCH_SIZE // 2)
            logger.warning(f"CUDA OOM error, retrying with reduced batch size {reduced_batch}")
            torch.cuda.empty_cache()  # Clear GPU memory
            return encode_texts(text_list, batch_size=reduced_batch)
        else:
            # If explicit batch size was provided or it's not an OOM error, raise it
            raise

def encode_text(text: str) -> torch.Tensor:
    """Encode a single text
    
    Args:
        text: Text string to encode
        
    Returns:
        Tensor embedding for the text
    """
    return encode_texts([text]).squeeze(0)

def batch_encode_texts(texts: List[str], max_batch_size: Optional[int] = None) -> np.ndarray:
    """Encode texts with automatic batching and memory management
    
    Useful for very large document collections that might not fit in memory at once.
    
    Args:
        texts: List of texts to encode
        max_batch_size: Maximum batch size (defaults to optimal size for device)
        
    Returns:
        NumPy array of embeddings
    """
    if not texts:
        return np.array([])
        
    model = get_embedding_model()
    embedding_dim = model.get_sentence_embedding_dimension()
    batch_size = max_batch_size or _OPTIMAL_BATCH_SIZE
    
    # Pre-allocate results array
    all_embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if i % (batch_size * 5) == 0 and i > 0:
            logger.info(f"Processed {i}/{len(texts)} embeddings")
            
        # Generate embeddings for this batch
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch,
                batch_size=batch_size,
                convert_to_tensor=True,
                device=_DEVICE,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        # Move to CPU and store in results array
        all_embeddings[i:i+len(batch)] = batch_embeddings.cpu().numpy()
        
        # Clear GPU memory if using CUDA
        if _DEVICE == "cuda":
            del batch_embeddings
            torch.cuda.empty_cache()
    
    return all_embeddings
