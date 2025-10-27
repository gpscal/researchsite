#!/usr/bin/env python3
"""
RAG Research Assistant with Anthropic Claude support
"""

import os
import sys
import json
from typing import List, Tuple, Optional
from pathlib import Path

# Set up HuggingFace token for model downloads
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    print("INFO: HuggingFace token configured for model downloads")
else:
    print("WARNING: HF_TOKEN not found in environment - some models may fail to download")

# Core dependencies
import chromadb
from chromadb.config import Settings

# Autonomous Web Search
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
    print("WARNING: autonomous_web_search not installed. Web search features disabled.")

# Check for required dependencies
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("WARNING: anthropic package not installed. Install with: pip install anthropic")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: transformers not installed. Install with: pip install transformers torch")

# Configuration defaults
DEFAULTS = {
    "PERSIST_DIR": "vector_store",
    "COLLECTION": "research_docs_v2",
    "EMBED_PROVIDER": "local",
    "EMBED_MODEL_LOCAL": "all-MiniLM-L6-v2",
    "LLM_PROVIDER": "anthropic",
    "ANTHROPIC_MODEL": "claude-3-haiku-20240307",
    "QWENVL_MODEL": "Qwen/Qwen2.5-VL-32B-Instruct",
    "CHUNK_SIZE": 1200,
    "CHUNK_OVERLAP": 200,

    # Autonomous Web Search Configuration
    "WEB_INDEX_PATH": "./web_index.db",
    "WEB_CRAWLER_MAX_PAGES": 100,
    "WEB_CRAWLER_MAX_DEPTH": 3,
    "WEB_CRAWLER_DELAY": 1.0,
}

def load_config():
    """Load configuration from environment or defaults."""
    cfg = {}
    for key, default in DEFAULTS.items():
        cfg[key] = os.getenv(key, default)
    
    # Convert numeric values
    try:
        cfg["CHUNK_SIZE"] = int(cfg["CHUNK_SIZE"])
        cfg["CHUNK_OVERLAP"] = int(cfg["CHUNK_OVERLAP"])
        cfg["WEB_CRAWLER_MAX_PAGES"] = int(cfg["WEB_CRAWLER_MAX_PAGES"])
        cfg["WEB_CRAWLER_MAX_DEPTH"] = int(cfg["WEB_CRAWLER_MAX_DEPTH"])
        cfg["WEB_CRAWLER_DELAY"] = float(cfg["WEB_CRAWLER_DELAY"])
    except ValueError as e:
        print(f"ERROR: Invalid numeric config: {e}")
        sys.exit(1)
    
    return cfg


# ============================================================================
# Embedding Models
# ============================================================================

class LocalEmbedder:
    """Local embedding using sentence-transformers (GPU-accelerated)."""
    
    def __init__(self, model_name: str):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")
        
        try:
            # Try GPU first with HF token if available
            if hf_token:
                self.model = SentenceTransformer(model_name, token=hf_token)
            else:
                self.model = SentenceTransformer(model_name)
            print(f"INFO: Using local embeddings: {model_name}")
            print(f"INFO: Use pytorch device_name: {self.model.device}")
        except Exception as e:
            if "CUDA" in str(e):
                print(f"WARNING: CUDA error with embeddings, falling back to CPU: {e}")
                if hf_token:
                    self.model = SentenceTransformer(model_name, device='cpu', token=hf_token)
                else:
                    self.model = SentenceTransformer(model_name, device='cpu')
                print(f"INFO: Using local embeddings on CPU: {model_name}")
            else:
                raise e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# ============================================================================
# LLM Models
# ============================================================================

class AnthropicLLM:
    """Anthropic Claude LLM interface."""

    def __init__(self, model_name: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed")

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Prioritize passed model_name, then env var, then default
        final_model_name = model_name or os.getenv("ANTHROPIC_MODEL") or DEFAULTS.get("ANTHROPIC_MODEL")
        
        if not final_model_name:
             raise RuntimeError("Anthropic model not specified via argument, ANTHROPIC_MODEL env var, or DEFAULTS")

        self.model = final_model_name
        print(f"INFO: Using Anthropic LLM: {self.model}")

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 8096, messages: Optional[List] = None) -> str:
        """Generate response from Anthropic Claude"""
        # Use provided messages (conversation history) or build new message list
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            # Messages already provided - filter out system messages (Anthropic uses separate param)
            msg_list = [msg for msg in messages if msg.get("role") != "system"]
            msg_list.append({"role": "user", "content": prompt})
            messages = msg_list

        # Anthropic requires system message as a separate parameter
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        if system:
            kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)
        return resp.content[0].text

    def generate_stream(self, prompt: str, system: Optional[str] = None, max_tokens: int = 8096, messages: Optional[List] = None):
        """Generate streaming response from Anthropic Claude"""
        # Use provided messages (conversation history) or build new message list
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            # Messages already provided - filter out system messages (Anthropic uses separate param)
            msg_list = [msg for msg in messages if msg.get("role") != "system"]
            msg_list.append({"role": "user", "content": prompt})
            messages = msg_list

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
            # Note: 'stream' parameter is NOT needed - .messages.stream() is already a streaming method
        }

        if system:
            kwargs["system"] = system

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            print(f"ERROR: Anthropic streaming failed: {e}")
            yield f"Error: {str(e)}"


class QwenVLLL:
    """Qwen2.5-VL-32B-Instruct vision-language model interface using transformers."""

    def __init__(self, model_name: Optional[str] = None, device: str = "cuda"):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers package not installed")

        # Get model name from config or environment
        final_model_name = model_name or os.getenv("QWENVL_MODEL") or DEFAULTS.get("QWENVL_MODEL")
        if not final_model_name:
            raise RuntimeError("QwenVL model not specified")

        self.model_name = final_model_name
        self.device = device
        self.model = None
        self.processor = None
        
        print(f"INFO: Initializing Qwen2.5-VL model: {self.model_name}")
        self._load_model()

    def _load_model(self):
        """Load the QwenVL model and processor."""
        try:
            print("INFO: Loading processor and model (this may take a while)...")
            
            # Load processor (handles both text and images)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=True
            )
            
            # Load model with optimized settings for A100
            print("INFO: Using bf16 precision for better quality on A100")
            
            if not torch.cuda.is_available():
                print("ERROR: CUDA is not available! Model will be very slow on CPU.")
                raise RuntimeError("CUDA not available - GPU is required for this model")
            
            print(f"INFO: Loading model on GPU with device_map='auto'")
            print(f"INFO: CUDA device count: {torch.cuda.device_count()}")
            print(f"INFO: Current CUDA device: {torch.cuda.current_device()}")
            
            # Try to use flash attention if available, otherwise fall back to eager
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print("INFO: Using flash_attention_2 for faster inference")
            except ImportError:
                attn_impl = "eager"
                print("INFO: flash_attn not installed, using eager attention")
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "75GB"},  # Use up to 75GB on GPU 0
                attn_implementation=attn_impl
            )
            
            # Verify model is on GPU
            first_param_device = next(self.model.parameters()).device
            print(f"INFO: Model first parameter device: {first_param_device}")
            
            # Print device info
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"INFO: QwenVL model loaded successfully")
            print(f"INFO: GPU memory allocated: {memory_allocated:.2f} GB")
            print(f"INFO: GPU memory reserved: {memory_reserved:.2f} GB")
            
        except Exception as e:
            print(f"ERROR: Failed to load QwenVL model: {e}")
            raise

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 256, messages: Optional[List] = None) -> str:
        """Generate response from QwenVL with optimized settings for speed."""
        try:
            # For Qwen2.5-VL, prepare messages in the format expected by the processor
            user_message = prompt
            if system:
                user_message = f"System: {system}\n\nUser: {prompt}"
            
            messages = [{"role": "user", "content": user_message}]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process text only (no images for now)
            inputs = self.processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to GPU
            first_device = next(self.model.parameters()).device
            inputs = inputs.to(first_device)
            
            # Generate response with optimized parameters for speed
            with torch.no_grad():
                torch.cuda.synchronize()  # Ensure GPU is ready
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=1,
                    do_sample=False,  # Greedy decoding for speed
                    num_beams=1,  # No beam search for speed
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache
                    repetition_penalty=1.05,  # Slight penalty to avoid loops
                )
                torch.cuda.synchronize()  # Wait for generation to finish
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip()

        except Exception as e:
            print(f"ERROR: QwenVL generation failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}"

    def generate_stream(self, prompt: str, system: Optional[str] = None, max_tokens: int = 256, messages: Optional[List] = None):
        """Generate streaming response from QwenVL."""
        try:
            # For now, generate the full response and yield it word by word
            # Future: implement proper streaming generation with TextIteratorStreamer
            response = self.generate(prompt, system, max_tokens, messages)
            
            # Split into words and yield them
            words = response.split()
            for word in words:
                yield word + " "
                
        except Exception as e:
            print(f"ERROR: QwenVL streaming failed: {e}")
            yield f"Error: {str(e)}"


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= text_len:
            break
    
    return chunks

