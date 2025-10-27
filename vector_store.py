"""Lightweight persistent vector store using NumPy arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from embedding_manager import encode_text, encode_texts


class VectorStore:
    """Stores embeddings and metadata on disk and provides cosine search."""

    def __init__(self, store_dir: Path | str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._data_path = self.store_dir / "entries.json"
        self._embeddings_path = self.store_dir / "embeddings.npy"
        self._entries: List[Dict] = []
        self._embeddings: np.ndarray | None = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._data_path.exists():
            with open(self._data_path, "r", encoding="utf-8") as fh:
                self._entries = json.load(fh)
        else:
            self._entries = []

        if self._embeddings_path.exists():
            self._embeddings = np.load(self._embeddings_path)
        else:
            self._embeddings = None

    def _save(self) -> None:
        with open(self._data_path, "w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, ensure_ascii=False, indent=2)

        if self._embeddings is None:
            np.save(self._embeddings_path, np.zeros((0, 1), dtype=np.float32))
        else:
            np.save(self._embeddings_path, self._embeddings.astype(np.float32))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self._entries)

    def add_texts(self, texts: Iterable[str], metadatas: Iterable[Dict], batch_size: int = 32) -> None:
        """Add texts to the vector store with memory-efficient batching
        
        Args:
            texts: Texts to add
            metadatas: Metadata for each text
            batch_size: Batch size for processing (lower for memory-constrained systems)
        """
        text_list = list(texts)
        metadata_list = list(metadatas)
        if len(text_list) != len(metadata_list):
            raise ValueError("texts and metadatas must be the same length")
        if not text_list:
            return
            
        # Try to determine appropriate batch size based on system RAM
        try:
            import psutil
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
            # Adjust batch size based on available RAM
            if system_ram_gb < 8:  # Very constrained system
                batch_size = min(batch_size, 8)
            elif system_ram_gb < 16:  # 8-16GB RAM
                batch_size = min(batch_size, 16)
        except ImportError:
            # If psutil isn't available, use provided batch_size
            pass

        # Process in batches to manage memory usage
        all_embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(text_list)-1)//batch_size + 1}")
            
            # Generate embeddings for this batch
            batch_embeddings = encode_texts(batch_texts).cpu().numpy().astype(np.float32)
            all_embeddings.append(batch_embeddings)
            
            # Force garbage collection after each batch if in memory-constrained environment
            if len(text_list) > batch_size * 2:  # Only for larger datasets
                try:
                    import gc
                    gc.collect()
                except:
                    pass
        
        # Combine all embeddings
        if len(all_embeddings) == 1:
            embeddings = all_embeddings[0]
        else:
            embeddings = np.concatenate(all_embeddings, axis=0)
            
        # Update store
        if self._embeddings is None or self._embeddings.size == 0:
            self._embeddings = embeddings
        else:
            self._embeddings = np.concatenate([self._embeddings, embeddings], axis=0)

        # Add entries
        for text, meta in zip(text_list, metadata_list):
            self._entries.append({"text": text, "metadata": meta})

        # Save to disk
        self._save()

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self._embeddings is None or self._embeddings.size == 0:
            return []

        query = query_embedding.astype(np.float32)
        scores = np.dot(self._embeddings, query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict] = []
        for idx in top_indices:
            entry = self._entries[int(idx)]
            score = float(scores[int(idx)])
            results.append({"text": entry["text"], "metadata": entry["metadata"], "score": score})
        return results

    def similarity_search_text(self, query_text: str, top_k: int = 5) -> List[Dict]:
        embedding = encode_text(query_text).cpu().numpy()
        return self.similarity_search(embedding, top_k=top_k)

    def clear(self) -> None:
        self._entries = []
        self._embeddings = None
        if self._data_path.exists():
            self._data_path.unlink()
        if self._embeddings_path.exists():
            self._embeddings_path.unlink()
