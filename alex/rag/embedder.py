"""Local embeddings via MLX, optimized for Apple Silicon.

Default model is BAAI's BGE-M3 in the MLX-community 4-bit quantization.
- 567M params, 1024-dim output, English + multilingual.
- ~10-30 ms per short query on M3/M4 once loaded; ~340 MB on disk.
- Strong on FinMTEB per the research; chosen for the plan.

Alternative repos that drop in here (set ``ALEX_EMBED_REPO``):
- ``mlx-community/bge-small-en-v1.5-4bit`` — fastest, 384-dim, lower accuracy
- ``mlx-community/snowflake-arctic-embed-l-v2.0-4bit`` — top MTEB, 1024-dim
- ``mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`` — top MTEB, larger
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import mlx.core as mx
import mlx_embeddings
import numpy as np
from loguru import logger

DEFAULT_MODEL = "mlx-community/bge-m3-mlx-4bit"


@dataclass
class EmbedResult:
    vectors: np.ndarray   # shape (n, dim), float32
    dim: int
    elapsed_ms: float


@lru_cache(maxsize=4)
def _load(repo: str):
    logger.info(f"loading MLX embedder: {repo}")
    t0 = time.perf_counter()
    model, tokenizer = mlx_embeddings.load(repo)
    logger.info(f"  loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")
    return model, tokenizer


class LocalEmbedder:
    """Thin wrapper that batches and normalizes vectors."""

    def __init__(self, repo: str = DEFAULT_MODEL) -> None:
        self.repo = repo
        self._model, self._tokenizer = _load(repo)
        # One throwaway encode to surface dimension and warm caches.
        warm = self.encode(["dimension probe"])
        self.dim = warm.dim
        logger.info(f"embedder ready: dim={self.dim}, repo={repo}")

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=np.zeros((0, 0), dtype=np.float32), dim=0, elapsed_ms=0.0)

        t0 = time.perf_counter()
        arr = mlx_embeddings.generate(self._model, self._tokenizer, list(texts))
        # mlx_embeddings returns either an mx.array OR an object with .text_embeds
        # depending on model class. Normalize to mx.array first.
        if hasattr(arr, "text_embeds"):
            arr = arr.text_embeds
        # If output is per-token, mean-pool. For BGE the model already emits a
        # pooled CLS-style vector, so this is a no-op then.
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        if normalize:
            norm = mx.linalg.norm(arr, axis=-1, keepdims=True)
            arr = arr / mx.maximum(norm, 1e-12)
        mx.eval(arr)
        np_arr = np.asarray(arr, dtype=np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return EmbedResult(vectors=np_arr, dim=np_arr.shape[1], elapsed_ms=elapsed_ms)
