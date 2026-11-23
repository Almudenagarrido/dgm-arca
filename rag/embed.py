"""Qwen3-Embedding with instructions, and an on-disk cache of the corpus embeddings.

Qwen3-Embedding is *instruction-aware*: queries are prefixed with a task description
(``Instruct: ...\\nQuery: ...``) and documents are embedded as they are. The asymmetry
matters because a question («¿cuánto es la multa por...?») and the passage that answers
it («multas administrativas de hasta...») do not look alike; the instruction tells the
model which relation to encode. ``rag.evaluate --no-instruction`` measures the effect.

All vectors are L2-normalised, so the dot product is the cosine similarity.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

INDEX_DIR = Path(__file__).resolve().parent / "index"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INSTRUCTION = (
    "Given a question about the EU AI Act (Regulation 2024/1689) or the Commission "
    "guidelines, retrieve the passage that answers it"
)


class Embedder:
    def __init__(
        self,
        model_name: str | None = None,
        instruction: str | None = DEFAULT_INSTRUCTION,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get("ARCA_RAG_EMBEDDER", DEFAULT_MODEL)
        self.instruction = instruction
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            kwargs = {"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {}
            self._model = SentenceTransformer(
                self.model_name, device=self.device, model_kwargs=kwargs
            )
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 500,
        )
        return vectors.astype(np.float32)

    def query_text(self, query: str) -> str:
        if not self.instruction:
            return query
        return f"Instruct: {self.instruction}\nQuery:{query}"

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([self.query_text(q) for q in queries])

    def embed_documents(self, texts: list[str], cache_name: str | None = None) -> np.ndarray:
        """Embed passages; with ``cache_name`` the result is stored in ``rag/index/``.

        The cache key includes the model and a hash of every text, so changing the
        chunking or the corpus never reuses stale vectors.
        """
        if cache_name is None:
            return self._encode(texts)
        digest = hashlib.sha1(self.model_name.encode())
        for text in texts:
            digest.update(text.encode())
            digest.update(b"\0")
        path = INDEX_DIR / f"{cache_name}-{digest.hexdigest()[:12]}.npy"
        if path.exists():
            return np.load(path)
        vectors = self._encode(texts)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors
