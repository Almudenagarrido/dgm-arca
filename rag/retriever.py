"""Retrieval: similarity search, keyword search (BM25) and the hybrid of both.

``BM25Retriever`` is the keyword baseline, ``DenseRetriever`` ranks by embedding similarity and
``HybridRetriever`` mixes both. All three return the same ``Hit`` objects so that
``rag/generate.py`` and ``rag/evaluate.py`` do not care which one is used.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Keyword ranking: term frequency, inverse document frequency, length normalisation."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


class DenseRetriever:
    """Cosine similarity over Qwen3-Embedding vectors, stored as a NumPy matrix.

    The matrix is persisted in ``persist_dir`` as ``<name>.npz`` together with a fingerprint
    of the embedding model and of every chunk id and text. If the chunks or the model change,
    the fingerprint does not match and the corpus is embedded again; otherwise the API starts
    without re-embedding anything.
    """

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder,
        persist_dir: str = "rag/index",
        index_name: str = "dense",
        query_instruction: str | None = None,
    ):
        self.chunks, self.embedder = chunks, embedder
        self.query_instruction = query_instruction
        path = Path(persist_dir) / f"{index_name}.npz"
        fingerprint = _fingerprint(embedder.model_id, chunks)
        if path.exists():
            stored = np.load(path, allow_pickle=False)
            if str(stored["fingerprint"]) == fingerprint:
                self.vectors = stored["vectors"]
                return
        self.vectors = embedder.embed_passages([c.text for c in chunks]).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=self.vectors, fingerprint=np.array(fingerprint))

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        kwargs = {"instruction": self.query_instruction} if self.query_instruction else {}
        query_vector = self.embedder.embed_queries([query], **kwargs)[0]
        scores = self.vectors @ query_vector
        top = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]


def _fingerprint(model_id: str, chunks: list[Chunk]) -> str:
    digest = hashlib.sha1(model_id.encode())
    for chunk in chunks:
        digest.update(chunk.id.encode())
        digest.update(chunk.text.encode())
    return digest.hexdigest()


def _min_max(hits: list[Hit]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [h.score for h in hits]
    low, high = min(scores), max(scores)
    span = high - low
    return {h.chunk.id: (h.score - low) / span if span > 0 else 1.0 for h in hits}


class HybridRetriever:
    """s_hybrid = λ · s_bm25 + (1 - λ) · s_dense, on scores rescaled to [0, 1].

    Each retriever contributes its top ``pool`` hits; scores are min-max normalised within
    each list so that BM25 (unbounded) and cosine similarity are comparable. A chunk that only
    one retriever found gets 0 from the other. λ (``weight_bm25``) is chosen on the gold set
    with ``rag/evaluate.py``.
    """

    name = "hybrid"

    def __init__(
        self, bm25: BM25Retriever, dense: DenseRetriever, weight_bm25: float = 0.5, pool: int = 20
    ):
        self.bm25, self.dense, self.weight_bm25, self.pool = bm25, dense, weight_bm25, pool

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        return self.combine(
            self.bm25.search(query, self.pool), self.dense.search(query, self.pool), top_k
        )

    def combine(self, bm25_hits: list[Hit], dense_hits: list[Hit], top_k: int) -> list[Hit]:
        """Merge two ranked lists (exposed so the λ sweep can reuse the same searches)."""
        sparse, dense = _min_max(bm25_hits), _min_max(dense_hits)
        by_id = {h.chunk.id: h.chunk for h in [*bm25_hits, *dense_hits]}
        lam = self.weight_bm25
        scored = {
            cid: lam * sparse.get(cid, 0.0) + (1 - lam) * dense.get(cid, 0.0) for cid in by_id
        }
        order = sorted(scored, key=scored.get, reverse=True)[:top_k]
        return [Hit(by_id[cid], scored[cid]) for cid in order]
