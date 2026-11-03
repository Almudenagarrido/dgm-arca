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


def interleave(*lists: list) -> list:
    """Merge ranked hit lists turn by turn, keeping the first occurrence of each chunk."""
    seen, out = set(), []
    for group in zip(*lists, strict=False):
        for hit in group:
            if hit.chunk.id not in seen:
                seen.add(hit.chunk.id)
                out.append(hit)
    return out


DEFAULT_RERANK_INSTRUCTION = (
    "Given a question about the EU AI Act, judge whether the passage answers it."
)


class Reranker:
    """Qwen3-Reranker as a cross-encoder: score = P("yes" | instruction, query, passage).

    Unlike the embedding model, which encodes query and passage separately, the reranker reads
    both together, so it can tell "the passage that answers" from "a passage on the same topic".
    It is too slow for 3000 chunks but fine for the 20-50 candidates of the first stage. The
    prompt is the one in the model card.
    """

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_id: str = "Qwen/Qwen3-Reranker-0.6B", max_length: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id, self.max_length = model_id, max_length
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device
        ).eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")

    def score(
        self,
        query: str,
        passages: list[str],
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
        batch_size: int = 16,
    ) -> list[float]:
        import torch

        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            texts = [
                f"{self.PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {p}"
                for p in passages[start : start + batch_size]
            ]
            # Truncate the passage, never the closing tags that ask for the answer.
            body = self.tokenizer(
                texts, truncation=True, add_special_tokens=False, max_length=self.max_length - 16
            )["input_ids"]
            suffix = self.tokenizer(self.SUFFIX, add_special_tokens=False)["input_ids"]
            batch = self.tokenizer.pad(
                {"input_ids": [b + suffix for b in body]}, return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                # Only the last position matters; full logits would be batch x length x 152k.
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            scores += torch.softmax(pair, dim=1)[:, 1].tolist()
        return scores


class RerankedRetriever:
    """First stage (any retriever) for ``pool`` candidates, then the cross-encoder reorders them."""

    def __init__(
        self,
        first_stage,
        reranker: Reranker,
        pool: int = 30,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
    ):
        self.first_stage, self.reranker = first_stage, reranker
        self.pool, self.instruction = pool, instruction
        self.name = f"{first_stage.name}+rerank"

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        return self.rerank(query, self.first_stage.search(query, top_k=self.pool), top_k)

    def rerank(self, query: str, candidates: list[Hit], top_k: int) -> list[Hit]:
        scores = self.reranker.score(query, [h.chunk.text for h in candidates], self.instruction)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(candidates[i].chunk, scores[i]) for i in order]
