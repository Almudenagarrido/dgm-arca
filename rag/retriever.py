"""Vector store, the three retrievers and the knowledge base that ties them to a chunking.

* ``DenseRetriever``: cosine similarity between the query embedding (with instruction)
  and the chunk embeddings held in a ``VectorStore``.
* ``BM25Retriever``: Okapi BM25 over accent-free lowercase tokens without stopwords.
* ``HybridRetriever``: ``λ · dense + (1 − λ) · bm25`` after min-max normalising each score
  over the whole corpus, so the two scales are comparable. ``λ = 1`` is pure dense and
  ``λ = 0`` pure BM25; ``rag.evaluate`` sweeps it on the gold set.

A *chunking config* is a string such as ``recursive-1000``, ``semantic-90`` or
``semantic-90+title`` (the ``+title`` suffix prefixes each chunk with its document title
before indexing it). A *per-folder* config gives each top-level corpus folder its own:
``aiact:recursive-1000+title,directrices:semantic-90+title``; ``*:`` sets the default for
folders not listed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rag.chunking import Chunk, recursive_chunks, semantic_chunks, split_sentences
from rag.embed import INDEX_DIR, Embedder
from rag.ingest import CORPUS_DIR, Document, load_corpus

RETRIEVERS = ("dense", "bm25", "hybrid")

# Spanish and English function words: the corpus mixes both languages.
STOPWORDS = frozenset(
    """a al algo como con contra cual cuales cuando de del desde donde e el ella ellas ello
    ellos en entre era es esa esas ese eso esos esta estas este esto estos fue ha han hasta
    hay la las le les lo los mas me mi mientras muy ni no nos o os otra otras otro otros para
    pero por que quien se sea segun ser si sin sobre son su sus tambien te tiene tienen u un
    una unas uno unos y ya the of and to in is are be for on or by with as that this it an
    which from at not may their its such""".split()
)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in re.findall(r"\w+", text) if len(t) > 1 and t not in STOPWORDS]


def _minmax(scores: np.ndarray) -> np.ndarray:
    low, high = float(scores.min()), float(scores.max())
    return np.zeros_like(scores) if high == low else (scores - low) / (high - low)


# --------------------------------------------------------------------------- vector store


@dataclass
class VectorStore:
    """In-memory matrix of normalised vectors with exact (brute-force) cosine search.

    A few thousand chunks fit in a few megabytes, and exact search means the retrieval
    numbers measure the embeddings, not an approximate index.
    """

    ids: list[str]
    vectors: np.ndarray  # (n, dim), L2-normalised

    def scores(self, query_vector: np.ndarray) -> np.ndarray:
        return self.vectors @ query_vector

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores = self.scores(query_vector)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        np.savez(path, ids=np.array(self.ids), vectors=self.vectors)

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        data = np.load(path)
        return cls(ids=data["ids"].tolist(), vectors=data["vectors"])


# ----------------------------------------------------------------------------- retrievers


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def scores(self, query: str) -> np.ndarray:
        return self.store.scores(self.embedder.embed_queries([query])[0])


class BM25Retriever:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, lam: float) -> None:
        self.dense, self.bm25, self.lam = dense, bm25, lam

    def combine(self, dense_scores: np.ndarray, bm25_scores: np.ndarray, lam: float | None = None):
        lam = self.lam if lam is None else lam
        return lam * _minmax(dense_scores) + (1 - lam) * _minmax(bm25_scores)

    def scores(self, query: str) -> np.ndarray:
        return self.combine(self.dense.scores(query), self.bm25.scores(query))


# ------------------------------------------------------------------------- knowledge base


def parse_config(config: str) -> tuple[str, float, bool]:
    """``semantic-90+title`` → ("semantic", 90.0, True)."""
    match = re.fullmatch(r"(recursive|semantic)-(\d+(?:\.\d+)?)(\+title)?", config)
    if not match:
        raise ValueError(f"Bad chunking config {config!r}; e.g. recursive-1000, semantic-90+title")
    return match[1], float(match[2]), bool(match[3])


def parse_folder_configs(config: str) -> dict[str, str]:
    """``aiact:recursive-1000,*:semantic-90`` → {"aiact": "recursive-1000", "*": "semantic-90"}.

    A plain config applies to every folder: ``recursive-1000`` → {"*": "recursive-1000"}.
    """
    if ":" not in config:
        parse_config(config)
        return {"*": config}
    mapping = {}
    for part in config.split(","):
        folder, sep, base = (x.strip() for x in part.partition(":"))
        if not sep or not folder or folder in mapping:
            raise ValueError(
                f"Bad per-folder config {config!r}; e.g. "
                "aiact:recursive-1000+title,directrices:semantic-90+title"
            )
        parse_config(base)
        mapping[folder] = base
    return mapping


def config_for(doc_id: str, mapping: dict[str, str]) -> str:
    """The chunking config that applies to ``doc_id`` (folder = first path component)."""
    folder = doc_id.split("/", 1)[0] if "/" in doc_id else ""
    base = mapping.get(folder, mapping.get("*"))
    if base is None:
        raise ValueError(f"No chunking config for {doc_id!r}: add '{folder}:...' or '*:...'")
    return base


def _chunk_docs(
    docs: list[Document], config: str, embedder: Embedder, sentence_docs: list[Document]
) -> dict[str, list[Chunk]]:
    """Chunks of each document under one plain config, keyed by doc id."""
    kind, param, _ = parse_config(config)
    if kind == "recursive":
        overlap = int(param) // 7
        return {d.doc_id: recursive_chunks(d, size=int(param), overlap=overlap) for d in docs}

    # Embed every sentence of the corpus once (cached on disk) and look them up per document,
    # so trying several percentiles costs no extra GPU time. ``sentence_docs`` is the whole
    # corpus even when only some folders are chunked semantically, so the cache is shared.
    sentences = sorted({d.text[s:e] for d in sentence_docs for s, e in split_sentences(d.text)})
    vectors = embedder.embed_documents(sentences, cache_name="sentences")
    row = {text: i for i, text in enumerate(sentences)}

    def lookup(texts: list[str]) -> np.ndarray:
        return vectors[[row[t] for t in texts]]

    return {d.doc_id: semantic_chunks(d, lookup, percentile=param) for d in docs}


def chunk_corpus(docs: list[Document], config: str, embedder: Embedder) -> list[Chunk]:
    """Chunks of the whole corpus, in document order; ``config`` may be per-folder."""
    mapping = parse_folder_configs(config)
    groups: dict[str, list[Document]] = {}
    for d in docs:
        groups.setdefault(config_for(d.doc_id, mapping), []).append(d)
    by_doc: dict[str, list[Chunk]] = {}
    for base, group in groups.items():
        by_doc.update(_chunk_docs(group, base, embedder, docs))
    return [c for d in docs for c in by_doc[d.doc_id]]


@dataclass
class KnowledgeBase:
    """Chunks of one config plus the three retrievers built over them."""

    config: str
    chunks: list[Chunk]
    dense: DenseRetriever
    bm25: BM25Retriever
    lam: float = 0.5
    by_id: dict[str, Chunk] = field(init=False)

    def __post_init__(self) -> None:
        self.by_id = {c.id: c for c in self.chunks}
        self.hybrid = HybridRetriever(self.dense, self.bm25, self.lam)

    @classmethod
    def build(
        cls,
        config: str,
        embedder: Embedder,
        lam: float = 0.5,
        corpus_dir: Path | str = CORPUS_DIR,
        docs: list[Document] | None = None,
    ) -> KnowledgeBase:
        docs = docs if docs is not None else load_corpus(corpus_dir)
        mapping = parse_folder_configs(config)
        chunks = chunk_corpus(docs, config, embedder)
        texts = [c.index_text(parse_config(config_for(c.doc_id, mapping))[2]) for c in chunks]
        name = re.sub(r"[^\w.+-]+", "_", config)  # per-folder configs carry ":", "," and "*"
        vectors = embedder.embed_documents(texts, cache_name=f"chunks-{name}")
        store = VectorStore([c.id for c in chunks], vectors)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        store.save(INDEX_DIR / f"store-{name}.npz")
        return cls(config, chunks, DenseRetriever(store, embedder), BM25Retriever(texts), lam)

    def scores(self, query: str, retriever: str) -> np.ndarray:
        if retriever not in RETRIEVERS:
            raise ValueError(f"retriever must be one of {RETRIEVERS}")
        return getattr(self, retriever).scores(query)

    def search(
        self, query: str, retriever: str = "hybrid", k: int = 5
    ) -> list[tuple[Chunk, float]]:
        scores = self.scores(query, retriever)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
