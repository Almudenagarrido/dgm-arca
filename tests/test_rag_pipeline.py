"""Phase 3 pieces that run without the embedding model: chunking, fusion, gold relevance."""

from __future__ import annotations

import numpy as np
import pytest

from rag.chunking import Chunk, semantic_chunks
from rag.evaluate import hit_at_k, relevant_chunk_ids
from rag.generate import NOT_FOUND, build_prompt, says_not_found, to_response
from rag.ingest import Document
from rag.retriever import DenseRetriever, Hit, HybridRetriever


def topic_embedder(sentences: list[str]) -> np.ndarray:
    """Two orthogonal 'topics': sentences about fines vs. everything else."""
    vectors = np.array([[1.0, 0.0] if "multa" in s else [0.0, 1.0] for s in sentences])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_semantic_chunks_split_where_the_topic_changes():
    doc = Document(
        id="d",
        source="x.md",
        text=(
            "Las multas llegan a 35 millones. La multa depende de la facturación. "
            "El chatbot debe avisar. El usuario sabrá que habla con una IA."
        ),
    )
    chunks = semantic_chunks([doc], topic_embedder, distance_threshold=0.5)
    assert [c.text.count(".") for c in chunks] == [2, 2]
    assert chunks[1].id == "d-0001" and "chatbot" in chunks[1].text


def test_semantic_chunks_respect_the_size_limit():
    doc = Document(id="d", source="x.md", text="Una multa alta. " * 20)
    chunks = semantic_chunks([doc], topic_embedder, distance_threshold=1.0, max_chunk_chars=60)
    assert all(len(c.text) <= 60 for c in chunks) and len(chunks) > 1


def _chunk(cid: str, text: str = "", source: str = "a.md") -> Chunk:
    return Chunk(id=cid, doc_id=cid.split("-")[0], source=source, text=text, index=0)


def test_hybrid_fusion_rescales_scores_and_weights_them():
    a, b, c = _chunk("a-0"), _chunk("b-0"), _chunk("c-0")
    bm25 = [Hit(a, 30.0), Hit(b, 10.0)]  # unbounded scores
    dense = [Hit(b, 0.80), Hit(c, 0.60)]  # cosine similarities
    only_dense = HybridRetriever(None, None, weight_bm25=0.0).combine(bm25, dense, 3)
    assert only_dense[0].chunk.id == "b-0"
    # Min-max puts the last hit of each list at 0, the same as a chunk the list never found.
    assert {h.chunk.id: h.score for h in only_dense[1:]} == {"a-0": 0.0, "c-0": 0.0}
    only_bm25 = HybridRetriever(None, None, weight_bm25=1.0).combine(bm25, dense, 3)
    assert only_bm25[0].chunk.id == "a-0"
    mixed = HybridRetriever(None, None, weight_bm25=0.5).combine(bm25, dense, 3)
    # b is second in BM25 (0) and first in dense (1): 0.5; a is 1 and 0: 0.5; c is 0 and 0.
    assert {h.chunk.id: round(h.score, 2) for h in mixed} == {"a-0": 0.5, "b-0": 0.5, "c-0": 0.0}


class FakeEmbedder:
    model_id = "fake"

    def __init__(self):
        self.calls = 0

    def embed_passages(self, texts, batch_size=32):
        self.calls += 1
        return topic_embedder(texts)

    def embed_queries(self, texts, instruction=None):
        return topic_embedder(texts)


def test_dense_index_is_persisted_and_invalidated(tmp_path):
    chunks = [_chunk("a-0", "una multa"), _chunk("b-0", "un chatbot")]
    embedder = FakeEmbedder()
    first = DenseRetriever(chunks, embedder, persist_dir=str(tmp_path))
    assert first.search("¿qué multa?", top_k=1)[0].chunk.id == "a-0"
    DenseRetriever(chunks, embedder, persist_dir=str(tmp_path))
    assert embedder.calls == 1  # second start reused rag/index
    changed = [*chunks, _chunk("c-0", "otra multa")]
    DenseRetriever(changed, embedder, persist_dir=str(tmp_path))
    assert embedder.calls == 2  # different chunks, different fingerprint


def test_gold_relevance_survives_a_chunk_boundary():
    evidence = "a más tardar en un plazo de veinticuatro horas desde el inicio del uso del sistema"
    item = {"evidence": [{"source": "art5.md", "text": evidence}]}
    whole = [_chunk("d-0", f"Texto previo, {evidence}. Más texto.", "art5.md")]
    assert relevant_chunk_ids(item, whole) == {"d-0"}
    split = [
        _chunk("d-0", f"previo {evidence[:65]}", "art5.md"),
        _chunk("d-1", f"{evidence[15:]} y sigue", "art5.md"),
        _chunk("e-0", evidence, "otro.md"),
    ]  # same words, wrong source
    assert relevant_chunk_ids(item, split) == {"d-0", "d-1"}
    assert hit_at_k(["x", "d-1"], {"d-0", "d-1"}, 2) == 1.0


@pytest.mark.parametrize("answer", [NOT_FOUND, f"«{NOT_FOUND}»", "No lo sé."])
def test_not_found_is_detected(answer):
    assert says_not_found(answer)


def test_prompt_lists_ids_and_response_drops_invented_citations():
    hits = [Hit(_chunk("d-0001", "texto del artículo 5"), 0.9)]
    prompt = build_prompt("¿Qué dice?", hits)
    assert "[d-0001]" in prompt and "No lo sé" in prompt
    response = to_response("Está prohibido [d-0001] [zz-9999].", hits, "hybrid")
    assert response.citations == ["d-0001"]
