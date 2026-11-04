"""Phase 3 pieces that run without the embedding model: chunking, fusion, gold relevance."""

from __future__ import annotations

import numpy as np
import pytest

from rag.chunking import Chunk, semantic_chunks
from rag.evaluate import hit_at_k, relevant_chunk_ids
from rag.generate import NOT_FOUND, build_prompt, resolve_labels, says_not_found, to_response
from rag.ingest import Document
from rag.retriever import DenseRetriever, Hit, HybridRetriever, interleave


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


def test_prompt_uses_labels_and_answers_get_real_ids():
    hits = [Hit(_chunk("d-0001", "texto del artículo 5"), 0.9), Hit(_chunk("e-0002", "otro"), 0.5)]
    prompt = build_prompt("¿Qué dice?", hits)
    assert "[F1]" in prompt and "[F2]" in prompt and "d-0001" not in prompt
    assert "No lo sé" in prompt
    answer, invalid = resolve_labels("Está prohibido [F1]. También [F2] y [F7].", hits)
    assert answer == "Está prohibido [d-0001]. También [e-0002] y." and invalid == ["[F7]"]
    response = to_response(answer, hits, "hybrid", invalid_citations=invalid)
    assert response.citations == ["d-0001", "e-0002"] and response.invalid_citations == ["[F7]"]


def test_interleave_merges_turn_by_turn_without_duplicates():
    a, b, c = _chunk("a-0"), _chunk("b-0"), _chunk("c-0")
    merged = interleave([Hit(a, 1), Hit(b, 0.5)], [Hit(b, 1), Hit(c, 0.2)])
    assert [h.chunk.id for h in merged] == ["a-0", "b-0", "c-0"]


def test_judge_verdicts_are_parsed_defensively():
    from rag.judge import parse_verdict

    raw = '<think>{"correct": "no"}</think> {"correct": "Si", "faithful": "no", "reason": "ok"}'
    assert parse_verdict(raw) == {
        "correct": "sí",
        "faithful": "no",
        "caveats": "?",
        "reason": "ok",
    }
    assert parse_verdict("sin json")["correct"] == "?"


def test_low_reranker_score_answers_not_found_without_generating():
    from rag.generate import RagPipeline

    pipeline = RagPipeline(rerank=True, translate_query=False)
    pipeline.min_score = 0.975
    calls = []
    pipeline.generate = lambda prompt, thinking=None: calls.append(prompt) or "Respuesta [F1]."
    pipeline.retrieve = lambda q, k, r: [Hit(_chunk("d-0001", "texto"), 0.42)]
    assert pipeline.answer("¿Capital de Australia?").answer == NOT_FOUND and not calls
    pipeline.retrieve = lambda q, k, r: [Hit(_chunk("d-0001", "texto"), 0.99)]
    response = pipeline.answer("¿Qué dice el artículo 5?")
    assert response.citations == ["d-0001"] and len(calls) == 1


def test_low_sufficiency_answers_not_found_without_generating():
    from rag.generate import RagPipeline

    pipeline = RagPipeline(rerank=True, translate_query=False)
    pipeline.min_score, pipeline.min_sufficiency = 0.5, 0.4
    calls = []
    pipeline.generate = lambda prompt, thinking=None: calls.append(prompt) or "Respuesta [F1]."
    pipeline.retrieve = lambda q, k, r: [Hit(_chunk("d-0001", "texto"), 0.99)]
    pipeline.sufficiency = lambda q, hits: 0.1
    assert pipeline.answer("¿Multa de la DSA?").answer == NOT_FOUND and not calls
    pipeline.sufficiency = lambda q, hits: 0.9
    assert pipeline.answer("¿Qué dice el artículo 5?").citations == ["d-0001"]


def test_wilson_interval_and_paired_bootstrap():
    from rag.stats import paired_bootstrap, wilson

    lo, hi = wilson(0, 46)
    assert lo == 0 and abs(hi - 0.077) < 0.001
    lo, hi = wilson(23, 46)
    assert lo < 0.5 < hi
    same = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0])
    assert same["diff"] == 0 and same["ci95"] == [0, 0] and same["p_not_better"] == 1


def test_calibration_picks_the_middle_of_the_tied_range():
    from rag.calibrate import candidates, choose

    rows = [
        {
            "answerable": True,
            "top_score": 0.99,
            "sufficiency": 1,
            "generator_refused": False,
            "answer": "Respuesta.",
            "keyword": True,
        },
        {
            "answerable": True,
            "top_score": 0.95,
            "sufficiency": 1,
            "generator_refused": False,
            "answer": "Respuesta.",
            "keyword": True,
        },
        {
            "answerable": False,
            "top_score": 0.50,
            "sufficiency": 1,
            "generator_refused": False,
            "answer": "Respuesta.",
            "keyword": None,
        },
    ]
    t, s = choose(rows, [(t, 0.0) for t in candidates([r["top_score"] for r in rows])])
    assert 0.5 < t < 0.95 and s == 0


@pytest.mark.parametrize(
    "answer, disguised",
    [
        (
            "La multa no se especifica directamente en los documentos proporcionados. Sin embargo, "
            "el artículo 99 dice... [F1]",
            True,
        ),
        ("La información proporcionada no especifica quién dirige la Oficina de IA.", True),
        ("La ley china no se menciona en los fragmentos. Los documentos tratan...", True),
        (NOT_FOUND, False),  # already a refusal, not a disguised one
        ("El reglamento no se aplicará a los sistemas de código abierto [F1], salvo...", False),
        ("No. El plazo general es de quince días [F2]; los documentos no indican otro.", False),
    ],
)
def test_disguised_refusals_are_detected_in_the_first_sentence(answer, disguised):
    from rag.generate import hedges

    assert hedges(answer) is disguised


def test_disguised_refusal_becomes_not_found():
    from rag.generate import RagPipeline

    pipeline = RagPipeline(rerank=True, translate_query=False)
    pipeline.min_score = 0.5
    pipeline.retrieve = lambda q, k, r: [Hit(_chunk("d-0001", "texto"), 0.99)]
    pipeline.generate = lambda prompt, thinking=None: (
        "El dato no se especifica en los documentos proporcionados. Sin embargo, [F1] dice..."
    )
    assert pipeline.answer("¿Multa del RGPD?").answer == NOT_FOUND
