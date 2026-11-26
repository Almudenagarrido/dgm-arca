"""Phase 3 tests that need neither GPU nor network: fake embedders, rerankers and generators."""

from __future__ import annotations

import numpy as np
import pytest

from rag.calibrate import choose_threshold, gate_metrics
from rag.chunking import Chunk, recursive_chunks, semantic_breaks, semantic_chunks, split_sentences
from rag.evaluate import metrics, ranks_of_first_relevant
from rag.generate import NOT_FOUND, RagPipeline, build_prompt, extract_citations, is_abstention
from rag.goldset import GOLD_DIR, Evidence, GoldItem, RelevanceJudge, find_span, load_gold
from rag.ingest import Document, load_corpus
from rag.retriever import (
    BM25Retriever,
    KnowledgeBase,
    VectorStore,
    _minmax,
    config_for,
    parse_config,
    parse_folder_configs,
    tokenize,
)

DOC = Document(
    "aiact/articulo_099.md",
    "Artículo 99. Sanciones",
    "# Artículo 99. Sanciones\n\n3. Multas de hasta 35 000 000 EUR por prácticas prohibidas.\n\n"
    "4. Multas de hasta 15 000 000 EUR por otras infracciones. "
    "Los Estados miembros comunicarán las normas.",
)


# ------------------------------------------------------------------------------- chunking


def test_recursive_chunks_keep_their_span():
    chunks = recursive_chunks(DOC, size=80, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert DOC.text[c.start : c.end] == c.text
        assert len(c.text) <= 80


def test_split_sentences_cuts_lines_and_full_stops():
    spans = split_sentences("Primera frase. Segunda frase.\na) una letra;\nb) otra")
    pieces = ["Primera frase.", "Segunda frase.", "a) una letra;", "b) otra"]
    assert [
        s for s in ("Primera frase. Segunda frase.\na) una letra;\nb) otra"[a:b] for a, b in spans)
    ] == pieces


def test_semantic_breaks_where_topic_changes():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    embeddings = np.stack([a, a, a, b, b])
    assert semantic_breaks(embeddings, percentile=50) == [2]


def test_semantic_chunks_group_similar_sentences():
    doc = Document("x.md", "x", "Multa uno. Multa dos. Multa tres.\nPlazo uno. Plazo dos.")

    def fake_embed(texts):
        return np.array([[1.0, 0.0] if t.startswith("Multa") else [0.0, 1.0] for t in texts])

    chunks = semantic_chunks(doc, fake_embed, percentile=50, min_chars=1, max_chars=1000)
    assert [c.text for c in chunks] == [
        "Multa uno. Multa dos. Multa tres.",
        "Plazo uno. Plazo dos.",
    ]
    assert all(doc.text[c.start : c.end] == c.text for c in chunks)


def test_semantic_chunks_respect_max_chars():
    doc = Document("x.md", "x", " ".join(f"Frase {i}." for i in range(40)))
    chunks = semantic_chunks(
        doc, lambda t: np.ones((len(t), 2)) / np.sqrt(2), max_chars=60, min_chars=1
    )
    assert all(len(c.text) <= 60 for c in chunks)


def test_index_text_with_title():
    chunk = Chunk("d#0", "d", "Artículo 5", "texto", 0, 5)
    assert chunk.index_text(True) == "Artículo 5\ntexto"
    assert chunk.index_text(False) == "texto"


def test_parse_config():
    assert parse_config("semantic-90+title") == ("semantic", 90.0, True)
    assert parse_config("recursive-1000") == ("recursive", 1000.0, False)
    with pytest.raises(ValueError):
        parse_config("legal")


def test_per_folder_configs():
    mapping = parse_folder_configs("aiact:recursive-1000+title, *:semantic-90")
    assert mapping == {"aiact": "recursive-1000+title", "*": "semantic-90"}
    assert config_for("aiact/articulo_005.md", mapping) == "recursive-1000+title"
    assert config_for("directrices/x.md", mapping) == "semantic-90"
    assert parse_folder_configs("recursive-1000") == {"*": "recursive-1000"}
    with pytest.raises(ValueError):
        config_for("otros/x.md", parse_folder_configs("aiact:recursive-1000"))
    for bad in ("aiact:legal", "aiact:recursive-1000,aiact:semantic-90", ":recursive-1000"):
        with pytest.raises(ValueError):
            parse_folder_configs(bad)


# ----------------------------------------------------------------------------- retrievers


def test_tokenize_strips_accents_and_stopwords():
    assert tokenize("La Comisión impondrá las MULTAS") == ["comision", "impondra", "multas"]


def test_bm25_ranks_the_matching_text_first():
    bm25 = BM25Retriever(
        ["plazo de notificación de incidentes", "multas por prácticas prohibidas", "otro texto"]
    )
    assert int(np.argmax(bm25.scores("¿qué multas hay?"))) == 1


def test_vector_store_search_is_cosine_top_k():
    store = VectorStore(["a", "b", "c"], np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32))
    assert [i for i, _ in store.search(np.array([0, 1], dtype=np.float32), 2)] == ["b", "c"]


def test_minmax_handles_constant_scores():
    assert _minmax(np.array([2.0, 2.0])).tolist() == [0.0, 0.0]
    assert _minmax(np.array([1.0, 3.0])).tolist() == [0.0, 1.0]


class FakeEmbedder:
    """Bag-of-words over a tiny vocabulary: enough to make dense retrieval deterministic."""

    vocab = ["multa", "plazo", "emociones", "pymes"]

    def _vec(self, text):
        v = np.array([text.lower().count(w) for w in self.vocab], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def embed_documents(self, texts, cache_name=None):
        return np.stack([self._vec(t) for t in texts])

    def embed_queries(self, queries):
        return self.embed_documents(queries)


@pytest.fixture
def kb(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.retriever.INDEX_DIR", tmp_path)
    docs = [
        Document(
            "a.md", "Sanciones", "La multa máxima es alta. La multa se fija por la autoridad."
        ),
        Document("b.md", "Incidentes", "El plazo de notificación es de quince días."),
        Document("c.md", "Trabajo", "Inferir emociones en el trabajo está prohibido."),
    ]
    return KnowledgeBase.build("recursive-200", FakeEmbedder(), lam=0.5, docs=docs)


def test_per_folder_config_chunks_and_titles_each_folder(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.retriever.INDEX_DIR", tmp_path)
    long_text = " ".join(["La multa se fija por la autoridad competente."] * 20)
    docs = [
        Document("aiact/a.md", "Sanciones", long_text),
        Document("guias/b.md", "Guía", long_text),
    ]
    kb = KnowledgeBase.build(
        "aiact:recursive-200+title,*:recursive-2000", FakeEmbedder(), docs=docs
    )
    by_doc = {d.doc_id: [c for c in kb.chunks if c.doc_id == d.doc_id] for d in docs}
    assert len(by_doc["aiact/a.md"]) > 1 and len(by_doc["guias/b.md"]) == 1
    assert [c.doc_id for c in kb.chunks] == sorted(c.doc_id for c in kb.chunks)  # document order
    assert list(tmp_path.glob("store-*.npz"))[0].name.count(":") == 0


@pytest.mark.parametrize("retriever", ["dense", "bm25", "hybrid"])
def test_three_retrievers_find_the_right_document(kb, retriever):
    top_chunk, _ = kb.search("¿Cuál es el plazo?", retriever, k=1)[0]
    assert top_chunk.doc_id == "b.md"


# ------------------------------------------------------------------------------- gold set


def test_gold_set_has_fifty_answerable_questions_and_some_without_answer():
    gold = load_gold()
    assert sum(g.answerable for g in gold) >= 50
    assert sum(not g.answerable for g in gold) >= 10
    assert len({g.id for g in gold}) == len(gold)


@pytest.mark.parametrize("name", ["gold.jsonl", "calibration.jsonl"])
def test_every_evidence_quote_is_in_the_corpus(name):
    judge = RelevanceJudge()
    for item in load_gold(GOLD_DIR / name):
        for ev in item.evidence:
            judge.span(ev)  # raises if the quote is not found


def test_gold_and_calibration_do_not_share_questions():
    gold = {g.question for g in load_gold()}
    calibration = {g.question for g in load_gold(GOLD_DIR / "calibration.jsonl")}
    assert not gold & calibration


def test_find_span_tolerates_whitespace():
    assert find_span("hasta  35 000\n000 EUR", "35 000 000 EUR") == (7, 21)


def test_relevance_needs_half_of_the_evidence(tmp_path):
    (tmp_path / "d.md").write_text("aaaa EVIDENCIA COMPLETA bbbb")
    judge = RelevanceJudge(tmp_path)
    item = GoldItem("g", "q", "a", (Evidence("d.md", "EVIDENCIA COMPLETA"),))
    start = 5
    assert judge.is_relevant(Chunk("d.md#0", "d.md", "", "", start, start + 18), item)
    assert judge.is_relevant(Chunk("d.md#1", "d.md", "", "", start + 8, start + 30), item)
    assert not judge.is_relevant(Chunk("d.md#2", "d.md", "", "", start + 12, start + 30), item)
    assert not judge.is_relevant(Chunk("x.md#0", "x.md", "", "", start, start + 18), item)


def test_corpus_loads_every_document():
    docs = load_corpus()
    assert len(docs) == 149
    assert all(d.title for d in docs)


# ------------------------------------------------------------------------------ metrics


def test_ranks_and_metrics():
    scores = np.array([[0.9, 0.1, 0.5], [0.1, 0.2, 0.3]])
    ranks = ranks_of_first_relevant(scores, [np.array([2]), np.array([], dtype=int)])
    assert ranks[0] == 2 and np.isinf(ranks[1])
    m = metrics(ranks)
    assert m["recall@1"] == 0.0 and m["recall@3"] == 0.5 and m["mrr@10"] == pytest.approx(0.25)


def test_choose_threshold_separates_the_two_groups():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    answerable = np.array([True, True, True, False, False])
    threshold = choose_threshold(scores, answerable)
    assert 0.2 < threshold <= 0.7
    assert gate_metrics(scores, answerable, threshold)["balanced_accuracy"] == 1.0


# ---------------------------------------------------------------------------- generation


def test_extract_citations_valid_and_invalid():
    valid, invalid = extract_citations("Sí [1][3]. También [2, 7] y [Fragmento 1].", n_fragments=3)
    assert valid == [1, 3, 2] and invalid == [7]


def test_is_abstention():
    assert is_abstention("No lo sé: el corpus no lo dice.")
    assert is_abstention("«No lo sé»")
    assert not is_abstention("La multa es de 35 M€ [1].")


def test_prompt_numbers_the_fragments():
    chunks = [
        Chunk("a#0", "a.md", "Título A", "texto A", 0, 7),
        Chunk("b#0", "b.md", "Título B", "texto B", 0, 7),
    ]
    user = build_prompt("¿pregunta?", chunks)[1]["content"]
    assert (
        "[1] (a.md · Título A)\ntexto A" in user
        and "[2] (b.md" in user
        and user.endswith("¿pregunta?")
    )


class FakeGenerator:
    model_name = "fake"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def chat(self, messages):
        self.calls += 1
        return self.reply


class FakeReranker:
    def __init__(self, score):
        self.score = score

    def scores(self, query, passages):
        return [self.score] * len(passages)


def test_pipeline_maps_citations_to_chunk_ids(kb):
    pipeline = RagPipeline(kb, FakeGenerator("Quince días [1]. Ver también [9]."))
    out = pipeline.answer("¿Cuál es el plazo?", top_k=2, retriever="hybrid")
    assert out["citations"] == [out["chunks"][0]["id"]]
    assert out["invalid_citations"] == ["[9]"]
    assert not out["abstained"]


def test_gate_abstains_without_calling_the_generator(kb):
    generator = FakeGenerator("no debería llamarse")
    pipeline = RagPipeline(kb, generator, FakeReranker(0.1), rerank_pool=3, min_score=0.5)
    out = pipeline.answer("¿Capital de Portugal?", top_k=2)
    assert out["answer"] == NOT_FOUND and out["abstained"] and out["citations"] == []
    assert generator.calls == 0 and len(out["chunks"]) == 2


def test_model_abstention_drops_citations(kb):
    out = RagPipeline(kb, FakeGenerator("No lo sé [1].")).answer("x", top_k=1)
    assert out["abstained"] and out["citations"] == []


def test_response_matches_api_contract(kb):
    from api.schemas import RagResponse

    out = RagPipeline(kb, FakeGenerator("Quince días [1].")).answer(
        "plazo", top_k=2, retriever="bm25"
    )
    response = RagResponse(**out)
    assert response.retriever == "bm25" and len(response.chunks) == 2
