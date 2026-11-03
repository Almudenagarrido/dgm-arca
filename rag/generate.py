"""Retrieval and generation: augment the prompt with the retrieved chunks and answer with citations.

``answer_with_rag`` is what ``POST /rag`` calls:

1. retrieve ``top_k`` chunks with the requested retriever (dense, bm25 or hybrid);
2. build the augmented prompt (the template from class, with chunk ids, plus our rules);
3. generate the answer (Qwen3-4B, greedy, no thinking);
4. extract the citations that exist among the returned chunks and return a ``RagResponse``.

``rag/evaluate_answers.py`` checks the answers on the gold set: cited ids exist, they are the
relevant chunks, the key fact is there, and unanswerable questions get ``NOT_FOUND``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from api.schemas import RagResponse, RetrievedChunk
from rag.retriever import Hit

RAG_PROMPT_TEMPLATE = """Use the provided up-to-date information to answer the question.
Cite the passages you rely on by writing their id in square brackets, like [{example_id}].
If the information is not enough to answer, say so instead of guessing.

Retrieved information:
{chunks}

Question: {question}
"""

# What the generator actually receives: the template from class plus the rules our domain
# needs (Spanish, every sentence backed by a citation, a fixed sentence when the passages do
# not answer, so that "I don't know" can be detected automatically).
NOT_FOUND = "No lo sé: los documentos recuperados no contienen esa información."
DOMAIN_RULES = f"""Responde en español, en dos a cinco frases.
Cada afirmación debe apoyarse en un fragmento citado con su etiqueta entre corchetes, por \
ejemplo [{{example_id}}]; usa solo las etiquetas de los fragmentos de arriba.
Usa solo la información recuperada, no lo que sepas por tu cuenta. Lee con atención las \
negaciones y las excepciones («no se aplicará», «salvo», «siempre se considerarán»): \
la respuesta tiene que decir lo mismo que el fragmento, no lo contrario.
Si los fragmentos no responden exactamente a lo que se pregunta, no completes la respuesta \
con información relacionada: contesta solo, exactamente: «{NOT_FOUND}»"""

DEFAULT_CHUNKING = "semantic-0.70+title"
DEFAULT_LAMBDA = "0.3"
DEFAULT_RERANK_POOL = "30"
DEFAULT_TRANSLATE = "1"
# Reranker probability below which the pipeline answers NOT_FOUND without generating. On the gold
# set the lowest answerable question scores 0.990 and four of the five unanswerable ones 0.960
# or less (docs/EXPERIMENTS.md).
DEFAULT_MIN_SCORE = "0.975"
FUSION_POOL = 75


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")


LABEL_PATTERN = re.compile(r"\[F(\d+)\]")


def label(position: int) -> str:
    """Short citation label for the chunk at ``position`` (1-based) in the prompt."""
    return f"F{position}"


def format_chunks(hits: list[Hit]) -> str:
    # The model cites short labels (F1, F2...) instead of the chunk ids: with ids such as
    # "a042e32725-0009" in front of it, it sometimes invented the neighbour "-0010".
    return "\n\n".join(
        f"[{label(i)}] (from {hit.chunk.source})\n{hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )


def build_prompt(question: str, hits: list[Hit]) -> str:
    example_id = label(1)
    prompt = RAG_PROMPT_TEMPLATE.format(
        example_id=example_id, chunks=format_chunks(hits), question=question
    )
    return f"{prompt}\n{DOMAIN_RULES.format(example_id=example_id)}"


def resolve_labels(answer: str, hits: list[Hit]) -> tuple[str, list[str]]:
    """Replace [F<n>] with the real chunk id; drop labels that point to no chunk.

    Returns the answer with ids, as the API contract expects, and the invalid labels found.
    """
    invalid: list[str] = []

    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(hits):
            return f"[{hits[n - 1].chunk.id}]"
        invalid.append(match.group(0))
        return ""

    return LABEL_PATTERN.sub(replace, answer).replace(" .", "."), invalid


def says_not_found(answer: str) -> bool:
    return answer.strip().strip('«»"').lower().startswith("no lo sé")


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match in valid_ids and match not in seen:
            seen.append(match)
    return seen


def to_response(
    answer: str,
    hits: list[Hit],
    retriever: str,
    model: str = "",
    invalid_citations: list[str] | None = None,
) -> RagResponse:
    chunks = [
        RetrievedChunk(id=h.chunk.id, source=h.chunk.source, score=h.score, text=h.chunk.text)
        for h in hits
    ]
    return RagResponse(
        answer=answer,
        chunks=chunks,
        citations=extract_citations(answer, {c.id for c in chunks}),
        retriever=retriever,
        model=model,
        invalid_citations=invalid_citations or [],
    )


def build_chunks(documents, chunking: str, embedder):
    """``recursive``, ``semantic-<threshold>``, either with an optional ``+title`` suffix."""
    from rag.chunking import recursive_chunks, semantic_chunks, with_titles

    base, _, suffix = chunking.partition("+")
    if base == "recursive":
        chunks = recursive_chunks(documents)
    elif base.startswith("semantic-"):
        chunks = semantic_chunks(
            documents,
            lambda s: embedder.embed_passages(s, batch_size=64),
            distance_threshold=float(base.removeprefix("semantic-")),
        )
    else:
        raise ValueError(f"unknown chunking {chunking!r}")
    return with_titles(chunks, documents) if suffix == "title" else chunks


def load_or_build_chunks(documents, chunking: str, embedder, index_dir: str = "rag/index"):
    """Chunks cached in ``rag/index/chunks-<chunking>.json``, keyed by a hash of the corpus.

    Semantic chunking embeds every sentence (~3 min for this corpus); with the cache the API
    starts in seconds. Any change in the documents or the embedding model rebuilds it.
    """
    from rag.chunking import Chunk

    digest = hashlib.sha1(embedder.model_id.encode())
    for doc in documents:
        digest.update(doc.id.encode())
        digest.update(doc.text.encode())
    path = Path(index_dir) / f"chunks-{chunking}.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if stored.get("corpus") == digest.hexdigest():
            return [Chunk(**c) for c in stored["chunks"]]
    chunks = build_chunks(documents, chunking, embedder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"corpus": digest.hexdigest(), "chunks": [c.__dict__ for c in chunks]},
            ensure_ascii=False,
        )
    )
    return chunks


class RagPipeline:
    """Everything ``POST /rag`` needs, built once and reused.

    Configuration (all optional): ``ARCA_RAG_CHUNKING`` (``recursive`` or ``semantic-<t>``,
    with an optional ``+title`` suffix),
    ``ARCA_RAG_LAMBDA`` (BM25 weight of the hybrid), ``ARCA_RAG_MODEL`` (generator, default
    the phase-2 model ``Qwen/Qwen3-4B``), ``ARCA_RAG_RERANK`` (``1``: rerank the first-stage
    candidates with Qwen3-Reranker), ``ARCA_RAG_RERANK_POOL`` (how many candidates),
    ``ARCA_RAG_TRANSLATE`` (``1``: also search with an English translation of the question),
    ``ARCA_RAG_THINKING`` (``1``: Qwen3 reasons before answering), ``ARCA_RAG_MIN_SCORE``
    (reranker score under which the answer is ``NOT_FOUND``).
    The defaults are the ones the gold set chose (``docs/EXPERIMENTS.md``).
    """

    def __init__(
        self,
        chunking: str | None = None,
        weight_bm25: float | None = None,
        model_id: str | None = None,
        max_new_tokens: int = 512,
        rerank: bool | None = None,
        translate_query: bool | None = None,
        thinking: bool | None = None,
    ):
        self.chunking = chunking or os.environ.get("ARCA_RAG_CHUNKING", DEFAULT_CHUNKING)
        self.weight_bm25 = (
            float(os.environ.get("ARCA_RAG_LAMBDA", DEFAULT_LAMBDA))
            if weight_bm25 is None
            else weight_bm25
        )
        self.model_id = model_id or os.environ.get("ARCA_RAG_MODEL", "Qwen/Qwen3-4B")
        self.max_new_tokens = max_new_tokens
        self.rerank = rerank if rerank is not None else _flag("ARCA_RAG_RERANK", "1")
        self.rerank_pool = int(os.environ.get("ARCA_RAG_RERANK_POOL", DEFAULT_RERANK_POOL))
        self.translate_query = (
            translate_query
            if translate_query is not None
            else _flag("ARCA_RAG_TRANSLATE", DEFAULT_TRANSLATE)
        )
        self.thinking = thinking if thinking is not None else _flag("ARCA_RAG_THINKING", "0")
        self.min_score = float(os.environ.get("ARCA_RAG_MIN_SCORE", DEFAULT_MIN_SCORE))
        self.retrievers: dict = {}
        self.reranker = None

    def load(self, generator: bool = True) -> None:
        """Ingest -> chunk -> the three retrievers (dense index cached in rag/index/) -> model."""
        from rag.embed import Embedder
        from rag.ingest import load_corpus
        from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

        documents = load_corpus()
        if not documents:
            raise NotImplementedError(
                "rag/corpus/ is empty: run `uv run python -m rag.download_corpus` first."
            )
        embedder = Embedder()
        chunks = load_or_build_chunks(documents, self.chunking, embedder)
        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, embedder, index_name=f"dense-{self.chunking}")
        self.retrievers = {
            "bm25": bm25,
            "dense": dense,
            "hybrid": HybridRetriever(bm25, dense, weight_bm25=self.weight_bm25),
        }
        if self.rerank:
            from rag.retriever import Reranker

            self.reranker = Reranker()
            # As in the experiment: each list contributes FUSION_POOL hits before the hybrid
            # fusion, so the min-max rescaling and the top of the merged list are the same.
            self.retrievers["hybrid"].pool = FUSION_POOL
        if generator or self.translate_query:
            from rlm.gpu import shared_causal_lm

            self.tokenizer, self.model = shared_causal_lm(self.model_id)

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        """First stage (the requested retriever), optional English query, then the reranker."""
        first_stage = self.retrievers[retriever]
        if not self.rerank:
            return first_stage.search(question, top_k=top_k)
        candidates = first_stage.search(question, top_k=self.rerank_pool)
        if self.translate_query:
            from rag.retriever import interleave

            english = first_stage.search(self.translate(question), top_k=self.rerank_pool)
            candidates = interleave(candidates, english)[: self.rerank_pool]
        scores = self.reranker.score(question, [h.chunk.text for h in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(candidates[i].chunk, scores[i]) for i in order]

    def generate(self, prompt: str, thinking: bool | None = None) -> str:
        """Greedy answer; with ``thinking`` Qwen3 reasons first (the reasoning is dropped)."""
        import torch

        thinking = self.thinking if thinking is None else thinking
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens * (4 if thinking else 1),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return raw.split("</think>", 1)[-1].strip()

    def translate(self, question: str) -> str:
        """English version of the question, to search the English-only documents too."""
        prompt = (
            "Translate this question into English. Do not answer it; write only the "
            f"translation.\n\n{question}"
        )
        saved, self.max_new_tokens = self.max_new_tokens, 96
        try:
            return self.generate(prompt, thinking=False).strip().strip('"')
        finally:
            self.max_new_tokens = saved

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        hits = self.retrieve(question, top_k, retriever)
        if self.rerank and hits and hits[0].score < self.min_score:
            # Not even the best candidate answers the question according to the reranker:
            # say so instead of letting the generator fill the gap with related text.
            return to_response(NOT_FOUND, hits, retriever, self.model_id)
        answer, invalid = resolve_labels(self.generate(build_prompt(question, hits)), hits)
        return to_response(answer, hits, retriever, self.model_id, invalid)


_pipeline: RagPipeline | None = None


def build_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        pipeline = RagPipeline()
        pipeline.load()
        _pipeline = pipeline
    return _pipeline


def answer_with_rag(question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_pipeline().answer(question, top_k=top_k, retriever=retriever)
