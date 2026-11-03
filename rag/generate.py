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
Cada afirmación debe apoyarse en un fragmento citado con su id entre corchetes, por ejemplo \
[{{example_id}}]; no cites ids que no aparezcan arriba.
Usa solo la información recuperada, no lo que sepas por tu cuenta.
Si los fragmentos no responden a la pregunta, contesta exactamente: «{NOT_FOUND}»"""

DEFAULT_CHUNKING = "semantic-0.70+title"
DEFAULT_LAMBDA = "0.3"

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def format_chunks(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.id}] (from {hit.chunk.source})\n{hit.chunk.text}" for hit in hits
    )


def build_prompt(question: str, hits: list[Hit]) -> str:
    example_id = hits[0].chunk.id if hits else "chunk-id"
    prompt = RAG_PROMPT_TEMPLATE.format(
        example_id=example_id, chunks=format_chunks(hits), question=question
    )
    return f"{prompt}\n{DOMAIN_RULES.format(example_id=example_id)}"


def says_not_found(answer: str) -> bool:
    return answer.strip().strip('«»"').lower().startswith("no lo sé")


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match in valid_ids and match not in seen:
            seen.append(match)
    return seen


def to_response(answer: str, hits: list[Hit], retriever: str, model: str = "") -> RagResponse:
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
    the phase-2 model ``Qwen/Qwen3-4B``). The defaults are the ones the gold set chose
    (``docs/EXPERIMENTS.md``).
    """

    def __init__(
        self,
        chunking: str | None = None,
        weight_bm25: float | None = None,
        model_id: str | None = None,
        max_new_tokens: int = 512,
    ):
        self.chunking = chunking or os.environ.get("ARCA_RAG_CHUNKING", DEFAULT_CHUNKING)
        self.weight_bm25 = (
            float(os.environ.get("ARCA_RAG_LAMBDA", DEFAULT_LAMBDA))
            if weight_bm25 is None
            else weight_bm25
        )
        self.model_id = model_id or os.environ.get("ARCA_RAG_MODEL", "Qwen/Qwen3-4B")
        self.max_new_tokens = max_new_tokens
        self.retrievers: dict = {}

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
        if generator:
            from rlm.gpu import shared_causal_lm

            self.tokenizer, self.model = shared_causal_lm(self.model_id)

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        return self.retrievers[retriever].search(question, top_k=top_k)

    def generate(self, prompt: str) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return raw.split("</think>", 1)[-1].strip()

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        hits = self.retrieve(question, top_k, retriever)
        answer = self.generate(build_prompt(question, hits))
        return to_response(answer, hits, retriever, self.model_id)


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
