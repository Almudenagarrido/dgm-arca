"""Phase 3 pipeline behind ``POST /rag``: retrieve, rerank, gate, generate with citations.

1. **Retrieve** a pool of candidates with the requested retriever (dense, bm25, hybrid).
2. **Rerank** the pool with Qwen3-Reranker and keep the best ``top_k`` (optional).
3. **Gate**: if even the best reranker score is below ``min_score`` (calibrated on a
   separate set by ``rag.calibrate``), answer «No lo sé» without calling the generator.
4. **Generate**: Qwen3-4B gets the fragments numbered ``[1]..[k]`` and must cite them.
5. **Check citations**: every ``[n]`` in the answer is mapped to a chunk id; numbers that
   do not correspond to a retrieved fragment are reported in ``invalid_citations``
   and never reach ``citations``.

The ``[n]`` in ``answer`` refer to the n-th element of ``chunks`` in the response.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from rag.chunking import Chunk
from rag.embed import Embedder
from rag.rerank import Reranker
from rag.retriever import RETRIEVERS, KnowledgeBase

DEFAULT_MODEL = "Qwen/Qwen3-4B"
NOT_FOUND = "No lo sé: el corpus no contiene información suficiente para responder a esta pregunta."

SYSTEM_PROMPT = """Eres un asistente experto en el Reglamento (UE) 2024/1689 de
Inteligencia Artificial (AI Act). Respondes solo con la información de los fragmentos que se
te dan, nunca con lo que sepas por tu cuenta.

Reglas:
- Cita cada afirmación con el número del fragmento entre corchetes, por ejemplo [2] o [1][3].
- Cita solo fragmentos que realmente contengan lo que afirmas.
- Si los fragmentos no contienen la respuesta, responde exactamente: «No lo sé»
  seguido de una frase breve.
- Si te apoyas en un borrador de directrices (documentos draft_*), indica que es un borrador.
- Responde en español, en dos a cuatro frases, aunque el fragmento esté en inglés."""

_CITATION = re.compile(r"\[(?:fragmento\s*)?(\d+(?:\s*(?:,|;|y|\]\[)\s*\d+)*)\]", re.IGNORECASE)


def extract_citations(answer: str, n_fragments: int) -> tuple[list[int], list[int]]:
    """(valid, invalid) fragment numbers cited in ``answer``, in order of first appearance."""
    valid: list[int] = []
    invalid: list[int] = []
    for match in _CITATION.finditer(answer):
        for number in map(int, re.findall(r"\d+", match.group(1))):
            target = valid if 1 <= number <= n_fragments else invalid
            if number not in target:
                target.append(number)
    return valid, invalid


def is_abstention(answer: str) -> bool:
    return answer.strip().lstrip("«\"'*").lower().startswith("no lo sé")


def build_prompt(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    fragments = "\n\n".join(
        f"[{i}] ({c.doc_id} · {c.title})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )
    user = f"Fragmentos:\n\n{fragments}\n\nPregunta: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class Generator:
    """Greedy chat generation with a Hugging Face causal LM (Qwen3, thinking off by default)."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL, thinking: bool = False, max_new_tokens: int = 400
    ):
        self.model_name = model_name
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens
        self._model = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self._model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def chat(self, messages: list[dict[str, str]]) -> str:
        import torch

        if self._model is None:
            self._load()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        text = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@dataclass
class Retrieved:
    chunk: Chunk
    retrieval_score: float
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.retrieval_score if self.rerank_score is None else self.rerank_score


class RagPipeline:
    def __init__(
        self,
        kb: KnowledgeBase,
        generator: Generator,
        reranker: Reranker | None = None,
        rerank_pool: int = 30,
        min_score: float | None = None,
    ) -> None:
        self.kb = kb
        self.generator = generator
        self.reranker = reranker
        self.rerank_pool = rerank_pool
        self.min_score = min_score if reranker is not None else None

    @classmethod
    def from_env(cls) -> RagPipeline:
        env = os.environ.get
        kb = KnowledgeBase.build(
            env("ARCA_RAG_CHUNKING", "recursive-1000+title"),
            Embedder(),
            lam=float(env("ARCA_RAG_LAMBDA", "0.7")),
        )
        generator = Generator(
            env("ARCA_RAG_MODEL", DEFAULT_MODEL), thinking=env("ARCA_RAG_THINKING", "0") == "1"
        )
        reranker = Reranker() if env("ARCA_RAG_RERANK", "1") == "1" else None
        min_score = env("ARCA_RAG_MIN_SCORE", "")
        return cls(
            kb,
            generator,
            reranker,
            int(env("ARCA_RAG_RERANK_POOL", "30")),
            float(min_score) if min_score else None,
        )

    def retrieve(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> list[Retrieved]:
        if retriever not in RETRIEVERS:
            raise ValueError(f"retriever must be one of {RETRIEVERS}")
        pool = max(top_k, self.rerank_pool) if self.reranker else top_k
        hits = [Retrieved(c, s) for c, s in self.kb.search(question, retriever, pool)]
        if self.reranker:
            for hit, score in zip(
                hits, self.reranker.scores(question, [h.chunk.text for h in hits]), strict=True
            ):
                hit.rerank_score = score
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
        return hits[:top_k]

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> dict:
        hits = self.retrieve(question, top_k, retriever)
        chunks = [h.chunk for h in hits]
        best = max((h.score for h in hits), default=0.0)
        gated = self.min_score is not None and best < self.min_score

        answer = NOT_FOUND if gated else self.generator.chat(build_prompt(question, chunks))
        abstained = gated or is_abstention(answer)
        valid, invalid = ([], []) if abstained else extract_citations(answer, len(chunks))
        return {
            "answer": answer,
            "chunks": [
                {
                    "id": h.chunk.id,
                    "source": h.chunk.doc_id,
                    "score": round(h.score, 4),
                    "text": h.chunk.text,
                }
                for h in hits
            ],
            "citations": [chunks[n - 1].id for n in valid],
            "retriever": retriever,
            "model": self.generator.model_name,
            "abstained": abstained,
            "gate_score": round(best, 4),
            "invalid_citations": [f"[{n}]" for n in invalid],
        }


def build_pipeline() -> RagPipeline:
    """Entry point used by ``api/backends.py``."""
    return RagPipeline.from_env()
