"""End-to-end evaluation of the answers on the gold set, with an LLM judge.

    uv run python -m rag.evaluate_answers                 # the three variants below
    uv run python -m rag.evaluate_answers --variants full

Variants (same chunking, λ and generator):

* ``hybrid``: hybrid top-5 straight to the generator, no reranker, no gate.
* ``rerank``: hybrid pool of 30 → reranker → top-5, no gate.
* ``full``: ``rerank`` + the «No lo sé» gate with the calibrated threshold.

Metrics on answerable questions:

* ``context_hit``: a fragment covering the gold evidence reached the prompt.
* ``correct``: the judge says the answer agrees with the reference answer.
* ``faithful``: the judge says every claim is supported by the fragments it cites.
* ``citations_valid``: no ``[n]`` outside the fragments given (checked by code, no judge).
* ``cites_evidence``: one of the cited fragments covers the gold evidence (by code).
* ``false_abstention``: answered «No lo sé» although the corpus has the answer.

On unanswerable questions: ``abstained``.

The judge defaults to the generator itself (Qwen3-4B, same instance, no extra memory): the
16 GB GPU partition does not fit Qwen3-4B and Qwen3-8B together. A model grading its own
answers can be lenient, so ``--judge`` takes another model when there is room, and the
verdicts were checked by hand on a sample (see rag/README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from rag.embed import Embedder
from rag.evaluate import REPORTS, bootstrap_ci
from rag.generate import DEFAULT_MODEL, Generator, RagPipeline
from rag.goldset import GOLD_DIR, RelevanceJudge, load_gold
from rag.rerank import Reranker
from rag.retriever import KnowledgeBase

JUDGE_MODEL = DEFAULT_MODEL

CORRECTNESS_PROMPT = """Eres un evaluador estricto.
Compara la RESPUESTA con la RESPUESTA DE REFERENCIA a la PREGUNTA.
La respuesta es correcta si contiene la información esencial de la referencia y no la contradice.
Puede añadir detalles si no contradicen la referencia. Una respuesta «No lo sé» es incorrecta.

PREGUNTA: {question}
RESPUESTA DE REFERENCIA: {reference}
RESPUESTA: {answer}

Contesta solo con una palabra: CORRECTA o INCORRECTA."""

FAITHFULNESS_PROMPT = """Eres un evaluador estricto.
Decide si todas las afirmaciones de la RESPUESTA están respaldadas por los FRAGMENTOS CITADOS.
Ignora si la respuesta es completa; solo importa que no afirme nada que los fragmentos no digan.

FRAGMENTOS CITADOS:
{fragments}

RESPUESTA: {answer}

Contesta solo con una palabra: FIEL o INFIEL."""


def judge_verdict(judge: Generator, prompt: str, positive: str) -> bool:
    reply = judge.chat([{"role": "user", "content": prompt}]).upper()
    words = re.findall(r"[A-ZÁÉÍÓÚ]+", reply)
    return bool(words) and words[0] == positive


def evaluate_variant(name, pipeline, gold, judge, relevance) -> tuple[dict, list[dict]]:
    rows = []
    for item in gold:
        out = pipeline.answer(item.question, top_k=5, retriever="hybrid")
        by_id = pipeline.kb.by_id
        row = {
            "id": item.id,
            "answerable": item.answerable,
            "question": item.question,
            "answer": out["answer"],
            "abstained": out["abstained"],
            "gate_score": out["gate_score"],
            "citations": out["citations"],
            "invalid_citations": out["invalid_citations"],
            "retrieved": [c["id"] for c in out["chunks"]],
        }
        if item.answerable:
            row["context_hit"] = any(
                relevance.is_relevant(by_id[i], item) for i in row["retrieved"]
            )
            row["cites_evidence"] = any(
                relevance.is_relevant(by_id[i], item) for i in row["citations"]
            )
            row["citations_valid"] = not row["invalid_citations"]
            row["false_abstention"] = out["abstained"]
            if out["abstained"]:
                row["correct"] = row["faithful"] = False
            else:
                row["correct"] = judge_verdict(
                    judge,
                    CORRECTNESS_PROMPT.format(
                        question=item.question, reference=item.answer, answer=out["answer"]
                    ),
                    "CORRECTA",
                )
                fragments = (
                    "\n\n".join(f"[{i}] {by_id[i].text}" for i in row["citations"]) or "(ninguno)"
                )
                row["faithful"] = judge_verdict(
                    judge,
                    FAITHFULNESS_PROMPT.format(fragments=fragments, answer=out["answer"]),
                    "FIEL",
                )
        rows.append(row)
        print(
            f"[{name}] {item.id} abst={row['abstained']} correct={row.get('correct')} "
            f"cites={row['citations'][:2]}"
        )

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {}
    for key in (
        "context_hit",
        "correct",
        "faithful",
        "citations_valid",
        "cites_evidence",
        "false_abstention",
    ):
        values = np.array([float(r[key]) for r in answerable])
        summary[key] = float(values.mean())
        summary[f"{key}_ci95"] = bootstrap_ci(values)
    answered = [r for r in answerable if not r["abstained"]]
    summary["faithful_when_answered"] = (
        float(np.mean([r["faithful"] for r in answered])) if answered else 0.0
    )
    summary["abstained_unanswerable"] = float(np.mean([r["abstained"] for r in unanswerable]))
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants", nargs="+", default=["hybrid", "rerank", "full"])
    parser.add_argument(
        "--config", default=os.environ.get("ARCA_RAG_CHUNKING", "recursive-1000+title")
    )
    parser.add_argument(
        "--lam", type=float, default=float(os.environ.get("ARCA_RAG_LAMBDA", "0.7"))
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="gate threshold (default: the one in reports/phase3_gate.json)",
    )
    parser.add_argument("--model", default=os.environ.get("ARCA_RAG_MODEL", DEFAULT_MODEL))
    parser.add_argument("--judge", default=JUDGE_MODEL)
    parser.add_argument("--out", default=str(REPORTS / "phase3_answers.json"))
    args = parser.parse_args()

    min_score = args.min_score
    if min_score is None and "full" in args.variants:
        min_score = json.loads((REPORTS / "phase3_gate.json").read_text())["rerank"]["threshold"]

    kb = KnowledgeBase.build(args.config, Embedder(), lam=args.lam)
    generator = Generator(args.model)
    reranker = Reranker()
    judge = generator if args.judge == args.model else Generator(args.judge, max_new_tokens=8)
    relevance = RelevanceJudge()
    gold = load_gold(GOLD_DIR / "gold.jsonl")

    pipelines = {
        "hybrid": RagPipeline(kb, generator, None),
        "rerank": RagPipeline(kb, generator, reranker, rerank_pool=30),
        "full": RagPipeline(kb, generator, reranker, rerank_pool=30, min_score=min_score),
    }
    report = {
        "config": args.config,
        "lambda": args.lam,
        "model": args.model,
        "judge": args.judge,
        "min_score": min_score,
        "variants": {},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        summary, rows = evaluate_variant(name, pipelines[name], gold, judge, relevance)
        report["variants"][name] = summary
        with out.with_name(f"phase3_answers_{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        "\n| Variante | Contexto | Correcta | Fiel | Citas válidas | Cita la evidencia "
        "| Abst. falsa | Abst. sin respuesta |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, s in report["variants"].items():
        print(
            f"| {name} | {s['context_hit']:.0%} | {s['correct']:.0%} | {s['faithful']:.0%} "
            f"| {s['citations_valid']:.0%} "
            f"| {s['cites_evidence']:.0%} | {s['false_abstention']:.0%} "
            f"| {s['abstained_unanswerable']:.0%} |"
        )
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
