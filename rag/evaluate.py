"""Retrieval evaluation: Recall@k and MRR for every chunking × retriever, and the λ sweep.

    uv run python -m rag.evaluate                       # all chunkings, writes reports/
    uv run python -m rag.evaluate --configs semantic-90+title --no-instruction

For each chunking config the dense and BM25 scores of every gold question are computed
once; the hybrid score for any λ is then a cheap combination of the two. λ is chosen on
the gold set, and a 5-fold cross-validation reports what that choice is worth on
questions it did not see (``lambda_cv_mrr``), so the selection is not overfitted to the
same 60 questions it is judged on.

Only answerable questions count here; the unanswerable ones are for the «No lo sé» gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag.chunking import chunk_stats
from rag.embed import Embedder
from rag.goldset import GOLD_DIR, RelevanceJudge, load_gold
from rag.ingest import load_corpus
from rag.retriever import KnowledgeBase, _minmax

REPORTS = Path(__file__).resolve().parent.parent / "reports"
DEFAULT_CONFIGS = [
    "recursive-500",
    "recursive-1000",
    "recursive-1500",
    "semantic-80",
    "semantic-90",
    "semantic-95",
    "recursive-1000+title",
    "semantic-90+title",
]
KS = (1, 3, 5, 10)
LAMBDAS = np.round(np.linspace(0, 1, 11), 2)


def ranks_of_first_relevant(scores: np.ndarray, relevant: list[np.ndarray]) -> np.ndarray:
    """1-based rank of the first relevant chunk for each question (inf if none)."""
    ranks = np.full(len(scores), np.inf)
    for q, (row, rel) in enumerate(zip(scores, relevant, strict=True)):
        if rel.size:
            best = row[rel].max()
            ranks[q] = 1 + int((row > best).sum())
    return ranks


def metrics(ranks: np.ndarray) -> dict[str, float]:
    out = {f"recall@{k}": float(np.mean(ranks <= k)) for k in KS}
    out["mrr@10"] = float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0)))
    return out


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hybrid_scores(dense: np.ndarray, bm25: np.ndarray, lam: float) -> np.ndarray:
    return np.stack(
        [lam * _minmax(d) + (1 - lam) * _minmax(b) for d, b in zip(dense, bm25, strict=True)]
    )


def mrr_of(ranks: np.ndarray) -> float:
    return metrics(ranks)["mrr@10"]


def cross_validated_lambda(dense, bm25, relevant, folds: int = 5, seed: int = 0) -> dict:
    """Choose λ on 4/5 of the questions, score it on the other 1/5."""
    n = len(relevant)
    order = np.random.default_rng(seed).permutation(n)
    ranks_by_lam = {
        lam: ranks_of_first_relevant(hybrid_scores(dense, bm25, lam), relevant) for lam in LAMBDAS
    }
    held_out = np.empty(n)
    chosen = []
    for fold in np.array_split(order, folds):
        train = np.setdiff1d(order, fold)
        best = max(LAMBDAS, key=lambda lam: mrr_of(ranks_by_lam[lam][train]))
        chosen.append(float(best))
        held_out[fold] = ranks_by_lam[best][fold]
    return {"lambda_per_fold": chosen, "lambda_cv_mrr": mrr_of(held_out)}


def evaluate_config(config, embedder, docs, gold, judge) -> dict:
    kb = KnowledgeBase.build(config, embedder, docs=docs)
    questions = [g.question for g in gold]
    query_vectors = embedder.embed_queries(questions)
    dense = np.stack([kb.dense.store.scores(v) for v in query_vectors])
    bm25 = np.stack([kb.bm25.scores(q) for q in questions])
    index = {c.id: i for i, c in enumerate(kb.chunks)}
    relevant = [
        np.array(sorted(index[i] for i in judge.relevant_ids(kb.chunks, g)), dtype=int)
        for g in gold
    ]
    missing = [g.id for g, r in zip(gold, relevant, strict=True) if not r.size]

    sweep = {}
    for lam in LAMBDAS:
        sweep[float(lam)] = metrics(
            ranks_of_first_relevant(hybrid_scores(dense, bm25, lam), relevant)
        )
    best_lam = max(sweep, key=lambda lam: (sweep[lam]["mrr@10"], sweep[lam]["recall@5"]))

    result = {"config": config, "stats": chunk_stats(kb.chunks), "no_relevant_chunk": missing}
    per_question = {}
    for name, scores in (
        ("dense", dense),
        ("bm25", bm25),
        ("hybrid", hybrid_scores(dense, bm25, best_lam)),
    ):
        ranks = ranks_of_first_relevant(scores, relevant)
        result[name] = metrics(ranks)
        result[name]["recall@5_ci95"] = bootstrap_ci((ranks <= 5).astype(float))
        per_question[name] = {
            g.id: (None if np.isinf(r) else int(r)) for g, r in zip(gold, ranks, strict=True)
        }
    result["best_lambda"] = best_lam
    result["lambda_sweep"] = sweep
    result.update(cross_validated_lambda(dense, bm25, relevant))
    result["ranks"] = per_question
    return result


def print_table(results: list[dict]) -> None:
    print("\n| Chunking | Chunks | Media (car.) | Retriever | R@1 | R@3 | R@5 | R@10 | MRR@10 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        for name in ("dense", "bm25", "hybrid"):
            m = r[name]
            label = f"hybrid λ={r['best_lambda']}" if name == "hybrid" else name
            print(
                f"| {r['config']} | {r['stats']['chunks']} "
                f"| {r['stats']['mean_chars']:.0f} | {label} "
                f"| {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['recall@5']:.2f} "
                f"| {m['recall@10']:.2f} | {m['mrr@10']:.3f} |"
            )


def plot_sweep(results: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in results:
        xs = list(r["lambda_sweep"])
        ax.plot(
            xs, [r["lambda_sweep"][x]["mrr@10"] for x in xs], marker="o", ms=3, label=r["config"]
        )
    ax.set_xlabel("λ  (0 = solo BM25, 1 = solo denso)")
    ax.set_ylabel("MRR@10 en el conjunto dorado")
    ax.set_title("Barrido de λ del retriever híbrido")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--gold", default=str(GOLD_DIR / "gold.jsonl"))
    parser.add_argument(
        "--no-instruction", action="store_true", help="embed queries without instruction"
    )
    parser.add_argument("--out", default=str(REPORTS / "phase3_retrieval.json"))
    args = parser.parse_args()

    gold = [g for g in load_gold(args.gold) if g.answerable]
    embedder = Embedder(instruction=None if args.no_instruction else Embedder().instruction)
    docs = load_corpus()
    judge = RelevanceJudge()
    results = [evaluate_config(c, embedder, docs, gold, judge) for c in args.configs]
    for r in results:
        if r["no_relevant_chunk"]:
            print(
                f"warning: {r['config']}: no chunk covers the evidence of {r['no_relevant_chunk']}"
            )
        print(
            f"{r['config']}: λ*={r['best_lambda']}  folds={r['lambda_per_fold']}  "
            f"CV MRR={r['lambda_cv_mrr']:.3f}"
        )
    print_table(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"instruction": embedder.instruction, "results": results}, indent=1, ensure_ascii=False
        )
    )
    if not args.no_instruction:
        plot_sweep(results, out.with_name("phase3_lambda_sweep.png"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
