"""Calibrate the «No lo sé» gate on a set that is *not* the gold set.

    uv run python -m rag.calibrate --config recursive-1000+title --lam 0.7

The gate abstains when the best reranker score among the retrieved fragments is below a
threshold. The threshold is picked on ``rag/gold/calibration.jsonl`` (15 answerable + 15
unanswerable questions written for this purpose) by maximising balanced accuracy, the mean
of «answers when it should» and «abstains when it should». Then it is applied, unchanged,
to the gold set to report what it does on questions it was not tuned on.

The same is done with the best first-stage score (dense cosine) as a baseline, to see
whether the cross-encoder is what makes the gate work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag.embed import Embedder
from rag.evaluate import REPORTS
from rag.goldset import GOLD_DIR, load_gold
from rag.rerank import Reranker
from rag.retriever import KnowledgeBase


def best_scores(
    kb: KnowledgeBase, reranker: Reranker, questions: list[str], pool: int
) -> dict[str, list[float]]:
    dense_best, rerank_best = [], []
    for q in questions:
        dense_best.append(float(kb.scores(q, "dense").max()))
        hits = kb.search(q, "hybrid", pool)
        rerank_best.append(max(reranker.scores(q, [c.text for c, _ in hits])))
    return {"dense": dense_best, "rerank": rerank_best}


def gate_metrics(scores: np.ndarray, answerable: np.ndarray, threshold: float) -> dict[str, float]:
    answers = scores >= threshold
    tpr = float(np.mean(answers[answerable])) if answerable.any() else 0.0
    tnr = float(np.mean(~answers[~answerable])) if (~answerable).any() else 0.0
    return {
        "answer_rate_answerable": tpr,
        "abstain_rate_unanswerable": tnr,
        "balanced_accuracy": (tpr + tnr) / 2,
    }


def choose_threshold(scores: np.ndarray, answerable: np.ndarray) -> float:
    """Threshold with the best balanced accuracy; ties go to the midpoint of the best range."""
    candidates = np.unique(np.concatenate([scores, [scores.max() + 1e-6]]))
    values = [gate_metrics(scores, answerable, t)["balanced_accuracy"] for t in candidates]
    best = max(values)
    tied = candidates[[i for i, v in enumerate(values) if v == best]]
    # Midpoint between the lowest tied threshold and the score just below it: a margin
    # on the side of answering, which is cheaper to undo than a refusal.
    low = tied.min()
    below = scores[scores < low]
    return float((low + below.max()) / 2) if below.size else float(low)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="recursive-1000+title")
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--pool", type=int, default=30)
    parser.add_argument("--out", default=str(REPORTS / "phase3_gate.json"))
    args = parser.parse_args()

    kb = KnowledgeBase.build(args.config, Embedder(), lam=args.lam)
    reranker = Reranker()
    calibration = load_gold(GOLD_DIR / "calibration.jsonl")
    gold = load_gold(GOLD_DIR / "gold.jsonl")

    report = {"config": args.config, "lambda": args.lam, "pool": args.pool}
    sets = {}
    for name, items in (("calibration", calibration), ("gold", gold)):
        sets[name] = (
            best_scores(kb, reranker, [g.question for g in items], args.pool),
            np.array([g.answerable for g in items]),
        )

    for kind in ("rerank", "dense"):
        cal_scores, cal_answerable = np.array(sets["calibration"][0][kind]), sets["calibration"][1]
        threshold = choose_threshold(cal_scores, cal_answerable)
        gold_scores, gold_answerable = np.array(sets["gold"][0][kind]), sets["gold"][1]
        report[kind] = {
            "threshold": threshold,
            "calibration": gate_metrics(cal_scores, cal_answerable, threshold),
            "gold": gate_metrics(gold_scores, gold_answerable, threshold),
            "gold_false_abstentions": [
                g.id
                for g, s in zip(gold, gold_scores, strict=True)
                if g.answerable and s < threshold
            ],
            "gold_missed_abstentions": [
                g.id
                for g, s in zip(gold, gold_scores, strict=True)
                if not g.answerable and s >= threshold
            ],
        }
        print(
            f"{kind:6s} threshold={threshold:.4f}  "
            f"calibration={report[kind]['calibration']}  gold={report[kind]['gold']}"
        )
    report["scores"] = {
        name: {"ids": [g.id for g in items], **sets[name][0]}
        for name, items in (("calibration", calibration), ("gold", gold))
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    plot(report, out.with_suffix(".png"))
    print(f"wrote {out}")


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, kind in zip(axes, ("rerank", "dense"), strict=True):
        for name, marker in (("calibration", "o"), ("gold", "x")):
            ids, scores = report["scores"][name]["ids"], report["scores"][name][kind]
            answerable = [not i.startswith(("u", "n")) for i in ids]
            ys = np.random.default_rng(0).uniform(-0.3, 0.3, len(ids)) + (
                0 if name == "calibration" else 1
            )
            colors = ["tab:blue" if a else "tab:red" for a in answerable]
            ax.scatter(scores, ys, c=colors, marker=marker, s=18, alpha=0.7)
        ax.axvline(report[kind]["threshold"], color="k", ls="--", lw=1)
        ax.set_yticks([0, 1], ["calibración", "dorado"])
        ax.set_title(f"Mejor puntuación {'del reranker' if kind == 'rerank' else 'densa'}")
    fig.suptitle("Azul: tiene respuesta en el corpus · Rojo: no la tiene · Línea: umbral")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
