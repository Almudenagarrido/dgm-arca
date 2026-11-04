"""Calibrate the "No lo sé" gates on a validation set that is not the gold set.

The first version picked the reranker threshold (0.975) on the same gold set that reports the
results, with a narrow margin. This script separates the two roles:

* ``--val`` (``rag/gold/threshold_val.jsonl``): questions only used to choose the thresholds;
  half of the unanswerable ones are on corpus topics but ask for a fact that is not there;
* ``--test`` (the gold set): where the chosen thresholds are measured, once.

For every question it records, with both gates off, the best reranker score, the generator's
P(the passages contain the fact) (``RagPipeline.sufficiency``), and the generated answer
(whether the generator refused by itself, the key fact). Every policy is then evaluated offline
on those records, so changing a threshold needs no GPU:

* ``generator``: only the prompt rule (the generator says ``NOT_FOUND`` by itself);
* ``rerank``: also refuse when the best reranker score is under ``t``;
* ``sufficiency``: also refuse when the sufficiency probability is under ``s``;
* ``both``: refuse when either gate fires;
* the disguised-refusal detector (``rag.generate.hedges``): an answer whose first sentence says
  the documents do not contain the fact counts as ``NOT_FOUND``.

A threshold is chosen by balanced accuracy on the validation set (mean of the answer rate on
answerable questions and the refusal rate on unanswerable ones); among equally good thresholds,
the one in the middle of the tied range, which leaves the widest margin on both sides.

Run::

    uv run python -m rag.calibrate --test rag/gold/gold.jsonl --out reports/phase3_calibration.json
    uv run python -m rag.calibrate --offline --out reports/phase3_calibration.json  # re-choose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.evaluate import load_gold, relevant_chunk_ids
from rag.generate import RagPipeline, build_prompt, hedges, resolve_labels, says_not_found
from rag.stats import wilson


def record(pipeline: RagPipeline, items: list[dict], chunks) -> list[dict]:
    rows = []
    for item in items:
        hits = pipeline.retrieve(item["question"], 5, "hybrid")
        sufficiency = pipeline.sufficiency(item["question"], hits)
        answer, _ = resolve_labels(pipeline.generate(build_prompt(item["question"], hits)), hits)
        answerable = item.get("answerable", True)
        keywords = [k.lower() for k in item.get("answer_keywords", [])]
        relevant = relevant_chunk_ids(item, chunks) if answerable else set()
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answerable": answerable,
                "top_score": round(hits[0].score, 5) if hits else 0.0,
                "sufficiency": round(sufficiency, 5),
                "generator_refused": says_not_found(answer),
                "keyword": any(k in answer.lower() for k in keywords) if answerable else None,
                "retrieved_relevant": bool({h.chunk.id for h in hits} & relevant)
                if answerable
                else None,
                "answer": answer,
            }
        )
        mark = "A" if answerable else "U"
        print(
            f"{mark} {item['id']} rerank={hits[0].score:.3f} suff={sufficiency:.3f} "
            f"gen_refused={rows[-1]['generator_refused']}",
            flush=True,
        )
    return rows


def refuses(row: dict, t: float = 0.0, s: float = 0.0, hedge: bool = True) -> bool:
    disguised = hedge and hedges(row["answer"])
    return row["generator_refused"] or disguised or row["top_score"] < t or row["sufficiency"] < s


def outcome(rows: list[dict], t: float = 0.0, s: float = 0.0, hedge: bool = True) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    false_ref = sum(refuses(r, t, s, hedge) for r in ans)
    correct_ref = sum(refuses(r, t, s, hedge) for r in una)
    # An answerable question only counts as answered right if it is not refused and has the fact.
    answered_right = sum(not refuses(r, t, s, hedge) and bool(r["keyword"]) for r in ans)
    return {
        "threshold_rerank": t,
        "threshold_sufficiency": s,
        "disguised_refusal_detector": hedge,
        "false_refusals": f"{false_ref}/{len(ans)}",
        "false_refusal_rate": _rate(false_ref, len(ans)),
        "correct_refusals": f"{correct_ref}/{len(una)}",
        "correct_refusal_rate": _rate(correct_ref, len(una)),
        "answered_with_fact": _rate(answered_right, len(ans)),
        "balanced_accuracy": round(
            ((1 - false_ref / max(len(ans), 1)) + correct_ref / max(len(una), 1)) / 2, 3
        ),
    }


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"value": round(k / max(n, 1), 3), "ci95": [round(lo, 3), round(hi, 3)]}


def candidates(values: list[float]) -> list[float]:
    """0 (gate off) plus the midpoints between consecutive distinct values."""
    v = sorted(set(values))
    return [0.0] + [round((a + b) / 2, 5) for a, b in zip(v, v[1:], strict=False)]


def choose(rows: list[dict], grid: list[tuple[float, float]]) -> tuple[float, float]:
    scored = [(outcome(rows, t, s)["balanced_accuracy"], t, s) for t, s in grid]
    best = max(b for b, _, _ in scored)
    tied = [(t, s) for b, t, s in scored if b == best]
    return tied[len(tied) // 2]


def calibrate(val: list[dict], test: list[dict]) -> dict:
    t_grid = candidates([r["top_score"] for r in val])
    s_grid = candidates([r["sufficiency"] for r in val])
    # (rerank threshold, sufficiency threshold, disguised-refusal detector)
    policies = {
        "generator only": (0.0, 0.0, False),
        "generator + detector": (0.0, 0.0, True),
        "rerank 0.975 (v1, chosen on gold)": (0.975, 0.0, False),
        "rerank 0.975 + detector": (0.975, 0.0, True),
        # 0.5 is the probe's own decision (A more likely than B), not a tuned value: the A/B
        # probe is almost binary (every answerable question scores > 0.9999), so a threshold
        # "chosen" between 0.99999 and 1.0 would only memorise the validation set.
        "rerank 0.975 + sufficiency 0.5 + detector": (0.975, 0.5, True),
        "rerank chosen on val + detector": (*choose(val, [(t, 0.0) for t in t_grid]), True),
        "both chosen on val + detector": (
            *choose(val, [(t, s) for t in t_grid for s in s_grid]),
            True,
        ),
    }
    return {
        name: {"val": outcome(val, t, s, h), "test": outcome(test, t, s, h)}
        for name, (t, s, h) in policies.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--val", default="rag/gold/threshold_val.jsonl")
    parser.add_argument("--test", default="rag/gold/gold.jsonl")
    parser.add_argument("--out", default="reports/phase3_calibration.json")
    parser.add_argument("--offline", action="store_true", help="reuse the records in --out")
    args = parser.parse_args()

    out = Path(args.out)
    if args.offline:
        records = json.loads(out.read_text())["records"]
    else:
        pipeline = RagPipeline(rerank=True)
        pipeline.min_score = pipeline.min_sufficiency = 0.0  # record everything ungated
        pipeline.load()
        chunks = pipeline.retrievers["bm25"].chunks
        records = {
            "val": record(pipeline, load_gold(args.val), chunks),
            "test": record(pipeline, load_gold(args.test), chunks),
        }
    results = calibrate(records["val"], records["test"])
    for name, r in results.items():
        print(f"{name:>44}  val: {_line(r['val'])}\n{'':>44} test: {_line(r['test'])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"val": args.val, "test": args.test, "policies": results, "records": records},
            indent=2,
            ensure_ascii=False,
        )
    )


def _line(o: dict) -> str:
    return (
        f"t={o['threshold_rerank']:.4f} s={o['threshold_sufficiency']:.4f} "
        f"false_ref={o['false_refusals']} correct_ref={o['correct_refusals']} "
        f"fact={o['answered_with_fact']['value']:.2f} bal={o['balanced_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
