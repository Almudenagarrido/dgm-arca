"""How much of a phase-3 difference is noise: confidence intervals for the gold-set metrics.

With 46 answerable questions one question is 2.2 points, so a table of rates says little
without its uncertainty. This module gives:

* ``wilson(k, n)``: the 95 % Wilson interval of a proportion (better than the normal one near
  0 % and 100 %, where most of our rates are);
* ``bootstrap(values)``: the 95 % percentile interval of a mean (for MRR);
* ``paired_bootstrap(a, b)``: the interval of the difference between two configurations
  measured on the *same* questions. Pairing matters: two systems that fail on the same hard
  questions differ much less than two independent samples would suggest.

Run, to write ``reports/phase3_ci.json`` and print the tables of ``docs/EXPERIMENTS.md``::

    uv run python -m rag.stats
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

Z95 = 1.959964
N_RESAMPLES = 10_000


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap(values: list[float], n: int = N_RESAMPLES, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def paired_bootstrap(a: list[float], b: list[float], n: int = N_RESAMPLES, seed: int = 0) -> dict:
    """Mean of b - a, its 95 % interval and the share of resamples where b is not better."""
    assert len(a) == len(b)
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {
        "diff": round(sum(diffs) / m, 3),
        "ci95": [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3)],
        "p_not_better": round(sum(d <= 0 for d in means) / n, 4),
    }


def rate(flags: list[bool]) -> dict:
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    return {"value": round(k / max(n, 1), 3), "k": k, "n": n, "ci95": [round(lo, 3), round(hi, 3)]}


def mean(values: list[float]) -> dict:
    lo, hi = bootstrap(values)
    return {"value": round(sum(values) / len(values), 3), "ci95": [round(lo, 3), round(hi, 3)]}


# ------------------------------------------------------------------------------- the reports


def _answers(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(path.read_text())["rows"]}


ANSWER_METRICS = {
    "relevant_chunk_retrieved": lambda r: bool(r["retrieved_relevant"]),
    "keyword": lambda r: bool(r["keyword"]),
    "judge_correct": lambda r: r.get("judge", {}).get("correct") == "sí",
    "judge_correct_or_partial": lambda r: r.get("judge", {}).get("correct") in ("sí", "parcial"),
    "judge_faithful": lambda r: r.get("judge", {}).get("faithful") == "sí",
    "answered": lambda r: not r["refusal"],  # higher is better, unlike the false refusals
}


def answer_intervals(path: Path) -> dict:
    return answer_intervals_rows(list(_answers(path).values()))


def answer_intervals_rows(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    has_judge = bool(ans) and all("judge" in r for r in ans)
    out = {
        name: rate([f(r) for r in ans])
        for name, f in ANSWER_METRICS.items()
        if (has_judge or not name.startswith("judge")) and name != "answered"
    }
    if ans:
        out["false_refusal"] = rate([bool(r["refusal"]) for r in ans])
        caveat_rows = [r for r in ans if r.get("judge", {}).get("caveats") in ("sí", "no")]
        if caveat_rows:
            out["judge_caveats_respected"] = rate(
                [r["judge"]["caveats"] == "sí" for r in caveat_rows]
            )
        if any(r.get("required_sources_retrieved") is not None for r in ans):
            out["required_sources_retrieved"] = rate(
                [bool(r["required_sources_retrieved"]) for r in ans]
            )
    else:
        out = {}
    if una:
        out["correct_refusal"] = rate([bool(r["refusal"]) for r in una])
    return out


def answer_differences(base: Path, other: Path) -> dict:
    """Paired differences other - base on the answerable questions both files share."""
    a, b = _answers(base), _answers(other)
    ids = [i for i in a if i in b and a[i]["answerable"]]
    out = {}
    for name, f in ANSWER_METRICS.items():
        if name.startswith("judge") and not all("judge" in a[i] and "judge" in b[i] for i in ids):
            continue
        out[name] = paired_bootstrap([float(f(a[i])) for i in ids], [float(f(b[i])) for i in ids])
    return out


def rerank_intervals(eval_path: Path, k: int = 5) -> dict:
    """hit@1, hit@k and MRR with intervals for every reranker run that kept per-question ranks."""
    per_question = json.loads(eval_path.read_text()).get("rerank", {}).get("per_question", {})
    out = {}
    for run, ranks in per_question.items():
        r = list(ranks.values())  # first relevant rank (1-based), or None when not retrieved
        out[run] = {
            "hit@1": rate([x == 1 for x in r]),
            f"hit@{k}": rate([x is not None and x <= k for x in r]),
            "mrr": mean([1 / x if x else 0.0 for x in r]),
        }
    return out


def chunking_difference(eval_path: Path, run: str, other_key: str, k: int = 5) -> dict:
    """Deployed rerank experiment minus another one (``other_key``) on the same questions.

    Positive diff = the deployed chunking (``rerank``) is better for that reranker run.
    """
    results = json.loads(eval_path.read_text())
    base = results.get("rerank", {}).get("per_question", {}).get(run)
    other = results.get(other_key, {}).get("per_question", {}).get(run)
    if not base or not other:
        return {}
    ids = sorted(set(base) & set(other))
    metrics = {
        "hit@1": lambda x: float(x == 1),
        f"hit@{k}": lambda x: float(x is not None and x <= k),
        "mrr": lambda x: 1 / x if x else 0.0,
    }
    return {
        "run": run,
        "against": other_key,
        **{
            name: paired_bootstrap([f(other[i]) for i in ids], [f(base[i]) for i in ids])
            for name, f in metrics.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--out", default="reports/phase3_ci.json")
    args = parser.parse_args()
    reports = Path(args.reports)
    final = reports / "phase3_answers.json"
    result = {"final": answer_intervals(final), "by_question_type": {}, "vs_final": {}}
    rows = json.loads(final.read_text())["rows"]
    for qtype in sorted({r.get("question_type") for r in rows if r.get("question_type")}):
        subset = [r for r in rows if r.get("question_type") == qtype]
        result["by_question_type"][qtype] = answer_intervals_rows(subset)
    # The variants were measured on gold v1, so they are compared with the v1 final answers.
    final_v1 = reports / "phase3_answers_gold_v1.json"
    base = final_v1 if final_v1.exists() else final
    for name in ("before_rerank", "pool20", "top8", "thinking"):
        other = reports / f"phase3_answers_{name}.json"
        if other.exists():
            # Positive diff = the final configuration is better.
            result["vs_final"][name] = answer_differences(other, base)
    if (reports / "phase3_eval.json").exists():
        result["rerank"] = rerank_intervals(reports / "phase3_eval.json")
        # With gold v2 the best first-stage MRR moved to semantic-0.50+title; is that real?
        result["deployed_vs_semantic_0.50"] = {
            run: chunking_difference(
                reports / "phase3_eval.json", run, "rerank_semantic-0.50+title"
            )
            for run in ("hybrid+en/aiact_en/pool30", "hybrid+en/aiact_en/pool75")
        }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("Final configuration (95 % Wilson intervals):")
    for name, r in result["final"].items():
        print(
            f"  {name:>26}: {r['value']:.1%}  [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]  "
            f"({r['k']}/{r['n']})"
        )
    for qtype, metrics in result["by_question_type"].items():
        print(f"{qtype}:")
        for name, r in metrics.items():
            print(
                f"  {name:>26}: {r['value']:.1%}  [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]  "
                f"({r['k']}/{r['n']})"
            )
    for other, diffs in result["vs_final"].items():
        print(f"Final minus {other} (paired bootstrap):")
        for name, d in diffs.items():
            print(
                f"  {name:>26}: {d['diff']:+.1%}  [{d['ci95'][0]:+.1%}, {d['ci95'][1]:+.1%}]  "
                f"P(not better)={d['p_not_better']:.3f}"
            )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
