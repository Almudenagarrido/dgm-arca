"""Phase 4 evaluation: success rate, steps and tokens on multi-tool tasks, per brain.

Each line of the benchmark (``agent/bench/tasks.jsonl``, built by ``build_tasks.py``) has a
task, the tools that must be involved, and a way to check the final answer automatically (a
number, a substring, a regex). Twenty tasks that need at least two tools is the minimum.
Run every brain::

    uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base rlm thinking

and put the resulting table in your report next to the JSON-vs-code-agent comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# A number as people write it in Spanish or English: 15.000.000 / 15,000,000 / 7,5 / 7.5,
# optionally followed by a scale word ("15 millones", "7,5 M€").
NUMBER = re.compile(
    r"(?<![\w.,])(-?\d{1,3}(?:([.,\s])\d{3})(?:\2\d{3})*(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?)"
    r"(?:\s*(millones|millón|million|mil millones|M\b|M€|k\b|mil\b))?",
    re.IGNORECASE,
)
SCALES = {"millones": 1e6, "millón": 1e6, "million": 1e6, "m": 1e6, "m€": 1e6, "k": 1e3, "mil": 1e3}


def parse_number(text: str, sep: str | None) -> float:
    """'15.000.000' -> 15000000, '7,5' -> 7.5, '1,000.25' -> 1000.25."""
    text = text.replace(" ", "")
    if sep:  # thousands groups: drop the separator, the other one is the decimal mark
        decimal = "," if sep == "." else "."
        return float(text.replace(sep, "").replace(decimal, "."))
    return float(text.replace(",", "."))


def numbers_in(text: str) -> list[float]:
    found = []
    for match in NUMBER.finditer(text.replace("\u00a0", " ")):
        value = parse_number(match.group(1), match.group(2))
        scale = (match.group(3) or "").lower()
        if scale == "mil millones":
            value *= 1e9
        elif scale:
            value *= SCALES.get(scale, 1)
        found.append(value)
    return found


def check_answer(task: dict, final_answer: str) -> bool:
    """Automatic check of the final answer, according to the task's ``check`` field.

    Kinds: ``contains`` (case-insensitive substring), ``regex``, ``number`` (any number in the
    answer within ``tolerance``; Spanish formats and "millones" are understood) and ``all`` /
    ``any`` over a list of checks, for tasks with several parts.
    """
    check = task.get("check", {})
    kind, value = check.get("type"), check.get("value")
    if kind == "all":
        return all(check_answer({"check": c}, final_answer) for c in value)
    if kind == "any":
        return any(check_answer({"check": c}, final_answer) for c in value)
    if kind == "contains":
        return str(value).lower() in final_answer.lower()
    if kind == "regex":
        return re.search(value, final_answer) is not None
    if kind == "number":
        tolerance = check.get("tolerance", 0)
        return any(abs(n - float(value)) <= tolerance for n in numbers_in(final_answer))
    return False


def tools_used(steps: list[dict]) -> set[str]:
    return {s["action"]["name"] for s in steps if s.get("action")}


def summarise(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r.get("kind", "?"), []).append(r)
    return {
        "success_rate": sum(r["correct"] for r in rows) / n,
        "required_tools_rate": sum(r["required_tools_used"] for r in rows) / n,
        "mean_steps": sum(r["n_steps"] for r in rows) / n,
        "mean_tokens": sum(r["tokens_used"] for r in rows) / n,
        "mean_seconds": sum(r["seconds"] for r in rows) / n,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
        "by_kind": {k: sum(r["correct"] for r in v) / len(v) for k, v in sorted(by_kind.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="agent/bench/tasks.jsonl")
    parser.add_argument("--brains", nargs="+", default=["rlm"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--ids", nargs="*", help="only these task ids")
    parser.add_argument("--out", default="reports/phase4_eval.json")
    args = parser.parse_args()

    load_dotenv()
    from agent.react_agent import ReActAgent

    tasks = [
        json.loads(line)
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.ids:
        tasks = [t for t in tasks if t["id"] in set(args.ids)]
    agent = ReActAgent.from_env()
    results = {
        "config": {
            "prompt": agent.prompt_style,
            "lenient_parse": agent.lenient_parse,
            "token_budget": agent.token_budget,
            "max_steps": args.max_steps,
            "brains": {b: agent.specs[b].label for b in args.brains},
        }
    }
    for brain in args.brains:
        rows = []
        for task in tasks:
            started = time.perf_counter()
            try:
                response = agent.run(task["task"], max_steps=args.max_steps, brain=brain)
                response = response.model_dump()
            except Exception as exc:  # noqa: BLE001 - one crash must not lose the whole run
                response = {
                    "final_answer": "",
                    "steps": [],
                    "n_steps": 0,
                    "succeeded": False,
                    "tokens_used": 0,
                    "stop_reason": f"crash: {type(exc).__name__}: {exc}",
                }
            ok = response["succeeded"] and check_answer(task, response["final_answer"])
            used = tools_used(response["steps"])
            rows.append(
                {
                    **task,
                    **response,
                    "seconds": round(time.perf_counter() - started, 1),
                    "correct": ok,
                    "tools_used": sorted(used),
                    "required_tools_used": set(task.get("required_tools", [])) <= used,
                }
            )
            print(
                f"{brain:>9} {task['id']:>14}: {'OK ' if ok else 'BAD'} "
                f"{response['stop_reason']:<14} steps {response['n_steps']}",
                flush=True,
            )
        results[brain] = {**summarise(rows), "rows": rows}
        summary = results[brain]
        print(
            f"{brain:>9}: success {summary['success_rate']:.2f} "
            f"· tools {summary['required_tools_rate']:.2f} · steps {summary['mean_steps']:.1f}"
            f" · tokens {summary['mean_tokens']:.0f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))


if __name__ == "__main__":
    main()
