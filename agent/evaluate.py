"""Phase 4 evaluation: success rate, steps and tokens on multi-tool tasks, per brain.

Each line of the benchmark (``agent/bench/example_tasks.jsonl``) has a task, the tools that
must be involved, and a way to check the final answer automatically (a number, a substring,
a regex). Twenty tasks that need at least two tools is the minimum. Run every brain::

    uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base rlm thinking

and put the resulting table in your report next to the JSON-vs-code-agent comparison.

Besides the success rate (the run ended well and the check passes) the report has the share of
tasks where every required tool was used, the stop reasons (``final_answer``, ``plain_answer``,
``max_steps``, ``loop_detected``) and the success per task family (the prefix of the ``id``).
Checks can be combined: ``{"type": "all", "checks": [...]}`` and ``{"type": "any", ...}``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path


def check_answer(task: dict, final_answer: str) -> bool:
    """Automatic check of the final answer, according to the task's ``check`` field."""
    check = task.get("check", {})
    kind, value = check.get("type"), check.get("value")
    if kind in ("all", "any"):
        results = [check_answer({"check": c}, final_answer) for c in check.get("checks", [])]
        return bool(results) and (all(results) if kind == "all" else any(results))
    if kind == "contains":
        return str(value).lower() in final_answer.lower()
    if kind == "regex":
        return re.search(value, final_answer) is not None
    if kind == "number":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", final_answer.replace(",", ""))
        return any(abs(float(n) - float(value)) <= check.get("tolerance", 0) for n in numbers)
    return False


def tools_used(steps: list[dict]) -> set[str]:
    """Tools the agent really called (a malformed block is in the trace, but it is not a call)."""
    return {
        s["action"]["name"]
        for s in steps
        if s.get("action") and s["action"]["name"] != "invalid_tool_call"
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate metrics of one brain (or one agent) over the benchmark rows."""
    n = max(len(rows), 1)
    families: dict[str, list[bool]] = {}
    for r in rows:
        families.setdefault(r.get("id", "task").rsplit("-", 1)[0], []).append(r["correct"])
    return {
        "success_rate": sum(r["correct"] for r in rows) / n,
        "required_tools_rate": sum(r["required_tools_used"] for r in rows) / n,
        "mean_steps": sum(r["n_steps"] for r in rows) / n,
        "mean_tokens": sum(r["tokens_used"] for r in rows) / n,
        "mean_seconds": sum(r.get("seconds", 0.0) for r in rows) / n,
        "stop_reasons": dict(Counter(r.get("stop_reason", "") for r in rows)),
        "by_family": {k: sum(v) / len(v) for k, v in sorted(families.items())},
    }


def load_previous(out: Path, resume: bool) -> dict:
    """Results already in ``out`` (with ``--resume``), so an interrupted run goes on from there."""
    if resume and out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    return {}


def save(out: Path, results: dict, brain: str, rows: list[dict]) -> None:
    """Write after every task: a run cut by the end of a GPU session loses one task at most."""
    results[brain] = {**summarize(rows), "rows": rows}
    # Write then rename: a process killed in the middle never leaves a half-written file.
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))
    tmp.replace(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="agent/bench/example_tasks.jsonl")
    parser.add_argument("--brains", nargs="+", default=["rlm"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out", default="reports/phase4_eval.json")
    parser.add_argument("--only", nargs="*", help="Run only these task ids")
    parser.add_argument(
        "--resume", action="store_true", help="Keep the rows already in --out and skip their tasks"
    )
    args = parser.parse_args()

    from agent.react_agent import ReActAgent

    tasks = [
        json.loads(line)
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.only:
        tasks = [t for t in tasks if t.get("id") in set(args.only)]
    agent = ReActAgent.from_env()
    results = load_previous(Path(args.out), args.resume)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for brain in args.brains:
        rows = [r for r in results.get(brain, {}).get("rows", [])]
        done = {r.get("id") for r in rows}
        for task in tasks:
            if task.get("id") in done:
                continue
            started = time.perf_counter()
            response = agent.run(task["task"], max_steps=args.max_steps, brain=brain).model_dump()
            seconds = time.perf_counter() - started
            ok = response["succeeded"] and check_answer(task, response["final_answer"])
            used = tools_used(response["steps"])
            rows.append(
                {
                    **task,
                    **response,
                    "seconds": round(seconds, 1),
                    "correct": ok,
                    "tools_used": sorted(used),
                    "required_tools_used": set(task.get("required_tools", [])) <= used,
                }
            )
            print(
                f"  {brain:>9} {task.get('id', ''):<20} {'OK ' if ok else 'BAD'} "
                f"{response['n_steps']:>2} steps {response['stop_reason']}",
                flush=True,
            )
            save(out, results, brain, rows)
        save(out, results, brain, rows)
        print(
            f"{brain:>9}: success {results[brain]['success_rate']:.2f} "
            f"· tools {results[brain]['required_tools_rate']:.2f} "
            f"· steps {results[brain]['mean_steps']:.1f} "
            f"· tokens {results[brain]['mean_tokens']:.0f}"
        )


if __name__ == "__main__":
    main()
