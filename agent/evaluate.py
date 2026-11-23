"""Phase-4 evaluation: success rate, steps and cost on the 20-task benchmark, per brain.

For each brain and task it records whether the run succeeded (the agent finished and the check
of the final answer passed), which tools it used, the number of steps, the tokens and the time.
The summary reports, per brain:

* success rate with a 95 % Wilson interval (20 tasks give wide intervals: say so when comparing);
* share of tasks where every required tool was used;
* mean steps, tokens and seconds per task;
* stop reasons (``final_answer``, ``plain_answer``, ``max_steps``, ``loop_detected``...);
* success per task family (the prefix of the task id).

Results are saved after every task, so an interrupted run can continue with ``--resume``::

    uv run python -m agent.evaluate --brains remote thinking rlm base --out reports/phase4_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).parent / "bench" / "tasks.jsonl"


# ---------------------------------------------------------------------------- checks


def check_answer(task: dict, final_answer: str) -> bool:
    """Automatic check of a final answer, according to the task's ``check`` field."""
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
        # Thousands separators are dropped so that "1.000" and "1,000" both read as 1000.
        text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", final_answer)
        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
        tolerance = check.get("tolerance", 0)
        return any(abs(float(n.replace(",", ".")) - float(value)) <= tolerance for n in numbers)
    raise ValueError(f"unknown check type: {kind}")


def load_tasks(path: str | Path = BENCH) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------- summary


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion, in percent."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(100 * (centre - half), 1), round(100 * (centre + half), 1)


def summarise(rows: list[dict]) -> dict:
    """Per-brain metrics of a list of task results."""
    summary = {}
    for brain in dict.fromkeys(r["brain"] for r in rows):
        runs = [r for r in rows if r["brain"] == brain]
        n, ok = len(runs), sum(r["success"] for r in runs)
        families = Counter(r["id"].rsplit("-", 1)[0] for r in runs)
        family_ok = Counter(r["id"].rsplit("-", 1)[0] for r in runs if r["success"])
        summary[brain] = {
            "tasks": n,
            "success_rate": round(100 * ok / n, 1),
            "success_ci95": wilson(ok, n),
            "required_tools_rate": round(100 * sum(r["required_tools_used"] for r in runs) / n, 1),
            "mean_steps": round(sum(r["steps"] for r in runs) / n, 2),
            "mean_tokens": round(sum(r["tokens"] for r in runs) / n),
            "mean_seconds": round(sum(r["seconds"] for r in runs) / n, 1),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in runs)),
            "success_by_family": {f: f"{family_ok[f]}/{families[f]}" for f in families},
        }
    return summary


# ---------------------------------------------------------------------------- running


def run_task(agent, task: dict, brain: str, max_steps: int) -> dict:
    """Run one task with one brain and score it."""
    start = time.perf_counter()
    response = agent.run(task["task"], max_steps=max_steps, brain=brain)
    seconds = time.perf_counter() - start
    used = [s.action.name for s in response.steps if s.action is not None]
    return {
        "id": task["id"],
        "brain": brain,
        "success": response.succeeded and check_answer(task, response.final_answer),
        "final_answer": response.final_answer,
        "tools_used": used,
        "required_tools_used": set(task["required_tools"]) <= set(used),
        "steps": response.n_steps,
        "tokens": response.tokens_used,
        "seconds": round(seconds, 1),
        "stop_reason": response.stop_reason,
        "trace": [s.model_dump() for s in response.steps],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=str(BENCH))
    parser.add_argument("--brains", nargs="+", default=["remote", "thinking", "rlm", "base"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out", default="reports/phase4_eval.json")
    parser.add_argument("--resume", action="store_true", help="skip runs already in --out")
    args = parser.parse_args()

    from dotenv import load_dotenv

    from agent.react_agent import ReActAgent  # heavy import: only when running

    load_dotenv()  # same settings as the API (remote model, adapter...)

    out = Path(args.out)
    rows = json.loads(out.read_text())["rows"] if args.resume and out.exists() else []
    done = {(r["id"], r["brain"]) for r in rows}
    agent = ReActAgent.from_env()
    for brain in args.brains:
        for task in load_tasks(args.bench):
            if (task["id"], brain) in done:
                continue
            row = run_task(agent, task, brain, args.max_steps)
            rows.append(row)
            print(f"{brain:<9} {task['id']:<20} {'OK ' if row['success'] else 'NO '} "
                  f"{row['steps']} steps  {row['stop_reason']}", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"summary": summarise(rows), "rows": rows}, indent=2,
                                      ensure_ascii=False))
    print(json.dumps(summarise(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
