"""Phase 2 evaluation: does the model pick the right tool with the right arguments?

The benchmark is a JSONL file (``tool_use/bench/cases.jsonl``, built by
``tool_use/bench/build_cases.py``) where each line holds a user query and the expected
first tool call::

    {"query": "...", "needs_tool": true, "expected_tool": "compute_max_fine",
     "expected_arguments": {"infringement": "prohibited_practice", ...},
     "expected_tools": ["classify_risk", "compute_max_fine"],   # optional, multi-step
     "answer_any": ["140 millones", "140.000.000"]}             # optional, final answer

Metrics:

* tool selection accuracy: the first call names the expected tool (or there is no call when
  ``needs_tool`` is false);
* argument accuracy: among all cases, selection is right and the arguments match. Arguments
  are compared after validating both sides with the tool's own schema, so omitted defaults
  (``is_sme=False``, all the false flags of ``classify_risk``) count as equal, and AI Act
  references are compared after parsing ('artículo 5.1.f' == 'Art. 5.1.f'). ``match`` can
  be ``subset`` (only the expected keys are checked) or ``name_only``;
* chain accuracy: every tool in ``expected_tools`` was called (multi-step and parallel);
* label accuracy (classification cases): the label that ``classify_risk`` returned;
* answer accuracy: the final answer contains one of ``answer_any``. It is the only metric
  that also applies to the baseline without tools (``--no-tools``).

Run::

    uv run python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl \\
        --out reports/phase2_eval.json
    uv run python -m tool_use.evaluate --no-tools --out reports/phase2_eval_no_tools.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tool_use.registry import ToolRegistry


def load_bench(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalise(value: Any) -> Any:
    """Make argument comparison tolerant to case, spacing and numeric formatting."""
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return sorted((normalise(v) for v in value), key=repr)
    return value


def canonical_arguments(
    name: str, arguments: dict[str, Any], registry: ToolRegistry | None
) -> dict[str, Any]:
    """Arguments as the tool would see them: defaults filled in, references parsed."""
    args = dict(arguments)
    if registry is not None and name in registry.tools:
        model = registry.tools[name].args_model
        try:
            args = registry.tools[name].validate(args)
        except Exception:  # noqa: BLE001 - invalid arguments simply do not match
            return normalise(arguments)
        # An explicit null means "not set": compare it as the field's default.
        args = {
            k: (model.model_fields[k].get_default(call_default_factory=True) if v is None else v)
            for k, v in args.items()
        }
    if name == "lookup_ai_act" and "reference" in args:
        from tool_use.ai_act_text import parse_reference

        try:
            args["reference"] = list(parse_reference(args["reference"]))
        except ValueError:
            pass
    return normalise(args)


def _arguments_match(
    case: dict[str, Any], call: dict[str, Any], registry: ToolRegistry | None
) -> bool:
    match = case.get("match", "exact")
    if match == "name_only":
        return True
    actual = canonical_arguments(call["name"], call["arguments"], registry)
    alternatives = case.get("expected_arguments_any") or [case.get("expected_arguments", {})]
    for expected in alternatives:
        if match == "subset":
            if all(
                normalise(call["arguments"].get(k)) == normalise(v) for k, v in expected.items()
            ):
                return True
        elif actual == canonical_arguments(call["name"], expected, registry):
            return True
    return False


def score_case(
    case: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    answer: str = "",
    registry: ToolRegistry | None = None,
) -> dict[str, bool | None]:
    """Compare the calls the model made against the expectation for one benchmark case."""
    first = tool_calls[0] if tool_calls else None
    answer_any = case.get("answer_any")
    answer_ok = any(a.lower() in answer.lower() for a in answer_any) if answer_any else None
    if not case.get("needs_tool", True):
        ok = first is None
        return {
            "selection_ok": ok,
            "arguments_ok": ok,
            "chain_ok": None,
            "label_ok": None,
            "answer_ok": answer_ok,
        }
    selection_ok = first is not None and first["name"] == case["expected_tool"]
    arguments_ok = selection_ok and _arguments_match(case, first, registry)
    chain_ok = None
    if case.get("expected_tools"):
        called = Counter(c["name"] for c in tool_calls if c.get("ok", True))
        chain_ok = all(called[n] >= k for n, k in Counter(case["expected_tools"]).items())
    label_ok = None
    if "expected_label" in case:
        labels = [
            c["result"]["label"]
            for c in tool_calls
            if c["name"] == "classify_risk" and c.get("ok") and isinstance(c.get("result"), dict)
        ]
        label_ok = bool(labels) and labels[-1] == case["expected_label"]
    return {
        "selection_ok": selection_ok,
        "arguments_ok": arguments_ok,
        "chain_ok": chain_ok,
        "label_ok": label_ok,
        "answer_ok": answer_ok,
    }


def _rate(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "tool_selection_accuracy": "selection_ok",
        "argument_accuracy": "arguments_ok",
        "chain_accuracy": "chain_ok",
        "label_accuracy": "label_ok",
        "answer_accuracy": "answer_ok",
    }
    summary: dict[str, Any] = {name: _rate(rows, key) for name, key in keys.items()}
    summary["n_cases"] = len(rows)
    summary["failed_calls"] = sum(1 for r in rows for c in r["tool_calls"] if not c["ok"])
    summary["mean_turns"] = round(sum(r["turns"] for r in rows) / max(len(rows), 1), 2)
    summary["mean_seconds"] = round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 1)
    by_kind: dict[str, list] = defaultdict(list)
    for row in rows:
        by_kind[row.get("kind", "")].append(row)
    summary["by_kind"] = {
        kind: {name: _rate(group, key) for name, key in keys.items()} | {"n": len(group)}
        for kind, group in by_kind.items()
    }
    return summary


def rescore(reports: list[str], bench: str) -> None:
    """Re-apply ``score_case`` to saved generations, e.g. after fixing an expectation."""
    from tool_use.tools import default_registry

    registry = default_registry()
    cases = {c["id"]: c for c in load_bench(bench)}
    for report in reports:
        data = json.loads(Path(report).read_text())
        rows = []
        for row in data["rows"]:
            case = cases[row["id"]]
            rows.append(
                {**row, **case, **score_case(case, row["tool_calls"], row["answer"], registry)}
            )
        data["rows"], data["summary"] = rows, summarise(rows)
        Path(report).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        print(f"{report}: answer_accuracy={data['summary']['answer_accuracy']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="tool_use/bench/cases.jsonl")
    parser.add_argument("--out", default="reports/phase2_eval.json")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--model", default=None, help="default: ARCA_TOOLS_MODEL or Qwen3-4B")
    parser.add_argument("--prompt", choices=["native", "slides"], default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--no-tools", action="store_true", help="baseline without tools")
    parser.add_argument(
        "--rescore",
        nargs="+",
        metavar="REPORT",
        help="recompute the metrics of saved reports against the current bench (no model)",
    )
    args = parser.parse_args()

    if args.rescore:
        rescore(args.rescore, args.bench)
        return

    from tool_use.executor import ToolLoop

    loop = ToolLoop(model_id=args.model, prompt_style=args.prompt, thinking=args.thinking)
    loop.load()
    cases = load_bench(args.bench)
    rows = []
    for case in cases:
        started = time.perf_counter()
        response = loop.run(case["query"], max_turns=args.max_turns, use_tools=not args.no_tools)
        calls = [c.model_dump() for c in response.tool_calls]
        scores = score_case(case, calls, response.answer, loop.registry)
        rows.append(
            {
                **case,
                "tool_calls": calls,
                "answer": response.answer,
                "turns": response.turns,
                "max_turns_reached": response.max_turns_reached,
                "seconds": round(time.perf_counter() - started, 1),
                **scores,
            }
        )
        mark = "✓" if (scores["answer_ok"] if args.no_tools else scores["arguments_ok"]) else "✗"
        print(f"{mark} {case.get('id', '')} {case['query'][:70]}", flush=True)

    summary = summarise(rows)
    config = {
        "model": loop.model_id,
        "prompt": loop.prompt_style,
        "thinking": loop.thinking,
        "tools": not args.no_tools,
    }
    print(
        json.dumps(
            {"config": config, **{k: v for k, v in summary.items() if k != "by_kind"}}, indent=2
        )
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"config": config, "summary": summary, "rows": rows},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
