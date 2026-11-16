"""Is the malformed tool call the model's fault or the prompt's? Measured on the first step only.

The base brain (Qwen3-0.6B) failed 100 of its 163 steps in the first benchmark run with a
``<tool_call>`` that had no ``name``: it wrote ``Action: classify_risk`` and put only the
arguments in the JSON. This script generates the first step of every benchmark task under four
prompts and counts well-formed calls, so the comparison costs minutes instead of a full run:

* ``a_text``: the ReAct prompt of the slides (tools pasted as text, ``Action:`` label) and the
  first version of our rules, with the example 'lookup_ai_act with a reference such as
  "Art. 50.1"';
* ``b_text_no_example``: the same without the example;
* ``c_native``: Qwen3's chat template renders the tools (``apply_chat_template(tools=...)``),
  with the example;
* ``d_native_no_example``: the native template without the example (what the agent uses now).

    uv run python -m agent.first_step_ablation --brain base
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from agent.prompts import DOMAIN_RULES, REACT_NATIVE_PROMPT, REACT_SYSTEM_PROMPT
from agent.react_agent import ReActAgent, final_answer, search_knowledge_base
from tool_use.parser import parse_tool_calls

WITH_EXAMPLE = DOMAIN_RULES.replace(
    "use lookup_ai_act.\n",
    'use lookup_ai_act with a reference such as "Art. 50.1" or "Anexo III.4".\n',
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="agent/bench/tasks.jsonl")
    parser.add_argument("--brain", default="base")
    parser.add_argument("--out", default="reports/phase4_first_step_ablation.json")
    args = parser.parse_args()

    assert WITH_EXAMPLE != DOMAIN_RULES, "the example could not be put back"
    agent = ReActAgent()
    agent.registry.register(final_answer, search_knowledge_base)
    tasks = [json.loads(line) for line in Path(args.bench).read_text().splitlines() if line]
    text = REACT_SYSTEM_PROMPT.format(
        max_steps=10, tools_description=agent.registry.render_for_prompt()
    )
    native = REACT_NATIVE_PROMPT.format(max_steps=10)
    variants = {
        "a_text": (f"{text}\n{WITH_EXAMPLE}", None),
        "b_text_no_example": (f"{text}\n{DOMAIN_RULES}", None),
        "c_native": (f"{native}\n{WITH_EXAMPLE}", agent.registry.definitions()),
        "d_native_no_example": (f"{native}\n{DOMAIN_RULES}", agent.registry.definitions()),
    }
    results = {}
    for name, (system, tools) in variants.items():
        counts: collections.Counter = collections.Counter()
        rows = []
        for task in tasks:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": task["task"]},
            ]
            raw = agent.generate(messages, args.brain, tools=tools)
            if raw.count("<tool_call>") > raw.count("</tool_call>"):
                raw += "\n</tool_call>"
            parsed = parse_tool_calls(raw)
            valid = [c for c in parsed.tool_calls if c.error is None]
            if valid:
                call = valid[0]
                counts["well_formed"] += 1
                counts["required_tool"] += call.name in task["required_tools"]
                counts["copies_example"] += "Art. 50.1" in json.dumps(call.arguments)
            elif parsed.tool_calls:
                counts["malformed"] += 1
                counts["no_name"] += "no 'name'" in (parsed.tool_calls[0].error or "")
            else:
                counts["no_call"] += 1
            rows.append({"id": task["id"], "raw": raw})
        results[name] = {"counts": dict(counts), "n": len(tasks), "rows": rows}
        print(f"{name:<20} {dict(counts)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
