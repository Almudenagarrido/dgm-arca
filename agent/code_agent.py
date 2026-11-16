"""The contrast for phase 4: the same tasks with a code agent (``smolagents.CodeAgent``).

Our agent writes one JSON tool call per step. A code agent (CodeAct, Wang et al. 2024) writes a
Python snippet instead: it can call several tools, keep results in variables and do arithmetic
in one step. To compare only the *action format*, everything else is shared with
``agent/react_agent.py``:

* the brain: ``BrainModel`` sends smolagents' messages to the same ``Brain`` objects (same
  weights, same chat template, same thinking switch, same decoding), so ``--brains rlm`` works
  as soon as ``ARCA_RLM_ADAPTER`` is set;
* the tools: each tool of our registry is wrapped as a ``smolagents.Tool`` whose ``forward``
  goes through ``ReActAgent.execute`` (same validation, timeouts and confirmation policy). The
  code agent uses its own ``final_answer``;
* the tasks, the step limit and the checks of ``agent/evaluate.py``.

What differs is the prompt: smolagents' own CodeAgent prompt (English, with few-shot examples)
plus our ``DOMAIN_RULES`` as instructions.

    uv run python -m agent.code_agent --bench agent/bench/tasks.jsonl --brains thinking base

Tools used are read from the code of each step (a call ``name(`` to a tool), so the "required
tools" rate is an approximation: a call written but never reached would also count.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from agent.evaluate import check_answer, load_previous, save
from agent.prompts import DOMAIN_RULES
from agent.react_agent import ReActAgent, RunState
from tool_use.parser import ToolCall
from tool_use.registry import Tool as RegistryTool

AUTHORIZED_IMPORTS = ["datetime", "math"]


def _json_type(schema: dict) -> str:
    """smolagents input type for one JSON-schema property (first non-null branch)."""
    if "anyOf" in schema:
        types = [_json_type(s) for s in schema["anyOf"] if s.get("type") != "null"]
        return types[0] if types else "any"
    kind = schema.get("type")
    if kind in ("string", "boolean", "integer", "number", "array", "object"):
        return kind
    return "any"


def wrap_tool(spec: RegistryTool, agent: ReActAgent, task: dict[str, str]):
    """A ``smolagents.Tool`` that runs one of our tools through ``ReActAgent.execute``."""
    from smolagents import Tool

    params = spec.definition["function"]["parameters"]
    required = set(params.get("required", []))
    inputs = {}
    for name, prop in params.get("properties", {}).items():
        entry = {"type": _json_type(prop), "description": prop.get("description", name)}
        if "enum" in prop:
            entry["description"] += f" One of {prop['enum']}."
        if name not in required:
            entry["nullable"] = True
        inputs[name] = entry

    class Wrapped(Tool):
        skip_forward_signature_validation = True

        def forward(self, **kwargs: Any) -> Any:
            arguments = {k: v for k, v in kwargs.items() if v is not None}
            record = agent.execute(ToolCall(spec.name, arguments), task["text"])
            if not record.ok:
                raise RuntimeError(record.error)
            return record.result

    Wrapped.name = spec.name
    Wrapped.description = spec.description
    Wrapped.inputs = inputs
    Wrapped.output_type = "any"
    return Wrapped()


def brain_model(agent: ReActAgent, brain: str):
    """A ``smolagents.Model`` backed by one of our brains."""
    from smolagents.models import (
        ChatMessage,
        MessageRole,
        Model,
        TokenUsage,
        get_clean_message_list,
        remove_content_after_stop_sequences,
        tool_role_conversions,
    )

    class BrainModel(Model):
        def generate(self, messages, stop_sequences=None, response_format=None, **kwargs):
            clean = get_clean_message_list(
                messages, role_conversions=tool_role_conversions, flatten_messages_as_text=True
            )
            raw = agent.generate(clean, brain, stop=list(stop_sequences or []))
            # The reasoning stays out of the code parser: a ``<code>`` written while thinking
            # must not be executed. The answer part is what smolagents reads.
            content = raw.split("</think>", 1)[-1].strip()
            content = remove_content_after_stop_sequences(content, stop_sequences)
            prompt_tokens, new_tokens = agent.last_usage
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                raw=raw,
                token_usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=new_tokens),
            )

    return BrainModel(model_id=agent.brains[brain].label)


def run_task(agent: ReActAgent, brain: str, task: str, max_steps: int) -> dict:
    """One task with a fresh CodeAgent; returns a row shaped like ``agent/evaluate.py``'s."""
    from smolagents import CodeAgent
    from smolagents.memory import ActionStep

    current = {"text": task}
    tools = [
        wrap_tool(spec, agent, current)
        for name, spec in agent.registry.tools.items()
        if name != "final_answer"
    ]
    code_agent = CodeAgent(
        tools=tools,
        model=brain_model(agent, brain),
        max_steps=max_steps,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        instructions=DOMAIN_RULES,
        verbosity_level=-1,  # LogLevel.OFF: the trace goes to the report
    )
    agent._state = RunState()
    result = code_agent.run(task, return_full_result=True)
    steps, used = [], set()
    for step in code_agent.memory.steps:
        if not isinstance(step, ActionStep):
            continue
        code = step.code_action or ""
        used |= {name for name in agent.registry.tools if re.search(rf"\b{name}\(", code)}
        steps.append(
            {
                "thought": step.model_output if isinstance(step.model_output, str) else "",
                "action": {"name": "python", "arguments": {"code": code}} if code else None,
                "observation": str(step.error) if step.error else step.observations,
            }
        )
    usage = result.token_usage
    return {
        "final_answer": "" if result.output is None else str(result.output),
        "succeeded": result.state == "success",
        "stop_reason": "final_answer" if result.state == "success" else "max_steps",
        "n_steps": len(steps),
        "tokens_used": (usage.input_tokens + usage.output_tokens) if usage else 0,
        "steps": steps,
        "tools_used": sorted(used - {"final_answer"}),
        "model": agent.brains[brain].label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="agent/bench/tasks.jsonl")
    parser.add_argument("--brains", nargs="+", default=["thinking"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out", default="reports/phase4_code_agent.json")
    parser.add_argument("--only", nargs="*", help="Run only these task ids")
    parser.add_argument(
        "--resume", action="store_true", help="Keep the rows already in --out and skip their tasks"
    )
    args = parser.parse_args()

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
            try:
                response = run_task(agent, brain, task["task"], args.max_steps)
            except Exception as exc:  # noqa: BLE001 - one broken run must not stop the bench
                response = {
                    "final_answer": "",
                    "succeeded": False,
                    "stop_reason": f"crash: {type(exc).__name__}: {exc}",
                    "n_steps": 0,
                    "tokens_used": agent._state.tokens,
                    "steps": [],
                    "tools_used": [],
                }
            ok = response["succeeded"] and check_answer(task, response["final_answer"])
            rows.append(
                {
                    **task,
                    **response,
                    "seconds": round(time.perf_counter() - started, 1),
                    "correct": ok,
                    "required_tools_used": set(task.get("required_tools", []))
                    <= set(response["tools_used"]),
                }
            )
            print(
                f"  code/{brain:<9} {task.get('id', ''):<20} {'OK ' if ok else 'BAD'} "
                f"{response['n_steps']:>2} steps {response['stop_reason'][:60]}",
                flush=True,
            )
            save(out, results, brain, rows)
        save(out, results, brain, rows)
        print(
            f"code/{brain}: success {results[brain]['success_rate']:.2f} "
            f"· tools {results[brain]['required_tools_rate']:.2f} "
            f"· steps {results[brain]['mean_steps']:.1f} "
            f"· tokens {results[brain]['mean_tokens']:.0f}"
        )


if __name__ == "__main__":
    main()
