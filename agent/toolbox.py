"""The agent's hands: a set of tools it can call by name, and the parser of its calls.

A tool is a plain Python function plus what the model needs to know about it (name,
description and JSON Schema of the arguments). The toolbox knows nothing about the AI Act:
phases 2 and 3 provide the functions, ``agent.react_agent`` plugs them in.

The model asks for a tool by writing, after its thought::

    <tool_call>
    {"name": "classify_risk", "arguments": {"emotion_recognition": true}}
    </tool_call>

``parse_tool_calls`` reads those blocks and ``Toolbox.run`` executes one with a time limit.
Nothing here raises to the caller: every problem (unknown tool, wrong arguments, malformed
JSON, exception, timeout) comes back as an error message, so the agent can show it to the
model as an observation and the model can fix its call.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
DEFAULT_TIMEOUT = 10.0


@dataclass
class ToolSpec:
    """One tool: the function and the description the model reads."""

    name: str
    description: str
    parameters: dict  # JSON Schema of the arguments
    fn: Callable[..., Any]
    timeout: float = DEFAULT_TIMEOUT
    requires_confirmation: bool = False  # side effects: only run when the user asked for it

    def definition(self) -> dict:
        """The OpenAI-style definition that Qwen3's chat template renders."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A call written by the model, or the reason it could not be read."""

    name: str = ""
    arguments: dict = field(default_factory=dict)
    raw: str = ""
    error: str | None = None


@dataclass
class ToolResult:
    ok: bool
    output: str


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Every ``<tool_call>`` block in ``text``, in order. Unreadable blocks carry an error."""
    calls = []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            calls.append(ToolCall(raw=raw, error=f"the JSON of the call is not valid ({exc.msg})"))
            continue
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            calls.append(ToolCall(raw=raw, error="the call has no 'name'"))
            continue
        arguments = data.get("arguments") or {}
        if isinstance(arguments, str):  # some models send the arguments as a JSON string
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                calls.append(ToolCall(raw=raw, error="'arguments' must be a JSON object"))
                continue
        if not isinstance(arguments, dict):
            calls.append(ToolCall(raw=raw, error="'arguments' must be a JSON object"))
            continue
        calls.append(ToolCall(name=data["name"], arguments=arguments, raw=raw))
    return calls


def _check_arguments(spec: ToolSpec, arguments: dict) -> str | None:
    """Required arguments present and no unknown ones. Deeper validation is the tool's job."""
    properties = spec.parameters.get("properties", {})
    missing = [a for a in spec.parameters.get("required", []) if a not in arguments]
    unknown = [a for a in arguments if properties and a not in properties]
    problems = []
    if missing:
        problems.append(f"missing required arguments: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown arguments: {', '.join(unknown)}")
    return "; ".join(problems) or None


@dataclass
class Toolbox:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def add(self, *specs: ToolSpec) -> Toolbox:
        for spec in specs:
            self.tools[spec.name] = spec
        return self

    def definitions(self) -> list[dict]:
        return [spec.definition() for spec in self.tools.values()]

    def describe(self) -> str:
        """The tools as plain text, for the text version of the prompt."""
        return "\n".join(
            f"- {s.name}: {s.description} Arguments (JSON Schema): {json.dumps(s.parameters)}"
            for s in self.tools.values()
        )

    def run(self, name: str, arguments: dict) -> ToolResult:
        """Execute one tool with its time limit. Errors become a failed result, never raise."""
        spec = self.tools.get(name)
        if spec is None:
            return ToolResult(False, f"unknown tool '{name}'. Available: {', '.join(self.tools)}")
        if problem := _check_arguments(spec, arguments):
            return ToolResult(False, f"invalid arguments for {name}: {problem}")
        # The tool runs in a worker thread so that a slow one cannot block the loop. Python
        # cannot kill a thread: after a timeout its late result is simply discarded.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(spec.fn, **arguments)
        try:
            output = future.result(timeout=spec.timeout)
        except concurrent.futures.TimeoutError:
            return ToolResult(False, f"{name} did not answer within {spec.timeout:g} s (timeout)")
        except Exception as exc:  # noqa: BLE001 - any tool failure goes back to the model
            return ToolResult(False, f"{name} failed: {type(exc).__name__}: {exc}")
        finally:
            pool.shutdown(wait=False)
        text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        return ToolResult(True, text)
