"""The ReAct agent: Thought -> Action -> Observation, in a loop, with every tool you built.

This is where phases 1 to 3 meet. The brain is a model (``brain="base"``, the phase-1 RLM with
``brain="rlm"``, or a thinking model with ``brain="thinking"``; see ``agent/brains.py``), the
hands are the phase-2 registry plus ``search_knowledge_base`` (the phase-3 retriever), and the
loop below is written by hand. ``ReActAgent.run`` is what ``POST /agent`` calls.

Decisions of the loop, each one a configurable guardrail:

* **Prompt.** ``ARCA_AGENT_PROMPT=native`` (default) gives the tools to Qwen's chat template
  (``tools=``), as phase 2 does; ``react`` pastes them as text into the ReAct prompt of the
  slides. With ``react`` the 0.6B brains copied the JSON schema into the call instead of the
  arguments (docs/EXPERIMENTS.md).
* **Lenient parsing.** ``ARCA_AGENT_LENIENT_PARSE=1`` (default) also accepts the ReAct-style
  ``Action: {"name": ..., "arguments": ...}`` and ``Action: tool_name {arguments}`` without
  ``<tool_call>`` tags, and unwraps ``{"function": {...}}``: the small brains write those.
* **One action per step.** If the model writes several ``<tool_call>`` blocks, only the first
  runs and the observation says so. Generation already stops at ``</tool_call>``, so in
  practice there is only one.
* **Timeouts.** Each tool runs in a worker thread with ``ARCA_AGENT_TOOL_TIMEOUT`` seconds
  (default 60: ``lookup_ai_act`` may download the regulation the first time). On timeout the
  agent gets an error observation and carries on; Python cannot kill the thread, so it
  finishes in the background and its result is dropped.
* **Errors are observations.** Unknown tools, invalid arguments, malformed JSON, exceptions and
  timeouts all go back to the model as ``{"error": ...}`` so it can fix the call.
* **No action.** A step without a tool call gets one reminder of the format. A second one is
  accepted as the answer (``stop_reason="plain_answer"``), flagged in the response.
* **Evidence before answering.** ``ARCA_AGENT_REQUIRE_EVIDENCE=1`` (default): a
  ``final_answer`` before any tool has returned a result is sent back once (``NO_EVIDENCE``);
  the second one is accepted. ``ARCA_AGENT_FEWSHOT=1`` (default) adds one worked example.
* **Loop detection.** The same call with the same arguments is not executed again: the model
  is shown the earlier result. The third identical call, or the third identical malformed
  message, stops the loop (``stop_reason="loop_detected"``).
* **Confirmation.** Tools with ``requires_confirmation`` (``write_assessment_report``) follow
  the phase-2 policy, ``ARCA_AGENT_AUTO_CONFIRM`` (default: ``ARCA_TOOLS_AUTO_CONFIRM``, ``1``).
* **Memory.** ``ConversationMemory`` with ``ARCA_AGENT_TOKEN_BUDGET`` tokens (default 12000:
  the tool definitions alone are ~4300). The task stays verbatim, observations enter the memory
  cut to ``ARCA_AGENT_MAX_OBSERVATION_CHARS`` and are cut harder before being summarised.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from agent.brains import Brain, BrainSpec, brain_specs
from agent.memory import ConversationMemory
from agent.prompts import (
    AGENT_DOMAIN_RULES,
    FEW_SHOT_EXAMPLE,
    REACT_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    annex_iii_table,
)
from api.schemas import AgentAction, AgentResponse, AgentStep, ToolCallRecord
from rlm.rewards import extract_answer
from tool_use.executor import CONFIRMATION_REFUSED
from tool_use.parser import (
    TOOL_CALL_PATTERN,
    ParsedOutput,
    ToolCall,
    format_tool_result,
    parse_tool_calls,
)
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import default_registry

KB_MAX_CHARS = 900  # per passage: five passages stay under ~1500 tokens

NO_ACTION_NUDGE = (
    "Format error: your message has no tool call. Call exactly one tool inside "
    "<tool_call></tool_call>, or call `final_answer` with your answer."
)
REPEATED_CALL = (
    "You already called {name} with these same arguments; it was not executed again. Its "
    "result was:\n{previous}\nUse it, try different arguments or call `final_answer`."
)
SCHEMA_COPIED = (
    "you copied the definition (JSON Schema) of {name} instead of calling it. Write "
    '{{"name": "{name}", "arguments": {{...}}}} with the values of the arguments.'
)
NO_EVIDENCE = (
    "Not accepted yet: you have not used any tool, so this answer is not based on any result. "
    "If the task needs a classification, a fine, an article or the knowledge base, call that "
    "tool first. If it really needs no tool, or the information is not available, call "
    "final_answer again."
)
EXTRA_CALLS_IGNORED = "\n(Only the first tool call was executed: call one tool per step.)"
MAX_STEPS_MESSAGE = "No he podido completar la tarea en {max_steps} pasos."
LOOP_MESSAGE = "He detenido la tarea: repetía la misma llamada a {name} sin avanzar."


@tool
def final_answer(answer: str) -> str:
    """Return the final answer to the user and stop the loop.

    Args:
        answer: The complete answer to the task.
    """
    return answer


def knowledge_base():
    """The phase-3 pipeline (shared with ``POST /rag``): same retrievers, reranker and gate."""
    from rag.generate import build_pipeline

    return build_pipeline()


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the knowledge base (the AI Act in Spanish, its recitals and the Commission
    guidelines on prohibited practices, the AI system definition and general-purpose models)
    and return the most relevant passages with their ids.

    Args:
        query: What to look for, phrased as a question.
        top_k: How many passages to return (5 to 8).
    """
    pipeline = knowledge_base()
    # At least 5: the retrieval of phase 3 was measured at k=5 (hit@5 0.84); asking for 3 the
    # brain missed the passage with the answer (docs/EXPERIMENTS.md).
    hits = pipeline.retrieve(query, top_k=max(5, min(top_k, 8)), retriever="hybrid")
    if not hits:
        return "No passages found."
    passages = "\n\n".join(
        f"[{h.chunk.id}] ({h.chunk.source})\n{h.chunk.text[:KB_MAX_CHARS]}" for h in hits
    )
    if pipeline.rerank and hits[0].score < pipeline.min_score:
        # The same gate that makes /rag answer "No lo sé": the agent has to see it too.
        return (
            f"WARNING: no passage answers this query with confidence (best reranker score "
            f"{hits[0].score:.3f} < {pipeline.min_score}). If the task depends on it, say that "
            f"the knowledge base does not contain it.\n\n{passages}"
        )
    return passages


ACTION_PATTERN = re.compile(r"Action:\s*`?(?P<name>[A-Za-z_]\w*)?`?\s*(?=\{)")


def _unwrap(payload: dict) -> dict:
    """{"type": "function", "function": {"name": ..., ...}} -> the inner call."""
    inner = payload.get("function")
    return inner if "name" not in payload and isinstance(inner, dict) else payload


def _call_from_payload(payload: Any, name: str | None = None) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None
    payload = _unwrap(payload)
    if isinstance(payload.get("name"), str) and payload["name"]:
        arguments = payload.get("arguments", payload.get("parameters", {}))
        if isinstance(arguments, dict) and "properties" in arguments:
            # The model pasted the JSON Schema of the tool. Pydantic ignores unknown fields,
            # so running it would silently use every default: tell the model instead.
            return ToolCall(name=payload["name"], error=SCHEMA_COPIED.format(name=payload["name"]))
        return ToolCall(
            name=payload["name"], arguments=arguments if isinstance(arguments, dict) else {}
        )
    return ToolCall(name=name, arguments=payload) if name else None


def parse_action(raw: str, lenient: bool = True) -> ParsedOutput:
    """The phase-2 parser, plus (``lenient``) the formats the small brains write instead."""
    parsed = parse_tool_calls(raw)
    if not lenient:
        return parsed
    if parsed.tool_calls:
        fixed = []
        # The parser adds exactly one entry per <tool_call> block, in order.
        matches = TOOL_CALL_PATTERN.finditer(raw)
        for call, match in zip(parsed.tool_calls, matches, strict=True):
            if call.error and "no 'name'" in call.error:
                try:
                    call = _call_from_payload(json.loads(match.group("body"))) or call
                except json.JSONDecodeError:
                    pass
            fixed.append(call)
        parsed.tool_calls = fixed
        return parsed
    answer_part = raw.split("</think>", 1)[-1]
    for match in ACTION_PATTERN.finditer(answer_part):
        try:
            payload, _ = json.JSONDecoder().raw_decode(answer_part[match.end() :])
        except json.JSONDecodeError:
            continue
        call = _call_from_payload(payload, match.group("name"))
        if call is not None:
            parsed.tool_calls = [call]
            parsed.text = answer_part[: match.start()].strip()
            break
    return parsed


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def _thought(thinking: str, text: str) -> str:
    """What the trace shows as the thought: the <think> block and the 'Thought:' text."""
    text = re.sub(r"^\s*Thought:\s*", "", text).strip()
    return "\n\n".join(part for part in (thinking, text) if part)


def _plain_answer(text: str) -> str:
    """Answer written without final_answer: the RLM's <answer> block, or the bare text."""
    answer = extract_answer(text)
    if answer is not None:
        return answer
    # A generation cut off while thinking has no </think>: that reasoning is not an answer.
    text = re.sub(r"<think>.*?(?:</think>|\Z)", "", text, flags=re.DOTALL)
    return re.sub(r"^\s*(Thought|Final Answer|Answer):\s*", "", text.strip()).strip()


def _history(messages: list[dict]) -> str:
    return "\n".join(
        f"{m['role']}{' (' + m['name'] + ')' if m.get('name') else ''}: {m['content']}"
        for m in messages
    )


@dataclass
class ReActAgent:
    registry: ToolRegistry = field(default_factory=default_registry)
    token_budget: int = field(
        default_factory=lambda: int(os.environ.get("ARCA_AGENT_TOKEN_BUDGET", "12000"))
    )
    tool_timeout: float = field(
        default_factory=lambda: float(os.environ.get("ARCA_AGENT_TOOL_TIMEOUT", "60"))
    )
    max_observation_chars: int = field(
        default_factory=lambda: int(os.environ.get("ARCA_AGENT_MAX_OBSERVATION_CHARS", "4000"))
    )
    auto_confirm: bool = field(
        default_factory=lambda: _flag(
            "ARCA_AGENT_AUTO_CONFIRM", os.environ.get("ARCA_TOOLS_AUTO_CONFIRM", "1")
        )
    )
    prompt_style: str = field(
        default_factory=lambda: os.environ.get("ARCA_AGENT_PROMPT", "native").strip().lower()
    )
    lenient_parse: bool = field(default_factory=lambda: _flag("ARCA_AGENT_LENIENT_PARSE", "1"))
    few_shot: bool = field(default_factory=lambda: _flag("ARCA_AGENT_FEWSHOT", "1"))
    require_evidence: bool = field(
        default_factory=lambda: _flag("ARCA_AGENT_REQUIRE_EVIDENCE", "1")
    )
    specs: dict[str, BrainSpec] = field(default_factory=dict)
    brains: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="arca-tool")

    @classmethod
    def from_env(cls) -> ReActAgent:
        agent = cls()
        agent.registry.register(final_answer, search_knowledge_base)
        agent.load()
        return agent

    def load(self) -> None:
        """Read the brain configuration. Models load lazily, the first time a brain is used:
        loading the three at start-up would make every ``/health`` call pay for all of them."""
        self.specs = brain_specs()

    def brain(self, name: str):
        if name not in self.brains:
            if name not in self.specs:
                raise ValueError(f"unknown brain '{name}'. Available: {sorted(self.specs)}")
            self.brains[name] = Brain(self.specs[name]).load()
        return self.brains[name]

    def system_prompt(self, max_steps: int) -> str:
        tools = (
            "(their signatures are in the <tools></tools> block below)"
            if self.prompt_style == "native"
            else self.registry.render_for_prompt()
        )
        react = REACT_SYSTEM_PROMPT.format(max_steps=max_steps, tools_description=tools)
        parts = [
            react,
            AGENT_DOMAIN_RULES,
            annex_iii_table(),
            FEW_SHOT_EXAMPLE if self.few_shot else "",
        ]
        return "\n".join(p for p in parts if p)

    def generate(self, messages: list[dict], brain: str) -> tuple[str, int]:
        """One generation with the chosen brain: (text, tokens processed)."""
        tools = self.registry.definitions() if self.prompt_style == "native" else None
        return self.brain(brain).generate(messages, tools=tools)

    def summarize(self, old: list[dict], brain: str) -> str:
        """The memory's summariser: the same brain, without thinking, with SUMMARY_PROMPT."""
        prompt = SUMMARY_PROMPT.format(history=_history(old))
        text, _ = self.brain(brain).generate(
            [{"role": "user", "content": prompt}], thinking=False, max_new_tokens=400, stop=False
        )
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # A 0.6B model can return nothing useful; then keep the (truncated) facts themselves.
        return text or _history(old)[:2000]

    def execute(self, name: str, arguments: dict) -> ToolCallRecord:
        """Run one tool with the confirmation policy and a timeout. Never raises."""
        spec = self.registry.tools.get(name)
        if spec is not None and spec.requires_confirmation and not self.auto_confirm:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=CONFIRMATION_REFUSED.format(name=name),
            )
        future = self._pool.submit(self.registry.call, name, arguments)
        try:
            return future.result(timeout=self.tool_timeout)
        except FutureTimeout:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=f"timeout: {name} did not answer in {self.tool_timeout:.0f} s. "
                "Try again later, with other arguments, or use another tool.",
                latency_ms=1000 * self.tool_timeout,
            )

    def _warm_up(self) -> None:
        """Load the knowledge base before the first timed call, so loading is not a timeout."""
        if "search_knowledge_base" in self.registry.tools:
            try:
                knowledge_base()
            except Exception:  # noqa: BLE001 - the tool will report the error when called
                pass

    def run(self, task: str, max_steps: int = 10, brain: str = "rlm") -> AgentResponse:
        """The ReAct loop. Every step in the response is one generation of the brain."""
        model = self.brain(brain)
        self._warm_up()
        # With the native prompt the template adds the tool definitions outside the memory's
        # text, so they come out of the budget here.
        reserved = (
            model.count_tokens(self.registry.render_for_prompt())
            if self.prompt_style == "native"
            else 0
        )
        memory = ConversationMemory(
            self.system_prompt(max_steps),
            count_tokens=model.count_tokens,
            token_budget=self.token_budget - reserved,
            pin_first=True,
            max_summarised_chars=600,
        )
        memory.add("user", task)
        steps: list[AgentStep] = []
        tokens = 0
        nudged = False
        evidence = False  # has any tool other than final_answer returned a result?
        challenged = False
        seen: Counter[str] = Counter()
        previous: dict[str, str] = {}

        def finish(answer: str, n_steps: int, reason: str, ok: bool = True) -> AgentResponse:
            return AgentResponse(
                final_answer=answer,
                steps=steps,
                n_steps=n_steps,
                succeeded=ok,
                model=model.spec.label,
                tokens_used=tokens,
                stop_reason=reason,
                memory_compressions=memory.compressions,
            )

        for step in range(max_steps):
            raw, used = self.generate(memory.render(), brain)
            tokens += used
            parsed = parse_action(raw, self.lenient_parse)
            thought = _thought(parsed.thinking, parsed.text)
            # Earlier turns go back without their <think> block, as Qwen's template does.
            content = raw.split("</think>", 1)[-1].strip()

            if not parsed.tool_calls:
                answer = _plain_answer(parsed.text)
                if not nudged and step < max_steps - 1:
                    nudged = True
                    steps.append(AgentStep(thought=thought, observation=NO_ACTION_NUDGE))
                    memory.add("assistant", content)
                    memory.add("user", NO_ACTION_NUDGE)
                    continue
                steps.append(AgentStep(thought=thought))
                return finish(answer, step + 1, "plain_answer", ok=bool(answer))

            call = parsed.tool_calls[0]
            action = AgentAction(name=call.name or "invalid_tool_call", arguments=call.arguments)
            if call.error:
                seen[f"invalid:{content}"] += 1
                if seen[f"invalid:{content}"] >= 3:
                    steps.append(AgentStep(thought=thought, action=action))
                    message = LOOP_MESSAGE.format(name="una herramienta (llamada mal formada)")
                    return finish(message, step + 1, "loop_detected", ok=False)
                observation = format_tool_result("", {"error": call.error})
            else:
                args = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
                key = f"{call.name}:{args}"
                seen[key] += 1
                if seen[key] >= 3:
                    steps.append(AgentStep(thought=thought, action=action))
                    message = LOOP_MESSAGE.format(name=call.name)
                    return finish(message, step + 1, "loop_detected", ok=False)
                if seen[key] == 2:
                    observation = REPEATED_CALL.format(name=call.name, previous=previous[key])
                else:
                    if (
                        call.name == "final_answer"
                        and self.require_evidence
                        and not (evidence or challenged)
                    ):
                        # Answering before looking anything up is how the small brains made up
                        # labels and fines. Ask once; a second final_answer is accepted (a task
                        # may really need no tool, or be impossible).
                        challenged = True
                        seen[key] -= 1
                        steps.append(
                            AgentStep(thought=thought, action=action, observation=NO_EVIDENCE)
                        )
                        memory.add("assistant", content)
                        memory.add("tool", NO_EVIDENCE, name="final_answer")
                        continue
                    record = self.execute(call.name, call.arguments)
                    evidence = evidence or (record.ok and call.name != "final_answer")
                    if record.ok and call.name == "final_answer":
                        steps.append(AgentStep(thought=thought, action=action))
                        return finish(str(record.result), step + 1, "final_answer")
                    payload = record.result if record.ok else {"error": record.error}
                    observation = format_tool_result(call.name, payload)
                    previous[key] = observation[: self.max_observation_chars]
            if len(parsed.tool_calls) > 1:
                observation += EXTRA_CALLS_IGNORED

            steps.append(AgentStep(thought=thought, action=action, observation=observation))
            memory.add("assistant", content)
            in_memory = observation
            if len(in_memory) > self.max_observation_chars:
                in_memory = in_memory[: self.max_observation_chars] + " [... truncated]"
            memory.add("tool", in_memory, name=action.name)
            memory.compress(lambda old: self.summarize(old, brain))

        return finish(MAX_STEPS_MESSAGE.format(max_steps=max_steps), max_steps, "max_steps", False)
