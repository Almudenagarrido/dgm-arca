"""The ReAct agent: Thought -> Action -> Observation, in a loop, with every tool we built.

This is where phases 1 to 3 meet. The brain is a model (``brain="base"``, our phase-1 RLM
with ``brain="rlm"``, or a thinking model with ``brain="thinking"``); the hands are the
phase-2 registry plus ``search_knowledge_base`` over the phase-3 retriever; the loop is ours,
with no agent framework. ``ReActAgent.run`` is what ``POST /agent`` calls.

Decisions of the loop (all measured or justified in ``docs/EXPERIMENTS.md``):

* **One action per step.** If the model writes several ``<tool_call>`` blocks, the first valid
  one is executed and the observation says the rest were ignored: the trace then has exactly
  one action per step, which is what ReAct and ``max_steps`` count.
* **Generation stops at ``</tool_call>``** (and at ``Observation:``), so the model cannot write
  an invented observation after its action.
* **Timeouts per tool** (``TOOL_TIMEOUTS``, ``ARCA_AGENT_TOOL_TIMEOUT``): the tool runs in a
  worker thread and, if it does not answer in time, the model receives a timeout error as
  observation and decides what to do. Python cannot kill a thread; the late result is dropped.
* **Error recovery.** Unknown tool, invalid arguments, malformed JSON, an exception inside the
  tool, a timeout, an empty ``final_answer``: all become an observation starting with
  ``Error:``, never an exception, so the model can fix the call or change tool.
* **Loop detection.** A call identical to an earlier one (same name and arguments) is not run
  again: the observation repeats the earlier result and tells the model to use it. After
  ``MAX_REPEATED_CALLS`` such repetitions the run stops with ``stop_reason="loop_detected"``.
* **Confirmation.** ``write_assessment_report`` has side effects. With
  ``ARCA_AGENT_CONFIRM=task`` (default) it runs only if the task asks for a report: the user's
  request is the confirmation. ``always`` and ``never`` are the other two policies.
* **Faithful trace.** A ``<tool_call>`` block that cannot be parsed is recorded as the action
  ``invalid_tool_call`` with the raw text of the block, next to the error the model received.
* **Plain answers.** If the model answers without calling any tool, the text is accepted as
  the final answer and flagged with ``stop_reason="plain_answer"``. A text with an ``Action:``
  line is not an answer but a call written in prose (the base model does it): it gets an
  error observation with the expected format.
* **Memory.** ``ConversationMemory`` with the policy in ``agent/memory.py``; with a
  ``session_id`` the memory survives between requests (multi-turn).

Configuration (all optional): ``ARCA_RLM_ADAPTER`` (phase-1 LoRA; its base model is the base
brain), ``ARCA_THINKING_MODEL`` (default ``Qwen/Qwen3-1.7B``), ``ARCA_AGENT_BASE_THINKING`` and
``ARCA_AGENT_RLM_THINKING`` (Qwen3 thinking switch for those brains, ``0`` and ``1``),
``ARCA_AGENT_PROMPT`` (``native``, default: Qwen3's chat template renders the tools, as in
phase 2; ``text``: the ReAct prompt of the slides with the tools pasted as text),
``ARCA_AGENT_TOKEN_BUDGET`` (memory budget, default 9000), ``ARCA_AGENT_CONFIRM``,
``ARCA_AGENT_TOOL_TIMEOUT`` (seconds, overrides every tool), ``ARCA_AGENT_SEED``.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from agent.memory import ConversationMemory
from agent.prompts import (
    CANNOT_ANSWER,
    DOMAIN_RULES,
    REACT_NATIVE_PROMPT,
    REACT_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
)
from api.schemas import AgentAction, AgentResponse, AgentStep, ToolCallRecord
from tool_use.parser import TOOL_CALL_PATTERN, ToolCall, format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import default_registry

log = logging.getLogger("arca.agent")

# Seconds. lookup_ai_act goes to the EU Publications Office; the knowledge base runs the
# reranker over 30 candidates (and a translation with the 4B model); the rest is local and fast.
TOOL_TIMEOUTS = {
    "lookup_ai_act": 30.0,
    "search_knowledge_base": 60.0,
    "write_assessment_report": 20.0,
}
DEFAULT_TOOL_TIMEOUT = 10.0
MAX_REPEATED_CALLS = 3
MAX_OBSERVATION_CHARS = 3000
KB_PASSAGE_CHARS = 700
REPORT_WORDS = re.compile(r"\b(informe|report|pdf)\b", re.IGNORECASE)
INVALID_CALL = "invalid_tool_call"  # trace name of a <tool_call> block that could not be parsed
ACTION_IN_PROSE = re.compile(r"^\s*Action\s*:", re.MULTILINE)
STOP_STRINGS = ["</tool_call>", "\nObservation:"]


# ---------------------------------------------------------------------------- tools


@tool
def final_answer(answer: str) -> str:
    """Return the final answer to the user and stop the loop.

    Args:
        answer: The complete answer to the task.
    """
    return answer


_kb = None
_kb_lock = threading.Lock()


def knowledge_base():
    """The phase-3 pipeline, shared with ``POST /rag`` (built once per process)."""
    global _kb
    with _kb_lock:
        if _kb is None:
            from rag.generate import build_pipeline

            _kb = build_pipeline()
    return _kb


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the official corpus of the AI Act (the regulation in Spanish, its recitals and
    the Commission guidelines) and return the most relevant passages with their ids.

    Args:
        query: What to look for, phrased as a question.
        top_k: How many passages to return (1 to 8).
    """
    top_k = max(1, min(int(top_k), 8))
    kb = knowledge_base()
    hits = kb.retrieve(query, top_k=top_k, retriever="hybrid")
    if not hits:
        return "No passages found."
    blocks = []
    # Same gate as /rag: when not even the best passage passes the reranker threshold, say it,
    # so that the agent can answer "I cannot answer" instead of using loosely related text.
    if kb.rerank and hits[0].score < kb.min_score:
        blocks.append(
            f"Warning: no passage reaches the relevance threshold (best score "
            f"{hits[0].score:.3f} < {kb.min_score}). The knowledge base probably does not "
            "contain the answer."
        )
    for hit in hits:
        text = hit.chunk.text
        if len(text) > KB_PASSAGE_CHARS:
            text = text[:KB_PASSAGE_CHARS] + " [...]"
        blocks.append(f"[{hit.chunk.id}] ({hit.chunk.source})\n{text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------- brains


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


@dataclass
class Brain:
    """One model that can drive the loop, loaded on first use."""

    name: str
    model_id: str
    adapter: str | None = None
    thinking: bool = False
    tokenizer: Any = None
    model: Any = None

    @property
    def label(self) -> str:
        suffix = f"+{os.path.basename(self.adapter.rstrip('/'))}" if self.adapter else ""
        return f"{self.model_id}{suffix}" + (" (thinking)" if self.thinking else "")

    def load(self) -> None:
        if self.model is not None:
            return
        if self.adapter is None and self.name == "rlm":
            raise NotImplementedError(
                "brain 'rlm' needs ARCA_RLM_ADAPTER pointing to the phase-1 LoRA adapter "
                "(see rlm/README.md); brains 'base' and 'thinking' work without it"
            )
        if self.adapter is None:
            from rlm.gpu import shared_causal_lm

            # Shared with phases 2 and 3 when the ids coincide (one copy on the GPU).
            self.tokenizer, self.model = shared_causal_lm(self.model_id)
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from rlm.gpu import disable_cudnn_attention

        disable_cudnn_attention()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        base = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, device_map=device)
        self.model = PeftModel.from_pretrained(base, self.adapter).eval()


def brain_specs() -> dict[str, Brain]:
    """The three brains from the environment. Base and RLM share the base model id."""
    from rlm.inference import _base_model_from_adapter

    adapter = os.environ.get("ARCA_RLM_ADAPTER") or None
    small = os.environ.get("ARCA_RLM_BASE_MODEL") or (
        _base_model_from_adapter(adapter) if adapter else "Qwen/Qwen3-0.6B"
    )
    return {
        "base": Brain("base", small, thinking=_flag("ARCA_AGENT_BASE_THINKING", "0")),
        "rlm": Brain("rlm", small, adapter=adapter, thinking=_flag("ARCA_AGENT_RLM_THINKING", "1")),
        "thinking": Brain(
            "thinking", os.environ.get("ARCA_THINKING_MODEL", "Qwen/Qwen3-1.7B"), thinking=True
        ),
    }


BRAINS = brain_specs


class StopOutsideThinking:
    """Stop at any of ``stops``, but never inside a ``<think>`` block.

    ``generate(stop_strings=...)`` looks for the strings in the whole completion, reasoning
    included, so a thinking model that drafts its ``<tool_call>`` (or ``<code>`` block) while
    reasoning would be cut before acting. This criterion follows the think tags (single tokens
    in Qwen3) and only looks for the stop strings in the tail of the visible part.
    """

    def __init__(self, tokenizer, stops: list[str], prompt_len: int, tail_tokens: int = 24):
        self.tokenizer, self.stops = tokenizer, stops
        self.start = prompt_len
        self.tail_tokens = tail_tokens
        self.think_open = tokenizer.convert_tokens_to_ids("<think>")
        self.think_close = tokenizer.convert_tokens_to_ids("</think>")
        self.inside = False

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        last = int(input_ids[0, -1])
        if last == self.think_open:
            self.inside = True
        elif last == self.think_close:
            self.inside = False
            self.start = input_ids.shape[1]
        hit = False
        if not self.inside:
            tail = input_ids[0, max(self.start, input_ids.shape[1] - self.tail_tokens) :]
            text = self.tokenizer.decode(tail, skip_special_tokens=True)
            hit = any(stop in text for stop in self.stops)
        return torch.full((input_ids.shape[0],), hit, dtype=torch.bool, device=input_ids.device)


# ---------------------------------------------------------------------------- the agent


@dataclass
class RunState:
    """What one run accumulates; returned to the caller inside ``AgentResponse``."""

    steps: list[AgentStep] = field(default_factory=list)
    tokens: int = 0
    seen_calls: dict[str, str] = field(default_factory=dict)
    repeated_calls: int = 0


@dataclass
class ReActAgent:
    registry: ToolRegistry = field(default_factory=default_registry)
    token_budget: int = field(
        default_factory=lambda: int(os.environ.get("ARCA_AGENT_TOKEN_BUDGET", "9000"))
    )
    keep_last: int = 6
    prompt_style: str = field(default_factory=lambda: os.environ.get("ARCA_AGENT_PROMPT", "native"))
    confirm_policy: str = field(
        default_factory=lambda: os.environ.get("ARCA_AGENT_CONFIRM", "task")
    )
    brains: dict[str, Brain] = field(default_factory=brain_specs)
    sessions: dict[str, ConversationMemory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._state = RunState()
        self.last_usage = (0, 0)

    @classmethod
    def from_env(cls) -> ReActAgent:
        agent = cls()
        agent.registry.register(final_answer, search_knowledge_base)
        agent.load()
        return agent

    def load(self) -> None:
        """Check the configuration. Models load lazily, per brain, on their first request.

        The knowledge base is built on the first ``search_knowledge_base`` call, outside the
        tool timeout. If it cannot be built (no corpus) the agent still runs: the tool fails at
        call time and the model sees the error, which is the behaviour we want with a broken tool.
        """
        adapter = self.brains["rlm"].adapter
        if adapter and not os.path.isdir(adapter):
            raise FileNotFoundError(f"ARCA_RLM_ADAPTER points to a missing folder: {adapter}")

    # -- model ------------------------------------------------------------------------------

    def brain(self, name: str) -> Brain:
        if name not in self.brains:
            raise ValueError(f"unknown brain '{name}'. Available: {sorted(self.brains)}")
        brain = self.brains[name]
        brain.load()
        return brain

    def system_prompt(self, max_steps: int) -> str:
        """``native``: the template adds the tools; ``text``: the ReAct prompt of the slides."""
        if self.prompt_style == "native":
            return f"{REACT_NATIVE_PROMPT.format(max_steps=max_steps)}\n{DOMAIN_RULES}"
        react = REACT_SYSTEM_PROMPT.format(
            max_steps=max_steps, tools_description=self.registry.render_for_prompt()
        )
        return f"{react}\n{DOMAIN_RULES}"

    def tool_definitions(self) -> list[dict] | None:
        """What goes to ``apply_chat_template(tools=...)`` in the loop (``native`` only)."""
        return self.registry.definitions() if self.prompt_style == "native" else None

    def count_tokens(self, text: str, brain: str) -> int:
        return len(self.brains[brain].tokenizer.encode(text))

    def generate(
        self,
        messages: list[dict],
        brain: str,
        thinking: bool | None = None,
        max_new_tokens: int | None = None,
        stop: bool | list[str] = True,
        tools: list[dict] | None = None,
    ) -> str:
        """One generation with the chosen brain. Adds prompt + new tokens to the run's cost.

        Thinking brains sample with the settings of the Qwen3 model card (greedy decoding loops
        while thinking); the others decode greedily. Generation stops after ``</tool_call>``
        (or after the strings in ``stop`` when it is a list: the code agent uses its own).
        """
        import torch

        b = self.brain(brain)
        stop_strings = stop if isinstance(stop, list) else (STOP_STRINGS if stop else None)
        thinking = b.thinking if thinking is None else thinking
        max_new_tokens = max_new_tokens or (2048 if thinking else 512)
        text = b.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
            **({"tools": tools} if tools else {}),
        )
        inputs = b.tokenizer(text, return_tensors="pt").to(b.model.device)
        sampling = (
            {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
            if thinking
            else {"do_sample": False}
        )
        criteria = (
            [StopOutsideThinking(b.tokenizer, stop_strings, inputs["input_ids"].shape[1])]
            if stop_strings
            else None
        )
        with torch.no_grad():
            out = b.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=b.tokenizer.pad_token_id or b.tokenizer.eos_token_id,
                stopping_criteria=criteria,
                **sampling,
            )
        new = out[0, inputs["input_ids"].shape[1] :]
        self.last_usage = (int(inputs["input_ids"].shape[1]), int(new.numel()))
        self._state.tokens += sum(self.last_usage)
        raw = b.tokenizer.decode(new, skip_special_tokens=True)
        # Qwen3 with thinking on writes "<think>" in the prompt, so the completion starts inside
        # the block: put the opening tag back so the parser finds the reasoning.
        if thinking and "</think>" in raw and "<think>" not in raw:
            raw = "<think>" + raw
        return raw.split("\nObservation:", 1)[0] if stop is True else raw

    def summarize(self, old: list[dict], brain: str) -> str:
        """The memory's compression call: the same brain, no thinking, a short summary."""
        history = "\n".join(
            f"{m['role']}{' (' + m['name'] + ')' if m.get('name') else ''}: {m['content']}"
            for m in old
        )
        raw = self.generate(
            [{"role": "user", "content": SUMMARY_PROMPT.format(history=history)}],
            brain,
            thinking=False,
            max_new_tokens=384,
            stop=False,
        )
        parsed = parse_tool_calls(raw)
        return re.sub(r"</?answer>", "", parsed.text).strip()

    # -- tools ------------------------------------------------------------------------------

    def _confirmed(self, name: str, task: str) -> bool:
        spec = self.registry.tools.get(name)
        if spec is None or not spec.requires_confirmation:
            return True
        if self.confirm_policy == "always":
            return True
        if self.confirm_policy == "never":
            return False
        return bool(REPORT_WORDS.search(task))

    def execute(self, call: ToolCall, task: str) -> ToolCallRecord:
        """Run one call with its timeout and the confirmation policy. Never raises."""
        if not self._confirmed(call.name, task):
            return ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=(
                    f"'{call.name}' has side effects and the user did not ask for it, so it "
                    "was not executed. Do not call it unless the user asks for a report."
                ),
            )
        timeout = float(
            os.environ.get("ARCA_AGENT_TOOL_TIMEOUT")
            or TOOL_TIMEOUTS.get(call.name, DEFAULT_TOOL_TIMEOUT)
        )
        if call.name == search_knowledge_base.name:
            try:  # first use builds the index and loads the reranker: not part of the timeout
                knowledge_base()
            except Exception as exc:  # noqa: BLE001
                return ToolCallRecord(
                    name=call.name,
                    arguments=call.arguments,
                    ok=False,
                    error=f"the knowledge base is unavailable ({type(exc).__name__}: {exc}). "
                    "Use the other tools or say that you cannot answer.",
                )
        future = self._pool.submit(self.registry.call, call.name, call.arguments)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=f"timeout: '{call.name}' did not answer in {timeout:.0f} s. Try again "
                "later with other arguments, use another tool, or answer with what you have.",
                latency_ms=1000 * timeout,
            )

    # -- the loop ---------------------------------------------------------------------------

    def _memory(self, session_id: str | None, brain: str, max_steps: int) -> ConversationMemory:
        memory = self.sessions.get(session_id) if session_id else None
        # With the native template the definitions are not in the system prompt string, but
        # they are in the context: the memory counts them as a fixed overhead.
        tools = self.tool_definitions()
        overhead = self.count_tokens(json.dumps(tools, ensure_ascii=False), brain) if tools else 0

        def count(text: str) -> int:
            return self.count_tokens(text, brain) + overhead

        if memory is None:
            memory = ConversationMemory(
                self.system_prompt(max_steps),
                count_tokens=count,
                token_budget=self.token_budget,
                keep_last=self.keep_last,
            )
            if session_id:
                self.sessions[session_id] = memory
        else:
            memory.system_prompt = self.system_prompt(max_steps)
            memory.count_tokens = count
            memory.unpin_all()  # the previous task may now be summarised
        return memory

    @staticmethod
    def _thought(raw: str) -> str:
        parsed = parse_tool_calls(raw)
        text = re.sub(r"^\s*Thought:\s*", "", parsed.text).strip()
        text = re.sub(r"\s*Action:\s*$", "", text).strip()
        return "\n\n".join(part for part in (parsed.thinking, text) if part) or "(no thought)"

    def run(
        self,
        task: str,
        max_steps: int = 10,
        brain: str = "rlm",
        session_id: str | None = None,
    ) -> AgentResponse:
        """The ReAct loop. One request at a time: the brains share one GPU."""
        with self._lock:
            return self._run(task, max_steps, brain, session_id)

    def _run(self, task: str, max_steps: int, brain: str, session_id: str | None) -> AgentResponse:
        if seed := os.environ.get("ARCA_AGENT_SEED", "0"):
            from transformers import set_seed

            set_seed(int(seed))
        b = self.brain(brain)
        self._state = state = RunState()
        memory = self._memory(session_id, brain, max_steps)
        memory.add("user", task, pinned=True)

        def finish(answer: str, n_steps: int, reason: str, succeeded: bool = True):
            if session_id and answer:
                memory.add("assistant", answer)
            return AgentResponse(
                final_answer=answer,
                steps=state.steps,
                n_steps=n_steps,
                succeeded=succeeded,
                model=b.label,
                tokens_used=state.tokens,
                stop_reason=reason,
                memory_compressions=memory.n_compressions,
            )

        for step in range(max_steps):
            raw = self.generate(memory.render(), brain, tools=self.tool_definitions())
            if "<think>" in raw and "</think>" not in raw:
                # The token limit cut the reasoning: there is no action to run.
                observation = (
                    "Error: your reasoning was cut off at the token limit before any Action. "
                    "Think more briefly, then call one tool."
                )
                state.steps.append(
                    AgentStep(thought=raw.replace("<think>", "").strip(), observation=observation)
                )
                memory.add("assistant", "(reasoning cut off)")
                memory.add("tool", observation, name="error")
                continue
            if raw.count("<tool_call>") > raw.count("</tool_call>"):
                raw += "\n</tool_call>"  # generation stopped right before the closing tag
            parsed = parse_tool_calls(raw)
            thought = self._thought(raw)
            # The history keeps the visible part only, as Qwen's template does with <think>.
            visible = raw.split("</think>", 1)[-1].strip()
            valid = [c for c in parsed.tool_calls if c.error is None]

            if not valid:
                text = re.sub(r"</?answer>", "", parsed.text).strip()
                # "Action: compute_max_fine" in prose is an attempted call, not an answer.
                if not parsed.tool_calls and text and not ACTION_IN_PROSE.search(text):
                    # Answered without the final_answer tool: accept it, flagged.
                    state.steps.append(AgentStep(thought=parsed.thinking or "(answered directly)"))
                    return finish(text, step + 1, "plain_answer")
                error = (
                    parsed.tool_calls[0].error
                    if parsed.tool_calls
                    else "your reply had no <tool_call> block. Write the Action as a <tool_call> "
                    "block with the JSON of the call, or call final_answer"
                )
                observation = (
                    f"Error: {error}. Write the call as <tool_call>"
                    '{"name": ..., "arguments": {...}}</tool_call>.'
                )
                # The trace keeps what the model actually wrote, so that a malformed call can be
                # read in the trace and not only its error message.
                blocks = TOOL_CALL_PATTERN.findall(visible)
                action = (
                    AgentAction(name=INVALID_CALL, arguments={"raw": blocks[0].strip()})
                    if blocks
                    else None
                )
                state.steps.append(
                    AgentStep(thought=thought, action=action, observation=observation)
                )
                memory.add("assistant", visible or "(empty)")
                memory.add("tool", observation, name="error")
                memory.compress(lambda old: self.summarize(old, brain))
                continue

            call = valid[0]
            action = AgentAction(name=call.name, arguments=call.arguments)
            if call.name == "final_answer":
                answer = call.arguments.get("answer")
                if not isinstance(answer, str):  # tolerate {"result": "..."} and friends
                    strings = [v for v in call.arguments.values() if isinstance(v, str)]
                    answer = strings[0] if strings else ""
                if answer.strip():
                    state.steps.append(AgentStep(thought=thought, action=action, observation=None))
                    return finish(answer.strip(), step + 1, "final_answer")
                observation = "Error: final_answer needs a non-empty 'answer' string."
            else:
                observation = self._observe(call, task)
            if len(parsed.tool_calls) > 1:
                observation += (
                    f"\n(Only the first tool call of each step is executed; "
                    f"{len(parsed.tool_calls) - 1} more were ignored.)"
                )
            state.steps.append(AgentStep(thought=thought, action=action, observation=observation))
            if state.repeated_calls >= MAX_REPEATED_CALLS:
                return finish(
                    f"{CANNOT_ANSWER}: el agente ha repetido la misma llamada "
                    f"{state.repeated_calls} veces sin avanzar.",
                    step + 1,
                    "loop_detected",
                    succeeded=False,
                )
            memory.add("assistant", visible)
            memory.add("tool", observation, name=call.name)
            memory.compress(lambda old: self.summarize(old, brain))

        return finish(
            f"No he podido completar la tarea en {max_steps} pasos.",
            max_steps,
            "max_steps",
            succeeded=False,
        )

    def _observe(self, call: ToolCall, task: str) -> str:
        """Execute (or not, if repeated) and render the observation the model will read."""
        state = self._state
        key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
        if key in state.seen_calls:
            state.repeated_calls += 1
            return (
                "Error: you already made this exact call; its result was:\n"
                f"{state.seen_calls[key]}\nUse it, try different arguments, or call final_answer."
            )
        record = self.execute(call, task)
        if record.ok:
            observation = format_tool_result(record.name, record.result)
        else:
            observation = f"Error: {record.error}"
        if len(observation) > MAX_OBSERVATION_CHARS:
            observation = observation[:MAX_OBSERVATION_CHARS] + " [...]"
        state.seen_calls[key] = observation
        return observation
