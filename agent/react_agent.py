"""The ReAct agent: Thought -> Action -> Observation, in a loop, until ``final_answer``.

This is where the phases meet: the brain (``agent.brains``, phase 1), the tools of phase 2
and the knowledge base of phase 3 (``agent.connectors``), and the memory (``agent.memory``).
The loop is ours, with no agent framework. ``ReActAgent.run`` is what ``POST /agent`` calls.

Decisions of the loop:

* **One action per step.** If the model writes several calls, only the first valid one runs
  and the observation says the rest were ignored, so every step of the trace has one action.
* **Errors are observations.** Unknown tool, invalid arguments, malformed JSON, an exception
  inside the tool, a timeout or an empty ``final_answer``: the model receives ``Error: ...``
  and can fix the call or change tool. The loop never crashes on the model's mistakes.
* **Loop detection.** A call identical to an earlier one (same tool and arguments) is not run
  again: the model gets the earlier result back with a warning. After ``MAX_REPEATED_CALLS``
  such repetitions the run stops with ``stop_reason="loop_detected"``.
* **Confirmation of side effects.** Tools with ``requires_confirmation`` (the PDF report) only
  run when the policy allows it: ``task`` (default: the user's task asks for a report),
  ``always`` or ``never`` (``ARCA_AGENT_CONFIRM``).
* **Plain answers.** If the model answers without calling any tool, its text is accepted as the
  final answer and flagged with ``stop_reason="plain_answer"``. Text that contains an
  ``Action:`` line is a call written in prose, not an answer: it gets an error with the format.
* **Memory.** ``ConversationMemory`` with a token budget; with a ``session_id`` the memory is
  kept between requests (multi-turn).

Configuration: ``ARCA_AGENT_PROMPT`` (``native``, default, or ``text``), ``ARCA_AGENT_RULES``
(``plain``, default, or ``example``), ``ARCA_AGENT_TOKEN_BUDGET`` (9000), ``ARCA_AGENT_CONFIRM``
and the brain variables of ``agent.brains``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from agent.brains import Brain
from agent.memory import ConversationMemory
from agent.prompts import (
    CANNOT_ANSWER,
    DOMAIN_RULES,
    DOMAIN_RULES_WITH_EXAMPLE,
    REACT_NATIVE_PROMPT,
    REACT_TEXT_PROMPT,
    SUMMARY_PROMPT,
)
from agent.toolbox import ToolCall, Toolbox, ToolSpec, parse_tool_calls
from api.schemas import AgentAction, AgentResponse, AgentStep

MAX_REPEATED_CALLS = 3
MAX_OBSERVATION_CHARS = 3000
REPORT_WORDS = re.compile(r"\b(informe|report|pdf)\b", re.IGNORECASE)
ACTION_IN_PROSE = re.compile(r"^\s*Action\s*:", re.MULTILINE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
INVALID_CALL = "invalid_tool_call"  # trace name of a call that could not be read

FINAL_ANSWER = ToolSpec(
    name="final_answer",
    description="Return the final answer to the user and stop.",
    parameters={
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "The complete answer."}},
        "required": ["answer"],
    },
    fn=lambda answer: answer,
)


@dataclass
class RunState:
    steps: list[AgentStep] = field(default_factory=list)
    tokens: int = 0
    seen_calls: dict[str, str] = field(default_factory=dict)  # call signature -> result
    repeated_calls: int = 0


@dataclass
class ReActAgent:
    brains: dict[str, Brain]
    toolbox: Toolbox
    prompt_style: str = "native"  # "native" (chat template renders the tools) or "text"
    rules: str = DOMAIN_RULES
    token_budget: int = 9000
    confirm: str = "task"  # "task", "always" or "never"
    sessions: dict[str, ConversationMemory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "final_answer" not in self.toolbox.tools:
            self.toolbox.add(FINAL_ANSWER)

    @classmethod
    def from_env(cls) -> ReActAgent:
        """The agent of the API: real brains and the tools of phases 2 and 3."""
        from agent.brains import brains_from_env
        from agent.connectors import default_toolbox

        rules = DOMAIN_RULES_WITH_EXAMPLE if os.environ.get("ARCA_AGENT_RULES") == "example" else (
            DOMAIN_RULES
        )
        return cls(
            brains=brains_from_env(),
            toolbox=default_toolbox(),
            prompt_style=os.environ.get("ARCA_AGENT_PROMPT", "native"),
            rules=rules,
            token_budget=int(os.environ.get("ARCA_AGENT_TOKEN_BUDGET", "9000")),
            confirm=os.environ.get("ARCA_AGENT_CONFIRM", "task"),
        )

    # ---------------------------------------------------------------- pieces of a step

    def system_prompt(self, max_steps: int) -> str:
        if self.prompt_style == "text":
            loop = REACT_TEXT_PROMPT.format(
                max_steps=max_steps, tools_description=self.toolbox.describe()
            )
        else:
            loop = REACT_NATIVE_PROMPT.format(max_steps=max_steps)
        return f"{loop}\n{self.rules}"

    def tool_definitions(self) -> list[dict] | None:
        """Native style: the chat template renders the tools. Text style: they are in the prompt."""
        return self.toolbox.definitions() if self.prompt_style == "native" else None

    def _memory(self, session_id: str | None, brain: Brain, max_steps: int) -> ConversationMemory:
        if session_id and session_id in self.sessions:
            memory = self.sessions[session_id]
            memory.unpin_all()  # the previous task may now be summarised
            return memory
        memory = ConversationMemory(
            system_prompt=self.system_prompt(max_steps),
            count_tokens=brain.count_tokens,
            token_budget=self.token_budget,
        )
        if session_id:
            self.sessions[session_id] = memory
        return memory

    def _summarize(self, brain: Brain, old: list[dict]) -> str:
        history = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        prompt = [{"role": "user", "content": SUMMARY_PROMPT.format(history=history)}]
        return THINK_RE.sub("", brain.generate(prompt)).strip()

    @staticmethod
    def _thought(raw: str) -> str:
        """The reasoning of a step: the <think> block if any, else the text before the call."""
        if match := THINK_RE.search(raw):
            return match.group(1).strip()
        before = raw.split("<tool_call>", 1)[0]
        return re.sub(r"^\s*Thought\s*:\s*", "", before.strip()).strip()

    def _confirmed(self, spec: ToolSpec, task: str) -> bool:
        if not spec.requires_confirmation or self.confirm == "always":
            return True
        if self.confirm == "never":
            return False
        return bool(REPORT_WORDS.search(task))

    def _observe(self, call: ToolCall, task: str, state: RunState) -> str:
        """Run one call (or explain why not) and return the observation for the model."""
        signature = json.dumps([call.name, call.arguments], sort_keys=True, ensure_ascii=False)
        if signature in state.seen_calls:
            state.repeated_calls += 1
            return (
                f"{state.seen_calls[signature]}\n(You already made this exact call: use this "
                "result and continue, or call final_answer.)"
            )
        spec = self.toolbox.tools.get(call.name)
        if spec is not None and not self._confirmed(spec, task):
            return (
                f"Error: {call.name} has side effects and the user did not ask for it. "
                "Do not call it unless the user asks for a report."
            )
        result = self.toolbox.run(call.name, call.arguments)
        observation = result.output if result.ok else f"Error: {result.output}"
        if len(observation) > MAX_OBSERVATION_CHARS:
            observation = observation[:MAX_OBSERVATION_CHARS] + " [...]"
        if result.ok:
            state.seen_calls[signature] = observation
        return observation

    # ---------------------------------------------------------------- the loop

    def run(
        self, task: str, max_steps: int = 10, brain: str = "remote", session_id: str | None = None
    ) -> AgentResponse:
        if brain not in self.brains:
            raise NotImplementedError(
                f"brain '{brain}' is not available (configured: {', '.join(self.brains)}); "
                "'rlm' needs ARCA_RLM_ADAPTER pointing to the phase-1 adapter"
            )
        b = self.brains[brain]
        state = RunState()
        memory = self._memory(session_id, b, max_steps)
        memory.add("user", task, pinned=True)

        def finish(answer: str, n_steps: int, reason: str, succeeded: bool = True):
            if session_id and answer:
                memory.add("assistant", answer)
            response = AgentResponse(
                final_answer=answer,
                steps=state.steps,
                n_steps=n_steps,
                succeeded=succeeded,
                model=b.label,
                tokens_used=state.tokens,
            )
            # Optional extras of the response (added to api/schemas.py when available).
            for extra, value in (
                ("stop_reason", reason),
                ("memory_compressions", memory.n_compressions),
            ):
                if extra in AgentResponse.model_fields:
                    setattr(response, extra, value)
            return response

        for step in range(max_steps):
            messages = memory.render()
            raw = b.generate(messages, tools=self.tool_definitions())
            state.tokens += b.count_tokens("".join(m["content"] for m in messages) + raw)

            if "<think>" in raw and "</think>" not in raw:
                observation = (
                    "Error: your reasoning was cut off at the token limit before any action. "
                    "Think more briefly, then call one tool."
                )
                state.steps.append(AgentStep(thought=raw.replace("<think>", "").strip(),
                                             observation=observation))
                memory.add("assistant", "(reasoning cut off)")
                memory.add("tool", observation, name="error")
                continue
            if raw.count("<tool_call>") > raw.count("</tool_call>"):
                raw += "\n</tool_call>"  # generation stopped right before the closing tag

            visible = raw.split("</think>", 1)[-1].strip()  # history keeps the visible part
            thought = self._thought(raw)
            calls = parse_tool_calls(visible)
            valid = [c for c in calls if c.error is None]

            if not valid:
                text = re.sub(r"<tool_call>.*?</tool_call>", "", visible, flags=re.DOTALL).strip()
                if not calls and text and not ACTION_IN_PROSE.search(text):
                    state.steps.append(AgentStep(thought=thought or "(answered directly)"))
                    return finish(text, step + 1, "plain_answer")
                error = calls[0].error if calls else (
                    "your reply had no <tool_call> block. Write the action as a <tool_call> "
                    "block with the JSON of the call, or call final_answer"
                )
                observation = (
                    f"Error: {error}. Write the call as <tool_call>"
                    '{"name": ..., "arguments": {...}}</tool_call>.'
                )
                action = (
                    AgentAction(name=INVALID_CALL, arguments={"raw": calls[0].raw}) if calls else None
                )
                state.steps.append(AgentStep(thought=thought, action=action, observation=observation))
                memory.add("assistant", visible or "(empty)")
                memory.add("tool", observation, name="error")
                memory.compress(lambda old: self._summarize(b, old))
                continue

            call = valid[0]
            action = AgentAction(name=call.name, arguments=call.arguments)
            if call.name == "final_answer":
                answer = call.arguments.get("answer")
                if not isinstance(answer, str):  # tolerate {"result": "..."} and friends
                    strings = [v for v in call.arguments.values() if isinstance(v, str)]
                    answer = strings[0] if strings else ""
                if answer.strip():
                    state.steps.append(AgentStep(thought=thought, action=action))
                    return finish(answer.strip(), step + 1, "final_answer")
                observation = "Error: final_answer needs a non-empty 'answer' string."
            else:
                observation = self._observe(call, task, state)
            if len(calls) > 1:
                observation += (
                    f"\n(Only the first tool call of each step is executed; "
                    f"{len(calls) - 1} more were ignored.)"
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
            memory.compress(lambda old: self._summarize(b, old))

        return finish(
            f"{CANNOT_ANSWER}: no he podido completar la tarea en {max_steps} pasos.",
            max_steps,
            "max_steps",
            succeeded=False,
        )
