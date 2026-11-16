"""The ReAct loop with a scripted brain: no GPU, no network, no corpus."""

from __future__ import annotations

import time

from agent.memory import ConversationMemory
from agent.react_agent import Brain, ReActAgent, final_answer
from tool_use.registry import tool
from tool_use.tools import default_registry


@tool
def slow_tool(seconds: float) -> str:
    """Sleep and return.

    Args:
        seconds: How long to sleep.
    """
    time.sleep(seconds)
    return "done"


@tool
def broken_tool(x: int) -> int:
    """Always fails.

    Args:
        x: Anything.
    """
    raise RuntimeError("service unavailable")


def call(name: str, arguments: str) -> str:
    body = f'{{"name": "{name}", "arguments": {arguments}}}'
    return f"Thought: I need {name}.\n<tool_call>\n{body}\n</tool_call>"


class ScriptedAgent(ReActAgent):
    """Replays fixed completions and records the prompts it receives."""

    def __init__(self, script: list[str], **kwargs):
        registry = default_registry().register(final_answer, slow_tool, broken_tool)
        brains = {name: Brain(name, "fake") for name in ("base", "rlm", "thinking")}
        super().__init__(registry=registry, brains=brains, **kwargs)
        self.script = list(script)
        self.prompts: list[list[dict]] = []

    def brain(self, name: str) -> Brain:
        return self.brains[name]

    def count_tokens(self, text: str, brain: str) -> int:
        return len(text) // 4

    def generate(
        self, messages, brain, thinking=None, max_new_tokens=None, stop=True, tools=None
    ) -> str:
        self.prompts.append(messages)
        self.tools_seen = tools
        self._state.tokens += 10
        if not stop:  # the memory's summary call
            return "summary of the old turns"
        return self.script.pop(0) if self.script else call("final_answer", '{"answer": "fin"}')


def test_chains_tools_and_stops_at_final_answer():
    agent = ScriptedAgent(
        [
            call(
                "compute_max_fine",
                '{"infringement": "prohibited_practice", "annual_turnover_eur": 1000000000}',
            ),
            call("calculator", '{"expression": "70000000 / 2"}'),
            call("final_answer", '{"answer": "La mitad son 35000000 EUR"}'),
        ]
    )
    response = agent.run("tarea", max_steps=5, brain="base")
    assert response.succeeded and response.stop_reason == "final_answer"
    assert response.final_answer == "La mitad son 35000000 EUR"
    assert response.n_steps == 3
    assert [s.action.name for s in response.steps] == [
        "compute_max_fine",
        "calculator",
        "final_answer",
    ]
    assert "70.000.000 EUR" in response.steps[0].observation  # the real tool result
    assert response.steps[1].observation == "35000000.0"
    assert response.steps[0].thought == "I need compute_max_fine."
    assert response.tokens_used == 30
    # The observation of step 1 is in the prompt of step 2, as a tool message.
    assert agent.prompts[1][-1]["role"] == "tool"


def test_respects_max_steps():
    agent = ScriptedAgent([call("calculator", f'{{"expression": "{i} + 1"}}') for i in range(10)])
    response = agent.run("tarea", max_steps=3, brain="rlm")
    assert not response.succeeded and response.stop_reason == "max_steps"
    assert response.n_steps == 3 and len(response.steps) == 3


def test_errors_become_observations_and_the_agent_recovers():
    agent = ScriptedAgent(
        [
            call("no_such_tool", "{}"),
            call("calculator", '{"expr": "1 + 1"}'),
            call("broken_tool", '{"x": 1}'),
            "<tool_call>{not json}</tool_call>",
            call("calculator", '{"expression": "1 + 1"}'),
            call("final_answer", '{"answer": "2"}'),
        ]
    )
    response = agent.run("tarea", max_steps=10, brain="base")
    observations = [s.observation or "" for s in response.steps]
    assert "unknown tool" in observations[0]
    assert "invalid arguments" in observations[1]
    assert "service unavailable" in observations[2]
    assert "not valid JSON" in observations[3]
    assert response.steps[3].action.name == "invalid_tool_call"
    assert response.steps[3].action.arguments["raw"] == "{not json}"
    assert observations[4] == "2.0"
    assert response.succeeded and response.final_answer == "2"


def test_tool_timeout(monkeypatch):
    monkeypatch.setenv("ARCA_AGENT_TOOL_TIMEOUT", "0.2")
    agent = ScriptedAgent([call("slow_tool", '{"seconds": 2}')])
    started = time.perf_counter()
    response = agent.run("tarea", max_steps=3, brain="base")
    assert time.perf_counter() - started < 1.5
    assert response.steps[0].observation.startswith("Error: timeout")
    assert response.succeeded


def test_repeated_calls_are_not_rerun_and_stop_the_loop():
    same = call("calculator", '{"expression": "2 * 3"}')
    agent = ScriptedAgent([same] * 6)
    response = agent.run("tarea", max_steps=10, brain="base")
    assert "already made this exact call" in response.steps[1].observation
    assert response.stop_reason == "loop_detected" and not response.succeeded
    assert response.n_steps == 4  # one real call + three repetitions


def test_report_needs_the_user_to_ask_for_it():
    args = (
        '{"system_name": "x", "description": "d", "label": "alto_riesgo", '
        '"legal_basis": "Anexo III.4.a", "justification": "j"}'
    )
    agent = ScriptedAgent([call("write_assessment_report", args)])
    response = agent.run("Clasifica este sistema", max_steps=3, brain="base")
    assert "not executed" in response.steps[0].observation
    assert agent._confirmed("write_assessment_report", "Clasifícalo y genera el informe en PDF")


def test_plain_answer_is_accepted_but_flagged():
    agent = ScriptedAgent(["<think>easy</think>Es riesgo mínimo."])
    response = agent.run("tarea", max_steps=3, brain="thinking")
    assert response.final_answer == "Es riesgo mínimo." and response.stop_reason == "plain_answer"
    assert response.steps[0].thought == "easy"


def test_truncated_reasoning_is_reported_to_the_model():
    agent = ScriptedAgent(["<think>a very long reasoning that never ends"])
    response = agent.run("tarea", max_steps=2, brain="thinking")
    assert "cut off" in response.steps[0].observation
    assert response.succeeded  # the default script then answers


def test_only_first_parallel_call_runs():
    two = (
        call("calculator", '{"expression": "1 + 1"}')
        + "\n"
        + call("calculator", '{"expression": "2 + 2"}')
    )
    agent = ScriptedAgent([two])
    response = agent.run("tarea", max_steps=3, brain="base")
    assert response.steps[0].observation.startswith("2.0")
    assert "1 more were ignored" in response.steps[0].observation


def test_memory_summarises_and_keeps_the_task(monkeypatch):
    agent = ScriptedAgent(
        [call("calculator", f'{{"expression": "{i} * 1000"}}') for i in range(8)],
        token_budget=1,  # every step is over budget
        keep_last=2,
    )
    response = agent.run("LA TAREA", max_steps=9, brain="base")
    assert response.memory_compressions > 0
    last_prompt = agent.prompts[-1]
    assert "summary of the old turns" in last_prompt[0]["content"]
    assert any(m["content"] == "LA TAREA" for m in last_prompt[1:])


def test_sessions_share_memory_between_requests():
    agent = ScriptedAgent(
        [
            call("final_answer", '{"answer": "primera"}'),
            call("final_answer", '{"answer": "segunda"}'),
        ]
    )
    agent.run("pregunta 1", max_steps=3, brain="base", session_id="s1")
    agent.run("pregunta 2", max_steps=3, brain="base", session_id="s1")
    contents = [m["content"] for m in agent.prompts[-1]]
    assert "pregunta 1" in contents and "primera" in contents and "pregunta 2" in contents


def test_memory_truncates_old_observations_before_summarising():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=800, keep_last=2)
    memory.add("user", "task", pinned=True)
    memory.add("assistant", "call")
    memory.add("tool", "x" * 1000)
    memory.add("assistant", "call 2")
    memory.add("tool", "short")
    assert memory.compress(lambda old: "never called")
    assert memory.summary == "" and memory.n_truncations == 1
    assert memory.messages[2]["content"].startswith("x" * 600)


def test_memory_recent_window_never_starts_with_a_tool_message():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=10, keep_last=1)
    memory.add("user", "task", pinned=True)
    memory.add("assistant", "call")
    memory.add("tool", "result")
    pinned, old, recent = memory._split()
    assert recent[0]["role"] == "assistant" and old == [] and pinned[0]["content"] == "task"


def test_action_written_in_prose_is_not_an_answer():
    agent = ScriptedAgent(["Thought: I need the fine.\n\nAction: compute_max_fine"])
    response = agent.run("tarea", max_steps=3, brain="base")
    assert "no <tool_call> block" in response.steps[0].observation
    assert response.final_answer == "fin" and response.n_steps == 2


def test_call_without_name_is_kept_in_the_trace():
    agent = ScriptedAgent(
        ['Action: classify_risk\n<tool_call>\n{"reference": "Art. 50.1"}\n</tool_call>']
    )
    response = agent.run("tarea", max_steps=3, brain="base")
    step = response.steps[0]
    assert "no 'name' field" in step.observation
    assert step.action.name == "invalid_tool_call"
    assert step.action.arguments["raw"] == '{"reference": "Art. 50.1"}'


def test_native_prompt_passes_the_tools_to_the_template(monkeypatch):
    agent = ScriptedAgent([], prompt_style="native")
    agent.run("tarea", max_steps=2, brain="base")
    assert {t["function"]["name"] for t in agent.tools_seen} >= {"classify_risk", "final_answer"}
    assert "classify_risk" in agent.prompts[0][0]["content"]  # named in the rules only
    assert '"parameters"' not in agent.prompts[0][0]["content"]

    text = ScriptedAgent([], prompt_style="text")
    text.run("tarea", max_steps=2, brain="base")
    assert text.tools_seen is None
    assert '"parameters"' in text.prompts[0][0]["content"]


def test_prompts_have_no_example_arguments():
    agent = ScriptedAgent([])
    assert "Art. 50.1" not in agent.system_prompt(10)
