"""The ReAct loop, checked with a scripted fake brain and fake tools: no model, no GPU."""

import json
import time

from agent.react_agent import ReActAgent
from agent.toolbox import Toolbox, ToolSpec, parse_tool_calls


class FakeBrain:
    """Returns the prepared outputs one after another, and records what it was shown."""

    label = "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen = []

    def generate(self, messages, tools=None):
        self.seen.append((messages, tools))
        return self.outputs.pop(0)

    def count_tokens(self, text):
        return len(text.split())


def call(name, **arguments):
    return f"Thought: I need {name}.\n<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def answer(text):
    return call("final_answer", answer=text)


def spec(name, fn, properties=None, required=(), **kw):
    schema = {"type": "object", "properties": properties or {}, "required": list(required)}
    return ToolSpec(name=name, description=f"{name} tool", parameters=schema, fn=fn, **kw)


def toolbox():
    return Toolbox().add(
        spec("classify_risk", lambda **flags: {"label": "prohibido", "article": "Art. 5.1.f"},
             {"emotion_recognition": {"type": "boolean"}}),
        spec("compute_max_fine", lambda infringement, annual_turnover_eur: "56.000.000 EUR",
             {"infringement": {"type": "string"}, "annual_turnover_eur": {"type": "number"}},
             required=["infringement", "annual_turnover_eur"]),
        spec("write_assessment_report", lambda **kw: "reports/x.pdf", {"label": {"type": "string"}},
             requires_confirmation=True),
        spec("slow_tool", lambda: time.sleep(2), timeout=0.2),
        spec("broken_tool", lambda: 1 / 0),
    )


def agent(outputs, **kw):
    brain = FakeBrain(outputs)
    return ReActAgent(brains={"base": brain}, toolbox=toolbox(), **kw), brain


def test_chains_two_tools_and_stops_at_final_answer():
    a, _ = agent([
        call("classify_risk", emotion_recognition=True),
        call("compute_max_fine", infringement="prohibited_practice", annual_turnover_eur=8e8),
        answer("Prohibido (Art. 5.1.f). Multa máxima: 56.000.000 EUR."),
    ])
    r = a.run("clasifica y calcula la multa", brain="base")
    assert r.succeeded and r.stop_reason == "final_answer" and r.n_steps == 3
    assert [s.action.name for s in r.steps] == ["classify_risk", "compute_max_fine", "final_answer"]
    assert "prohibido" in r.steps[0].observation  # the real tool result is in the trace
    assert r.steps[0].thought == "I need classify_risk."


def test_errors_become_observations_and_the_agent_recovers():
    a, _ = agent([
        call("unknown_tool"),
        "<tool_call>{not json}</tool_call>",
        call("compute_max_fine", infringement="prohibited_practice"),  # missing argument
        call("broken_tool"),
        answer("done"),
    ])
    r = a.run("task", brain="base")
    observations = [s.observation for s in r.steps[:4]]
    assert all(o.startswith("Error:") for o in observations)
    assert "unknown tool" in observations[0] and "not valid" in observations[1]
    assert "missing required arguments" in observations[2] and "ZeroDivisionError" in observations[3]
    assert r.steps[1].action.name == "invalid_tool_call"  # the trace keeps what was written
    assert r.succeeded and r.final_answer == "done"


def test_tool_timeout():
    a, _ = agent([call("slow_tool"), answer("ok")])
    start = time.perf_counter()
    r = a.run("task", brain="base")
    assert "timeout" in r.steps[0].observation and time.perf_counter() - start < 1.5


def test_max_steps():
    a, _ = agent([call("classify_risk", emotion_recognition=True)] + [call("broken_tool")] * 5)
    r = a.run("task", max_steps=3, brain="base")
    assert not r.succeeded and r.stop_reason == "max_steps" and r.n_steps == 3
    assert r.final_answer.startswith("No puedo responder")


def test_repeated_calls_are_not_rerun_and_stop_the_loop():
    calls = []
    a, _ = agent([call("classify_risk", emotion_recognition=True)] * 5)
    original = a.toolbox.tools["classify_risk"].fn
    a.toolbox.tools["classify_risk"].fn = lambda **kw: calls.append(kw) or original(**kw)
    r = a.run("task", brain="base")
    assert len(calls) == 1  # executed once, then served from the earlier result
    assert r.stop_reason == "loop_detected" and not r.succeeded


def test_report_needs_the_user_to_ask_for_it():
    a, _ = agent([call("write_assessment_report", label="alto_riesgo"), answer("ok")])
    r = a.run("clasifica este sistema", brain="base")
    assert "side effects" in r.steps[0].observation
    a, _ = agent([call("write_assessment_report", label="alto_riesgo"), answer("ok")])
    r = a.run("clasifica y genera el informe en PDF", brain="base")
    assert r.steps[0].observation == "reports/x.pdf"


def test_plain_answer_is_accepted_but_flagged():
    a, _ = agent(["Es de riesgo mínimo."])
    r = a.run("task", brain="base")
    assert r.final_answer == "Es de riesgo mínimo." and r.stop_reason == "plain_answer"


def test_action_written_in_prose_is_not_an_answer():
    a, _ = agent(["Action: classify_risk\nAction Input: {}", answer("ok")])
    r = a.run("task", brain="base")
    assert r.steps[0].observation.startswith("Error:") and r.final_answer == "ok"


def test_only_the_first_of_several_calls_runs():
    a, _ = agent([call("classify_risk") + call("broken_tool"), answer("ok")])
    r = a.run("task", brain="base")
    assert "1 more were ignored" in r.steps[0].observation


def test_truncated_reasoning_is_reported_to_the_model():
    a, _ = agent(["<think>thinking for too long", answer("ok")])
    r = a.run("task", brain="base")
    assert "cut off" in r.steps[0].observation and r.final_answer == "ok"


def test_native_prompt_passes_the_tools_to_the_template():
    a, brain = agent([answer("ok")])
    a.run("task", brain="base")
    _, tools = brain.seen[0]
    assert {t["function"]["name"] for t in tools} >= {"classify_risk", "final_answer"}


def test_text_prompt_pastes_the_tools_into_the_system_prompt():
    a, brain = agent([answer("ok")], prompt_style="text")
    a.run("task", brain="base")
    messages, tools = brain.seen[0]
    assert tools is None and "classify_risk" in messages[0]["content"]


def test_sessions_share_memory_between_requests():
    a, brain = agent([answer("primera"), answer("segunda")])
    a.run("primera pregunta", brain="base", session_id="s1")
    a.run("segunda pregunta", brain="base", session_id="s1")
    second_context = " ".join(m["content"] for m in brain.seen[1][0])
    assert "primera pregunta" in second_context and "primera" in second_context


def test_parser_reads_string_arguments_and_rejects_calls_without_name():
    calls = parse_tool_calls(
        '<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>'
        '<tool_call>{"arguments": {}}</tool_call>'
    )
    assert calls[0].arguments == {"a": 1} and calls[1].error == "the call has no 'name'"
