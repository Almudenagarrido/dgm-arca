"""The smolagents adapter: our tools as ``smolagents.Tool``, run through ``ReActAgent.execute``."""

import pytest

pytest.importorskip("smolagents")

from agent.code_agent import _json_type, wrap_tool  # noqa: E402
from agent.react_agent import Brain, ReActAgent  # noqa: E402


def make_agent() -> ReActAgent:
    return ReActAgent(brains={"base": Brain("base", "fake")})


def test_every_registry_tool_wraps_and_validates():
    agent = make_agent()
    for spec in agent.registry.tools.values():
        wrapped = wrap_tool(spec, agent, {"text": "tarea"})
        assert wrapped.name == spec.name and wrapped.output_type == "any"
    classify = wrap_tool(agent.registry.tools["classify_risk"], agent, {"text": "x"})
    assert classify.inputs["emotion_recognition"]["type"] == "boolean"
    assert classify.inputs["annex_iii_use_cases"]["type"] == "array"
    assert classify.inputs["emotion_recognition"]["nullable"]


def test_forward_runs_the_real_tool_and_raises_on_errors():
    agent = make_agent()
    fine = wrap_tool(agent.registry.tools["compute_max_fine"], agent, {"text": "x"})
    result = fine(infringement="prohibited_practice", annual_turnover_eur=2e9, is_sme=None)
    assert result["max_fine_eur"] == 140_000_000
    with pytest.raises(RuntimeError, match="invalid arguments"):
        fine(infringement="murder", annual_turnover_eur=1)


def test_report_tool_keeps_the_confirmation_policy():
    agent = make_agent()
    report = wrap_tool(
        agent.registry.tools["write_assessment_report"], agent, {"text": "clasifica"}
    )
    with pytest.raises(RuntimeError, match="not executed"):
        report(
            system_name="x",
            description="d",
            label="alto_riesgo",
            legal_basis="Anexo III.4.a",
            justification="j",
        )


def test_json_type():
    assert _json_type({"anyOf": [{"type": "boolean"}, {"type": "null"}]}) == "boolean"
    assert _json_type({"enum": ["a"], "type": "string"}) == "string"
    assert _json_type({"$ref": "#/x"}) == "any"
