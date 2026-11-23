"""The agent sees the real phase 2 tools and the phase 3 search (retriever faked: no models)."""

import json
from types import SimpleNamespace

from agent.connectors import phase2_specs, search_spec
from agent.toolbox import Toolbox


def test_phase2_tools_are_plugged_in():
    box = Toolbox().add(*phase2_specs())
    assert set(box.tools) == {
        "classify_risk",
        "lookup_ai_act",
        "compute_max_fine",
        "write_assessment_report",
        "calculator",
    }
    assert box.tools["write_assessment_report"].requires_confirmation


def test_phase2_tool_runs_and_validates():
    box = Toolbox().add(*phase2_specs())
    ok = box.run("calculator", {"expression": "2 + 3"})
    assert ok.ok and json.loads(ok.output) == 5
    bad = box.run("classify_risk", {"not_a_fact": True})
    assert not bad.ok


def test_classify_risk_through_the_agent_toolbox():
    box = Toolbox().add(*phase2_specs())
    result = box.run("classify_risk", {"emotion_recognition": True})
    assert result.ok and "label" in json.loads(result.output)


class FakePipeline:
    def retrieve(self, query, top_k, retriever):
        chunk = SimpleNamespace(title="Art. 5", text="x" * 2000)
        return [SimpleNamespace(chunk=chunk, score=0.91234)] * top_k


def test_search_returns_short_fragments_with_source():
    box = Toolbox().add(search_spec(lambda: FakePipeline()))
    result = box.run("search_knowledge_base", {"query": "¿Qué prohíbe el artículo 5?", "top_k": 9})
    hits = json.loads(result.output)
    assert result.ok and len(hits) == 5
    assert hits[0]["source"] == "Art. 5" and len(hits[0]["text"]) == 700
