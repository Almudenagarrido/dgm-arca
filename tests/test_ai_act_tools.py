# ruff: noqa: E501  (the XHTML fixture keeps the original one-row-per-line markup)
"""Phase 2 tools, executor and scoring, all without network or GPU."""

from __future__ import annotations

import json

import pytest

from tool_use import ai_act_text
from tool_use.evaluate import canonical_arguments, score_case
from tool_use.executor import ToolLoop
from tool_use.tools import (
    classify_risk,
    compute_max_fine,
    default_registry,
    lookup_ai_act,
    write_assessment_report,
)

# A trimmed copy of the CELLAR markup: one article with a lettered list and a closing
# subparagraph, one article without numbered paragraphs, and Annex III.
XHTML = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<div class="eli-subdivision" id="art_5">
  <p class="oj-ti-art">Art\xc3\xadculo 5</p>
  <div class="eli-title"><p class="oj-sti-art">Pr\xc3\xa1cticas de IA prohibidas</p></div>
  <div id="005.001">
    <p class="oj-normal">1. Quedan prohibidas las siguientes pr\xc3\xa1cticas de IA:</p>
    <table><tbody><tr><td><p>a)</p></td><td><p>t\xc3\xa9cnicas subliminales;</p></td></tr></tbody></table>
    <table><tbody><tr><td><p>f)</p></td><td><p>inferir emociones en el trabajo;</p></td></tr></tbody></table>
    <p class="oj-normal">El p\xc3\xa1rrafo primero se entiende sin perjuicio de otras normas.</p>
  </div>
</div>
<div class="eli-subdivision" id="art_3">
  <p class="oj-ti-art">Art\xc3\xadculo 3</p>
  <div class="eli-title"><p class="oj-sti-art">Definiciones</p></div>
  <p class="oj-normal">A los efectos del presente Reglamento, se entender\xc3\xa1 por:</p>
  <table><tbody><tr><td><p>1)</p></td><td><p>\xc2\xabsistema de IA\xc2\xbb: un sistema basado en una m\xc3\xa1quina;</p></td></tr></tbody></table>
</div>
<div id="anx_III">
  <p class="oj-doc-ti">ANEXO III</p>
  <p class="oj-doc-ti">Sistemas de IA de alto riesgo</p>
  <table><tbody><tr><td><p>4.</p></td><td><p>Empleo:</p>
    <table><tbody><tr><td><p>a)</p></td><td><p>contrataci\xc3\xb3n o selecci\xc3\xb3n de personas</p></td></tr></tbody></table>
  </td></tr></tbody></table>
</div>
</body></html>"""


@pytest.fixture
def offline_document(monkeypatch):
    document = ai_act_text.parse(XHTML)
    monkeypatch.setattr(ai_act_text, "load_document", lambda language="es": document)
    return document


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("Art. 5.1.f", ("article", "5", ["1", "f"])),
        ("artículo 50", ("article", "50", [])),
        ("Article 5(1)(a)", ("article", "5", ["1", "a"])),
        ("Anexo III.4.a", ("annex", "III", ["4", "a"])),
        ("annex iii", ("annex", "III", [])),
    ],
)
def test_references_are_parsed_in_spanish_and_english(reference, expected):
    assert ai_act_text.parse_reference(reference) == expected


def test_parser_resolves_points_definitions_and_annex(offline_document):
    canonical, title, unit = ai_act_text.resolve(offline_document, "Art. 5.1.f")
    assert canonical == "Art. 5.1.f" and title == "Prácticas de IA prohibidas"
    assert "emociones" in unit.text
    paragraph = ai_act_text.resolve(offline_document, "Art. 5.1")[2].render()
    # The closing subparagraph stays after the list, not before it.
    assert paragraph.index("f)") < paragraph.index("sin perjuicio")
    assert "máquina" in ai_act_text.resolve(offline_document, "Art. 3.1")[2].text
    assert "selección" in ai_act_text.resolve(offline_document, "Anexo III.4.a")[2].text


def test_lookup_tool_reports_missing_points_to_the_model(offline_document):
    result = lookup_ai_act(reference="Art. 5.1.a")
    assert result["reference"] == "Art. 5.1.a" and "subliminales" in result["text"]
    record = default_registry().call("lookup_ai_act", {"reference": "Art. 5.1.z"})
    assert not record.ok and "available: a, f" in record.error


def test_classify_risk_runs_the_rule_engine_and_rejects_unknown_fields():
    assert classify_risk(annex_iii_use_cases=["4.a"])["label"] == "alto_riesgo"
    result = classify_risk(annex_iii_use_cases=["4.a"], derogation_6_3="preparatory_task")
    assert (result["label"], result["article"]) == ("riesgo_minimo", "Art. 6.3.d")
    record = default_registry().call("classify_risk", {"system_name": "x"})
    assert (
        not record.ok and "Extra inputs" in record.error and "call the tool again" in record.error
    )
    derived = default_registry().call("classify_risk", {"annex_iii_use_cases": ["1.a"]})
    assert not derived.ok  # biometric cases come from the flags, not from the list


@pytest.mark.parametrize(
    ("infringement", "turnover", "sme", "expected"),
    [
        ("prohibited_practice", 2e9, False, 140_000_000),  # 7 % beats 35 M
        ("prohibited_practice", 1e8, False, 35_000_000),  # 35 M beats 7 M
        ("other_obligation", 1e7, True, 300_000),  # SME: the lower one
        ("incorrect_information", 1e8, False, 7_500_000),
    ],
)
def test_fines_follow_article_99(infringement, turnover, sme, expected):
    result = compute_max_fine(infringement=infringement, annual_turnover_eur=turnover, is_sme=sme)
    assert result["max_fine_eur"] == expected
    assert result["summary"].startswith("Multa máxima:")


def test_report_writes_a_pdf_and_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_REPORTS_DIR", str(tmp_path))
    assert write_assessment_report.requires_confirmation
    result = write_assessment_report(
        system_name="Filtro de CV",
        description="Filtra candidaturas.",
        label="alto_riesgo",
        legal_basis="Anexo III.4.a",
        justification="Selección de personal — «contratación» con coste en €.",
        obligations=["Gestión de riesgos (Art. 9)"],
    )
    path = tmp_path / result["path"].rsplit("/", 1)[-1]
    assert path.exists() and path.read_bytes().startswith(b"%PDF")


class ScriptedLoop(ToolLoop):
    """The real loop with the model replaced by a list of canned generations."""

    def __init__(self, outputs: list[str], **kwargs):
        super().__init__(prompt_style="slides", thinking=False, **kwargs)
        self.outputs = list(outputs)
        self.seen: list[list[dict]] = []

    def generate(self, messages):
        self.seen.append([dict(m) for m in messages])
        return self.outputs.pop(0)


def _call(name: str, **arguments) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def test_loop_chains_parallel_calls_and_feeds_results_back():
    loop = ScriptedLoop(
        [
            _call("classify_risk", emotion_recognition=True, workplace_or_education=True),
            _call("compute_max_fine", infringement="prohibited_practice", annual_turnover_eur=1e9)
            + _call("calculator", expression="0.07 * 1000000000"),
            "Está prohibido y la multa máxima es de 70 millones.",
        ]
    )
    response = loop.run("Clasifica y calcula la multa", max_turns=5)
    assert [c.name for c in response.tool_calls] == [
        "classify_risk",
        "compute_max_fine",
        "calculator",
    ]
    assert response.turns == 3 and not response.max_turns_reached
    last_prompt = loop.seen[-1]
    assert [m["role"] for m in last_prompt].count("tool") == 3
    assert "prohibido" in last_prompt[3]["content"]


def test_loop_returns_errors_to_the_model_and_respects_max_turns():
    loop = ScriptedLoop(["<tool_call>{not json}</tool_call>", _call("unknown_tool")])
    response = loop.run("x", max_turns=2)
    assert response.max_turns_reached and len(response.tool_calls) == 2
    assert all(not c.ok for c in response.tool_calls)
    assert "not valid JSON" in loop.seen[1][-1]["content"]


def test_tools_with_side_effects_are_refused_without_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_REPORTS_DIR", str(tmp_path))
    args = dict(
        system_name="S",
        description="d",
        label="alto_riesgo",
        legal_basis="Anexo III.4.a",
        justification="j",
    )
    loop = ScriptedLoop(
        [_call("write_assessment_report", **args), "Necesito tu confirmación."], auto_confirm=False
    )
    response = loop.run("Genera el informe")
    assert not response.tool_calls[0].ok and "confirmation" in response.tool_calls[0].error
    assert not list(tmp_path.iterdir())


def test_scoring_fills_defaults_and_parses_references():
    registry = default_registry()
    assert canonical_arguments("lookup_ai_act", {"reference": "artículo 5.1.f"}, registry) == (
        canonical_arguments(
            "lookup_ai_act", {"reference": "Art. 5.1.f", "language": "es"}, registry
        )
    )
    case = {
        "needs_tool": True,
        "expected_tool": "classify_risk",
        "expected_arguments": {"annex_iii_use_cases": ["4.a"]},
        "expected_label": "alto_riesgo",
        "answer_any": ["alto riesgo"],
    }
    explicit_defaults = {"annex_iii_use_cases": ["4.a"], "real_time": False, "rbi_objective": None}
    calls = [
        {
            "name": "classify_risk",
            "arguments": explicit_defaults,
            "ok": True,
            "result": {"label": "alto_riesgo"},
        }
    ]
    scores = score_case(case, calls, "Es de alto riesgo.", registry)
    assert scores["arguments_ok"] and scores["label_ok"] and scores["answer_ok"]
    extra_flag = [{**calls[0], "arguments": {"annex_iii_use_cases": ["4.a"], "real_time": True}}]
    assert not score_case(case, extra_flag, "", registry)["arguments_ok"]
    assert score_case({"needs_tool": False}, [], "Hola")["selection_ok"]
