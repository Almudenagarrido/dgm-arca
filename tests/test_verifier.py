"""Tests for the verifiers. Add one class of tests per domain verifier you write."""

import pytest

from rlm.verifier import (
    AIActVerifier,
    ExactMatchVerifier,
    NumericVerifier,
    find_aiact_labels,
    parse_aiact_label,
)


def test_numeric_verifier_exact():
    v = NumericVerifier()
    assert v.is_correct("42", "42")
    assert v.is_correct("$42.00", "42")
    assert not v.is_correct("41", "42")
    assert not v.is_correct(None, "42")
    assert not v.is_correct("forty-two", "42")


def test_numeric_verifier_with_tolerance():
    v = NumericVerifier(tolerance=0.05)
    assert v.is_correct("3.14", "3.1416")
    assert not v.is_correct("3.0", "3.1416")


def test_verify_extracts_from_full_completion():
    result = NumericVerifier().verify("<think>...</think><answer>18</answer>", "18")
    assert result.is_correct and result.predicted == "18"
    missing = NumericVerifier().verify("no tags at all", "18")
    assert not missing.is_correct and "no <answer>" in missing.detail


def test_exact_match_ignores_case_and_spacing():
    v = ExactMatchVerifier()
    assert v.is_correct("  Madrid ", "madrid")
    assert not v.is_correct("Barcelona", "Madrid")


# ---------------------------------------------------------------- AI Act verifier


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("alto_riesgo (Anexo III.4.a)", "alto_riesgo"),
        ("Alto riesgo", "alto_riesgo"),
        ("ALTO-RIESGO", "alto_riesgo"),
        ("high-risk (Annex III point 4(a))", "alto_riesgo"),
        ("prohibido (Art. 5.1.a)", "prohibido"),
        ("Es una práctica prohibida", "prohibido"),
        ("riesgo_limitado (Art. 50.1)", "riesgo_limitado"),
        ("riesgo minimo", "riesgo_minimo"),
        ("Riesgo mínimo", "riesgo_minimo"),
        ("fuera del ámbito", "fuera_de_ambito"),
    ],
)
def test_aiact_label_variants_are_recognised(text, label):
    assert parse_aiact_label(text) == label


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "no lo sé",
        "riesgo",  # a bare word is not a label
        "prohibido o alto_riesgo",  # hedging
        "alto_riesgo, aunque podría ser riesgo_limitado",
        "riesgo_minimo (no es alto riesgo)",
        "medio_riesgo",  # invented label
    ],
)
def test_aiact_answers_without_exactly_one_label_are_rejected(text):
    assert parse_aiact_label(text) is None
    assert not AIActVerifier().is_correct(text, "alto_riesgo")


def test_aiact_verifier_compares_only_the_label():
    v = AIActVerifier()
    assert v.is_correct("alto_riesgo (Anexo III.5.b)", "alto_riesgo (Anexo III.4.a)")
    assert v.is_correct("alto riesgo", "alto_riesgo")
    assert not v.is_correct("riesgo_limitado (Art. 50.1)", "alto_riesgo")


def test_aiact_verify_reports_why_it_failed():
    v = AIActVerifier()
    ok = v.verify(
        "<think>Anexo III</think><answer>alto_riesgo (Anexo III.4.a)</answer>", "alto_riesgo"
    )
    assert ok.is_correct and ok.detail == "label=alto_riesgo"
    hedge = v.verify("<think>x</think><answer>prohibido o alto riesgo</answer>", "prohibido")
    assert not hedge.is_correct and "several labels" in hedge.detail
    assert (
        v.verify("<answer>depende</answer>", "prohibido").detail == "no AI Act label in the answer"
    )
    assert "no <answer>" in v.verify("alto_riesgo", "alto_riesgo").detail


def test_aiact_labels_in_thinking_do_not_count():
    # The model may weigh several labels while thinking; only the answer block is judged.
    completion = "<think>¿prohibido? no, es alto riesgo</think><answer>alto_riesgo</answer>"
    assert AIActVerifier().verify(completion, "alto_riesgo").is_correct


def test_find_aiact_labels_keeps_order():
    assert find_aiact_labels("riesgo_limitado y luego prohibido") == [
        "riesgo_limitado",
        "prohibido",
    ]
