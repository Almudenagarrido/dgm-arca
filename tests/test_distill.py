"""Distillation post-processing: no GPU needed, the teacher's output is given as text."""

from rlm.distill import canonicalize, cap_per_problem, score_trace, summarize
from rlm.rewards import has_valid_format
from rlm.verifier import AIActVerifier

PROBLEM = {"answer": "alto_riesgo", "valid_articles": ["Anexo III.4.a"]}


def test_canonical_trace_is_kept_as_is():
    text = "<think>razono</think><answer>alto_riesgo (Anexo III.4.a)</answer>"
    assert canonicalize(text) == (text, False)


def test_prose_after_think_is_rewritten_into_an_answer_block():
    text = "<think>razono</think>\n\nEs un sistema de **alto riesgo** según el Anexo III, 4.a."
    trace, rewritten = canonicalize(text)
    assert rewritten and has_valid_format(trace)
    assert trace.endswith("<answer>alto_riesgo (Anexo III.4.a)</answer>")


def test_answer_tag_with_trailing_text_is_cleaned():
    text = "<think>x</think><answer>prohibido (Art. 5.1.f)</answer>\nEspero que ayude."
    trace, rewritten = canonicalize(text)
    assert rewritten and trace.endswith("<answer>prohibido (Art. 5.1.f)</answer>")


def test_hedging_prose_and_truncated_traces_are_not_rescued():
    hedge = "<think>x</think> Podría ser prohibido o alto riesgo."
    assert canonicalize(hedge) == (hedge, False)
    truncated = "<think>razonamiento que nunca termina"
    assert canonicalize(truncated) == (truncated, False)


def test_score_trace_requires_label_and_citation():
    v = AIActVerifier()
    good = score_trace(
        "<think>x</think><answer>alto_riesgo (Anexo III.4.a)</answer>", PROBLEM, v, True
    )
    assert good["verified"] and good["citation_score"] == 1.0
    no_cite = score_trace("<think>x</think><answer>alto_riesgo</answer>", PROBLEM, v, True)
    assert no_cite["label_correct"] and not no_cite["verified"]
    assert score_trace("<think>x</think><answer>alto_riesgo</answer>", PROBLEM, v, False)[
        "verified"
    ]
    wrong = score_trace("<think>x</think><answer>prohibido (Art. 5.1.a)</answer>", PROBLEM, v, True)
    assert not wrong["label_correct"] and not wrong["verified"]


def _row(pid, verified, label="alto_riesgo"):
    return {
        "problem_id": pid,
        "verified": verified,
        "detail": "",
        "answer": label,
        "family": "f",
        "format_ok": True,
        "label_correct": verified,
        "citation_score": 1.0 if verified else 0.0,
        "truncated": False,
        "rewritten": False,
        "new_tokens": 100,
    }


def test_cap_per_problem_keeps_at_most_n_verified_traces():
    rows = [_row(0, True), _row(0, True), _row(0, True), _row(1, True)]
    cap_per_problem(rows, 2)
    assert [r["verified"] for r in rows] == [True, True, False, True]
    assert "dropped" in rows[2]["detail"]


def test_summarize_reports_per_label_and_problem_coverage():
    rows = [_row(0, True), _row(0, False), _row(1, False, "prohibido")]
    stats = summarize(rows)
    assert stats["overall"]["traces"] == 3
    assert stats["overall"]["problems_with_a_verified_trace"] == 1
    assert stats["by_label"]["prohibido"]["label_correct"] == 0.0
    assert stats["verified_traces_for_sft"] == 1
