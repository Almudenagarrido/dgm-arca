"""Rewards for the AI Act task: no reward for hedging, citations only count with the right label."""

import pytest

from rlm.aiact_rewards import aiact_accuracy_reward, citation_reward, citation_score
from rlm.rewards import accuracy_reward


def wrap(answer: str) -> str:
    return f"<think>razonamiento</think><answer>{answer}</answer>"


def test_numeric_accuracy_reward_is_hackable_with_labels_and_ours_is_not():
    # The documented bug: the given reward pays 1.0 for an invented label.
    completions = [wrap("etiqueta_inventada")]
    assert accuracy_reward(None, completions, answer=["alto_riesgo"]) == [1.0]
    assert aiact_accuracy_reward(None, completions, answer=["alto_riesgo"]) == [0.0]


def test_aiact_accuracy_reward():
    completions = [
        wrap("alto_riesgo (Anexo III.4.a)"),
        wrap("riesgo_limitado"),
        wrap("alto_riesgo o prohibido"),
        "alto_riesgo sin etiquetas",
        [{"role": "assistant", "content": wrap("Alto riesgo")}],
    ]
    answers = ["alto_riesgo"] * 5
    assert aiact_accuracy_reward(None, completions, answer=answers) == [1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("answer", "expected", "valid", "score"),
    [
        ("alto_riesgo (Anexo III.4.a)", "alto_riesgo", ["Anexo III.4.a"], 1.0),
        ("alto_riesgo (Anexo III, punto 4, letra a)", "alto_riesgo", ["Anexo III.4.a"], 1.0),
        ("alto_riesgo (Anexo III.4.b)", "alto_riesgo", ["Anexo III.4.a"], 0.5),
        ("alto_riesgo (Anexo III.5.b)", "alto_riesgo", ["Anexo III.4.a"], 0.0),
        ("alto_riesgo", "alto_riesgo", ["Anexo III.4.a"], 0.0),
        # right article, wrong label: nothing
        ("riesgo_limitado (Anexo III.4.a)", "alto_riesgo", ["Anexo III.4.a"], 0.0),
        # any article of the decisive tier is fine (deep fake: 50.2 and 50.4)
        ("riesgo_limitado (Art. 50.4)", "riesgo_limitado", ["Art. 50.2", "Art. 50.4"], 1.0),
        # minimal risk with nothing to cite: citing nothing is right, inventing is wrong
        ("riesgo_minimo", "riesgo_minimo", [], 1.0),
        ("riesgo_minimo (Art. 95)", "riesgo_minimo", [], 0.0),
        # minimal risk through the Art. 6.3 derogation
        ("riesgo_minimo (Art. 6.3.a)", "riesgo_minimo", ["Art. 6.3.a"], 1.0),
        # listing many articles does not pay: only the first one counts
        ("prohibido (Art. 50.1, Art. 5.1.a)", "prohibido", ["Art. 5.1.a"], 0.0),
    ],
)
def test_citation_score(answer, expected, valid, score):
    assert citation_score(answer, expected, valid) == score


def test_citation_reward_reads_dataset_columns():
    completions = [wrap("prohibido (Art. 5.1.f)"), wrap("riesgo_minimo")]
    rewards = citation_reward(
        None,
        completions,
        answer=["prohibido", "riesgo_minimo"],
        valid_articles=[["Art. 5.1.f"], None],
    )
    assert rewards == [1.0, 1.0]
