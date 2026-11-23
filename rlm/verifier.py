"""Verifier interface for reinforcement learning with verifiable rewards.

A verifier answers one question deterministically: *is this answer correct for
this problem?* Everything in phase 1 hangs on it. If the verifier is sloppy, the
model will learn to exploit the sloppiness instead of learning to reason.

We ship two verifiers:

* ``NumericVerifier`` compares numbers after normalisation. It is what GSM8K needs
  and what the smoke test uses.
* ``ExactMatchVerifier`` compares normalised strings. Useful for multiple-choice
  or short factual answers.

Your domain verifier goes in this module too. Subclass ``Verifier``, implement
``is_correct`` and add a test for it in ``tests/test_verifier.py``. Common shapes:
run unit tests on generated code, execute a SQL query and compare result sets,
validate a JSON document against a schema, check that a date falls in a range.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlm.rewards import extract_answer, normalize_number


@dataclass(frozen=True)
class VerificationResult:
    """What a verifier reports back. ``detail`` is free text for logging and debugging."""

    is_correct: bool
    predicted: str | None
    expected: str
    detail: str = ""


class Verifier(ABC):
    """Base class for all verifiers."""

    name: str = "verifier"

    @abstractmethod
    def is_correct(self, predicted: str | None, expected: str) -> bool:
        """Return True when ``predicted`` should be accepted as a correct answer."""

    def verify(self, completion: str, expected: str) -> VerificationResult:
        """Extract the final answer from a full completion and check it."""
        predicted = extract_answer(completion)
        ok = self.is_correct(predicted, expected)
        detail = "no <answer> block found" if predicted is None else ""
        return VerificationResult(ok, predicted, expected, detail)


class NumericVerifier(Verifier):
    """Numeric comparison with an optional absolute tolerance."""

    name = "numeric"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        p, e = normalize_number(predicted), normalize_number(expected)
        if p is None or e is None:
            return False
        if self.tolerance == 0.0:
            return p == e
        return abs(float(p) - float(e)) <= self.tolerance


class ExactMatchVerifier(Verifier):
    """Case- and whitespace-insensitive string comparison."""

    name = "exact_match"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        return self._normalize(predicted) == self._normalize(expected)


# Every way a model may write each label. "fuera_de_ambito" is recognised (it is a label the
# engine can produce) so that hedging between it and another label is caught as two labels.
_LABEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "prohibido": re.compile(r"\bprohibid[oa]s?\b|\bprohibited\b", re.IGNORECASE),
    "alto_riesgo": re.compile(r"\balto[\s_-]+riesgo\b|\bhigh[\s_-]+risk\b", re.IGNORECASE),
    "riesgo_limitado": re.compile(
        r"\briesgo[\s_-]+limitado\b|\blimited[\s_-]+risk\b", re.IGNORECASE
    ),
    "riesgo_minimo": re.compile(
        r"\briesgo[\s_-]+m[ií]nimo\b|\bminimal[\s_-]+risk\b", re.IGNORECASE
    ),
    "fuera_de_ambito": re.compile(
        r"\bfuera[\s_-]+(?:del?[\s_-]+)?[áa]mbito\b|\bout[\s_-]+of[\s_-]+scope\b", re.IGNORECASE
    ),
}


def find_aiact_labels(text: str) -> list[str]:
    """Every distinct AI Act label mentioned in ``text``, in order of first appearance."""
    hits = []
    for label, pattern in _LABEL_PATTERNS.items():
        match = pattern.search(text)
        if match:
            hits.append((match.start(), label))
    return [label for _, label in sorted(hits)]


def parse_aiact_label(text: str | None) -> str | None:
    """The single label in ``text``, or None if there is none or there are several.

    Several labels count as no answer on purpose: "prohibido o alto_riesgo" must not be
    rewarded, otherwise the model learns to hedge instead of deciding.
    """
    if text is None:
        return None
    labels = find_aiact_labels(text)
    return labels[0] if len(labels) == 1 else None


class AIActVerifier(Verifier):
    """AI Act risk tier: exactly one label in the answer, equal to the rule engine's.

    The expected answer may be the bare label (``alto_riesgo``) or label plus citation
    (``alto_riesgo (Anexo III.4.a)``); only the label decides correctness. The citation is
    scored separately by ``rlm.aiact_rewards.citation_reward``.
    """

    name = "aiact"

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        label = parse_aiact_label(predicted)
        return label is not None and label == parse_aiact_label(expected)

    def verify(self, completion: str, expected: str) -> VerificationResult:
        predicted = extract_answer(completion)
        ok = self.is_correct(predicted, expected)
        if predicted is None:
            detail = "no <answer> block found"
        else:
            labels = find_aiact_labels(predicted)
            if not labels:
                detail = "no AI Act label in the answer"
            elif len(labels) > 1:
                detail = f"several labels in the answer: {', '.join(labels)}"
            else:
                detail = f"label={labels[0]}"
        return VerificationResult(ok, predicted, expected, detail)
