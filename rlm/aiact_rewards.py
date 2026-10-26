"""Rewards for the AI Act task, with the ``trl.GRPOTrainer`` signature.

Why not ``rlm.rewards.accuracy_reward``: it compares numbers, and with text labels both sides
normalise to ``None``, so *any* answer scores 1.0 (see EXPERIMENTS.md, 2026-10-26). These two
replace it for our dataset; ``format_reward`` from ``rlm.rewards`` is still used as is.

* ``aiact_accuracy_reward``: 1.0 when the answer holds exactly one label and it is the
  engine's label. Several labels score 0.0, so hedging is never rewarded.
* ``citation_reward`` (the domain reward): how well the cited legal basis matches the
  engine's, and only when the label is right. A compliance officer needs the article to act
  on the answer; a correct label with an invented article is still a bad answer, and a right
  article for a wrong label is meaningless. Scores:

  - 1.0  the first citation in the answer is one of ``valid_articles``;
  - 0.5  same article or same Annex III point, wrong paragraph or letter;
  - 0.0  other article, no citation when one was expected, or wrong label;
  - for ``riesgo_minimo`` with nothing to cite, 1.0 for citing nothing and 0.0 for citing
    something (an invented legal basis is the failure we most want to avoid).

  Only the first citation in the ``<answer>`` block counts, so listing many articles to hit
  one of them does not pay.

Dataset columns arrive as keyword arguments: ``answer`` and ``valid_articles`` come from the
JSONL written by ``rlm.generate_aiact``.
"""

from __future__ import annotations

from collections.abc import Sequence

from rlm.aiact_rules import article_match_score, extract_articles
from rlm.rewards import _completion_text, extract_answer
from rlm.verifier import parse_aiact_label


def aiact_accuracy_reward(
    prompts: Sequence, completions: Sequence, answer: Sequence[str], **kwargs
) -> list[float]:
    """1.0 if the answer holds exactly one label and it matches the ground truth."""
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        predicted = parse_aiact_label(extract_answer(_completion_text(completion)))
        rewards.append(1.0 if predicted is not None and predicted == expected else 0.0)
    return rewards


def citation_score(answer_text: str | None, expected_label: str, valid: Sequence[str]) -> float:
    """Citation quality of one answer block (see the module docstring for the scale)."""
    if answer_text is None or parse_aiact_label(answer_text) != expected_label:
        return 0.0
    cited = extract_articles(answer_text)
    first = cited[0] if cited else None
    if not valid:
        return 1.0 if first is None else 0.0
    return max(article_match_score(first, article) for article in valid)


def citation_reward(
    prompts: Sequence,
    completions: Sequence,
    answer: Sequence[str],
    valid_articles: Sequence[Sequence[str] | None],
    **kwargs,
) -> list[float]:
    """The domain reward: legal basis cited correctly, conditional on the right label."""
    rewards = []
    for completion, expected, valid in zip(completions, answer, valid_articles, strict=True):
        answer_text = extract_answer(_completion_text(completion))
        rewards.append(citation_score(answer_text, expected, list(valid or [])))
    return rewards
