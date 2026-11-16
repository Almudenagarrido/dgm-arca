"""The AI Act generator is the dataset: labels must match the engine, splits must not leak."""

import json
import random

import pytest

from rlm.aiact_rules import LABELS, classify
from rlm.generate_aiact import (
    FAMILIES,
    INSTRUCTION,
    LEAK_RE,
    AIActGenerator,
    describe,
    write_jsonl,
)

GEN = AIActGenerator()


@pytest.mark.parametrize(
    ("split", "family"),
    [(split, name) for split in ("train", "test", "ood") for name in FAMILIES[split]],
)
def test_every_family_produces_its_label_and_states_its_flags(split, family):
    label, sample = FAMILIES[split][family]
    rng = random.Random(family)
    for _ in range(30):
        case = sample(rng)
        assert classify(case.profile).label == label
        assert case.sentences and all(
            "{" not in s.replace("{Sys}", "").replace("{sys}", "") for s in case.sentences
        )


def test_every_label_has_families_in_train_and_ood():
    for split in ("train", "test", "ood"):
        assert {label for label, _ in FAMILIES[split].values()} == set(LABELS)


def test_ood_families_never_appear_in_train():
    assert not set(FAMILIES["train"]) & set(FAMILIES["ood"])


@pytest.fixture(scope="module")
def splits():
    train = GEN.generate(400, "train", seed=1)
    seen = {GEN.key(p.params) for p in train}
    texts = {GEN.case_text(p.params) for p in train}
    test = GEN.generate(80, "test", seed=1, exclude=seen, exclude_texts=texts)
    ood = GEN.generate(40, "ood", seed=1, exclude=seen, exclude_texts=texts)
    return train, test, ood


def test_labels_are_balanced_exactly(splits):
    for problems in splits:
        counts = describe(problems, GEN)["labels"]
        assert set(counts) == set(LABELS)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_no_case_is_shared_between_splits(splits):
    train, test, ood = splits
    keys = [{GEN.key(p.params) for p in s} for s in (train, test, ood)]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]


def _fact_sentences(split: str) -> set[str]:
    """Every fact sentence a split's families can produce (sampled until it stops growing)."""
    found: set[str] = set()
    for name, (_, sample) in FAMILIES[split].items():
        rng = random.Random(name)
        for _ in range(400):
            found.update(sample(rng).sentences)
    return found


def test_test_and_ood_never_reuse_a_train_fact_sentence():
    """The in-distribution test measures new cases: no sentence that states a legal fact is
    shared with train (until 2026-11, 193/200 test cases reused train's sentences)."""
    train = _fact_sentences("train")
    assert not train & _fact_sentences("test")
    assert not train & _fact_sentences("ood")


def test_test_and_ood_fillers_and_frames_agree_with_feminine_subjects():
    from rlm.audit_data import MASCULINE_AFTER_SUBJECT
    from rlm.generate_aiact import FILLERS, FRAMES_BY_SPLIT

    for split in ("test", "ood"):
        for text in FILLERS[split] + FRAMES_BY_SPLIT[split]:
            assert not MASCULINE_AFTER_SUBJECT.search(text), (split, text)


def test_test_and_ood_cases_have_unique_texts(splits):
    train, test, ood = splits
    texts = [{GEN.case_text(p.params) for p in s} for s in (train, test, ood)]
    assert len(texts[1]) == len(test) and len(texts[2]) == len(ood)
    assert not texts[0] & texts[1] and not texts[0] & texts[2]


def test_descriptions_do_not_leak_the_label(splits):
    for problems in splits:
        assert describe(problems, GEN)["label_words_in_description"] == 0


def test_answer_and_citations_come_from_the_engine(splits):
    for problem in (p for s in splits for p in s):
        result = classify(problem.flags)
        assert problem.answer == result.label
        assert problem.article == result.article
        if problem.article:
            assert problem.article in problem.valid_articles
        assert problem.question.endswith(INSTRUCTION)


def test_generation_is_deterministic():
    first = [p.question for p in GEN.generate(40, "train", seed=7)]
    second = [p.question for p in GEN.generate(40, "train", seed=7)]
    assert first == second


def test_leak_regex_catches_label_words():
    assert LEAK_RE.search("es claramente de alto riesgo")
    assert LEAK_RE.search("según el Anexo III")
    assert not LEAK_RE.search("evalúa el riesgo de los viajeros")


def test_jsonl_rows_have_the_fields_the_verifier_needs(tmp_path, splits):
    path = tmp_path / "train.jsonl"
    write_jsonl(splits[0][:5], path, "train")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    for row in rows:
        assert {"question", "answer", "article", "valid_articles", "family", "flags"} <= set(row)
        assert classify(row["flags"]).label == row["answer"]
