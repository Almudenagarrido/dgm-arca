"""Quality audit of the phase 1 splits (train, test, test_ood).

The generator already guarantees two things by construction: the label comes from the rule
engine and no semantic key repeats across splits. This script checks what that guarantee does
*not* cover, on the files that were actually used for training and evaluation:

* **labels**: every row still matches the engine (catches stale files after a rule change);
* **near-duplicates**: a test row whose description, without the subject and the filler
  sentences, is identical to a train row. The dedup key includes a scenario id with indices
  that do not always change the text (for example the use index in ``_derogation_case`` when
  two uses share the same organisation), so two keys can render the same case;
* **template overlap**: test rows whose case sentences (the part that carries the legal facts)
  are exactly those of some train row, with only the frame, organisation, subject or fillers
  changed. A high rate means the in-distribution test measures recall of seen cases and the OOD
  split is the real generalisation number;
* **agreement**: feminine subjects (``la herramienta``…) rendered into sentences written in
  the masculine (``está integrado``, ``Está construido``…).

The overlap check needs the generator parameters, so the splits are regenerated with the seed
in ``generation_stats.json`` and compared with the files first; if they differ, that check is
skipped and the others still run.

    uv run python -m rlm.audit_data                       # summary
    uv run python -m rlm.audit_data --details             # plus the offending rows
    uv run python -m rlm.audit_data --out reports/phase1_data_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from rlm.aiact_rules import classify
from rlm.generate_aiact import FILLERS, SUBJECTS, AIActGenerator

SPLITS = {"train": "train.jsonl", "test": "test.jsonl", "ood": "test_ood.jsonl"}
FEMININE = tuple(s for s in SUBJECTS if s.startswith("la "))
# Masculine forms that, in our templates, can only refer to the subject: participles and the
# pronoun ``lo`` in the fillers and frames (``Lo ha desarrollado…``, ``Quien lo despliega…``).
MASCULINE_AFTER_SUBJECT = re.compile(
    r"\b(?:está|es) (?:integrado|construido|diseñado|instalado|basado|entrenado)\b|"
    r"^Está construido\b|\blo ha desarrollado\b|\bdesplegarlo\b|\blo usen\b|"
    r"\blo despliega\b",
    re.IGNORECASE | re.MULTILINE,
)


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def description(question: str) -> str:
    """The case text: everything before the fixed instruction."""
    return question.split("\n\n")[0]


def normalise(question: str) -> str:
    """Description without filler sentences and with the subject replaced by a placeholder."""
    text = description(question)
    for filler in {f for fillers in FILLERS.values() for f in fillers}:
        text = text.replace(filler, "")
    for subject in sorted(SUBJECTS, key=len, reverse=True):
        text = re.sub(re.escape(subject), "<SYS>", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def label_mismatches(rows: list[dict[str, Any]]) -> list[int]:
    return [i for i, r in enumerate(rows) if classify(r["flags"]).label != r["answer"]]


def agreement_issues(rows: list[dict[str, Any]]) -> list[tuple[int, list[str]]]:
    issues = []
    for i, row in enumerate(rows):
        text = description(row["question"])
        if not any(subject.lower() in text.lower() for subject in FEMININE):
            continue
        # one sentence per line, so ^Está matches a filler that starts a sentence
        found = MASCULINE_AFTER_SUBJECT.findall(re.sub(r"(?<=[.;»]) ", "\n", text))
        if found:
            issues.append((i, found))
    return issues


def near_duplicates(
    reference: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[tuple[int, list[int]]]:
    index: dict[str, list[int]] = {}
    for i, row in enumerate(reference):
        index.setdefault(normalise(row["question"]), []).append(i)
    pairs = []
    for j, row in enumerate(rows):
        hits = index.get(normalise(row["question"]))
        if hits:
            pairs.append((j, hits))
    return pairs


def regenerate(seed: int, sizes: dict[str, int]) -> dict[str, list] | None:
    """Rebuild the splits exactly as ``generate_aiact.main`` does."""
    generator = AIActGenerator()
    seen: set[str] = set()
    seen_texts: set[str] = set()
    out = {}
    for split in ("train", "test", "ood"):
        problems = generator.generate(
            sizes[split], split, seed, exclude=seen, exclude_texts=seen_texts
        )
        seen |= {generator.key(p.params) for p in problems}
        seen_texts |= {generator.case_text(p.params) for p in problems}
        out[split] = problems
    return out


def closest_train_similarity(train: list, rows: list) -> list[float]:
    """For each row, the highest character similarity between its case sentences and those of
    any train case of the same family, or of any family for OOD (difflib ratio, 1.0 = same
    text)."""
    by_family: dict[str, set[str]] = {}
    for p in train:
        by_family.setdefault(p.family, set()).add(" ".join(p.params["sentences"]))
    scores = []
    for p in rows:
        text = " ".join(p.params["sentences"])
        # OOD families never appear in train: compare with every train case instead
        candidates = by_family.get(p.family) or {c for cs in by_family.values() for c in cs}
        scores.append(
            max((SequenceMatcher(None, text, c).ratio() for c in candidates), default=0.0)
        )
    return scores


def audit(data_dir: Path, details: bool = False) -> dict[str, Any]:
    rows = {split: load(data_dir / name) for split, name in SPLITS.items()}
    report: dict[str, Any] = {"sizes": {s: len(r) for s, r in rows.items()}}

    for split, data in rows.items():
        mismatches = label_mismatches(data)
        agreement = agreement_issues(data)
        exact = len(data) - len({description(r["question"]) for r in data})
        entry: dict[str, Any] = {
            "labels": dict(Counter(r["answer"] for r in data).most_common()),
            "label_mismatches": len(mismatches),
            "exact_duplicate_descriptions": exact,
            "agreement_issues": len(agreement),
        }
        if details:
            entry["label_mismatch_rows"] = mismatches
            entry["agreement_rows"] = agreement
        report[split] = entry

    for split in ("test", "ood"):
        pairs = near_duplicates(rows["train"], rows[split])
        report[split]["near_duplicates_of_train"] = len(pairs)
        report[split]["near_duplicate_pairs"] = [{"row": j, "train_rows": t} for j, t in pairs]

    stats_path = data_dir / "generation_stats.json"
    seed = json.loads(stats_path.read_text())["seed"] if stats_path.exists() else 0
    regenerated = regenerate(seed, report["sizes"])
    same = all(
        [p.question for p in regenerated[s]] == [r["question"] for r in rows[s]] for s in rows
    )
    report["regenerated_matches_files"] = same
    if same:
        train_cases = {tuple(p.params["sentences"]) for p in regenerated["train"]}
        train_sentences = {s for case in train_cases for s in case}
        for split in ("test", "ood"):
            shared = [
                i
                for i, p in enumerate(regenerated[split])
                if tuple(p.params["sentences"]) in train_cases
            ]
            report[split]["case_sentences_seen_in_train"] = len(shared)
            # stricter: any single fact sentence of the case also appears in some train case
            report[split]["any_fact_sentence_seen_in_train"] = sum(
                any(s in train_sentences for s in p.params["sentences"]) for p in regenerated[split]
            )
            scores = closest_train_similarity(regenerated["train"], regenerated[split])
            report[split]["similarity_to_closest_train_case"] = {
                "mean": round(sum(scores) / len(scores), 3),
                "max": round(max(scores), 3),
                "rows_above_0.9": sum(s > 0.9 for s in scores),
            }
            if details:
                report[split]["novel_case_rows"] = [
                    i for i in range(len(regenerated[split])) if i not in set(shared)
                ]
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(
        f"{'':<6} {'filas':>6} {'etiq. ≠ motor':>14} {'duplic.':>8} {'concord.':>9}"
        f" {'casi copia train':>17} {'caso visto train':>17} {'alguna frase':>13}"
        f" {'similitud media/máx':>20}"
    )
    for split in SPLITS:
        entry = report[split]
        n = report["sizes"][split]
        dup = entry.get("near_duplicates_of_train", "-")
        seen = entry.get("case_sentences_seen_in_train", "-")
        seen_txt = f"{seen} ({100 * seen / n:.0f} %)" if isinstance(seen, int) else seen
        any_seen = entry.get("any_fact_sentence_seen_in_train", "-")
        sim = entry.get("similarity_to_closest_train_case")
        sim_txt = f"{sim['mean']:.2f} / {sim['max']:.2f}" if sim else "-"
        print(
            f"{split:<6} {n:>6} {entry['label_mismatches']:>14} "
            f"{entry['exact_duplicate_descriptions']:>8} {entry['agreement_issues']:>9} "
            f"{dup!s:>17} {seen_txt:>17} {any_seen!s:>13} {sim_txt:>20}"
        )
    if not report["regenerated_matches_files"]:
        print("\nAviso: el generador ya no reproduce los ficheros; no se mide el solapamiento.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="rlm/data")
    parser.add_argument("--out", default=None, help="write the full report as JSON")
    parser.add_argument("--details", action="store_true", help="include the offending rows")
    args = parser.parse_args()

    report = audit(Path(args.data_dir), details=args.details)
    print_summary(report)
    if args.details:
        for split in ("test", "ood"):
            for pair in report[split]["near_duplicate_pairs"]:
                print(f"{split} #{pair['row']} ~ train {pair['train_rows']}")
            for row, found in report[split].get("agreement_rows", []):
                print(f"{split} #{row} concordancia: {found}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
