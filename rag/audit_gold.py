"""Consistency checks for the gold set and the threshold validation set.

Every number of phase 3 is measured against ``rag/gold/gold.jsonl``, so a broken row silently
moves the metrics. This script finds the mechanical errors a reviewer can miss reading row by
row; the legal review itself (is the expected answer right?) is still the team's job.

Checks, per file:

* unique ids and no repeated questions; no question shared between the gold set and the
  threshold validation set (the threshold must be chosen on data it is not evaluated on);
* answerable rows have evidence and unanswerable rows have none;
* every ``evidence.source`` and every file in ``required_sources`` exists in ``rag/corpus/``,
  and every ``evidence.text`` appears verbatim in its source (after normalising whitespace);
* ``answer_keywords`` is consistent with ``expected_answer``: at least one keyword must be in
  the reference answer, with the same substring test ``rag.evaluate_answers`` applies to the
  model. If not, a model that answers exactly like the reference fails the keyword metric;
* the distribution of ``question_type`` and ``review`` status.

The corpus is not versioned (``rag/download_corpus.py`` rebuilds it); without it the corpus
checks are skipped.

    uv run python -m rag.audit_gold
    uv run python -m rag.audit_gold --out reports/phase3_gold_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from rag.evaluate import load_gold


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


class Corpus:
    def __init__(self, root: Path):
        self.root, self._cache = root, {}
        self.available = root.is_dir() and any(root.iterdir())

    def text(self, source: str) -> str | None:
        if source not in self._cache:
            path = self.root / source
            self._cache[source] = (
                normalise(path.read_text(encoding="utf-8")) if path.is_file() else None
            )
        return self._cache[source]


def check_rows(rows: list[dict[str, Any]], corpus: Corpus) -> dict[str, Any]:
    problems: dict[str, list] = {
        "duplicate_ids": [k for k, v in Counter(r["id"] for r in rows).items() if v > 1],
        "duplicate_questions": [
            k for k, v in Counter(normalise(r["question"]) for r in rows).items() if v > 1
        ],
        "answerable_without_evidence": [],
        "unanswerable_with_evidence": [],
        "keywords_not_in_expected_answer": [],
        "missing_sources": [],
        "evidence_not_verbatim": [],
    }
    for row in rows:
        answerable = row.get("answerable", True)
        evidence = row.get("evidence") or []
        if answerable and not evidence:
            problems["answerable_without_evidence"].append(row["id"])
        if not answerable and evidence:
            problems["unanswerable_with_evidence"].append(row["id"])

        keywords = row.get("answer_keywords") or []
        expected = row.get("expected_answer")
        if answerable and keywords and expected:
            if not any(k.lower() in expected.lower() for k in keywords):
                problems["keywords_not_in_expected_answer"].append(
                    {"id": row["id"], "keywords": keywords, "expected_answer": expected}
                )

        if not corpus.available:
            continue
        sources = {e["source"] for e in evidence}
        sources |= {s for group in row.get("required_sources") or [] for s in group}
        for source in sorted(sources):
            if corpus.text(source) is None:
                problems["missing_sources"].append({"id": row["id"], "source": source})
        for item in evidence:
            text = corpus.text(item["source"])
            if text is not None and normalise(item["text"]) not in text:
                problems["evidence_not_verbatim"].append(
                    {"id": row["id"], "source": item["source"], "text": item["text"][:120]}
                )
    return {
        "rows": len(rows),
        "question_types": dict(Counter(r.get("question_type", "-") for r in rows)),
        "review": dict(Counter(r.get("review", "-") for r in rows)),
        "unanswerable": sum(not r.get("answerable", True) for r in rows),
        "problems": problems,
        "n_problems": sum(len(v) for v in problems.values()),
    }


def audit(gold: Path, validation: Path, corpus_dir: Path) -> dict[str, Any]:
    corpus = Corpus(corpus_dir)
    files = {"gold": load_gold(gold)}
    if validation.exists():
        files["threshold_val"] = load_gold(validation)
    report: dict[str, Any] = {"corpus_checked": corpus.available}
    for name, rows in files.items():
        report[name] = check_rows(rows, corpus)
    if "threshold_val" in files:
        gold_q = {normalise(r["question"]) for r in files["gold"]}
        shared = [r["id"] for r in files["threshold_val"] if normalise(r["question"]) in gold_q]
        report["questions_shared_gold_and_val"] = shared
    return report


def print_summary(report: dict[str, Any]) -> None:
    if not report["corpus_checked"]:
        print("rag/corpus/ vacío: no se comprueban fuentes ni evidencias.\n")
    for name in ("gold", "threshold_val"):
        if name not in report:
            continue
        entry = report[name]
        print(f"== {name}: {entry['rows']} filas, {entry['unanswerable']} sin respuesta")
        print(f"   tipos: {entry['question_types']}")
        print(f"   revisión: {entry['review']}")
        for check, items in entry["problems"].items():
            mark = "ok" if not items else f"{len(items)} PROBLEMAS"
            print(f"   {check:<34} {mark}")
            for item in items:
                print(f"      {item}")
    shared = report.get("questions_shared_gold_and_val")
    if shared is not None:
        print(f"\npreguntas compartidas entre dorado y validación: {len(shared)} {shared or ''}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="rag/gold/gold.jsonl")
    parser.add_argument("--validation", default="rag/gold/threshold_val.jsonl")
    parser.add_argument("--corpus", default="rag/corpus")
    parser.add_argument("--out", default=None, help="write the full report as JSON")
    args = parser.parse_args()

    report = audit(Path(args.gold), Path(args.validation), Path(args.corpus))
    print_summary(report)
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
