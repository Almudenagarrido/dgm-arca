"""Gold set loading and the rule that decides which chunks are relevant.

Each answerable question is annotated with *evidence*: the document and the exact text
that answers it (``rag/gold/gold.jsonl``). The annotation does not name chunk ids, because
ids change with every chunking. Instead, a chunk is relevant when it is from the same
document and covers at least half of the evidence span. That makes one gold set valid for
every chunking strategy, and it is what the enunciado calls "el fragmento que las responde".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import Chunk
from rag.ingest import CORPUS_DIR

GOLD_DIR = Path(__file__).resolve().parent / "gold"
MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class Evidence:
    source: str
    text: str


@dataclass(frozen=True)
class GoldItem:
    id: str
    question: str
    answer: str | None
    evidence: tuple[Evidence, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def load_gold(path: Path | str = GOLD_DIR / "gold.jsonl") -> list[GoldItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        evidence = tuple(Evidence(e["source"], e["text"]) for e in row.get("evidence", []))
        if row.get("answerable", True) and not evidence:
            raise ValueError(f"{row['id']}: answerable item without evidence")
        items.append(GoldItem(row["id"], row["question"], row.get("answer"), evidence))
    return items


def find_span(document: str, quote: str) -> tuple[int, int] | None:
    """Span of ``quote`` in ``document``, tolerant to whitespace differences."""
    pattern = r"\s*".join(re.escape(ch) for ch in re.sub(r"\s+", "", quote))
    match = re.search(pattern, document)
    return (match.start(), match.end()) if match else None


class RelevanceJudge:
    """Maps gold evidence to the relevant chunk ids of a given chunking."""

    def __init__(self, corpus_dir: Path | str = CORPUS_DIR) -> None:
        self.corpus_dir = Path(corpus_dir)
        self._spans: dict[Evidence, tuple[int, int]] = {}

    def span(self, evidence: Evidence) -> tuple[int, int]:
        if evidence not in self._spans:
            text = (self.corpus_dir / evidence.source).read_text(encoding="utf-8")
            span = find_span(text, evidence.text)
            if span is None:
                raise ValueError(f"Evidence not found in {evidence.source}: {evidence.text[:60]!r}")
            self._spans[evidence] = span
        return self._spans[evidence]

    def is_relevant(self, chunk: Chunk, item: GoldItem) -> bool:
        for ev in item.evidence:
            if chunk.doc_id != ev.source:
                continue
            start, end = self.span(ev)
            overlap = min(end, chunk.end) - max(start, chunk.start)
            if overlap >= MIN_COVERAGE * (end - start):
                return True
        return False

    def relevant_ids(self, chunks: list[Chunk], item: GoldItem) -> set[str]:
        return {c.id for c in chunks if self.is_relevant(c, item)}
