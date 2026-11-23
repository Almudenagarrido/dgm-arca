"""Two chunking strategies, plus the statistics used to compare them.

* **Recursive** (``recursive_chunks``): split on the largest separator that keeps the
  piece under ``size`` characters (paragraph, line, sentence, word), with ``overlap``
  characters shared between neighbours. Blind to meaning, but predictable.
* **Semantic** (``semantic_chunks``): split the document into sentences, embed each one
  and cut where two consecutive sentences are far apart in embedding space. A cut happens
  when the cosine distance is above the ``percentile``-th percentile of the distances in
  that document, so the threshold adapts to how homogeneous each document is.
  ``min_chars`` and ``max_chars`` keep the pieces from being a lone heading or a whole
  annex.

Every chunk keeps its character span in the source document. That is how the evaluation
decides whether a chunk contains the gold evidence, whatever the chunking.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from rag.ingest import Document


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str
    text: str
    start: int  # character span in the source document
    end: int

    def index_text(self, with_title: bool) -> str:
        """Text that gets embedded and indexed. The title gives an article fragment its context."""
        return f"{self.title}\n{self.text}" if with_title else self.text


def _make_chunks(doc: Document, spans: Sequence[tuple[int, int]]) -> list[Chunk]:
    chunks = []
    for start, end in spans:
        # Trim whitespace but keep the span pointing at the trimmed text.
        raw = doc.text[start:end]
        lead = len(raw) - len(raw.lstrip())
        text = raw.strip()
        if not text:
            continue
        s = start + lead
        chunks.append(
            Chunk(f"{doc.doc_id}#{len(chunks)}", doc.doc_id, doc.title, text, s, s + len(text))
        )
    return chunks


# ------------------------------------------------------------------------------ recursive

RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def recursive_chunks(doc: Document, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=RECURSIVE_SEPARATORS,
        add_start_index=True,
    )
    spans = [
        (piece.metadata["start_index"], piece.metadata["start_index"] + len(piece.page_content))
        for piece in splitter.create_documents([doc.text])
    ]
    return _make_chunks(doc, spans)


# ------------------------------------------------------------------------------- semantic

# A sentence ends at a line break, or at ". " / "; " / ": " followed by something that
# starts a new sentence (capital, digit, opening quote or a list letter like "b)").
_SENTENCE_END = re.compile(r"\n+|(?<=[.;:])\s+(?=[A-ZÁÉÍÓÚÑ0-9«(\"]|[a-z]\))")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences in ``text`` (empty pieces dropped)."""
    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        if text[start : match.start()].strip():
            spans.append((start, match.start()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def semantic_breaks(embeddings: np.ndarray, percentile: float) -> list[int]:
    """Indices ``i`` such that a cut goes between sentence ``i`` and ``i + 1``.

    ``embeddings`` must be L2-normalised, one row per sentence.
    """
    if len(embeddings) < 2:
        return []
    distances = 1.0 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    threshold = np.percentile(distances, percentile)
    return [i for i, d in enumerate(distances) if d > threshold]


def semantic_chunks(
    doc: Document,
    embed_sentences: Callable[[list[str]], np.ndarray],
    percentile: float = 90,
    min_chars: int = 200,
    max_chars: int = 1500,
) -> list[Chunk]:
    sentences = split_sentences(doc.text)
    if not sentences:
        return []
    embeddings = embed_sentences([doc.text[s:e] for s, e in sentences])
    breaks = set(semantic_breaks(embeddings, percentile))

    spans, group_start = [], sentences[0][0]
    for i, (_, e) in enumerate(sentences):
        length = e - group_start
        next_end = sentences[i + 1][1] if i + 1 < len(sentences) else None
        too_long_with_next = next_end is not None and next_end - group_start > max_chars
        if next_end is None or (i in breaks and length >= min_chars) or too_long_with_next:
            spans.append((group_start, e))
            if next_end is not None:
                group_start = sentences[i + 1][0]
    # A single sentence longer than max_chars stays whole: cutting it mid-sentence would
    # defeat the point of this strategy, and the stats below make those cases visible.
    return _make_chunks(doc, spans)


# ---------------------------------------------------------------------------------- stats


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, float]:
    lengths = [len(c.text) for c in chunks]
    return {
        "chunks": len(chunks),
        "mean_chars": round(statistics.mean(lengths), 1),
        "median_chars": statistics.median(lengths),
        "p10_chars": float(np.percentile(lengths, 10)),
        "p90_chars": float(np.percentile(lengths, 90)),
        "max_chars": max(lengths),
        "under_100_chars": sum(n < 100 for n in lengths),
    }
