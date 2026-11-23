"""Corpus ingestion: read every document under ``rag/corpus/`` into ``Document`` objects.

Supports Markdown, plain text, HTML and PDF so the corpus can grow beyond what
``download_corpus.py`` produces. The document id is its path relative to the corpus
root (``aiact/articulo_005.md``): stable, readable and the same string the gold set
uses to point at evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
SUFFIXES = (".md", ".txt", ".html", ".htm", ".pdf")


@dataclass(frozen=True)
class Document:
    doc_id: str  # path relative to the corpus root, with forward slashes
    title: str
    text: str


class _TextExtractor(HTMLParser):
    """Keeps the visible text of an HTML page, one block per line."""

    _BLOCKS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section"}
    _SKIP = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skipping -= 1

    def handle_data(self, data):
        if not self._skipping:
            self.parts.append(data)


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix in (".html", ".htm"):
        parser = _TextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raise ValueError(f"Unsupported file type: {path}")


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_corpus(root: Path | str = CORPUS_DIR) -> list[Document]:
    """All documents under ``root``, sorted by id so every run sees the same order."""
    root = Path(root)
    docs = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUFFIXES or not path.is_file():
            continue
        text = _read_text(path)
        if not text.strip():
            continue
        doc_id = path.relative_to(root).as_posix()
        docs.append(Document(doc_id=doc_id, title=_title(text, path.stem), text=text))
    if not docs:
        raise FileNotFoundError(f"No documents found under {root}")
    return docs
