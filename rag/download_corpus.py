"""Build ``rag/corpus/`` from official EU sources, as Markdown files that ``rag/ingest.py`` reads.

The corpus is not in git (it is ~2 MB and regenerable). Run::

    uv run python -m rag.download_corpus

What it writes (sources and licences in ``rag/CORPUS.md``):

* ``aiact/articulo_NNN.md``, ``aiact/anexo_X.md``: Regulation (EU) 2024/1689 in Spanish, one
  file per article (113) and annex (13), from the Publications Office (CELLAR) through
  ``tool_use.ai_act_text``. Each paragraph is a block separated by a blank line, with its
  points on their own lines, so chunkers can split on the legal structure.
* ``aiact/considerandos_NNN-MMM.md``: the 180 recitals in groups of ten.
* ``directrices/*.md``: Commission guidelines (prohibited practices and AI-system definition,
  in Spanish) and the draft guidelines on high-risk classification (May 2026, English only),
  converted from PDF. Page numbers and table-of-contents lines are dropped and hard-wrapped
  lines are joined back into paragraphs.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import requests

from tool_use import ai_act_text

CORPUS_DIR = Path(__file__).parent / "corpus"
NEWSROOM = "https://ec.europa.eu/newsroom/dae/redirection/document/{id}"

GUIDELINES = {
    "directrices_practicas_prohibidas_es": (
        118659,
        "Directrices de la Comisión sobre prácticas de IA prohibidas (C(2025) 5052), español",
    ),
    "directrices_definicion_sistema_ia_es": (
        118640,
        "Directrices de la Comisión sobre la definición de sistema de IA, español",
    ),
    "draft_high_risk_general_principles_en": (
        128559,
        "Draft guidelines on high-risk classification: general principles (May 2026), English",
    ),
    "draft_high_risk_annex_i_en": (
        128560,
        "Draft guidelines on high-risk classification: Annex I (May 2026), English",
    ),
    "draft_high_risk_annex_iii_en": (
        128561,
        "Draft guidelines on high-risk classification: Annex III (May 2026), English",
    ),
}


# ---------------------------------------------------------------------------- regulation


def _unit_markdown(unit: ai_act_text.Unit, depth: int = 0) -> list[str]:
    lines = [("  " * max(depth - 1, 0)) + unit.text] if unit.text else []
    for child in unit.children:
        lines += _unit_markdown(child, depth + 1)
    return lines


def article_markdown(document: ai_act_text.Document, number: str) -> str:
    article = document.articles[number]
    title = document.article_titles.get(number, "")
    blocks = [f"# Artículo {number}. {title}".strip(". ")]
    for paragraph in article.children:
        blocks.append("\n".join(_unit_markdown(paragraph)))
    if not article.children:  # short articles keep their text after the heading
        head = f"Artículo {number}. {title}".strip(". ")
        blocks.append(article.text.removeprefix(head).strip())
    return "\n\n".join(b for b in blocks if b.strip()) + "\n"


def annex_markdown(document: ai_act_text.Document, numeral: str) -> str:
    annex = document.annexes[numeral]
    blocks = [f"# Anexo {numeral}. {document.annex_titles.get(numeral, '')}".strip(". ")]
    blocks += ["\n".join(_unit_markdown(point)) for point in annex.children]
    return "\n\n".join(b for b in blocks if b.strip()) + "\n"


def write_regulation(out: Path) -> int:
    document = ai_act_text.load_document("es")
    out.mkdir(parents=True, exist_ok=True)
    for number in document.articles:
        (out / f"articulo_{int(number):03d}.md").write_text(article_markdown(document, number))
    for numeral in document.annexes:
        (out / f"anexo_{numeral}.md").write_text(annex_markdown(document, numeral))
    numbers = sorted(document.recitals, key=int)
    for start in range(0, len(numbers), 10):
        group = numbers[start : start + 10]
        text = "\n\n".join(document.recitals[n] for n in group)
        name = f"considerandos_{int(group[0]):03d}-{int(group[-1]):03d}.md"
        (out / name).write_text(f"# Considerandos {group[0]} a {group[-1]}\n\n{text}\n")
    return len(document.articles) + len(document.annexes) + -(-len(numbers) // 10)


# ---------------------------------------------------------------------------- PDF guidelines

PARAGRAPH_START = re.compile(
    r"^(\(?\d{1,3}\)\s|\d+(\.\d+)*\.?\s+[A-ZÁÉÍÓÚÑ]|[–•-]\s|Por ejemplo|For example|[a-z]\)\s)"
)


def clean_pdf_text(pages: list[str]) -> str:
    """Join hard-wrapped lines into paragraphs; drop page numbers and TOC lines."""
    paragraphs: list[str] = []
    current = ""
    for page in pages:
        for raw in page.splitlines():
            line = " ".join(raw.split())
            if not line or re.fullmatch(r"\d{1,3}", line) or re.search(r"\.{5,}", line):
                continue
            if PARAGRAPH_START.match(line) and current:
                paragraphs.append(current)
                current = line
            else:
                current = f"{current} {line}".strip()
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def write_guidelines(out: Path) -> int:
    from pypdf import PdfReader

    out.mkdir(parents=True, exist_ok=True)
    for name, (doc_id, title) in GUIDELINES.items():
        response = requests.get(
            NEWSROOM.format(id=doc_id), headers={"User-Agent": "arca-corpus/1.0"}, timeout=120
        )
        response.raise_for_status()
        reader = PdfReader(io.BytesIO(response.content))
        text = clean_pdf_text([page.extract_text() or "" for page in reader.pages])
        source = NEWSROOM.format(id=doc_id)
        (out / f"{name}.md").write_text(f"# {title}\n\nFuente: {source}\n\n{text}\n")
        print(f"  {name}: {len(reader.pages)} pages, {len(text):,} chars")
    return len(GUIDELINES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()
    n_reg = write_regulation(args.out / "aiact")
    print(f"regulation: {n_reg} files")
    n_guid = write_guidelines(args.out / "directrices")
    print(f"guidelines: {n_guid} files -> {args.out}")


if __name__ == "__main__":
    main()
