"""Build ``rag/corpus/`` from official EU sources, as Markdown files that ``rag/ingest.py`` reads.

The corpus is versioned in git (~2 MB); this script regenerates it from the sources. Run::

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
import xml.etree.ElementTree as ET
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


# The Markdown is rendered straight from the XHTML, in document order, instead of from the
# ``ai_act_text.Unit`` tree: that tree is built for citing points, and it keeps text that is
# not a numbered point (an article's lead-in, an annex's numbered paragraphs, the sentence
# that opens a list inside a table cell) apart from the points or not at all.

_X = ai_act_text.XHTML
_SKIP = ("oj-ti-art", "oj-doc-ti")  # "Artículo 5" / "ANEXO III": replaced by the heading


def _inline(elem: ET.Element) -> str:
    return ai_act_text._text(elem)


def _flow(elem: ET.Element, depth: int) -> list[str]:
    """Lines of ``elem``'s content in document order; nested lists indented two spaces."""
    lines: list[str] = []
    inline: list[str] = [elem.text or ""]

    def flush() -> None:
        if text := " ".join(" ".join(inline).split()):
            lines.append("  " * depth + text)
        inline.clear()

    for child in elem:
        tag = child.tag.removeprefix(_X)
        if tag == "p" and "inline" in child.get("style", ""):
            inline.append(_inline(child))  # Annex VI: "1." and its text are inline siblings
        elif tag == "p":
            flush()
            if text := _inline(child):
                lines.append("  " * depth + text)
        elif tag == "div":
            flush()
            lines += _flow(child, depth)
        elif tag == "table":
            flush()
            lines += _table(child, depth)
        elif tag not in ("col", "colgroup"):
            inline.append(_inline(child))  # <span>, <a>: text that runs on
        inline.append(child.tail or "")
    flush()
    return lines


def _table(table: ET.Element, depth: int) -> list[str]:
    """A list item: one row with a label cell (``a)``, ``3.1.``) and a content cell."""
    lines: list[str] = []
    # Only this table's rows: rows of nested tables are rendered by the recursion.
    for row in table.findall(f"{_X}tbody/{_X}tr") + table.findall(f"{_X}tr"):
        # Annex I adds an empty indentation cell before the label.
        cells = [c for c in row if c.tag == f"{_X}td" and (_inline(c) or len(c))]
        if len(cells) != 2:
            lines += [line for c in cells for line in _flow(c, depth)]
            continue
        label = _inline(cells[0])
        body = _flow(cells[1], depth + 1)
        if body:
            lines.append("  " * depth + f"{label} {body[0].strip()}".strip())
            lines += body[1:]
        else:
            lines.append("  " * depth + label)
    return lines


def _blocks(div: ET.Element) -> list[str]:
    """One block per top-level element (paragraph, list item, lead-in sentence)."""
    blocks = []
    for child in div:
        cls = child.get("class", "")
        if any(c in cls.split() for c in _SKIP) or "eli-title" in cls.split():
            continue
        wrapper = ET.Element("div")
        wrapper.append(child)
        if lines := _flow(wrapper, 0):
            blocks.append("\n".join(lines))
    return blocks


def article_markdown(div: ET.Element, number: str, title: str) -> str:
    heading = f"# Artículo {number}. {title}".strip(". ")
    return "\n\n".join([heading, *_blocks(div)]) + "\n"


def annex_markdown(div: ET.Element, numeral: str, title: str) -> str:
    heading = f"# Anexo {numeral}. {title}".strip(". ")
    return "\n\n".join([heading, *_blocks(div)]) + "\n"


def write_regulation(out: Path) -> int:
    xhtml = ai_act_text.fetch_xhtml("es")
    document = ai_act_text.parse(xhtml, "es")
    out.mkdir(parents=True, exist_ok=True)
    for div in ai_act_text.parse_root(xhtml).iter(f"{_X}div"):
        ident = div.get("id", "")
        if re.fullmatch(r"art_\d+", ident):
            number = ident.removeprefix("art_")
            markdown = article_markdown(div, number, document.article_titles.get(number, ""))
            (out / f"articulo_{int(number):03d}.md").write_text(markdown)
        elif re.fullmatch(r"anx_[IVX]+", ident):
            numeral = ident.removeprefix("anx_")
            markdown = annex_markdown(div, numeral, document.annex_titles.get(numeral, ""))
            (out / f"anexo_{numeral}.md").write_text(markdown)
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
