"""The text of the AI Act, fetched from the EU Publications Office and split into citable units.

EUR-Lex's HTML pages sit behind a bot challenge, but the Publications Office's CELLAR
repository serves the same consolidated XHTML through plain content negotiation::

    GET http://publications.europa.eu/resource/celex/32024R1689
    Accept: application/xhtml+xml
    Accept-Language: spa

The document is well-formed XHTML (about 1.4 MB) where every article is a
``<div id="art_N">``, every paragraph a ``<div id="NNN.MMM">`` and every annex a
``<div id="anx_X">``. Lettered points (``a)``, ``b)``...) and numbered annex points (``1.``,
``2.``...) are two-column tables: the label in the first cell and the content in the second.
This module turns that into a tree of ``Unit`` objects and resolves references such as
``Art. 5.1.f``, ``Art. 50`` or ``Anexo III.4.a`` to their text.

The download is cached per process: the first lookup pays the HTTP call (~0.5 s), the rest
are lookups in memory. Phase 3 reuses ``load_units`` to build the RAG corpus.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache

import requests

CELLAR_URL = "http://publications.europa.eu/resource/celex/32024R1689"
EUR_LEX_URL = "https://eur-lex.europa.eu/legal-content/{lang}/TXT/?uri=CELEX:32024R1689"
LANGUAGES = {"es": "spa", "en": "eng"}
XHTML = "{http://www.w3.org/1999/xhtml}"

ANNEX_NUMERALS = (
    "I II III IV V VI VII VIII IX X XI XII XIII".split()
)  # the AI Act has thirteen annexes


@dataclass
class Unit:
    """A citable piece of the regulation: an article, a paragraph, a point or a sub-point."""

    label: str  # "5", "1", "f" ...; for annexes "III"
    text: str = ""  # the unit's own text, without its children
    children: list[Unit] = field(default_factory=list)

    def child(self, label: str) -> Unit | None:
        return next((c for c in self.children if c.label == label), None)

    def render(self, depth: int = 0) -> str:
        lines = [("  " * depth) + self.text] if self.text else []
        lines += [c.render(depth + 1) for c in self.children]
        return "\n".join(line for line in lines if line.strip())


@dataclass
class Document:
    language: str
    articles: dict[str, Unit]
    article_titles: dict[str, str]
    annexes: dict[str, Unit]
    annex_titles: dict[str, str]


# ---------------------------------------------------------------------------- fetching


def fetch_xhtml(language: str = "es", timeout: float = 60) -> bytes:
    """Download the regulation in ``language`` ("es" or "en") from CELLAR."""
    response = requests.get(
        CELLAR_URL,
        headers={"Accept": "application/xhtml+xml", "Accept-Language": LANGUAGES[language]},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------- parsing


def _text(elem: ET.Element) -> str:
    return " ".join("".join(elem.itertext()).split())


def _tag(elem: ET.Element) -> str:
    return elem.tag.removeprefix(XHTML)


def _items(container) -> tuple[list[str], list[Unit]]:
    """Paragraph texts and lettered/numbered items directly inside ``container``.

    ``container`` is an element or a list of elements. A cell with no block children (Annex I
    puts its entries in a ``<span>``) contributes its whole text.
    """
    texts: list[str] = []
    items: list[Unit] = []
    blocks = [e for e in container if _tag(e) in ("p", "table", "div")]
    if not blocks and isinstance(container, ET.Element):
        return ([_text(container)] if _text(container) else []), []
    for elem in blocks:
        tag = _tag(elem)
        if tag == "p":
            if text := _text(elem):
                # Text after a list (a closing subparagraph) stays after it, as an unlabelled unit.
                if items:
                    items.append(Unit(label="", text=text))
                else:
                    texts.append(text)
        elif tag == "table":
            for row in elem.iter(f"{XHTML}tr"):
                # Two cells (label, content); Annex I adds an empty indentation cell first.
                cells = [c for c in row if _tag(c) == "td" and (_text(c) or len(c))]
                if len(cells) != 2:
                    continue
                label = _text(cells[0]).rstrip(").").strip("(") or "?"
                sub_texts, sub_items = _items(cells[1])
                items.append(
                    Unit(
                        label=label,
                        text=f"{_text(cells[0])} {' '.join(sub_texts)}".strip(),
                        children=sub_items,
                    )
                )
                break  # one item per table; nested tables are handled by the recursion
        elif tag == "div":
            sub_texts, sub_items = _items(elem)
            texts += sub_texts
            items += sub_items
    return texts, items


PARAGRAPH_ID = re.compile(r"^\d{3}\.(\d{3})$")


def _article(div: ET.Element) -> tuple[Unit, str]:
    number = div.get("id", "").removeprefix("art_")
    title, intro, paragraphs = "", [], []
    for elem in div:
        cls, ident = elem.get("class", ""), elem.get("id", "")
        if _tag(elem) == "p" and "oj-ti-art" in cls:
            continue
        if _tag(elem) == "div" and "eli-title" in cls:
            title = _text(elem)
        elif _tag(elem) == "div" and (m := PARAGRAPH_ID.match(ident)):
            texts, items = _items(elem)
            paragraphs.append(
                Unit(label=str(int(m.group(1))), text=" ".join(texts), children=items)
            )
        else:
            texts, items = _items([elem])
            intro += texts
            paragraphs += items  # articles without numbered paragraphs (e.g. Art. 3)
    head = f"Artículo {number}. {title}" if title else f"Artículo {number}"
    return Unit(label=number, text=" ".join([head, *intro]).strip(), children=paragraphs), title


def _annex(div: ET.Element) -> tuple[Unit, str]:
    numeral = div.get("id", "").removeprefix("anx_")
    texts, items = _items(div)
    title = texts[1] if len(texts) > 1 else ""
    return Unit(label=numeral, text=" ".join(texts), children=items), title


def parse(xhtml: bytes, language: str = "es") -> Document:
    root = ET.fromstring(xhtml)
    articles, article_titles, annexes, annex_titles = {}, {}, {}, {}
    for div in root.iter(f"{XHTML}div"):
        ident = div.get("id", "")
        if re.fullmatch(r"art_\d+", ident):
            unit, title = _article(div)
            articles[unit.label], article_titles[unit.label] = unit, title
        elif re.fullmatch(r"anx_[IVX]+", ident):
            unit, title = _annex(div)
            annexes[unit.label], annex_titles[unit.label] = unit, title
    if not articles:
        raise ValueError("no articles found: the CELLAR document format may have changed")
    return Document(language, articles, article_titles, annexes, annex_titles)


@lru_cache(maxsize=2)
def load_document(language: str = "es") -> Document:
    """Fetch and parse once per process and language."""
    return parse(fetch_xhtml(language), language)


# ---------------------------------------------------------------------------- references

REFERENCE = re.compile(
    r"^\s*(?:(?P<art>art(?:[íi]culo|icle)?\.?)\s*(?P<num>\d+)|(?P<anx>anexo|annex)\s*(?P<roman>[IVX]+))"
    r"(?P<rest>(?:\s*[.,(]?\s*[0-9a-z]{1,3}\)?)*)\s*$",
    re.IGNORECASE,
)


def parse_reference(reference: str) -> tuple[str, str, list[str]]:
    """``"Art. 5.1.f"`` -> ``("article", "5", ["1", "f"])``; ``"Anexo III.4"`` -> annex."""
    match = REFERENCE.match(reference)
    if not match:
        raise ValueError(
            f"unrecognised reference {reference!r}; use forms like 'Art. 5', 'Art. 5.1.f', "
            "'Art. 50.2' or 'Anexo III.4.a'"
        )
    path = [p.lower() for p in re.findall(r"[0-9a-z]{1,3}", match.group("rest"), re.IGNORECASE)]
    if match.group("art"):
        return "article", match.group("num"), path
    return "annex", match.group("roman").upper(), path


def resolve(document: Document, reference: str) -> tuple[str, str, Unit]:
    """Return (canonical reference, title of the article/annex, unit) for a reference."""
    kind, top, path = parse_reference(reference)
    table = document.articles if kind == "article" else document.annexes
    unit = table.get(top)
    if unit is None:
        limit = f"1-{len(document.articles)}" if kind == "article" else "I-XIII"
        raise ValueError(
            f"{'Article' if kind == 'article' else 'Annex'} {top} does not exist ({limit})"
        )
    canonical = f"Art. {top}" if kind == "article" else f"Anexo {top}"
    for part in path:
        child = unit.child(part)
        if child is None:
            available = ", ".join(c.label for c in unit.children) or "none"
            raise ValueError(f"{canonical} has no point '{part}' (available: {available})")
        unit, canonical = child, f"{canonical}.{part}"
    title = (document.article_titles if kind == "article" else document.annex_titles).get(top, "")
    return canonical, title, unit
