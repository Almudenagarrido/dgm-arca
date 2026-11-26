# ruff: noqa: E501  (the XHTML fixture keeps the original markup)
"""The regulation's Markdown keeps every word of the XHTML, in order (no network)."""

from __future__ import annotations

import re

import pytest

from rag import download_corpus
from tool_use import ai_act_text

# Trimmed copies of the CELLAR markup for the cases the Unit-based renderer lost: a lead-in
# before a list (Art. 16), an article without numbered paragraphs or lists (Art. 113), an
# annex of inline-numbered paragraphs (Annex VI) and a list item whose own text is a <span>
# before a nested list (Annex VII, point 3.1).
XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<div class="eli-subdivision" id="art_16">
  <p class="oj-ti-art">Artículo 16</p>
  <div class="eli-title"><p class="oj-sti-art">Obligaciones de los proveedores</p></div>
  <p class="oj-normal">Los proveedores de sistemas de IA de alto riesgo:</p>
  <table><col width="4%"/><col width="96%"/><tbody><tr><td><p>a)</p></td><td><p>velarán por que cumplan los requisitos;</p></td></tr></tbody></table>
  <table><col width="4%"/><col width="96%"/><tbody><tr><td><p>b)</p></td><td><p>indicarán su nombre;</p></td></tr></tbody></table>
</div>
<div class="eli-subdivision" id="art_113">
  <p class="oj-ti-art">Artículo 113</p>
  <div class="eli-title"><p class="oj-sti-art">Entrada en vigor y aplicación</p></div>
  <p class="oj-normal">El presente Reglamento entrará en vigor a los veinte días de su publicación.</p>
  <p class="oj-normal">Será aplicable a partir del 2 de agosto de 2026.</p>
  <p class="oj-normal">No obstante:</p>
  <table><tbody><tr><td><p>a)</p></td><td><p>los capítulos I y II serán aplicables a partir del 2 de febrero de 2025;</p></td></tr></tbody></table>
</div>
<div class="eli-container" id="anx_VI">
  <p class="oj-doc-ti">ANEXO VI</p>
  <p class="oj-doc-ti">Procedimiento fundamentado en un control interno</p>
  <div class="oj-enumeration-spacing">
    <p style="display: inline;">1.   </p>
    <p style="display: inline;"><span>El procedimiento se basa en los puntos 2, 3 y 4.</span></p>
  </div>
  <div class="oj-enumeration-spacing">
    <p style="display: inline;">2.   </p>
    <p style="display: inline;"><span>El proveedor comprueba el sistema de gestión de la calidad.</span></p>
  </div>
</div>
<div class="eli-container" id="anx_VII">
  <p class="oj-doc-ti">ANEXO VII</p>
  <p class="oj-doc-ti">Conformidad fundamentada en el sistema de gestión de la calidad</p>
  <p class="oj-ti-grseq-1">3.   Sistema de gestión de la calidad</p>
  <table><tbody><tr><td valign="top"/><td><p>3.1.</p></td><td>
    <span>La solicitud del proveedor incluirá:</span>
    <table><tbody><tr><td><p>a)</p></td><td><p>el nombre y la dirección del proveedor;</p></td></tr></tbody></table>
    <table><tbody><tr><td><p>b)</p></td><td><p>la lista de los sistemas de IA;</p></td></tr></tbody></table>
  </td></tr></tbody></table>
</div>
</body></html>""".encode()


@pytest.fixture(scope="module")
def rendered() -> dict[str, tuple[str, str]]:
    """{div id: (markdown, whitespace-free text of the div without its title lines)}."""
    root = ai_act_text.parse_root(XHTML)
    document = ai_act_text.parse(XHTML)
    out = {}
    for div in root.iter(f"{ai_act_text.XHTML}div"):
        ident = div.get("id", "")
        if ident.startswith("art_"):
            n = ident[4:]
            markdown = download_corpus.article_markdown(div, n, document.article_titles[n])
        elif ident.startswith("anx_"):
            n = ident[4:]
            markdown = download_corpus.annex_markdown(div, n, document.annex_titles[n])
        else:
            continue
        titles = {"oj-ti-art", "oj-doc-ti", "eli-title"}
        body = [c for c in div if not titles & set(c.get("class", "").split())]
        out[ident] = (markdown, "".join("".join(ai_act_text._text(c) for c in body).split()))
    return out


def test_every_character_of_the_source_is_kept_in_order(rendered):
    for ident, (markdown, source) in rendered.items():
        body = markdown.split("\n", 1)[1]
        assert "".join(body.split()) == source, ident


def test_lead_in_comes_before_the_list(rendered):
    markdown = rendered["art_16"][0]
    assert markdown.startswith("# Artículo 16. Obligaciones de los proveedores\n\n")
    assert markdown.index("Los proveedores de sistemas") < markdown.index("a) velarán")


def test_article_without_lists_keeps_its_dates(rendered):
    markdown = rendered["art_113"][0]
    assert "entrará en vigor a los veinte días" in markdown
    assert "Será aplicable a partir del 2 de agosto de 2026." in markdown


def test_inline_numbered_annex_points_stay_on_one_line(rendered):
    blocks = rendered["anx_VI"][0].strip().split("\n\n")
    assert blocks[1:] == [
        "1. El procedimiento se basa en los puntos 2, 3 y 4.",
        "2. El proveedor comprueba el sistema de gestión de la calidad.",
    ]


def test_list_item_text_in_a_span_is_kept_and_nested_items_indented(rendered):
    markdown = rendered["anx_VII"][0]
    assert (
        "3. Sistema de gestión de la calidad\n\n3.1. La solicitud del proveedor incluirá:\n"
        in markdown
    )
    assert re.search(r"\n  a\) el nombre y la dirección del proveedor;\n  b\) ", markdown)
