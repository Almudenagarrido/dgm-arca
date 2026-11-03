# Corpus de la fase 3

El corpus no está en git: `uv run python -m rag.download_corpus` lo regenera en `rag/corpus/`
en menos de un minuto (unos 2 MB de Markdown). Todo procede de fuentes oficiales de la UE.

| Carpeta | Documentos | Fuente | Idioma |
|---|---|---|---|
| `aiact/articulo_NNN.md` | 113 artículos | Reglamento (UE) 2024/1689, DO L de 12.7.2024, vía CELLAR (`publications.europa.eu/resource/celex/32024R1689`) | español |
| `aiact/anexo_X.md` | 13 anexos | ídem | español |
| `aiact/considerandos_NNN-MMM.md` | 180 considerandos en 18 ficheros | ídem | español |
| `directrices/directrices_practicas_prohibidas_es.md` | 164 páginas | Directrices de la Comisión sobre prácticas de IA prohibidas, C(2025) 5052 (documento 118659 del newsroom de la DG CONNECT) | español |
| `directrices/directrices_definicion_sistema_ia_es.md` | 15 páginas | Directrices de la Comisión sobre la definición de sistema de IA (documento 118640) | español |
| `directrices/draft_high_risk_*_en.md` | 6 + 13 + 148 páginas | Borrador de directrices sobre la clasificación de sistemas de alto riesgo (Art. 6), consulta de mayo de 2026 (documentos 128559-128561) | inglés |

En total son 149 documentos y 1,76 millones de caracteres.

**Licencias.** Los actos jurídicos publicados en EUR-Lex y en el Diario Oficial se pueden
reutilizar libremente citando la fuente (Decisión 2011/833/UE de la Comisión). Los documentos de
la Comisión se reutilizan bajo CC BY 4.0 (Decisión C(2019) 1655). Los borradores de directrices
de alto riesgo son un documento de consulta, no la versión final. Las respuestas que se apoyen en
ellos lo dicen así.

**Calidad del texto.** El reglamento sale del XHTML oficial: está limpio y estructurado por
apartados y letras, y los exponentes se conservan (`10^25`). Las directrices salen de PDF con
pypdf, que a veces parte palabras («recom endaciones», «solution s»). Se eliminan los números de
página y las líneas del índice, y las líneas cortadas se vuelven a unir en párrafos. Esa partición
de palabras perjudica a BM25 en esos documentos; el retriever denso es más tolerante.
