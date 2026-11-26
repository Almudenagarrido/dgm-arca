# Cómo se ha extraído el corpus

Este documento explica **solo** de dónde sale el corpus que se conserva al reiniciar el proyecto,
cómo se descargó y limpió, y qué problemas aparecieron por el camino. No describe nada de lo que
se hace después con él (chunking, embeddings, retrieval...).

## 1. Qué hay en el corpus

El corpus está en `rag/corpus/`: **149 documentos Markdown**, unos 2,1 MB en disco y
1,76 millones de caracteres de texto. Todo sale de **fuentes oficiales de la UE**.

```
rag/corpus/
├── aiact/                      144 ficheros · Reglamento (UE) 2024/1689 en español
│   ├── articulo_001.md … articulo_113.md       113 artículos, uno por fichero
│   ├── anexo_I.md … anexo_XIII.md               13 anexos, uno por fichero
│   └── considerandos_001-010.md … _171-180.md   180 considerandos, de diez en diez (18 ficheros)
└── directrices/                5 ficheros · documentos de la Comisión
    ├── directrices_practicas_prohibidas_es.md     C(2025) 5052 · 164 páginas · español · ~536 000 car.
    ├── directrices_definicion_sistema_ia_es.md    C(2025) 5053 ·  15 páginas · español · ~45 000 car.
    ├── draft_high_risk_general_principles_en.md   borrador mayo 2026 ·   6 páginas · inglés · ~14 000 car.
    ├── draft_high_risk_annex_i_en.md              borrador mayo 2026 ·  13 páginas · inglés · ~39 000 car.
    └── draft_high_risk_annex_iii_en.md            borrador mayo 2026 · 148 páginas · inglés · ~492 000 car.
```

Más del 60 % del texto son las directrices sobre prácticas prohibidas y el borrador del Anexo III.
El reglamento está muy troceado (144 ficheros) pero cada fichero es corto.

## 2. Las fuentes

| Documento | Organismo | Dónde se descarga | Formato original | Idioma |
|---|---|---|---|---|
| Reglamento (UE) 2024/1689 (AI Act), DO L de 12.7.2024 | Oficina de Publicaciones de la UE | CELLAR: `http://publications.europa.eu/resource/celex/32024R1689` | XHTML (~1,4 MB) | español |
| Directrices sobre prácticas de IA prohibidas, C(2025) 5052 | Comisión Europea, DG CONNECT | newsroom, documento **118659**: `https://ec.europa.eu/newsroom/dae/redirection/document/118659` | PDF | español |
| Directrices sobre la definición de sistema de IA, C(2025) 5053 | Comisión Europea, DG CONNECT | newsroom, documento **118640** | PDF | español |
| Borrador de directrices de clasificación de alto riesgo (Art. 6): principios generales | Comisión Europea (consulta de mayo de 2026) | newsroom, documento **128559** | PDF | inglés |
| ídem: Anexo I | ídem | newsroom, documento **128560** | PDF | inglés |
| ídem: Anexo III | ídem | newsroom, documento **128561** | PDF | inglés |

### Licencias

- **Reglamento**: los actos publicados en EUR-Lex y en el Diario Oficial se pueden reutilizar
  libremente citando la fuente (Decisión 2011/833/UE de la Comisión).
- **Documentos de la Comisión**: CC BY 4.0 (Decisión C(2019) 1655).
- Los **borradores de alto riesgo** son un documento de consulta, **no la versión final**. Lo
  que se apoye en ellos debe decirlo.

## 3. Cómo se extrajo el Reglamento

Lo hacen dos ficheros: `tool_use/ai_act_text.py` (descarga y parseo) y `rag/download_corpus.py`
(escritura en Markdown).

### 3.1 Descarga

1. **Primer intento, EUR-Lex.** La página normal
   (`eur-lex.europa.eu/legal-content/ES/TXT/?uri=CELEX:32024R1689`) responde **HTTP 202 con el
   cuerpo vacío** a cualquier petición automática: es un desafío anti-bots. No se puede descargar
   con `requests`.
2. **Solución, CELLAR.** El repositorio de la Oficina de Publicaciones sirve el mismo documento por
   *content negotiation*, sin desafío:

   ```http
   GET http://publications.europa.eu/resource/celex/32024R1689
   Accept: application/xhtml+xml
   Accept-Language: spa
   ```

   Devuelve unos 1,4 MB de XHTML bien formado en menos de medio segundo. Cambiando
   `Accept-Language` a `eng` se obtiene la versión inglesa (no está en el corpus).

### 3.2 Estructura del XHTML

El XHTML del Diario Oficial está muy bien etiquetado, y el parser se apoya en esos `id`:

| Elemento | Cómo aparece en el XHTML |
|---|---|
| Artículo N | `<div id="art_N">`; el título está en un `<div class="eli-title">` |
| Apartado M del artículo N | `<div id="NNN.MMM">` (p. ej. `005.001` = Art. 5.1) |
| Anexo X | `<div id="anx_X">` (números romanos I-XIII) |
| Considerando N | `<div id="rct_N">` |
| Letras `a)`, `b)`... y puntos `1.`, `2.` de los anexos | **tablas de dos columnas**: la etiqueta en la primera celda y el texto en la segunda |

El parser (`xml.etree.ElementTree`) construye un árbol de unidades citables
(artículo → apartado → letra → inciso) para poder resolver referencias como `Art. 5.1.f` o
`Anexo III.4.a`.

### 3.3 Escritura en Markdown

`rag/download_corpus.py` escribe el Markdown recorriendo el XHTML **en el orden del documento**,
no el árbol de 3.2. El árbol sirve para citar puntos, pero deja fuera el texto que no es un punto
numerado (ver «Texto perdido» en 3.4). Un test comprueba que cada fichero conserva todos los
caracteres del XHTML y en el mismo orden. Escribe:

- **Un fichero por artículo** (`articulo_005.md`): `# Artículo 5. Prácticas de IA prohibidas`,
  después **un bloque por apartado separado por una línea en blanco**, y dentro de cada apartado
  las letras en su propia línea y los incisos (`i)`, `ii)`) sangrados. Así cualquier troceado
  posterior puede cortar respetando la estructura legal.
- **Un fichero por anexo** (`anexo_III.md`), con la misma lógica para sus puntos.
- **Los considerandos de diez en diez** (`considerandos_001-010.md`), porque cada uno por separado
  es demasiado corto.

### 3.4 Problemas encontrados en el Reglamento y cómo se resolvieron

| Problema | Síntoma | Solución |
|---|---|---|
| EUR-Lex bloquea bots | HTTP 202 con cuerpo vacío | Descargar de CELLAR por content negotiation (3.1) |
| **Exponentes perdidos** | El umbral de cómputo del Art. 51 aparecía como «**1025** FLOPs» en vez de 10^25, porque `itertext()` pega el texto del `<span class="oj-super">` | Antes de extraer el texto, cada `oj-super` con dígitos se reescribe como `^25`. Los `oj-super` con letras (el «º» de «n.º») se dejan como están |
| Letras en tablas | Las letras `a)`, `b)` no son listas HTML sino tablas | Cada fila de dos celdas se lee como (etiqueta, contenido) |
| **Anexo I distinto** | Sus tablas tienen una celda vacía de sangría delante, y el texto va en un `<span>` en lugar de `<p>` | Se descartan las celdas vacías; si una celda no tiene bloques hijos, se toma su texto entero |
| Párrafos de cierre tras una lista | En el Art. 5.1, el párrafo «El párrafo primero, letra h), se entiende sin perjuicio...» va **después** de las letras y se colaba al principio | El texto que sigue a una lista se guarda como unidad sin etiqueta, detrás de la lista |
| Artículos sin apartados numerados | El Art. 3 (definiciones) y otros cortos no tienen `<div id="NNN.MMM">` | El contenido se toma de la introducción del artículo; en los cortos el texto va justo después del título |
| Tablas anidadas | Algunas letras contienen sub-tablas | Se procesa una fila por tabla y las anidadas se tratan por recursión |
| **Texto perdido** (arreglado el 24 nov) | El Markdown se escribía desde el árbol de 3.2 y solo incluía los hijos de cada unidad. Se perdían ~3.700 caracteres (sin contar espacios) en 19 ficheros: el Anexo VI quedaba vacío, el Art. 113 sin «entrará en vigor… aplicable a partir del 2 de agosto de 2026», el Anexo VII sin sus puntos 1-2 ni el «La solicitud del proveedor incluirá:» del 3.1, las frases de entrada de los Arts. 3, 16, 66, 108 y 110 y de casi todos los anexos, y en otros 34 los subpárrafos quedaban pegados en una línea | El Markdown se escribe recorriendo el XHTML en orden. Los `<p style="display: inline">` (Anexo VI: «1.» y su texto) se juntan en una línea. Test: `tests/test_download_corpus.py` |
| Formato de CELLAR cambiante | Si la Oficina de Publicaciones cambia el XHTML, el parser no encontraría artículos | El parser lanza un error explícito si no encuentra ningún `art_N`, en lugar de escribir un corpus vacío |

**Calidad final:** el texto del Reglamento está limpio, completo (113 + 13 + 180) y estructurado.

## 4. Cómo se extrajeron las directrices (PDF)

### 4.1 Descarga

Cada documento del newsroom de la DG CONNECT se descarga en
`https://ec.europa.eu/newsroom/dae/redirection/document/<id>`, con un `User-Agent` genérico
(`arca-corpus/1.0`) y 120 s de timeout.

**Problema: encontrar la versión española.** Cada directriz está publicada en **24 traducciones**,
cada una con su propio identificador. Hubo que localizar a mano cuáles eran las españolas (118659
y 118640). El **borrador de alto riesgo solo existe en inglés**, por eso esos tres documentos
están en inglés y el corpus es **bilingüe**.

### 4.2 Extracción del texto

1. El PDF se lee en memoria con **pypdf** (`PdfReader`) y se extrae el texto página a página.
2. Limpieza (`clean_pdf_text`):
   - se normalizan los espacios de cada línea;
   - se **eliminan los números de página** (líneas que solo tienen 1-3 dígitos);
   - se **eliminan las líneas del índice** (las que tienen cinco o más puntos seguidos `.....`);
   - se **vuelven a unir las líneas cortadas** del PDF en párrafos. Empieza un párrafo nuevo
     cuando la línea comienza por un número de párrafo `(12)`, un epígrafe `3.2 Título`, una viñeta
     `–`/`•`, una letra `a)` o «Por ejemplo» / «For example». Si no, la línea se pega a la anterior.
3. Se escribe como Markdown con el título y la URL de origen en la cabecera:

   ```markdown
   # Directrices de la Comisión sobre prácticas de IA prohibidas (C(2025) 5052), español

   Fuente: https://ec.europa.eu/newsroom/dae/redirection/document/118659

   <texto en párrafos>
   ```

### 4.3 Problemas que quedan en las directrices (no resueltos)

| Problema | Ejemplo | Consecuencia |
|---|---|---|
| **pypdf parte palabras** | «recom endaciones», «solution s» | Una búsqueda por palabras exactas (BM25) no las encuentra; un buscador por embeddings es más tolerante |
| Cabeceras de portada | Todos empiezan por «ES ES COMISIÓN EUROPEA Bruselas...» o «EN EN EUROPEAN COMMISSION...» | Ruido en el primer párrafo |
| Marcadores de borrador | «Brussels, XXX […](2026) XXX draft» | Los borradores no tienen fecha ni número oficial |
| Sin estructura | Las directrices quedan como párrafos planos: no hay títulos Markdown ni identificadores de sección | Para citarlas hay que apoyarse en el número de párrafo que aparezca en el texto |
| Corte de párrafos heurístico | La regla de «empieza párrafo» a veces une dos párrafos o parte uno | Párrafos irregulares, sobre todo en tablas y notas al pie del PDF |
| Notas al pie | El texto de las notas se mezcla con el cuerpo | Algún fragmento con frases sueltas |

Si en el proyecto nuevo esto molesta, las opciones son: otro extractor (pdfplumber, PyMuPDF), o una
pasada de limpieza que una palabras partidas comparando con un diccionario.

## 5. Advertencias para el proyecto nuevo

1. **El corpus no está en git.** `.gitignore` excluye `rag/corpus/*`. Si se borra la carpeta local,
   desaparece. Antes de reiniciar el proyecto hay que **guardar una copia** (por ejemplo
   `tar czf corpus_2026-11-03.tar.gz rag/corpus`) o decidir versionarlo (2 MB, licencias que lo
   permiten).
2. **El script de descarga depende de otro módulo.** `rag/download_corpus.py` importa
   `tool_use/ai_act_text.py`. Si se borra todo menos el corpus, ya no se puede regenerar.
   Conviene conservar también esos dos ficheros, aunque solo sea como referencia.
3. **Es una foto fija.** El corpus se descargó el **3 de noviembre de 2026**. Si se vuelve a
   descargar, CELLAR podría servir una versión consolidada distinta, y los borradores de alto
   riesgo pueden ser sustituidos por la versión final. Las cifras de evaluación solo son
   comparables con la misma foto.
4. **Dependencias para regenerarlo:** `requests` y `pypdf`. Orden:
   `uv run python -m rag.download_corpus` (menos de un minuto).
5. **Mezcla de autoridad.** El corpus junta texto vinculante (artículos y anexos), interpretativo
   (considerandos), orientaciones no vinculantes (directrices) y un borrador en consulta. Conviene
   saber de qué tipo es cada fragmento al usarlo.
6. **Mezcla de idiomas.** 144 + 2 documentos en español y 3 en inglés. Las preguntas en español
   sobre alto riesgo pueden tener la respuesta solo en el texto inglés.
