# `rag/` — Fase 3 · Conocimiento (`POST /rag`)

Responde preguntas sobre el AI Act y las directrices de la Comisión con fragmentos recuperados
del corpus oficial, citando cada afirmación y diciendo «No lo sé» cuando el corpus no tiene la
respuesta. No usa nada de las fases 1 y 2.

```
pregunta ──► híbrido 0,7·denso + 0,3·BM25 (30 candidatos)
              │
              ▼
          Qwen3-Reranker-0.6B ──► top-k ──► ¿mejor P(sí) < 0,9939? ──sí──► «No lo sé»
                                                │ no
                                                ▼
                         Qwen3-4B con los fragmentos numerados [1]..[k]
                                                │
                                                ▼
                    citas [n] ──► ids de fragmento (las que no existen se descartan)
```

## Ficheros

| Fichero | Qué hace |
|---|---|
| `ingest.py` | Lee el corpus (Markdown, texto, HTML, PDF) en `Document(doc_id, title, text)`. El id es la ruta relativa (`aiact/articulo_005.md`) |
| `chunking.py` | Troceado recursivo y semántico. Cada fragmento guarda su posición en el documento. Estadísticas |
| `embed.py` | Qwen3-Embedding-0.6B: consultas con instrucción, documentos sin ella. Caché en `rag/index/` |
| `retriever.py` | Almacén vectorial (matriz normalizada, búsqueda exacta por coseno), denso, BM25, híbrido con λ y `KnowledgeBase` |
| `rerank.py` | Qwen3-Reranker-0.6B como cross-encoder: puntuación = P(«yes») |
| `generate.py` | `RagPipeline`, el punto de entrada de `/rag`: recuperar, reordenar, umbral, generar y validar citas |
| `goldset.py` | Carga el conjunto dorado y decide qué fragmentos son relevantes |
| `evaluate.py` | Recall@k y MRR por troceado × retriever, barrido de λ con validación cruzada |
| `calibrate.py` | Umbral del «No lo sé» sobre un conjunto aparte |
| `evaluate_answers.py` | Respuestas de punta a punta: correctas, fieles, citas válidas, abstenciones |
| `gold/gold.jsonl` | 60 preguntas con respuesta + 10 sin respuesta |
| `gold/calibration.jsonl` | 15 + 15 preguntas distintas, solo para el umbral |
| `corpus/`, `CORPUS.md`, `download_corpus.py` | El corpus (149 documentos, 1,76 M caracteres), sus licencias y el script que lo regenera |

## Cómo ejecutarlo

```bash
uv sync --extra train --extra rag
uv run python -m rag.evaluate            # recuperación: 8 troceados × 3 retrievers + barrido de λ
uv run python -m rag.calibrate           # umbral del «No lo sé»
uv run python -m rag.evaluate_answers    # respuestas, con LLM como juez
uv run pytest tests/test_rag.py          # sin GPU ni red
```

Los resultados se escriben en `reports/phase3_*` (fuera de git). Las cifras de abajo salen de
ahí, y la versión fechada con lo que se probó está en
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## El conjunto dorado

Lo escribí leyendo el corpus, no generándolo. Hay 60 preguntas con respuesta repartidas por
todo el reglamento: multas (Art. 99-101), prohibiciones (Art. 5), clasificación de alto riesgo
(Art. 6 y Anexo III), obligaciones de proveedores y responsables del despliegue, transparencia
(Art. 50), modelos de uso general (Art. 51-55), sandboxes, incidentes, gobernanza, ámbito de
aplicación y fechas. Seis son de las directrices: dos de la definición de sistema de IA, dos de
las prácticas prohibidas y dos de los borradores de alto riesgo en inglés. Hay además 10
preguntas del dominio que el corpus no responde (la autoridad española, precios, la ley de
California, multas ya impuestas...). Se comprobó con `grep` que el corpus no contiene esa
información.

Las preguntas están **parafraseadas** a propósito: «¿qué rasgos sensibles no se pueden deducir
con categorización biométrica?» en lugar de copiar las palabras del artículo. Si se copiaran,
BM25 ganaría por construcción.

Cada pregunta lleva su **evidencia**: el documento y el texto literal que la responde. No
apunta a un id de fragmento, porque los ids cambian con cada troceado. Un fragmento es relevante
si es del mismo documento y cubre al menos la mitad de la evidencia, así que el mismo dorado
sirve para todos los troceados (`goldset.py`). Un test comprueba que las 75 citas aparecen
literalmente en el corpus.

## Recuperación

Las 60 preguntas con respuesta, evaluadas con el retriever híbrido en su mejor λ.

| Troceado | Fragmentos | Media (car.) | Denso MRR | BM25 MRR | Híbrido MRR | Híbrido R@5 | Híbrido R@10 | λ* | MRR con val. cruzada |
|---|---|---|---|---|---|---|---|---|---|
| recursive-500 | 5429 | 326 | 0,323 | 0,241 | 0,363 | 0,45 | 0,48 | 0,9 | 0,363 |
| recursive-1000 | 2589 | 689 | 0,416 | 0,268 | 0,439 | 0,58 | 0,62 | 0,9 | 0,439 |
| recursive-1500 | 1663 | 1082 | 0,394 | 0,311 | 0,457 | 0,58 | 0,68 | 0,8 | 0,432 |
| semantic-80 | 1977 | 890 | 0,416 | 0,319 | 0,449 | 0,57 | 0,67 | 0,7 | 0,449 |
| semantic-90 | 1686 | 1044 | 0,399 | 0,322 | 0,456 | 0,53 | 0,65 | 0,7 | 0,456 |
| semantic-95 | 1556 | 1132 | 0,391 | 0,334 | 0,447 | 0,58 | 0,68 | 0,7 | 0,417 |
| **recursive-1000+title** | 2589 | 689 | 0,443 | 0,296 | **0,476** | **0,63** | 0,73 | 0,7 | **0,476** |
| semantic-90+title | 1686 | 1044 | 0,374 | 0,371 | 0,475 | 0,62 | **0,75** | 0,6 | 0,461 |

`semantic-N` corta donde la distancia entre frases consecutivas supera el percentil N del
documento. `+title` antepone el título del documento («Artículo 99. Sanciones») al texto que se
indexa.

**Lo que dicen los números:**

- **El híbrido siempre gana a sus dos componentes.** El denso es el mejor retriever individual
  (las preguntas están parafraseadas), pero BM25 aporta las coincidencias exactas de cifras y
  términos legales («ultrasuplantación», «sesgo de automatización»), y las dos señales se
  complementan.
- **Semántico frente a recursivo: empate.** Con el mismo tamaño medio, `semantic-90` (1044
  caracteres) y `recursive-1500` (1082) quedan a 0,001 de MRR. El corpus ya viene troceado por
  artículo y apartado, así que el recursivo, que corta primero por párrafos, ya respeta la
  estructura legal, y el semántico tiene poco que añadir. Donde sí se nota es en los fragmentos
  minúsculos: el recursivo de 1000 deja 46 por debajo de 100 caracteres y el semántico 1.
- **Fragmentos demasiado pequeños hacen daño**: con 500 caracteres el MRR cae a 0,36. Una
  respuesta legal necesita el sujeto, la obligación y el plazo en el mismo fragmento.
- **El título es la mejora más barata**: +0,02 a +0,04 de MRR en los dos troceados. Un apartado suelto
  («3. ... hasta 35 000 000 EUR») no dice de qué artículo es, y el título se lo da.
- **Cómo se eligió λ.** El barrido va de 0 (solo BM25) a 1 (solo denso). Para no elegirlo y
  evaluarlo con las mismas 60 preguntas, se hizo validación cruzada en 5 pliegues: se elige λ
  con 48 preguntas y se mide con las otras 12. `recursive-1000+title` saca la mejor cifra con
  validación cruzada (0,476) y su λ es 0,7 en los cinco pliegues. `semantic-90+title` queda a
  0,001 en MRR directo (0,475), pero su λ baila entre 0,5 y 0,7 según el pliegue y con validación
  cruzada baja a 0,461. **Me quedo con `recursive-1000+title`, λ = 0,7.**
- **Con 60 preguntas el margen de error es grande.** El intervalo de confianza al 95 % de R@5
  del híbrido elegido es [0,52; 0,75]. Las diferencias de ±0,02 entre filas no son
  significativas. Lo que sí es robusto: híbrido > denso > BM25, y 500 caracteres es peor.

**Instrucción en la consulta** (medido con el corpus anterior al arreglo del 24 de noviembre).
Qwen3-Embedding admite una instrucción delante de la pregunta.
Al quitarla, el MRR del híbrido baja de 0,438 a 0,420 con `recursive-1000` y sube de 0,478 a
0,486 con `semantic-90+title`. Es un efecto pequeño y de signo variable. Esperaba más, porque
pregunta y respuesta no se parecen, pero con preguntas tan del dominio la consulta ya dice qué
se busca. La instrucción se mantiene porque es lo que recomienda el modelo y no empeora en
media.

**Análisis de fallos.** 16 de las 60 preguntas no tienen su evidencia en el top-10 del
retriever elegido (el reranker parte de 30 candidatos, y en el top-30 está el 87 %). Mirando
qué se recupera en su lugar:

- **Fallos que no lo son (g07, g08, g11...).** En «¿está prohibido detectar las emociones de
  los empleados?», los cuatro primeros fragmentos son de las directrices de prácticas
  prohibidas, con ejemplos como «está prohibido que un centro de llamadas utilice cámaras web...
  para controlar las emociones de su empleado». Responden a la pregunta, y mejor que el
  artículo, pero el dorado solo acepta el Art. 5. Las directrices citan y desarrollan cada letra
  del Art. 5, con 778 fragmentos frente a los 19 del artículo. La métrica es estricta y se
  queda por debajo de la calidad real. No amplié el dorado después de ver los resultados para
  que no se ajuste al sistema. Lo correcto sería anotar desde el principio más de una evidencia
  válida por pregunta.
- **Fallos reales de vocabulario (g54).** «¿Qué tiene que hacer una empresa de fuera de la UE
  antes de vender...?» recupera el Art. 49 (registro), que empieza igual («Antes de introducir
  en el mercado...»), y no el Art. 22 (representante autorizado), que habla de «terceros
  países», no de «fuera de la UE».
- **Fallos entre idiomas (g59, g60).** Las dos preguntas cuya respuesta solo está en los
  borradores en inglés fallan en todos los troceados. Qwen3-Embedding es multilingüe, pero una
  pregunta en español se parece más a cualquier fragmento en español del mismo tema que al
  párrafo inglés correcto, y BM25 no comparte ni una palabra. La mejora natural sería traducir
  la consulta al inglés y fusionar las dos búsquedas.

## «No lo sé»

El generador ya se abstiene solo cuando los fragmentos no le sirven: sin umbral, Qwen3-4B
responde «No lo sé» en 8 de las 10 preguntas sin respuesta. El umbral es una segunda barrera
que no depende de que el modelo lo reconozca. Si la mejor puntuación del reranker entre los
fragmentos recuperados no llega al umbral, se responde «No lo sé» sin llamar al generador.

Se calibra con `gold/calibration.jsonl`, 15 preguntas con respuesta y 15 sin ella, todas
distintas del dorado. Se elige el umbral que maximiza la exactitud equilibrada (la media de
«responde cuando debe» y «se abstiene cuando debe»), y después se aplica sin tocarlo al dorado:

| Puntuación del umbral | Umbral | Calibración: responde / se abstiene | Dorado: responde / se abstiene |
|---|---|---|---|
| **Mejor P(sí) del reranker** | 0,9939 | 100 % / 93 % | 93 % / 90 % |
| Mejor coseno denso | 0,677 | 100 % / 80 % | 77 % / 60 % |

El coseno denso no sirve como umbral: las preguntas fuera del corpus («¿cuánto cuesta registrar
un sistema?») se parecen a muchos fragmentos aunque ninguno las responda. El reranker lee la
pregunta y el fragmento juntos y separa mucho mejor, aunque sus puntuaciones se amontonan cerca
de 1 (las preguntas con respuesta del conjunto de calibración están todas por encima de 0,996).
Por eso el umbral queda en 0,9939 y no en algo como 0,5.

El umbral se calibró con el retriever híbrido. Con `retriever: "bm25"` los candidatos son
peores, la mejor puntuación baja y se abstiene más: «¿qué multa hay por incumplir las
obligaciones de transparencia?» se queda en 0,978 con BM25 y responde con el híbrido. Para
desactivar el umbral basta con dejar `ARCA_RAG_MIN_SCORE` vacío.

## Respuestas de punta a punta

Las 70 preguntas del dorado con `recursive-1000+title`, λ = 0,7, top-5 y Qwen3-4B sin
razonamiento. Hay tres variantes que van añadiendo piezas:

| Variante | Evidencia en el contexto | Correcta | Fiel | Citas válidas | Cita la evidencia | Abstención falsa | Se abstiene sin respuesta |
|---|---|---|---|---|---|---|---|
| híbrido top-5 | 63 % | 77 % | 80 % | 100 % | 57 % | 3 % | 80 % |
| + reranker (30 → 5) | 67 % | **83 %** | **88 %** | 100 % | 60 % | 2 % | 80 % |
| + reranker + umbral (`/rag`) | 67 % | 78 % | 85 % | 100 % | 57 % | 7 % | **90 %** |

*Correcta* y *fiel* las decide un LLM como juez (`evaluate_answers.py`). *Citas válidas* y
*cita la evidencia* las comprueba el código: que cada `[n]` corresponda a un fragmento recuperado
y que alguno de los citados cubra la evidencia del dorado.

- **Las citas siempre existen.** En las 210 respuestas, ninguna cita queda fuera de los
  fragmentos recuperados después de la validación. En una prueba a mano con `top_k=3` el modelo
  escribió «[4]», y la validación lo apartó a `invalid_citations`.
- **El reranker es la pieza que más aporta**: +3 puntos de evidencia en el contexto y +7 de
  respuestas correctas. Saca a las primeras posiciones fragmentos que el híbrido tenía entre el
  6 y el 30.
- **El umbral cambia aciertos por abstenciones.** Gana una abstención (u01, «¿qué organismo
  supervisa la IA en España?», donde sin umbral el modelo divaga sobre «las autoridades
  designadas por el Estado miembro»), pero se abstiene sin motivo en tres preguntas que el
  reranker sí respondía bien (g06, g30, g33, con puntuaciones entre 0,94 y 0,99). Con 60 preguntas
  con respuesta frente a 10 sin ella, en este dorado sale a cuenta quitarlo. Lo mantengo porque
  inventar una respuesta legal es peor que no darla, y porque en un uso real la proporción de
  preguntas fuera del corpus no tiene por qué ser 1 de cada 7. Es una decisión de producto, no
  algo que diga el dato, y se cambia con una variable de entorno.
- **La pregunta sin respuesta que se cuela (u02).** «¿Cuánto cuesta registrar un sistema de alto
  riesgo?» supera el umbral (0,997), porque el reranker ve fragmentos del Art. 71 sobre la base de
  datos. El modelo deduce «la base de datos es de acceso público y gratuito, por lo tanto el
  registro no tiene coste». El fragmento habla del acceso, no del registro. Es la alucinación
  más peligrosa: plausible, con cita y falsa.
- **Correcta sin citar la evidencia.** La diferencia entre «correcta» (83 %) y «cita la
  evidencia» (60 %) son sobre todo respuestas que citan las directrices en lugar del artículo.
  Responden bien con otra fuente válida que el dorado no anota, lo mismo que se vio en la
  recuperación.
- **Un fallo de generación con el contexto correcto.** «¿Qué multa hay por incumplir las
  obligaciones de transparencia?» (prueba a mano con la API): el modelo responde 35 M€ o el 7 %
  (la de las prácticas prohibidas) en lugar de 15 M€ o el 3 % del Art. 99.4.g, y cita el Art. 100
  y las directrices. El fragmento correcto estaba entre los cinco. Un modelo de 4B mezcla las
  cuantías de apartados contiguos.

**El juez.** Por memoria (porción MIG de 16 GB) no caben Qwen3-4B y Qwen3-8B a la vez, así que
el juez es el propio generador. Revisé a mano 20 respuestas de la variante híbrida (una de cada
tres) y coincido con su veredicto de *correcta* en 18. Los dos desacuerdos son de un juez
demasiado blando: g13 cita las condiciones del Art. 7.3 (quitar casos del Anexo III) en lugar de
las excepciones del Art. 6.3, y g49 empieza con «Sí, tienes obligaciones», al revés que la
referencia, y las dos se dan por correctas. La cifra de «correcta» tiene un sesgo optimista de
unos 10 puntos. Las comparaciones entre variantes valen igual, porque el sesgo es el mismo en
las tres. *Fiel* es más estricto: marca como infiel cualquier detalle que no esté en los
fragmentos citados, aunque sea cierto (g10 añade «Art. 5.1.g» por su cuenta).

## Elegir el generador: local o servidor vLLM

La recuperación, el reranker, el umbral, el prompt y la validación de citas son siempre los
mismos. Lo único que cambia es quién escribe la respuesta, y se elige con variables de entorno
(en `.env` o en el propio comando):

| Variable | Valor por defecto | Qué hace |
|---|---|---|
| `LLM_BACKEND` | `local` | `local`: `ARCA_RAG_MODEL` (Qwen3-4B) en la GPU propia. `openai`: un servidor con API compatible con OpenAI |
| `LLM_BASE_URL` | `http://0.0.0.0:8001/v1` | URL del servidor (el vLLM de la universidad; `localhost` no funciona) |
| `LLM_API_KEY` | vacía | Clave del servidor, enviada como `Authorization: Bearer`. Solo en `.env`, que está en `.gitignore` |
| `LLM_MODEL` | vacía | Modelo que se pide. Si está vacía, el primero que anuncie `GET /v1/models` |
| `LLM_TIMEOUT` | `120` | Segundos de espera por respuesta |

```bash
LLM_BACKEND=openai ARCA_API_PORT=8017 uv run arca-api     # generación en el servidor vLLM
```

Con `openai` no se carga Qwen3-4B: la API ocupa ~3,5 GB de GPU (embedder y reranker) en lugar
de ~13,4 GB. La generación es en greedy, sin razonamiento (`enable_thinking: false`) y con el
mismo límite de 400 tokens que el modelo local. El campo `model` de la respuesta dice qué
modelo generó.

Si el servidor no responde a tiempo, está caído o rechaza la clave, `/rag` devuelve **HTTP
503** con un mensaje claro (por ejemplo, «El servidor LLM rechazó la clave (HTTP 401). Revisa
LLM_API_KEY en .env.») y la API sigue funcionando. El mensaje nunca incluye la clave.
`tests/test_rag_remote.py` lo comprueba con un servidor simulado, sin red.

**Comparación en la API real** (23-nov-2026, híbrido, `top_k` 5). El servidor sirve
`qwen3.8-27b-fp8`:

| Pregunta | Qwen3-4B local | qwen3.8-27b-fp8 remoto |
|---|---|---|
| ¿Qué multa hay por usar una práctica de IA prohibida? | Correcta: 35 M€ o 7 %. No cita el Art. 99 | Correcta: 35 M€ o 7 %, y 1,5 M€ para instituciones de la UE. Cita las directrices y el Art. 99 ([1][2]) y el Art. 100 ([3]) · `invalid_citations` vacío |
| ¿Está prohibido detectar las emociones de los empleados? | Correcta: sí, con matices | Correcta: sí, con la excepción de la formación personal y la prohibición en contratación y período de prueba. Cita 2 fragmentos de las directrices · `invalid_citations` vacío |
| ¿Cuál es la capital de Portugal? | Se abstiene | Se abstiene (umbral: 0,113) |
| ¿Qué organismo supervisa la IA en España? | Se abstiene | Se abstiene (umbral: 0,915) |

En las dos preguntas sin respuesta se abstiene el **umbral**, antes de llamar al generador,
así que el resultado no depende del backend. En las dos con respuesta, el modelo de 27B cita
más fragmentos y más precisos (ahora sí el Art. 99), y en esta prueba tardó entre 4 y 10 s por respuesta
(la primera incluye cargar el índice). Las cifras del dorado de arriba siguen siendo las de Qwen3-4B.
Para medir el 27B habría que repetir `rag.evaluate_answers` con este backend, que todavía no
lo admite.

## Límites y siguientes pasos

- **Dorado de una sola fuente por pregunta.** Anotar las evidencias alternativas (las
  directrices que desarrollan cada artículo) subiría las métricas de recuperación a su valor
  real, pero debería hacerse antes de ver los resultados, con otro conjunto.
- **Preguntas en español, respuesta en inglés.** Traducir la consulta y fusionar las dos
  búsquedas es lo más prometedor para los borradores de alto riesgo.
- **Memoria.** `/rag` ocupa ~13,4 GB de GPU con todo cargado (Qwen3-4B, embedder y reranker). En
  la porción de 16 GB no cabe junto a la fase 2, que carga su propio Qwen3-4B. `GET /health`
  carga todas las fases a la vez y se queda sin memoria. Las fases 1 a 3 juntas suman ~23 GB:
  con 24 GB va justo, y con 32 GB hay margen.
