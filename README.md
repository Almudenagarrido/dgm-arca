# ARCA

Agente con Razonamiento, Conocimiento y Acción: un asistente que clasifica sistemas de IA según
el AI Act (Reglamento (UE) 2024/1689). Es la práctica final de Modelos Generativos Profundos
(MIA, ICAI).

Le describes un sistema de IA en lenguaje natural («un banco usa un modelo que puntúa la
solvencia de quienes piden un préstamo») y te dice si es prohibido, de alto riesgo, de
riesgo limitado o de riesgo mínimo, con el artículo o el punto del anexo que lo justifica.

El repositorio se reinició el 18-nov-2026 a partir del corpus oficial (plan en
[`docs/CRONOGRAMA.md`](docs/CRONOGRAMA.md)). Estado actual:

| Fase | Endpoint | Estado |
|---|---|---|
| 1 · Razonamiento | `POST /reasoning` | Lista (los pesos se descargan aparte, ver «Pesos») |
| 2 · Herramientas | `POST /tools` | Lista |
| 3 · Conocimiento (RAG) | `POST /rag` | Solo el corpus |
| 4 · Agente | `POST /agent` | Pendiente |

---

## La pieza común: el motor de reglas

`rlm/aiact_rules.py` aplica la precedencia del reglamento sobre la descripción estructurada de un
sistema (`SystemProfile`):

1. Art. 5, prácticas prohibidas → `prohibido`.
2. Art. 6.1 y Anexo I, componente de seguridad de un producto regulado → `alto_riesgo`.
3. Anexo III, salvo las excepciones del Art. 6.3 (que nunca se aplican si hay elaboración de
   perfiles) → `alto_riesgo`.
4. Art. 50, obligaciones de transparencia (chatbots, deepfakes, contenido generado) →
   `riesgo_limitado`.
5. Todo lo demás → `riesgo_minimo`.

Devuelve la etiqueta y los artículos que la justifican. Se usa tres veces: genera y etiqueta los
datos de la fase 1, hace de verificador en sus recompensas y es la herramienta `classify_risk` de
la fase 2. Ninguna etiqueta del proyecto se anota a mano: todas salen de este código, que tiene
sus propios tests.

---

## Fase 1 · Razonamiento (`/reasoning`)

El objetivo es que un modelo pequeño, Qwen3-0.6B, razone antes de responder: primero escribe su
razonamiento entre `<think>…</think>` y después da la etiqueta y el artículo entre
`<answer>…</answer>`. Se sigue el camino de DeepSeek-R1 a pequeña escala: arranque en frío con
trazas de un profesor (SFT) y después aprendizaje por refuerzo con recompensas verificables (GRPO).

### Cómo se construye

```
generate_aiact.py ──► train / test / test_ood  (etiquetados por el motor de reglas)
        │
distill.py  Qwen3-8B (thinking) genera 4 trazas por problema ──► el verificador descarta las malas
        │
train_sft.py   LoRA sobre las trazas que sobreviven      ──► adaptador SFT
        │
train_grpo.py  GRPO desde el SFT con 3 recompensas       ──► adaptador final (ARCA_RLM_ADAPTER)
        │
evaluate.py / plots.py   pass@1 base vs SFT vs GRPO, curvas de recompensa y longitud
```

`rlm/generate_aiact.py` genera 1000 problemas de entrenamiento, 200 de test y 100 fuera de
distribución (OOD), repartidos a partes iguales entre las cuatro etiquetas. Cada problema sale de
un perfil del sistema y una plantilla de redacción. La primera versión del test compartía frases
con train en el 96 % de los casos y daba resultados inflados, así que se reescribió para que no
comparta ninguna. El OOD es de educación, un dominio que no aparece en el entrenamiento.

`rlm/distill.py` pide a Qwen3-8B 4000 trazas y se queda con las que tienen el formato, la
etiqueta y el artículo correctos, que son el 77,5 %. Como mucho se guardan 2 por problema, para
que los fáciles no dominen, y quedan 1758 trazas para el SFT.

GRPO usa tres recompensas (`rlm/rewards.py` y `rlm/aiact_rewards.py`). La de formato comprueba
que la respuesta respeta `<think>` y `<answer>`. La de exactitud compara la etiqueta con la del
motor de reglas. La tercera, propia del dominio, premia citar el artículo que justifica la
etiqueta: a un responsable de compliance una etiqueta sin su base legal no le sirve de mucho.

`rlm/grpo_step.py` implementa a mano, sin TRL, un paso del algoritmo: ventajas normalizadas por
grupo, ratio de políticas, recorte y penalización KL. `tests/test_grpo_step.py` lo comprueba.

### Resultados

Evaluación greedy sobre 200 casos de test y 100 OOD. *Etiqueta + cita* exige además citar el
artículo correcto.

| Modelo | Test: etiqueta | Test: etiqueta + cita | OOD: etiqueta | OOD: etiqueta + cita |
|---|---|---|---|---|
| Qwen3-0.6B base | 26,5 % | 5,0 % | 17,0 % | 0,0 % |
| + SFT | 63,0 % | 51,5 % | 67,0 % | 61,0 % |
| + GRPO | 75,5 % | 67,0 % | 83,0 % | 74,0 % |

El SFT aprende el formato, pero escribe unos 1000 tokens por respuesta, y todos sus fallos de
formato son respuestas que se cortan dando vueltas al mismo paso. GRPO sube la etiqueta 12,5
puntos en test y 16 en OOD, y la cita todavía más. Que mejore más en OOD que en test encaja con
la idea de que el SFT memoriza y el RL generaliza.

Donde más falla es en `alto_riesgo`: 58 % en test y 44 % en OOD. Para acertar ahí hay que
encadenar varias condiciones (las excepciones del Art. 6.3, la elaboración de perfiles) y el
modelo de 0,6B se pierde.

### Pesos

Los adaptadores LoRA no van en git (`rlm/weights/` está en `.gitignore`). Están en el release
[`rlm-weights`](https://github.com/Almudenagarrido/dgm-arca/releases/tag/rlm-weights),
en un archivo de 76 MB con el adaptador final de GRPO (`final_rlm_lora`) y el de SFT
(`sft_lora`), los dos sobre Qwen3-0.6B:

```bash
curl -L -o /tmp/rlm-weights.tar.gz \
  https://github.com/Almudenagarrido/dgm-arca/releases/download/rlm-weights/arca-v1-rlm-weights.tar.gz
mkdir -p rlm/weights && tar -xzf /tmp/rlm-weights.tar.gz -C rlm/weights
```

`.env.example` ya apunta `ARCA_RLM_ADAPTER` a `rlm/weights/final_rlm_lora`. Para entrenarlos de
nuevo hace falta una GPU de 16 GB: ver [`rlm/README.md`](rlm/README.md) y
[`smoke/README.md`](smoke/README.md).

---

## Fase 2 · Herramientas (`/tools`)

Aquí el modelo consulta el texto oficial, calcula multas y escribe informes, cosas que no puede
hacer solo. Es Qwen3-4B con un bucle de tool use escrito a
mano, sin frameworks.

### Las herramientas (`tool_use/tools.py`)

| Herramienta | Tipo | Qué hace |
|---|---|---|
| `lookup_ai_act` | Consulta externa | Descarga el reglamento de la API CELLAR de la UE y devuelve el artículo o el punto del anexo pedido (`Art. 5.1.f`, `Anexo III.4.a`) |
| `classify_risk` | Cálculo | Llama al motor de reglas con los hechos del sistema y devuelve la etiqueta y su base legal |
| `compute_max_fine` | Cálculo | Multa máxima del Art. 99 según el tipo de infracción y la facturación |
| `write_assessment_report` | Acción con efecto | Escribe un informe de evaluación en PDF. Pide confirmación |
| `calculator` | Cálculo | Aritmética acotada (sin `eval`) |

### Cómo funciona el bucle (`tool_use/executor.py`)

1. `registry.py` convierte cada función en un JSON Schema a partir de sus type
   hints y su docstring. Se pasan a la plantilla de chat nativa de Qwen3
   (`apply_chat_template(..., tools=...)`).
2. El modelo responde con texto o con una o varias llamadas
   `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
3. `parser.py` detecta las llamadas, incluidas varias en la misma respuesta
   (paralelas), y separa el `<think>` si lo hay.
4. Pydantic valida los argumentos antes de ejecutar nada. Si fallan,
   el error vuelve al modelo como observación para que se corrija.
5. Cada resultado vuelve como mensaje con rol `tool` y el modelo sigue. Esto se
   repite hasta que da una respuesta final o se alcanza `max_turns` (multi-step).

### Resultados

Banco de 34 casos (`tool_use/bench/cases.jsonl`): clasificar, consultar artículos, calcular
multas, escribir informes, cadenas de varias herramientas y casos que no necesitan ninguna.

| Configuración | Selección | Argumentos | Etiqueta | Respuesta correcta |
|---|---|---|---|---|
| Qwen3-4B con herramientas | 97 % | 85 % | 80 % | 89 % |
| Qwen3-4B con el prompt de las diapositivas | 91 % | 71 % | 80 % | 93 % |
| Qwen3-4B con razonamiento (thinking) | 97 % | 68 % | 70 % | 86 % |
| Qwen3-1.7B con herramientas | 91 % | 53 % | 10 % | 57 % |
| Qwen3-4B sin herramientas | n/a | n/a | 0 % | 11 % |

Sin herramientas, el mismo modelo no acierta ninguna clasificación y solo el 11 % de las
respuestas verificables, porque no se sabe los artículos ni las cuantías de las multas.

Lo que más subió los argumentos fue reescribir las descripciones de las herramientas. Con un
índice de artículos válidos y mensajes de error que explican cómo corregir la llamada, pasaron
del 62 % al 85 %. La plantilla nativa de Qwen3 también da argumentos más precisos que el prompt
de las diapositivas, probablemente porque es el formato con el que se entrenó el modelo.

Son solo 34 casos y cada uno vale 3 puntos, así que las diferencias pequeñas entre filas no son
significativas.

---

## Levantarlo

Hace falta Python 3.11 o 3.12 y [`uv`](https://docs.astral.sh/uv/), o Docker con
`nvidia-container-toolkit`.

```bash
cp .env.example .env                 # configuración de las fases
uv sync --extra train                # fases 1 y 2
uv run arca-api                      # API en http://localhost:8000/docs
```

Con Docker: `docker compose up api`. `GET /health` dice qué fases están listas.

```bash
curl -s -X POST localhost:8000/reasoning -H 'Content-Type: application/json' \
  -d '{"question": "Una empresa usa un sistema que filtra currículums para preseleccionar candidatos."}'

curl -s -X POST localhost:8000/tools -H 'Content-Type: application/json' \
  -d '{"query": "¿Qué dice el artículo 5.1.f y qué multa máxima tiene una empresa que factura 50 M€ si lo incumple?"}'
```

### Regenerar datos y resultados

```bash
uv run python -m rlm.generate_aiact          # datos de la fase 1
uv run python -m rlm.distill                 # trazas del profesor (GPU)
uv run python -m rlm.train_sft               # SFT (GPU)
uv run python -m rlm.train_grpo              # GRPO (GPU)
uv run python -m rlm.evaluate                # pass@1 base / SFT / GRPO
uv run python -m tool_use.bench.build_cases  # banco de la fase 2
uv run python -m tool_use.evaluate           # métricas de la fase 2, con y sin herramientas
```

Las opciones de cada script están en su `--help` y en [`rlm/README.md`](rlm/README.md) y
[`tool_use/README.md`](tool_use/README.md).

### Comprobaciones

```bash
uv run pytest    # tests sin GPU ni red
make lint
```

---

## Estructura

| Carpeta | Qué hay |
|---|---|
| [`rlm/`](rlm/README.md) | Fase 1: motor de reglas, generador de datos, verificador, recompensas, destilación, SFT, GRPO (y un paso a mano), evaluación e inferencia |
| [`tool_use/`](tool_use/README.md) | Fase 2: herramientas, registro, parser, bucle de tool use, banco de 34 casos y evaluación |
| [`smoke/`](smoke/README.md) | Comprobación del entorno de GPU y GRPO de prueba |
| [`rag/`](rag/) | Corpus oficial (149 documentos) y script que lo regenera |
| [`api/`](api/README.md) | API FastAPI con los cuatro endpoints |
| [`docs/`](docs/) | Cronograma y extracción del corpus |

### El corpus

```
rag/corpus/                149 documentos Markdown (~2 MB): AI Act en español + directrices de la Comisión
rag/CORPUS.md              fuentes y licencias
rag/download_corpus.py     regenera el corpus desde las fuentes oficiales
docs/CORPUS_EXTRACCION.md  cómo se descargó y limpió el corpus, y los problemas que hubo
```

```bash
uv run python -m rag.download_corpus              # sobrescribe rag/corpus/
uv run python -m rag.download_corpus --out /tmp/c # o en otra carpeta, para comparar
```
