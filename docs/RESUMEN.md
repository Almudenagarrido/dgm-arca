# Resumen del proyecto (fases 0 a 3)

Guía rápida de qué hemos hecho, de dónde salen los datos y qué es cada fichero. El detalle de
cada experimento está en [`EXPERIMENTS.md`](EXPERIMENTS.md). La fase 4 (agente, `agent/`) no
se cubre aquí.

## Qué es ARCA

Un asistente que recibe la descripción de un sistema de IA y lo clasifica según el AI Act
(Reglamento (UE) 2024/1689) en **prohibido**, **alto riesgo**, **riesgo limitado** o **riesgo
mínimo**, citando el artículo o el anexo que lo justifica.

La pieza central es un **motor de reglas** (`rlm/aiact_rules.py`) que aplica el reglamento en
orden: Art. 5, Art. 6.1 + Anexo I, Anexo III con las excepciones del Art. 6.3 y Art. 50. Ese
motor genera y etiqueta los datos de la fase 1, sirve de verificador en las recompensas y es la
herramienta `classify_risk` de la fase 2.

## Lo que hemos hecho

| Fase | Fechas | Qué hicimos | Resultado principal |
|---|---|---|---|
| 0 · Propuesta | 26-10 | Tema, usuario objetivo, diez preguntas reales | `docs/propuesta.md` aprobada |
| 1 · Razonamiento | 26-10 a 13-11 | Motor de reglas → dataset sintético → destilación con Qwen3-8B → SFT → GRPO sobre Qwen3-0.6B | Etiqueta correcta: 26,5 % (base) → 63 % (SFT) → 75,5 % (GRPO) |
| 2 · Herramientas | 03-11 | Cuatro herramientas y un bucle de tool use escrito a mano con Qwen3-4B | 89 % de respuestas correctas con herramientas frente a 11 % sin ellas |
| 3 · RAG | 03-11 a 04-11 | Corpus oficial, chunking, BM25 + denso + reranker, respuestas con citas y «No lo sé» calibrado | 89 % de respuestas correctas y 96 % fieles a las citas |
| Auditoría | 10-11 a 13-11 | Revisión automática de los datos; el test de la fase 1 se rehízo porque compartía frases con train | Nuevo `test.jsonl`; las cifras de la fase 1 de arriba ya son las del test nuevo |

## De dónde salen los datos

| Datos | Origen | Fichero |
|---|---|---|
| Casos de la fase 1 (train/test/OOD) | **Sintéticos**: `rlm/generate_aiact.py` compone descripciones de sistemas y el motor de reglas les pone la etiqueta. El OOD es un dominio (educación) que no aparece en train | `rlm/data/train.jsonl` (1000), `test.jsonl` (200), `test_ood.jsonl` (100) |
| Trazas de razonamiento para el SFT | Generadas por un **profesor** (Qwen3-8B en modo thinking) con `rlm/distill.py`; solo se quedan las que el verificador da por buenas | `rlm/data/sft_traces*.jsonl` |
| Benchmark externo | *AI Act Evaluation Benchmark* (Davvetas et al., 2026, arXiv:2603.09435, CC BY 4.0), 339 escenarios. Solo sirve para auditar el motor de reglas, no para entrenar | `rlm/data/external/aiact_benchmark_scenarios.json` |
| Texto del AI Act para las herramientas | API CELLAR de la Oficina de Publicaciones de la UE, en tiempo de ejecución | lo descarga `tool_use/ai_act_text.py` |
| Banco de la fase 2 | 34 casos escritos por nosotras; la etiqueta esperada la calcula el motor | `tool_use/bench/cases.jsonl` |
| Corpus de la fase 3 | Fuentes oficiales de la UE: reglamento (español), dos guías de la Comisión (traducción al español) y tres borradores de alto riesgo (inglés). **No está en git**: se regenera con `rag/download_corpus.py` | `rag/corpus/`; descrito en [`rag/CORPUS.md`](../rag/CORPUS.md) |
| Conjunto dorado de la fase 3 | 71 preguntas escritas a mano con la evidencia exacta en el corpus (57 con respuesta y 14 sin ella). Aparte hay otras 36 para calibrar el umbral de «No lo sé» | `rag/gold/gold.jsonl`, `rag/gold/threshold_val.jsonl` |
| Pesos | Salen de nuestro entrenamiento. **No están en git**; solo existen en la DGX | `rlm/weights/` |

Modelos base (de Hugging Face, se descargan en `hf_cache/`): Qwen3-0.6B (fase 1), Qwen3-4B
(fases 2 y 3), Qwen3-8B (profesor), Qwen3-Embedding-0.6B y Qwen3-Reranker (fase 3).

## Estructura

```
dgm-arca/
├── rlm/        fase 1 · razonamiento
├── tool_use/   fase 2 · herramientas
├── rag/        fase 3 · conocimiento
├── agent/      fase 4 · agente (no se cubre aquí)
├── api/        API FastAPI que sirve las cuatro fases
├── smoke/      comprobaciones del entorno en la DGX
├── tests/      tests sin GPU ni red
├── reports/    métricas y predicciones de cada evaluación (JSON)
└── docs/       enunciado, propuesta, cuaderno de experimentos y guías
```

### Raíz

| Fichero | Qué es |
|---|---|
| `README.md` | Presentación del proyecto, métricas y cómo levantarlo |
| `pyproject.toml`, `uv.lock` | Dependencias (con `uv`) |
| `Makefile` | Atajos: `make test`, `make audit`, `make lint`, `make api`, `make smoke`... |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Imagen con GPU; el repo se monta en `/app` |
| `.env.example` | Configuración de las fases (modelos, ruta del adaptador...). Se copia a `.env` |
| `practica1_arca-3.pdf` | Copia idéntica de `docs/enunciado.pdf` |
| `LICENSE` | Licencia |

### `rlm/`: fase 1, razonamiento

| Fichero | Qué hace |
|---|---|
| `aiact_rules.py` | **Motor de reglas**: rasgos del sistema → nivel de riesgo + base legal |
| `generate_aiact.py` | Genera el dataset de casos del AI Act a partir del motor |
| `generate_problems.py` | Ejemplo de la plantilla (problemas aritméticos); no se usa en nuestro dominio |
| `data.py` | Utilidades para cargar los JSONL |
| `verifier.py` | Interfaz de verificadores (comprueba si una respuesta es correcta) |
| `rewards.py` | Recompensas genéricas de la plantilla: formato y acierto |
| `aiact_rewards.py` | Recompensas del AI Act: formato, etiqueta y cita del artículo |
| `distill.py` | Destilación: el profesor genera trazas y el verificador las filtra |
| `train_sft.py` | SFT con LoRA sobre las trazas verificadas |
| `train_grpo.py` | GRPO con TRL sobre el modelo del SFT |
| `grpo_step.py` | Un paso de GRPO escrito a mano, para entender lo que hace TRL |
| `evaluate.py` | Mide el pass@1 de base, SFT y GRPO en test y OOD |
| `plots.py` | Curvas de entrenamiento y gráfica comparativa |
| `inference.py` | Carga el base + adaptador LoRA; es lo que hay detrás de `POST /reasoning` |
| `gpu.py` | Ajustes de GPU compartidos por los scripts de entrenamiento |
| `compare_benchmark.py` | Compara el motor de reglas con el benchmark externo |
| `audit_benchmark.py` | Auditoría manual de 40 escenarios del benchmark |
| `audit_data.py` | Auditoría de calidad de train/test/OOD (duplicados, fugas, balance) |
| `data/` | Datasets (ver la tabla siguiente) |
| `weights/` | Adaptadores LoRA `sft_lora/` y `final_rlm_lora/` (fuera de git) |

En `rlm/data/` están `train`, `test` y `test_ood` (los finales) y los `*_v1` (el test antiguo
que tenía fugas). `sft_traces.jsonl` son las trazas de la destilación completa,
`sft_traces_resample.jsonl` el remuestreo de los problemas que se quedaron sin traza y
`sft_traces_merged.jsonl` la unión de ambos, que es lo que entra al SFT. `sft_traces_pilot*` son
los pilotos de la destilación (los `_aborted` se pararon a medias) y `*.stats.json` sus
estadísticas. `benchmark_comparison.*` y `audit/benchmark_audit.csv` son el resultado de comparar
el motor con el benchmark.

### `tool_use/`: fase 2, herramientas

| Fichero | Qué hace |
|---|---|
| `tools.py` | Las herramientas: `lookup_ai_act` (texto de un artículo), `classify_risk` (motor de reglas), `compute_max_fine` (multa máxima), `write_assessment_report` (PDF) y `calculator` |
| `ai_act_text.py` | Descarga el AI Act de CELLAR y lo trocea en artículos, apartados y anexos citables |
| `registry.py` | Genera el esquema de cada herramienta a partir de los type hints y valida las llamadas |
| `parser.py` | Detecta las llamadas a herramientas en la salida del modelo |
| `executor.py` | El bucle de tool use hecho a mano: llamadas encadenadas y en paralelo |
| `evaluate.py` | Mide la selección de herramienta, los argumentos, la etiqueta y la respuesta final |
| `bench/build_cases.py`, `bench/cases.jsonl` | Construye y contiene el banco de 34 casos |

### `rag/`: fase 3, conocimiento

| Fichero | Qué hace |
|---|---|
| `download_corpus.py` | Descarga el corpus oficial y lo guarda como Markdown en `rag/corpus/` |
| `ingest.py` | Paso 1: lee los documentos del corpus |
| `chunking.py` | Paso 2: trocea los documentos (por estructura legal y semántico) |
| `embed.py` | Paso 3: embeddings con Qwen3-Embedding-0.6B |
| `retriever.py` | Búsqueda densa, BM25 e híbrida (λ = 0,3) |
| `generate.py` | Reranker, prompt con los fragmentos, respuesta con citas y umbral de «No lo sé» |
| `evaluate.py` | Métricas de recuperación: Recall@k y MRR en el conjunto dorado |
| `evaluate_answers.py` | Métricas de respuesta: correcta, citada, fiel y rechazos |
| `judge.py` | Juez LLM de corrección y fidelidad |
| `calibrate.py` | Calibra los umbrales de «No lo sé» con `threshold_val.jsonl` |
| `stats.py` | Intervalos de confianza (Wilson) de las métricas |
| `audit_gold.py` | Comprueba que el conjunto dorado es coherente con el corpus |
| `CORPUS.md` | Qué documentos hay, su fuente, su idioma y su licencia |
| `gold/` | Conjunto dorado y conjunto de calibración |

### `api/`: la API

| Fichero | Qué hace |
|---|---|
| `app.py` | FastAPI con `/health`, `/reasoning`, `/tools`, `/rag` y `/agent` |
| `backends.py` | Conecta cada endpoint con el código de su fase |
| `schemas.py` | Formato de peticiones y respuestas |

### `smoke/`, `tests/` y `reports/`

- `smoke/`: `check_gpu.py` diagnostica la GPU, `smoke_grpo.py` lanza un GRPO diminuto para
  probar el entorno y `dgx_env.sh` deja preparadas las variables de la DGX.
- `tests/`: un fichero por módulo (`test_aiact_rules.py`, `test_rag_pipeline.py`...). Se lanzan
  con `uv run pytest`.
- `reports/`: un JSON por evaluación. Por prefijo:
  - `eval_test*`: fase 1. `_grpo` es el modelo GRPO, `_ood` el dominio nuevo y `_v1` el test antiguo.
  - `history_*`: curvas de entrenamiento.
  - `phase1_data_audit`: auditoría de los datos de la fase 1.
  - `phase2_eval*`: fase 2 y sus variantes (sin herramientas, 1.7B, thinking...).
  - `phase3_*`: recuperación, respuestas, calibración, intervalos de confianza y auditoría del dorado.

### `docs/`

| Fichero | Qué es |
|---|---|
| `enunciado.pdf` / `.tex` | Enunciado de la práctica (del profesor) |
| `rubrica.md` | Rúbrica de evaluación (del profesor) |
| `propuesta.md` | **Nuestra** propuesta de la fase 0 |
| `00_propuesta.md` | Plantilla vacía de la propuesta (del profesor) |
| `EXPERIMENTS.md` | **Nuestro** cuaderno de experimentos, con fecha, de todas las fases |
| `ARQUITECTURA.md` | Mapa de piezas y ficheros |
| `RESUMEN.md` | Este documento |
| `datasets.md`, `temas_ejemplo.md`, `dgx.md`, `github.md` | Guías del profesor: cómo conseguir datos, ejemplos de temas, uso de la DGX y de GitHub |
| `*_plantilla.md` | Plantillas originales del README, del cuaderno y del informe |
