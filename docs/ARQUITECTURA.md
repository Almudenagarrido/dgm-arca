# Arquitectura del repositorio

Mapa de cómo se comunican los ficheros de ARCA aplicado a nuestro tema: **clasificación de
sistemas de IA según el AI Act (Reglamento (UE) 2024/1689)** en `prohibido`, `alto_riesgo`,
`riesgo_limitado` o `riesgo_minimo`, con cita del artículo o anexo.

Convenciones:

- `──►` dependencia que ya existe en el código (import real).
- `- - ->` conexión que tenemos que crear nosotras.
- `★` fichero o función nueva de nuestro dominio.
- **dado**: viene hecho en la plantilla. **nuestro**: lo completamos nosotras.

---

## 1. Visión general

```
                    ┌──────────────────────────────┐
                    │  ★ rlm/aiact_rules.py         │  motor de reglas:
                    │  flags → etiqueta + artículo  │  Art. 5, Art. 6.1, Anexo III,
                    └──────────────────────────────┘  Art. 6.3, Art. 50
                       │            │            │
        genera datos   │   verdad   │   herram.  │
                       ▼            ▼            ▼
                 FASE 1 (rlm)   verificador   FASE 2 (tool_use: classify_risk)
                       │                         │
                       ▼                         ▼
                 adaptador GRPO ─────────► FASE 4 (agent) ◄──── FASE 3 (rag: retriever)
                                                 │
                                                 ▼
                                             api/ (HTTP)
```

El motor de reglas es la pieza que conecta el dominio con todo lo demás: genera y etiqueta el
dataset de la fase 1, es la verdad del verificador y es la herramienta de cálculo de la fase 2.
La fase 4 no añade lógica de dominio nueva: reutiliza las piezas de las fases 1, 2 y 3.

---

## 2. Fase 1 offline: del dataset al adaptador

```
                         ★ rlm/aiact_rules.py  (motor de reglas: flags → etiqueta + artículo)
                          │            │                      │
                          ▼            ▼                      ▼ (fase 2)
          rlm/generate_problems.py   rlm/verifier.py       tool_use/tools.py
          (muestrea flags, plantillas) │  ▲ usa extract_answer
                          │            │  └──────── rlm/rewards.py ◄──── smoke/smoke_grpo.py
                          ▼            │            (format_reward,        (usa rewards + data,
          rlm/data/train|test|ood.jsonl│             extract_answer)        no se toca)
                          │            │                 ▲
                          ▼            │                 │
                     rlm/data.py ◄─────┼─────────────────┼──────────────┐
          (build_prompt, load_domain_dataset)            │              │
           │          │           │                      │              │
           ▼          ▼           ▼                      │              │
  rlm/distill.py  rlm/train_sft.py  rlm/train_grpo.py ───┘        rlm/evaluate.py
  (profesor +     (LoRA sobre       (format + accuracy★ +          (pass@1 base/SFT/GRPO,
   verifier)       trazas)           domain_reward★)                curvas, confusión)
      │              ▲   │              ▲     │                         ▲
      ▼              │   ▼              │     ▼                         │
  data/sft_traces ───┘  weights/sft ────┘  weights/grpo ────────────────┘
                                              │
  rlm/grpo_step.py  (aislado: solo lo usa     │  ARCA_RLM_ADAPTER
  tests/test_grpo_step.py; no entrena)        ▼
                                        rlm/inference.py  → /reasoning
```

Importante: `rlm/train_grpo.py` importa hoy la `accuracy_reward` de `rlm/rewards.py`, que con
etiquetas de texto da 1.0 a cualquier respuesta (ver `docs/EXPERIMENTS.md`). La sustituimos por
una recompensa de exactitud basada en `AIActVerifier`.

| Fichero | Estado | Qué ponemos |
|---|---|---|
| `rlm/aiact_rules.py` ★ | nuestro | Motor de reglas con precedencia prohibido > alto_riesgo > riesgo_limitado > riesgo_minimo. |
| `rlm/generate_problems.py` | ejemplo → nuestro | Muestrea flags, etiqueta con el motor, varias plantillas + paráfrasis, deduplica por hash de flags, separa train / test / OOD. |
| `rlm/verifier.py` | dado + nuestro | `AIActVerifier`: normaliza la etiqueta; rechaza vacía, inventada o con varias etiquetas. |
| `rlm/rewards.py` | dado | `format_reward` y `extract_answer` (no usar su `accuracy_reward`). |
| `rlm/data.py` | dado | Prompt `<think>/<answer>` y carga de JSONL. |
| `rlm/distill.py` | nuestro | Trazas del profesor filtradas por el verificador; tasa de aceptación por clase. |
| `rlm/train_sft.py` | esqueleto | SFT con LoRA sobre `sft_traces.jsonl`. |
| `rlm/train_grpo.py` | esqueleto | Formato + exactitud (`AIActVerifier`) + `domain_reward` (cita del artículo correcto). |
| `rlm/grpo_step.py` | nuestro | Paso de GRPO a mano: ventajas por grupo, ratio, clip, KL. |
| `rlm/evaluate.py` | nuestro | pass@1 base / SFT / GRPO en test y OOD, matriz de confusión, curvas, 5 fallos. |
| `rlm/inference.py` | dado | Registrar `AIActVerifier` en `VERIFIERS`; apuntar `ARCA_RLM_ADAPTER`. |
| `rlm/data/` | nuestro | `train.jsonl`, `test.jsonl`, `test_ood.jsonl`, `sft_traces.jsonl` con `question`, `answer`, `article`, `flags`. |
| `rlm/weights/` | nuestro | Adaptadores LoRA (fuera de git). |

---

## 3. Fase 3 offline: indexado del corpus

```
rag/corpus/*.pdf ──► rag/ingest.py ──► rag/chunking.py ──► rag/embed.py ──► rag/retriever.py
                     (Document)        (Chunk: id         (Qwen3-Embedding)  (BM25 / denso / híbrido λ)
                                        <doc>-<idx>)                              │
                                                                                  ▼
rag/gold/*.jsonl  (50 preguntas → chunk_ids) ─────────────────────────────► rag/evaluate.py
                                                                            (Recall@k, MRR → elegir λ)
```

| Fichero | Estado | Qué ponemos |
|---|---|---|
| `rag/ingest.py` | dado | Lee el reglamento y las guías de la Comisión (PDF/HTML). |
| `rag/chunking.py` | dado + nuestro | Recursivo dado; añadimos el semántico (y probamos trocear por artículo). |
| `rag/embed.py` | dado | Qwen3-Embedding con instrucción adaptada al dominio legal. |
| `rag/retriever.py` | dado + nuestro | BM25 dado; denso e híbrido con λ. |
| `rag/evaluate.py` | dado + nuestro | Recall@k y MRR de los tres retrievers; λ elegido con datos. |
| `rag/generate.py` | dado + nuestro | Respuesta con citas y comprobación de que existen. |
| `rag/corpus/` | nuestro | Reglamento 2024/1689, guías de prácticas prohibidas, borrador de guías de alto riesgo. |
| `rag/gold/` | nuestro | 50 preguntas anotadas con el `chunk_id` que las responde. |

---

## 4. En ejecución: la API y cómo reutiliza cada fase

```
      corrector / ngrok / agent/ui.py (HTTP, httpx)
                     │
                     ▼
               api/app.py ──────────────► api/schemas.py  ◄── contratos que importan TODAS las fases
                     │                     (ReasoningResponse, ToolsResponse,
                     ▼                      RagResponse, AgentResponse…)
              api/backends.py   (si una fase lanza NotImplementedError → 501 / "pending" en /health)
     ┌───────────────┼──────────────────┬───────────────────────────┐
     ▼               ▼                  ▼                           ▼
 /reasoning       /tools              /rag                       /agent
 rlm/inference   tool_use/executor   rag/generate               agent/react_agent
 .ReasoningModel .build_tool_loop    .build_pipeline            .ReActAgent
   │               │                   │                          │
   ├ rlm/data      ├ tool_use/parser   └ rag/retriever ◄ - - - - -┤ search_knowledge_base
   ├ rlm/rewards   ├ tool_use/registry    (+ chunking,            ├ tool_use/tools.default_registry
   └ rlm/verifier  └ tool_use/tools          embed, ingest)       ├ tool_use/registry
     (+ ★AIAct        │                                           ├ tool_use/parser
      Verifier)       ├ search_eurlex★   → API HTTP EUR-Lex       ├ agent/prompts
                      ├ classify_risk★  - - -► ★aiact_rules       ├ agent/memory  - - - (política nuestra)
                      └ generate_report★ → escribe PDF            └ cerebro: base | weights/grpo | thinking
                                                                      (ARCA_RLM_BASE_MODEL /
                                                                       ARCA_RLM_ADAPTER /
                                                                       ARCA_THINKING_MODEL)
```

### API (`api/`, dado, no se toca)

| Fichero | Qué es |
|---|---|
| `api/app.py` | FastAPI con `/health`, `/reasoning`, `/tools`, `/rag`, `/agent`. |
| `api/schemas.py` | Contratos de entrada/salida. Se pueden añadir campos opcionales, nunca renombrar ni quitar. |
| `api/backends.py` | Conecta cada endpoint con su fase; traduce `NotImplementedError` en 501. |

### Fase 2 (`tool_use/`)

| Fichero | Estado | Qué ponemos |
|---|---|---|
| `tool_use/registry.py` | dado | Funciones tipadas → JSON Schema, validación de argumentos. |
| `tool_use/parser.py` | dado | Detecta `<tool_call>` en la salida del modelo. |
| `tool_use/tools.py` | ejemplo → nuestro | `search_eurlex` (consulta HTTP), `classify_risk` (motor de reglas), `generate_risk_report` (escribe PDF). |
| `tool_use/executor.py` | nuestro | Bucle de cinco pasos a mano, multi-step y llamadas paralelas. |
| `tool_use/evaluate.py` | dado | Precisión de selección y de argumentos. |
| `tool_use/bench/` | nuestro | 30 casos, incluidos casos sin herramienta. |

### Fase 4 (`agent/`)

| Fichero | Estado | Qué ponemos |
|---|---|---|
| `agent/prompts.py` | dado | Prompt ReAct. |
| `agent/memory.py` | dado + nuestro | Memoria con presupuesto de tokens; política de resumen nuestra. |
| `agent/react_agent.py` | nuestro | Bucle ReAct con `final_answer`, límite de pasos, timeouts, recuperación de errores, cerebro intercambiable. |
| `agent/evaluate.py` | dado | Éxito, pasos y tokens por cerebro. |
| `agent/ui.py` | dado | Visor de trazas contra la API. |
| `agent/bench/` | nuestro | 20 tareas de al menos dos herramientas (clasificar → citar → generar informe). |

---

## 5. Evaluadores y tests

```
tool_use/evaluate.py ──► executor.run_tool_loop   + tool_use/bench/*.jsonl  (30 casos)
agent/evaluate.py    ──► react_agent.ReActAgent    + agent/bench/*.jsonl     (20 tareas)
rag/evaluate.py      ──► retrievers               + rag/gold/*.jsonl        (50 preguntas)
rlm/evaluate.py      ──► modelos base/SFT/GRPO    + rlm/data/test(_ood).jsonl

tests/test_verifier.py       → rlm/verifier.py (+ ★ tests del motor de reglas)
tests/test_rewards.py        → rlm/rewards.py
tests/test_grpo_step.py      → rlm/grpo_step.py
tests/test_generate_problems → rlm/generate_problems.py
tests/test_tool_*.py         → tool_use/parser, registry
tests/test_rag.py            → rag/evaluate (métricas), rag/generate (citas)
tests/test_agent.py          → agent/memory, agent/evaluate
tests/test_api.py            → api/app (arranca con todo en 501)
```

---

## 6. Resto del repositorio

| Ruta | Qué es |
|---|---|
| `smoke/` | `check_gpu.py`, `smoke_grpo.py`, `dgx_env.sh`: comprobación del entorno y GRPO de prueba sobre GSM8K. No se toca; sirve de modelo para `train_grpo.py`. |
| `docs/propuesta.md` | Nuestra propuesta aprobada. |
| `docs/EXPERIMENTS.md` | Cuaderno de experimentos fechado (todas las fases). |
| `docs/datasets.md`, `docs/rubrica.md`, `docs/enunciado.pdf` | Guías y enunciado del curso. |
| `docs/informe_plantilla.md` | Plantilla del informe final (≤ 10 páginas). |
| `pyproject.toml`, `uv.lock`, `Makefile` | Entorno (`uv sync`) y atajos (`make test`, `make api`, `make smoke`). |
| `Dockerfile`, `docker-compose.yml` | `docker compose up api` debe levantar en una máquina limpia. |
| `.env.example` | Copiar a `.env` y rellenar `ARCA_TEAM`, `ARCA_DOMAIN`, `ARCA_RLM_ADAPTER`… |

---

## 7. Ideas clave

- **`api/schemas.py` es el centro**: todas las fases devuelven sus tipos.
- **`aiact_rules.py` es el nexo del dominio**: dataset y verificador en la fase 1, herramienta `classify_risk` en la fase 2, y ambas cosas en el agente.
- **El agente no añade lógica de dominio**: reutiliza `parser`, `registry` y `tools` (fase 2), el retriever (fase 3) y el adaptador (fase 1), seleccionado por variable de entorno.
- **`grpo_step.py` no se usa en ejecución**: existe para los tests y para la defensa oral.
