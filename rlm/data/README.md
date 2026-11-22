# Datos de la fase 1

Todos los ficheros son JSONL, una línea por problema o por traza. Las etiquetas salen del motor de
reglas (`rlm/aiact_rules.py`), nunca se escriben a mano. El proceso completo, con las cifras de
cada paso, está en [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md).

## Problemas

| Fichero | Filas | Qué es |
|---|---|---|
| `train.jsonl` | 1000 | Entrenamiento (SFT y GRPO) |
| `test.jsonl` | 200 | Test actual, sin ninguna frase compartida con train (13-nov) |
| `test_ood.jsonl` | 100 | Fuera de distribución: dominio educativo, que no aparece en train |
| `test_v1.jsonl`, `test_ood_v1.jsonl` | 200, 100 | Versiones antiguas. El test compartía frases con train en el 96 % de los casos |
| `generation_stats.json` | | Semilla y recuento por etiqueta y familia de cada split |

Se regeneran con `uv run python -m rlm.generate_aiact`. `rlm/audit_data.py` (`make audit`)
comprueba que train sale byte a byte igual con la semilla guardada.

## Trazas del profesor (destilación)

Cada línea es una traza de Qwen3-8B con el campo `verified`: solo las verificadas (formato,
etiqueta y cita correctos) se usan para el SFT.

| Fichero | Filas | Verificadas | Qué es |
|---|---|---|---|
| `sft_traces.jsonl` | 4000 | 1758 | Destilación completa: 1000 problemas × 4 muestras |
| `sft_traces_resample.jsonl` | 632 | 67 | Segundo pase: los 79 problemas sin traza válida × 8 muestras |
| **`sft_traces_merged.jsonl`** | 4632 | **1825** | Unión de las dos. **Es la que usó el SFT** |

Los `.stats.json` de cada fichero dan la tasa de aceptación por etiqueta y por familia.

## Pilotos de destilación (se conservan a propósito)

Antes de la destilación completa se probaron varios profesores y versiones de la ficha legal del
prompt. Estos ficheros **no se usan para entrenar**. Se guardan para poder consultar cómo se
llegó a la configuración final: cada uno corresponde a una fila de las tablas de las entradas del
27-oct en `docs/EXPERIMENTS.md`.

| Fichero | Profesor y ficha | Problemas | Para qué sirvió |
|---|---|---|---|
| `sft_traces_pilot_v0.jsonl` | Qwen3-4B, sin ficha | 0-15 | Línea base: el profesor inventa la numeración de los artículos (3 % de citas correctas) |
| `sft_traces_pilot_v1.jsonl` | Qwen3-4B, ficha v1 | 0-39 | La ficha sube las citas correctas al 59 % |
| `sft_traces_pilot_v2a_aborted.jsonl`, `_v2b_aborted` | Qwen3-4B, ficha v2 | 0-15 | Parados: el 4B aplica de más cada regla que se le enfatiza |
| `sft_traces_pilot_8b16.jsonl` | Qwen3-8B, ficha v2 | 0-15 | Primera prueba del profesor 8B |
| `sft_traces_pilot_v2c_8b_aborted.jsonl`, `_v3a_8b_aborted` | Qwen3-8B, fichas intermedias | 0-39 y 0-7 | Parados a medias mientras se ajustaba la ficha |
| `sft_traces_pilot.jsonl` | Qwen3-8B, ficha v3 | 0-79 | 86 % en los problemas usados para ajustar el prompt, 72 % en los no vistos |
| `sft_traces_pilot_v4_heldout.jsonl` | Qwen3-8B, ficha v4 (final) | 80-119 | Problemas no vistos: 74 %. Es la configuración de la destilación completa |

## Contraste con un benchmark externo

| Fichero | Qué es |
|---|---|
| `external/aiact_benchmark_scenarios.json` | 339 escenarios del AI Act Evaluation Benchmark (Davvetas et al., 2026, CC-BY-4.0) |
| `benchmark_comparison.jsonl` y `.stats.json` | El motor de reglas frente a ese benchmark (`rlm/compare_benchmark.py`) |
| `audit/benchmark_audit.csv` | Muestra de 40 escenarios para revisar a mano el motor (`rlm/audit_benchmark.py`) |
