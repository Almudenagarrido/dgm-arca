# EXPERIMENTS.md — cuaderno de experimentos

Copiad esta plantilla a la raíz de vuestro repositorio como `EXPERIMENTS.md` y añadid una
entrada cada vez que lancéis algo o toméis una decisión. Con fecha. No lo escribáis al final
de memoria: dentro de un mes no recordaréis por qué bajasteis la tasa de aprendizaje.

Una entrada honesta que diga "probamos X, no funcionó, creemos que por Y" vale más que diez
que digan "todo bien". Esto es lo primero que leo cuando corrijo interpretación.

---

## Formato de cada entrada

```
### AAAA-MM-DD · Fase N · Título corto

**Qué queríamos saber.** La pregunta o la hipótesis.
**Qué hicimos.** Comando exacto, modelo, datos, hiperparámetros que cambian respecto a la
entrada anterior.
**Qué pasó.** Números, curva (enlace a reports/), ejemplos.
**Qué concluimos.** Y qué hacemos a continuación.
```

---

## Ejemplo

### 2026-10-02 · Fase 1 · Primer GRPO sobre nuestro dataset

**Qué queríamos saber.** Si el modelo aprende el formato con nuestro prompt de sistema antes
de preocuparnos por la exactitud.

**Qué hicimos.** `uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --steps 100
--num-generations 8 --max-completion-length 512` partiendo del adaptador de SFT. Sin KL.

**Qué pasó.** La recompensa de formato pasa de 0.31 a 0.97 en 40 pasos. La de exactitud sube
de 0.18 a 0.26 y se estanca. La longitud media baja de 480 a 210 tokens: el modelo aprende a
cerrar la etiqueta antes de quedarse sin presupuesto. Curva en `reports/grpo_run01.png`.

**Qué concluimos.** El formato está resuelto. La exactitud se estanca porque el 40 % de los
problemas del dataset tienen respuestas con unidades y el verificador numérico las ignora:
"120 litros" y "120" cuentan igual, pero "0.12 m³" cuenta como fallo. Siguiente paso:
normalizar unidades en el verificador y repetir.

---

## Entradas

(vuestras entradas, de la más antigua a la más reciente)

## 2026-10-26 — Entorno y punto de partida

**Entorno.** `uv run arca-check-gpu`: porción MIG 1g.18gb de una H200 (~16 GB útiles),
bf16 soportado, torch 2.11.0+cu128, transformers 5.17, trl 1.13, peft 0.20. El smoke test
(`uv run arca-smoke`) ocupa ~8,5 GB, así que tenemos ~7 GB de margen antes de tener que
bajar `--num-generations` o `--max-completion-length`.

**Fallo detectado en la recompensa de exactitud dada.** `accuracy_reward` (rlm/rewards.py)
compara con `numbers_match`, pensada para respuestas numéricas. Con nuestras etiquetas de
texto, `normalize_number` devuelve `None` para la predicción y para la referencia, y como
`None == None` es cierto, cualquier etiqueta (incluso una inventada) recibe recompensa 1.0.
Usarla tal cual en GRPO haría que el modelo solo aprendiese el formato (reward hacking).
Decisión: no modificar rewards.py; implementar un `AIActVerifier` en rlm/verifier.py y una
recompensa de exactitud propia basada en él.

### 2026-10-26 · Fase 1 · Motor de reglas del AI Act

**Qué queríamos saber.** Si podíamos traducir el Reglamento (UE) 2024/1689 a reglas
deterministas que sirvan a la vez de etiquetador del dataset, de verdad del verificador y, en
la fase 2, de herramienta `classify_risk`.

**Qué hicimos.** `rlm/aiact_rules.py`: un `SystemProfile` (~40 flags) entra en `classify()`
y sale la etiqueta, el artículo decisivo, todas las reglas evaluadas (incluidas las
excepciones) y las obligaciones. Cubre Art. 5.1.a-h con sus excepciones, Art. 6.1 + Anexo I
(secciones A y B, solo con evaluación por terceros), los 25 puntos del Anexo III con sus
exclusiones internas (verificación biométrica, fraude en 5.b, documentos de viaje en 7.d,
logística de campañas en 8.b), Art. 6.3.a-d anulado por perfilado y Art. 50.1-4. Precedencia:
prohibido > alto_riesgo > riesgo_limitado > riesgo_minimo; un sistema con varias funciones
toma la más grave (`classify_components`). Utilidades para lo que viene: `flags_hash()` para
deduplicar por flags y no por texto, `normalize_article()` / `article_match_score()` para la
recompensa de dominio (1 cita exacta, 0.5 mismo artículo o punto, 0 otra) y `explain()` para
trazas e informes. Tests en `tests/test_aiact_rules.py` (60), con los diez ejemplos de la
propuesta como casos.

Buscamos si existía algo parecido: AI Act Companion (Python), RiskLine (Go), ai-act-check,
el toolkit de Microsoft y el árbol de decisión de TU Delft. Ninguno expone flags con los que
generar un dataset, así que el motor es propio; los usaremos como contraste en el informe.

**Qué pasó.** 60/60 tests en verde y `ruff` limpio. Decisiones legales tomadas:

- Texto congelado en la versión del DOUE de 12-07-2024, sin el Digital Omnibus.
- Fuera del modelo: GPAI (Arts. 51-56), software libre (Art. 2.12), ámbito territorial y la
  cláusula "en la medida en que esté permitido" del Anexo III.
- Biometría policial en tiempo real con objetivo permitido **y** autorización se etiqueta
  `alto_riesgo` (Anexo III.1.a), no `prohibido`.
- `riesgo_minimo` sin excepción aplicada no tiene artículo (`None`); con Art. 6.3 cita
  `Art. 6.3.x`.
- `fuera_de_ambito` existe solo si se marca explícitamente; el dataset no lo usará.

**Qué concluimos.** El motor está listo para etiquetar, pero las reglas se escribieron sin
el PDF del Reglamento en el repositorio: hay que revisarlas contra el texto oficial (sobre
todo 5.1.h, 6.3 y 50). Siguiente paso: el generador de problemas con clases equilibradas por
familia, varias plantillas y una familia reservada para OOD.
