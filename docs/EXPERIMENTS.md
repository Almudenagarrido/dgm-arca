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
