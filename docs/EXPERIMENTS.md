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

### 2026-10-26 · Fase 1 · Generador del dataset (estrategia 1 de datasets.md)

**Qué queríamos saber.** Si podíamos producir cientos de casos en lenguaje natural, con las
cuatro clases equilibradas y etiquetas correctas por construcción, sin anotar nada a mano.

**Qué hicimos.** `rlm/generate_aiact.py`, que hereda de `ProblemGenerator` (el patrón de
`generate_problems.py`, que dejamos intacto). Se elige la etiqueta, luego una *familia* (una
por rama legal: cada letra del Art. 5, casos del Anexo III, excepciones del Art. 6.3 con y sin
perfilado, cada deber del Art. 50, las exclusiones explícitas…) y la familia devuelve el
perfil y las frases que lo describen. `solve` pasa el perfil por el motor y aborta si no da la
etiqueta que espera la familia. El texto se envuelve en uno de 7 marcos (ficha de proyecto,
correo interno, ticket de compliance…), con 5 sujetos distintos y 0-2 frases de relleno
neutras, y termina con una instrucción fija que pide «etiqueta (artículo)».

`uv run python -m rlm.generate_aiact --out-dir rlm/data` (semilla 0).

**Qué pasó.**

- Primera versión sin cuotas: train 301/279/218/202 y OOD 38/37/15/10. Al agotarse las
  familias pequeñas (`vulnerability` tenía ~11 casos distintos, `social_scoring` 7), los
  duplicados se descartaban y se sobrerrepresentaban las clases con familias grandes.
  Solución: cuotas exactas por etiqueta y más escenarios y organizaciones en las familias
  pequeñas.
- Versión final: train 1000, test 200, OOD 100, exactamente 25 % por clase; 7 marcos usados en
  los tres; 38 artículos distintos en train.
- Fuga entre conjuntos: 0. Se deduplica por (familia, perfil, escenario, organización), no
  por el texto, y test/OOD se generan excluyendo las claves de train.
- Etiqueta en el enunciado: 0 tras cambiar «publica artículos» (de prensa) por «publica
  textos», que el control detectaba como falso positivo.
- OOD = dominio educativo (Anexo III.3, emociones en el aula, tutores con chat), ausente de
  train y test, con las cuatro clases.
- Estadísticas completas en `rlm/data/generation_stats.json`.

**Qué concluimos.** El dataset está listo para la destilación. Limitaciones a vigilar:

- Peso por rama desigual dentro de cada clase: Art. 50.1 (142) y riesgo mínimo sin artículo
  (93) dominan; algunos puntos del Anexo III salen 3-5 veces. Hay que reportar la exactitud
  por familia, no solo global.
- Deep fakes y textos de interés público también generan contenido sintético, así que su
  artículo principal es 50.2; `valid_articles` incluye 50.4 para no penalizarlo en la
  recompensa de citas.
- Algunos casos de «excepción 6.3 + perfilado» son poco naturales (p. ej. anonimizar escritos
  elaborando perfiles). Son la frontera que queremos enseñar, pero conviene revisarlos en la
  auditoría manual.
- Pendiente: auditar a mano ~50 casos de test y, opcionalmente, una pasada de paráfrasis con
  un LLM para más diversidad léxica.

### 2026-10-26 · Fase 1 · Verificador y recompensas del dominio

**Qué queríamos saber.** Cómo comprobar automáticamente una respuesta «etiqueta (artículo)»
sin dejar huecos para el *reward hacking*, y qué recompensa propia del dominio tiene sentido.

**Qué hicimos.**

- `AIActVerifier` (`rlm/verifier.py`): extrae la etiqueta del bloque `<answer>` (acepta
  variantes: «alto riesgo», «ALTO-RIESGO», «high-risk», «práctica prohibida»…) y solo la da
  por buena si hay **exactamente una**. Las etiquetas que aparezcan en `<think>` no cuentan.
  Registrado en `VERIFIERS` de `rlm/inference.py` como `aiact` (`ARCA_RLM_VERIFIER=aiact`).
- `rlm/aiact_rewards.py`: `aiact_accuracy_reward` sustituye a la `accuracy_reward` numérica
  en `train_grpo.py` cuando los datos no son GSM8K. `citation_reward` es la `domain_reward`:
  1 si la primera cita del `<answer>` está en `valid_articles`, 0,5 si coincide el artículo
  o el punto del Anexo III pero no el apartado o la letra, 0 en otro caso, y **0 siempre que
  la etiqueta sea incorrecta**. En riesgo mínimo sin artículo, no citar nada vale 1 e
  inventarse una cita vale 0.
- Pesos por defecto `--reward-weights 0.5,1.0,0.5` (formato, exactitud, cita).

**Qué pasó.** Tests en verde (variantes de etiqueta, respuestas con dos etiquetas, etiquetas
inventadas, citas en formato español e inglés). Con un «oráculo» que responde
`etiqueta (artículo)` del motor, los tres conjuntos dan exactitud = cita = verificador = 1.0,
así que datos, verificador y recompensas son coherentes entre sí. Un test deja documentado el
fallo de la recompensa dada: `accuracy_reward` paga 1.0 por `etiqueta_inventada` y la nuestra 0.

**Qué concluimos.** Decisiones contra el *reward hacking* que hay que defender en la oral:

- Varias etiquetas = 0, para que no compense cubrirse («prohibido o alto_riesgo»).
- La cita solo cuenta con la etiqueta correcta, para que no se optimice por separado.
- Solo cuenta la primera cita, para que enumerar artículos no garantice acertar.
- Citar algo cuando no hay nada que citar vale 0: un fundamento legal inventado es el peor
  error para un responsable de compliance.

Los pesos 0,5/1/0,5 son una hipótesis: la exactitud domina, el formato es un requisito previo
y la cita, como mucho, iguala la mitad de la exactitud. Probaremos variantes en GRPO.
Siguiente paso: destilación (`rlm/distill.py`) con un profesor Qwen3 en modo thinking,
reportando la tasa de aceptación por familia.

### 2026-10-27 · Fase 1 · Destilación: piloto con Qwen3-4B y ficha legal en el prompt

**Qué queríamos saber.** Qué fracción de las trazas de un profesor Qwen3-4B en modo thinking
pasa el verificador completo (formato + etiqueta + cita exacta), y por tanto cuántos problemas
hay que destilar para tener un conjunto de SFT útil.

**Qué hicimos.**

- `rlm/distill.py`: muestreo con Qwen3-4B (`temperature=0.6`, `top_p=0.95`, `top_k=20`,
  4 muestras por problema, 2048 tokens máx.), canonicalización de respuestas que cierran
  `</think>` y contestan en prosa sin `<answer>`, verificación con `AIActVerifier` +
  `citation_score`, escritura incremental con `--resume` y como máximo 2 trazas verificadas
  por problema (`--max-per-problem`) para que los problemas fáciles no dominen el SFT.
  Estadísticas por etiqueta y familia en `<output>.stats.json`.
- **Piloto v0** (16 problemas × 4, solo el prompt R1-Zero): ver resultados abajo.
- **Cambio**: `reference_card()` en `rlm/aiact_rules.py` genera, a partir de las mismas tablas
  del motor, una ficha con todos los puntos citables (Art. 5.1.a-h, Art. 6.1, Anexo III,
  Art. 6.3.a-d, Art. 50.1-4), la precedencia entre etiquetas y el formato exacto de la cita.
  Se añade al prompt R1-Zero como `AIACT_SYSTEM_PROMPT` (`rlm/data.py`) y se usa **en todas las
  etapas**: destilación, SFT, GRPO (`load_domain_dataset`), inferencia con
  `ARCA_RLM_VERIFIER=aiact` y evaluación. Un test comprueba que la ficha contiene todas las
  `valid_articles` de train, test y OOD.
- **Piloto v1** (40 primeros problemas de train × 4, con la ficha). Unos 80 min en la H200
  (MIG) con `transformers.generate` y lotes de 8 problemas.

**Qué pasó.**

| | v0 (sin ficha, 16 problemas) | v1 (con ficha, 40 problemas) |
|---|---|---|
| Formato válido | 100 % | 97,5 % |
| Etiqueta correcta | 57,8 % | 80,6 % |
| Etiqueta + cita exacta | 3,1 % | 59,4 % |
| Problemas con ≥1 traza verificada | 1/16 | 32/40 |
| Trazas para SFT (tras el tope de 2) | 2 | 59 |
| Tokens medios / truncadas | 702 / 0 % | 853 / 2,5 % |

En los mismos 16 primeros problemas, v1 verifica 28/64 frente a 2/64 en v0. En v0 el profesor
inventaba la numeración («Anexo III.12», «Art. 5(1)», «Anexo III, punto 14») y fallaba todas
las prohibiciones (0/8); con la ficha acierta el 94 % de las prohibiciones con cita exacta.

Por etiqueta en v1 (etiqueta correcta / etiqueta + cita): prohibido 0,94 / 0,92; alto_riesgo
0,72 / 0,53; riesgo_limitado 0,66 / 0,61; riesgo_minimo 0,90 / 0,38.

Errores que quedan:

- **Excepción del Art. 6.3** (familia `derogation`: etiqueta 0,89 pero cita 0,25). El profesor
  acierta `riesgo_minimo` pero no cita nada (12 casos) o confunde la letra, sobre todo con la
  6.3.b (9 casos). `derogation_profiling`: 0/1 problemas con traza válida.
- **riesgo_limitado → riesgo_minimo** (9 trazas): no ve el deber de transparencia del Art. 50
  (contenido sintético, textos de interés público, deep fakes).
- **Anexo III.5.b vs 5.c** (solvencia vs seguros) y 4 trazas de alto riesgo sin etiqueta
  legible (truncadas).

**Qué concluimos.**

- La dificultad del profesor no era el razonamiento sino **recordar la numeración del
  reglamento**. Dar la ficha en el prompt es legítimo para el caso de uso (un responsable de
  compliance trabaja con el reglamento delante) y convierte la tarea en *aplicar* la norma.
  Como se usa igual en todas las etapas, no hay desajuste entre entrenamiento y evaluación.
  Limitación a declarar: la ficha mapea artículo → etiqueta, así que parte del mérito es
  elegir bien la regla, no conocerla.
- Con el 37 % de las trazas y el 80 % de los problemas aprovechables, los 1000 problemas de
  train darían ~800 problemas con traza y ~1450 trazas para SFT, de sobra para el arranque en
  frío. Las familias débiles (6.3, Art. 50.2/50.4) estarán infrarrepresentadas: hay que
  vigilarlo en la evaluación por familia y, si hace falta, sobremuestrearlas.
- **Coste**: a ~2 min por problema, la ejecución completa con `transformers` tardaría ~33 h.
  Antes de lanzarla conviene usar vLLM o lotes más grandes, o lanzar por tramos con `--resume`.

Siguiente paso: reforzar en la ficha que el riesgo mínimo por la excepción del 6.3 **debe**
citar la letra, repetir un piloto corto sobre las familias `derogation*` y lanzar la
destilación completa.

### 2026-10-27 · Fase 1 · Destilación: profesor Qwen3-8B y ficha legal v3/v4

**Qué queríamos saber.** Si el piloto anterior (4B + ficha v1: 59 % etiqueta + cita) se podía
llevar a un rendimiento útil para la destilación completa, y si las mejoras generalizan a
problemas que no se usaron para ajustar el prompt.

**Qué hicimos.** Iteramos sobre la ficha (`reference_card()`) y el profesor, siempre midiendo
sobre los mismos problemas y guardando cada piloto en `rlm/data/sft_traces_pilot_*.jsonl`:

- **v2 (4B)**: añadimos un procedimiento en 5 pasos que sigue el orden de `classify` y aclaraciones
  legales para cada error del piloto v1 (cita obligatoria del 6.3, Art. 50 en chatbots y texto
  generado, 5.1.h sin autorización, emociones fuera del trabajo = Anexo III.1.c, verificación
  1:1 excluida). El 4B **sobreaplica cada regla que se enfatiza**: con «revisa siempre el Art. 50»
  ve un 50.1 en el triaje de urgencias; con «la cita del 6.3 es obligatoria» cita el 6.3 en un
  videojuego. Lo paramos tras 16 problemas.
- **Profesor Qwen3-8B** (misma ficha v2). Cabe en la partición MIG de 71 GB con lotes de 8
  problemas y `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (con 16 se queda sin memoria).
- **v3 (8B)**: el 8B razonaba bien el 6.3 pero contestaba `riesgo_minimo` sin cita, porque el
  enunciado dice «si no hay ningún artículo que lo fundamente, escribe solo riesgo_minimo». La
  ficha separa ahora los dos casos de riesgo mínimo (con 6.3 → se cita la letra; fuera del
  Anexo III → sin cita) y dice que la excepción nunca da riesgo limitado. Quitamos los
  marcadores genéricos (`Art. 6.3.x`, `Art. 50.N`) porque el modelo llegó a copiar la «x».
- **v4 (8B)**: tras mirar los problemas 40-47, dos aclaraciones más: las exclusiones propias del
  Anexo III (verificación 1:1, fraude, documentos de viaje, logística de campañas) son riesgo
  mínimo sin cita, no 6.3; y la identificación biométrica policial para delitos fuera de los tres
  objetivos del 5.1.h está prohibida. Como 40-47 ya estaban vistos, la v4 se evalúa en problemas
  nuevos (80-119) con la nueva opción `--offset` de `distill.py`.

**Qué pasó.** Etiqueta + cita exacta, por traza (4 muestras por problema):

| Piloto | Profesor | Ficha | Problemas | Etiqueta | Etiqueta + cita | Problemas con traza válida |
|---|---|---|---|---|---|---|
| v0 | 4B | — | 0-15 | 58 % | 3 % | 1/16 |
| v1 | 4B | v1 | 0-39 | 81 % | 59 % | 32/40 |
| v3 | 8B | v3 | 0-39 (ajuste) | 91 % | **86 %** | 39/40 |
| v3 | 8B | v3 | 40-79 (no vistos) | 89 % | 72 % | 37/40 |
| v4 | 8B | v4 | 80-119 (no vistos) | 83 % | **74 %** | 36/40 |

En los 8 primeros problemas (los más difíciles: 6.3 y triaje) la progresión fue 15/32 (4B v1) →
25/32 (8B v2) → 30/32 (8B v3). El 8B además razona más corto (~760-870 tokens frente a ~1000).
El 55-83 % de sus respuestas necesitan la canonicalización de `distill.py` (cierra `</think>` y
contesta en prosa sin `<answer>`).

Qué arregló la v4 y qué no, en problemas no vistos:

- **5.1.h con objetivo no permitido: arreglado.** `police_rbi_prohibited` 17/20 en v4, frente a
  0/4 en v3 (40-79).
- **6.3 citado donde no aplica: no arreglado.** 22 trazas en v4 (80-119) frente a 23 en v3
  (40-79). Afecta a `exception` (7/12), `plain` (2/4) y a `derogation_chatbot` (4/8), donde
  cita el 6.3 en vez del Art. 50.1.
- **`derogation_profiling`: la familia más difícil** (8/28). El texto dice claramente «elabora un
  perfil de cada persona», pero el profesor aplica el 6.3 igualmente. En el tramo 80-119 hay 7
  problemas de esta familia, frente a 1 en 0-39, lo que explica buena parte de la caída en
  `alto_riesgo` (62 % con cita).

Las etiquetas del dataset se revisaron en todos los casos fallidos inspeccionados y eran correctas
según el motor; los errores son del profesor. Excepción a auditar: algunos casos de «6.3 +
perfilado» son poco naturales (p. ej. anonimizar escritos elaborando perfiles, problema 83).

**Qué concluimos.**

- El cambio decisivo fue el **profesor 8B**. La ficha era necesaria (sin ella no cita bien) pero
  con el 4B cada aclaración desplazaba el error a otro sitio.
- **Hay sobreajuste del prompt, y es moderado**: 86 % en los problemas usados para ajustarlo frente
  a 72-74 % en problemas nuevos. La cifra a reportar es la de problemas no vistos (~73 %).
  Paramos de iterar el prompt aquí: el error que queda (ver el 6.3 en todas partes) empeora
  cuanto más se explica la excepción, y seguir ajustando con los mismos datos aumentaría el
  sobreajuste.
- Para el SFT esto es suficiente: el verificador descarta las trazas incorrectas, así que el
  sesgo del profesor reduce el **rendimiento** por familia, no la calidad de los datos. Con
  ~90 % de problemas cubiertos, los 1000 de train darían ~900 problemas con traza.
- Riesgo: `derogation_profiling`, `exception` y `plain` quedarán infrarrepresentadas y son
  justamente las fronteras que queremos enseñar. Para la ejecución completa proponemos
  remuestrear (p. ej. `--samples 8`) los problemas de esas familias sin traza verificada, y
  reportar la exactitud por familia en la evaluación.
- **Ficha final: v4** (la que queda en el código). Se usa igual en destilación, SFT, GRPO,
  inferencia y evaluación.
- **Coste**: ~2,5 min por problema con el 8B y `transformers` (lotes de 8). Los 1000 problemas
  tardarían ~40 h; hace falta vLLM o lanzar por tramos con `--offset`/`--resume`.

Ficheros: `rlm/data/sft_traces_pilot.jsonl` (v3, 0-79, 145 trazas para SFT),
`rlm/data/sft_traces_pilot_v4_heldout.jsonl` (v4, 80-119, 70 trazas) y sus `.stats.json`; los
pilotos intermedios y abortados se conservan con sufijo `_v0`, `_v1`, `_v2*`, `_v3a*`, `_8b16`.

### 2026-10-28 · Fase 1 · Motor de reglas frente al AI Act Evaluation Benchmark

**Por qué.** El profesor (Andrés) propuso usar benchmarks existentes en lugar de datos
sintéticos. Antes de decidir, contrastamos nuestro motor con el AI Act Evaluation Benchmark
(Davvetas et al., 2026, arXiv:2603.09435, CC-BY-4.0): 339 escenarios en inglés, etiquetados con
ayuda de un LLM (70 prohibidos, 86 alto riesgo, 83 limitado, 99 mínimo), sin partición y con
citas solo a nivel de artículo (`[6, 8, 9, …]`). La Risk Classification Database de appliedAI
se descartó: es de 2023 (borrador, sin Art. 6.3), sin clase «limitado» y sin descarga.

**Cómo.** `rlm/compare_benchmark.py`: el motor no lee texto, así que Qwen3-8B (greedy, sin
thinking) rellena el `SystemProfile` de cada escenario en JSON, `classify` lo etiqueta y se
compara con el benchmark. La cifra mezcla tres fuentes de error: extractor, motor y benchmark.

| Extracción | Coincidencia | Limitado+mínimo fusionados |
|---|---|---|
| Ficha del Anexo III en español, sin «ninguno» por defecto | 40 % | 40 % |
| Con thinking (16 casos) | 6/16 | — |
| Anexo III en inglés + «la mayoría no encaja en ninguno» (final) | **57 %** | **73 %** |

Por clase del benchmark (final): prohibido 66 %, alto riesgo 35 %, limitado 20 %, mínimo 100 %.

**Qué vimos al revisar los desacuerdos (muestra, no exhaustivo).**

- **Errores del extractor (la mayoría).** No marca casos obvios del Anexo III (selección de
  personal → III.4.a, prestaciones por discapacidad → III.5.a, referéndum → III.8.b, puntuación
  ciudadana → Art. 5.1.c) y se inventa productos del Anexo I. Explica casi todo el «alto riesgo
  → mínimo» (42 casos).
- **«Limitado» del benchmark ≠ Art. 50** (54 casos «limitado → mínimo»): planificador de viajes,
  asistente de cocina, hoja de presupuestos… no tienen obligaciones de transparencia salvo que
  conversen con el usuario. El 95 % de sus «mínimos» también lista el Art. 50.
- **Benchmark discutible en prohibidos**: identificación biométrica remota en tiempo real con
  fines comerciales (centros comerciales, parques) aparece como prohibida, pero el Art. 5.1.h
  solo prohíbe el uso policial; sería alto riesgo (Anexo III.1.a). El reconocimiento de
  emociones en exámenes aparece como alto riesgo, pero el Art. 5.1.f lo prohíbe en educación.
- **Interpretación del «daño significativo» (Art. 5.1.a)**: el benchmark considera prohibida
  cualquier manipulación subliminal para vender; el motor exige además daño significativo.

**Qué concluimos.** El 57 % es una cota inferior dominada por el extractor, no una medida del
motor. En los desacuerdos revisados el motor seguía el reglamento y las discrepancias legales
venían del benchmark o de una interpretación abierta. Para medirlo de forma limpia hace falta
anotar a mano los flags de una muestra (~40 escenarios, 10 por clase). El benchmark sirve como
test externo de etiqueta, pero sus clases «limitado» y «prohibido» requieren reetiquetar o
reportar por separado.

Ficheros: `rlm/data/external/aiact_benchmark_scenarios.json`,
`rlm/data/benchmark_comparison.jsonl` y `.stats.json`, `logs/compare_benchmark.log`.

### 2026-10-28 · Fase 1 · Auditoría manual del motor con 40 escenarios del benchmark

**Por qué.** La comparación anterior mide sobre todo al extractor. Para medir el motor, las
casillas del `SystemProfile` tienen que ponerlas personas.

**Cómo.** Muestra estratificada de 40 escenarios (10 por clase del benchmark, semilla 2026) en
`rlm/data/audit/benchmark_audit.csv`. Claude rellenó un **borrador** de casillas
(`flags_borrador`) con una nota jurídica por caso; el equipo debe revisarlo (`flags_revisados`) y
dar su propio veredicto (`etiqueta_humana`, `articulo_humano`). `rlm/audit_benchmark.py` pasa la
hoja por el motor y calcula motor vs benchmark, motor vs humano y humano vs benchmark.

**Resultado provisional (borrador sin revisar).** Motor vs benchmark: **26/40 (65 %)**; alto
riesgo 10/10, mínimo 9/10, prohibido 6/10, limitado 1/10. Los 14 desacuerdos son de los tipos
ya vistos:

- **9 «limitado» sin obligación del Art. 50** (previsiones de demanda, iluminación, plagas…).
- **4 «prohibidos» que el reglamento no prohíbe**: identificación biométrica remota comercial
  (#64, #65, Anexo III.1.a), emociones en transporte público (#50, III.1.c) y FOMO en adultos
  jóvenes (#15, sin grupo vulnerable ni perjuicio significativo).
- **1 caso del Art. 50.2**: generador de pies de foto (#251), que el benchmark da como mínimo.

Con las casillas puestas a mano, el alto riesgo pasa del 35 % (extractor) al 100 %: el bajo
resultado de la comparación automática era del extractor. Pendiente: revisión del equipo, que es
la que vale; el borrador de Claude no es una anotación independiente.

### 2026-10-30 · Fase 1 · Destilación completa (1000 problemas de train)

**Configuración.** Igual que el piloto final: Qwen3-8B en modo thinking, ficha legal v4, 4
muestras por problema, lotes de 8, 2048 tokens nuevos como máximo, `--max-per-problem 2`.
Lanzada el 28-10 por la mañana y terminada el 28-10 a las 23:37: **~15 h** (≈0,9 min por
problema), bastante menos que las ~40 h estimadas con los pilotos.

**Resultado.**

| | Etiqueta + cita | Etiqueta | Problemas con traza verificada |
|---|---|---|---|
| Total (4000 trazas) | **77,5 %** | 86,8 % | **921/1000** |
| prohibido | 96 % | 97 % | 249/250 |
| riesgo_limitado | 84 % | 84 % | 240/250 |
| riesgo_minimo | 69 % | 98 % | 226/250 |
| alto_riesgo | 61 % | 68 % | 206/250 |

**1758 trazas verificadas para SFT** tras el tope de 2 por problema. Formato válido 97,5 %,
truncadas 2,6 %, 806 tokens nuevos de media.

**Dónde falla el profesor** (coincide con lo visto en los pilotos):

- **Abuso del Art. 6.3.** Es el error dominante. En `plain` y `exception` (mínimo sin cita)
  cita el 6.3.a igualmente (~140 trazas con la etiqueta bien y la cita mal). En
  `derogation_profiling` aplica el 6.3 aunque haya perfilado (alto riesgo → mínimo, ~120
  trazas). En `derogation_chatbot` aplica el 6.3 en lugar del Art. 50.1 (limitado → mínimo).
- **`police_rbi_authorised`**: con autorización previa y objetivo permitido sigue diciendo
  prohibido (Art. 5.1.h) en 57 trazas.
- Confusiones menores entre puntos vecinos del Anexo III (6.a/6.d, 6.b/6.e).

El verificador descarta todas esas trazas, así que el SFT no aprende el error, pero esas
familias quedan infrarrepresentadas: `derogation_profiling` 46/66 problemas con traza,
`exception` 18/27, `police_rbi_authorised` 30/39, `plain` 59/66.

**Siguiente paso.** Remuestrear con más muestras los 79 problemas sin traza (cuestión de 1-2 h al
ritmo observado) y lanzar el SFT con el resultado.

Ficheros: `rlm/data/sft_traces.jsonl`, `rlm/data/sft_traces.stats.json`, `logs/distill_full.log`.

### 2026-10-30 · Fase 1 · Remuestreo de problemas sin traza y lanzamiento del SFT

**Remuestreo.** Nueva opción `distill.py --only-unsolved`: los 79 problemas sin traza verificada,
8 muestras cada uno (2 h 57 min). Rescata **44/79** problemas con 67 trazas (etiqueta + cita
14,9 %, etiqueta 45,7 %, 1127 tokens de media, 8,9 % truncadas: son los casos difíciles).
Total para SFT (`rlm/data/sft_traces_merged.jsonl`): **1825 trazas de 965 problemas**
(prohibido 499, limitado 471, mínimo 442, alto riesgo 413).

**Longitud de secuencia.** La ficha legal hace que el prompt solo ocupe **2625 tokens**; los
ejemplos completos miden 3051-4942 (mediana 3488). Con el `--max-length 2048` por defecto se
habría truncado toda la respuesta en el 100 % de los ejemplos. SFT lanzado con
`--max-length 5120`. Para GRPO, TRL 1.13 no trunca el prompt, pero `--max-completion-length`
(768 por defecto) se queda corto para trazas de ~800-1100 tokens: subirlo a ~1536.

**SFT.** Qwen3-0.6B + LoRA r=16 (all-linear), 2 épocas, lr 2e-4, lote efectivo 16 (4×4),
230 pasos, ~14 s/paso (~55 min). Pérdida solo sobre la respuesta (formato prompt/completion).

### 2026-10-30 · Fase 1 · Resultados del SFT y causa de la generación lenta

**SFT.** 230 pasos en 39 min. Pérdida 0,82 → 0,45; exactitud por token 75 % → 84 %, sin
saltos. Adaptador en `rlm/weights/sft_lora`.

**Evaluación** (`rlm/evaluate.py`, ahora implementado: greedy, lotes, 1536 tokens nuevos máx.):

| Modelo | Conjunto | Etiqueta | Etiqueta + cita | Formato | Tokens medios |
|---|---|---|---|---|---|
| Qwen3-0.6B base | test (200) | 19,5 % | 5,5 % | 57,5 % | 471 |
| **SFT** | test (200) | **71,0 %** | **62,0 %** | 81,5 % | 895 |
| Qwen3-0.6B base | test OOD (100) | 19,0 % | 3,0 % | 56,0 % | 428 |
| **SFT** | test OOD (100) | **67,0 %** | **57,0 %** | 81,0 % | 880 |

- El base queda por debajo del azar (25 %): no sigue el formato ni conoce el procedimiento.
- **Todos los fallos de formato del SFT son truncamientos** (37/200 y 19/100 llegan a 1536
  tokens dando vueltas: «Let me recheck the steps…»). Sin contar los truncados, el SFT acierta
  la etiqueta en el 87 % del test. GRPO, con la recompensa de formato y un límite de longitud,
  debería atacar esto primero.
- Por clase (test): prohibido 86 %, mínimo 90 % (66 % con cita), limitado 66 %, **alto riesgo
  42 %**. Por familia, lo peor es lo mismo que fallaba el profesor: `derogation_profiling`
  (23 %), `derogation_chatbot` (38 %), `annex_iii` (47 %). En OOD, `ood_derogation_profiling`
  0/12 y `ood_emotion_medical` 0/4. El alumno hereda los sesgos del profesor (6.3 en exceso),
  aunque el verificador filtrara las trazas: le faltan ejemplos de esas fronteras.
- Generaliza razonablemente: OOD 67 % frente a test 71 %.

**Generación lenta: era el backend cuDNN de SDPA.** La evaluación iba a ~0,66 s por paso de
decodificación (lote 100). Perfilando: el forward sin caché tarda 12,8 ms, pero en `generate`
cada llamada a `scaled_dot_product_attention` tarda ~7 ms (×28 capas). Con
`attn_implementation="eager"` bajaba a 0,016 s/paso con prompt corto, lo que señalaba al
backend. Desactivando solo cuDNN SDPA (`torch.backends.cuda.enable_cudnn_sdp(False)`):

| Qwen3-0.6B, prompt de 2600 tokens | cuDNN (por defecto) | sin cuDNN |
|---|---|---|
| lote 32 | 0,46 s/paso | **0,09 s/paso** (13,7 GB) |
| lote 100 | ~0,66 s/paso | **0,26 s/paso** (40 GB) |

Aplicado en todos los scripts con `rlm/gpu.py::disable_cudnn_attention`. Probablemente explica
también parte de la lentitud de la destilación con el 8B.

### 2026-10-30 · Fase 1 · GRPO: prueba de humo y lanzamiento

**Prueba (3 pasos).** Con un prompt por lote y 4 lotes generados juntos (`--grad-accum 4`, TRL
genera `steps_per_generation = grad_accum` lotes a la vez), cada paso de optimización son 4
problemas × 8 respuestas = 32 generaciones y tarda **~190 s**. Con los 32 en un solo lote de
entrenamiento se agotaba la memoria en el backward (logits de 32 × 1536 × 152k en fp32 ≈ 28 GB).
Partiendo del SFT: recompensa media 1,55-1,70 sobre 2,0; formato 91-97 %, etiqueta 75-81 %,
cita 69-81 %, longitud media 750-980 tokens, 3-9 % truncadas; `frac_reward_zero_std` = 0 (hay
señal en todos los grupos).

**Configuración lanzada.** 250 pasos = 1000 problemas (una época de train), ~13 h, checkpoint
cada 25 pasos. 8 generaciones, temperatura 1,0, `--max-completion-length 1536`, beta 0 (sin KL),
epsilon 0,2, pesos de recompensa 0,5 / 1,0 / 0,5 (formato / etiqueta / cita). **lr 2e-5** en
lugar del 5e-6 por defecto: entrenamos un adaptador LoRA, que necesita una tasa ~10× mayor que
el ajuste completo para el que está pensado ese valor por defecto. Salida:
`rlm/weights/final_rlm_lora`, log en `logs/train_grpo.log`.

### 2026-11-03 · Fase 1 · GRPO: reanudación en GPU de 16 GB y evaluación final

**Corte y reanudación.** La ejecución del 30-10 se cortó en el paso 184/250 a las 22:00 (fin de la
sesión). Esta sesión tiene una MIG 1g.18gb (16 GB) en lugar de la GPU grande del 30-10, y
reanudar desde `checkpoint-175` con la misma configuración agotaba la memoria:

1. En la generación: 32 respuestas a la vez (4 lotes × 8, `steps_per_generation = grad_accum`)
   con un prompt de ~2600 tokens + 1536 nuevos no caben.
2. Con `--steps-per-generation 1` (8 a la vez), en el backward: logits de 8 × 1536 × 152k.

Solución (dos opciones nuevas en `train_grpo.py`): `--per-device-batch-size 2 --grad-accum 16
--steps-per-generation 4`. Se genera un problema (8 respuestas) por tanda y el paso de optimización
sigue siendo 4 problemas × 8 = 32 respuestas, así que el algoritmo es el mismo (beta 0, la política
solo cambia en el paso de optimización). Resto de hiperparámetros idénticos al checkpoint. Pico de
~7,5 GB; ~3 min por paso (frente a ~2-3 min en la GPU grande); los 75 pasos, 3 h 52 min.
Salvedad: al cambiar la acumulación al reanudar, el orden de datos tras el paso 175 puede no
coincidir exactamente con el de la ejecución original (algunos problemas repetidos u omitidos).

**Curva de recompensa** (medias por tramo de 50 pasos; máximo 2,0):

| Pasos | Recompensa | Etiqueta | Cita | Tokens | Truncadas |
|---|---|---|---|---|---|
| 1-50 | 1,64 | 0,80 | 0,73 | 699 | 5,2 % |
| 51-100 | 1,74 | 0,85 | 0,79 | 617 | 1,9 % |
| 101-150 | 1,77 | 0,87 | 0,82 | 610 | 1,8 % |
| 151-200 | 1,77 | 0,88 | 0,81 | 596 | 2,7 % |
| 201-250 | 1,80 | 0,89 | 0,84 | 584 | 2,0 % |

**Evaluación** (`rlm/evaluate.py`, greedy, 1536 tokens máx., lotes de 16):

| Modelo | Conjunto | Etiqueta | Etiqueta + cita | Formato | Tokens medios |
|---|---|---|---|---|---|
| SFT | test (200) | 71,0 % | 62,0 % | 81,5 % | 895 |
| **GRPO** | test (200) | **82,5 %** | **79,0 %** | 92,0 % | 652 |
| SFT | test OOD (100) | 67,0 % | 57,0 % | 81,0 % | 880 |
| **GRPO** | test OOD (100) | **80,0 %** | **73,0 %** | 88,0 % | 681 |

- Todos los fallos de formato siguen siendo truncamientos (16/200, 12/100). Sin ellos, etiqueta
  89,7 % (test) y 90,9 % (OOD).
- Por clase (test, etiqueta + cita, SFT → GRPO): limitado 66 → 92 %, alto riesgo 30 → 54 %,
  mínimo 66 → 78 % (etiqueta sola 90 → 86 %), prohibido 86 → 92 %. En OOD el alto riesgo sigue
  siendo lo peor: 4 → 24 %.
- Lo que GRPO arregla: `derogation_chatbot` 38 → 83 % (deja de usar el 6.3 en lugar del 50.1),
  `police_rbi_authorised` 55 → 82 %, `annex_iii` 26 → 58 %, `ood_emotion_classroom` 72 → 92 %.
- Lo que no: `derogation_profiling` 18 → 29 % (test) y 0 → 17 % (OOD), `ood_emotion_medical`
  0 → 25 % (n=4). El sesgo del profesor (Art. 6.3 aunque haya perfilado) persiste. Ligera
  regresión de etiqueta en `derogation` (97 → 87 %), aunque la cita mejora (58 → 77 %).
- La ganancia se mantiene fuera de distribución (+13 puntos en OOD frente a +11,5 en test).

Ficheros: `rlm/weights/final_rlm_lora`, `reports/eval_test_grpo.json`,
`reports/eval_test_ood_grpo.json`, `logs/train_grpo_resume3.log`, `logs/eval_grpo.log`.
