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

### 2026-11-03 · Fase 1 · Cierre: GRPO a mano, curvas, análisis de fallos y endpoint

**GRPO a mano (`rlm/grpo_step.py`).** Ventajas por grupo `(r - media) / (std + eps)` (std
muestral, como TRL), ratio `exp(logp_new - logp_old)`, objetivo recortado
`min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)`, KL de DeepSeekMath `exp(Δ) - Δ - 1` con `Δ = logp_ref - logp`, y
pérdida = −(media por respuesta de la media por token válido − β·KL). Los 6 tests dados pasan;
añadimos dos más: el padding enmascarado no cambia la pérdida, y con β > 0 la pérdida sube
exactamente β·KL.

**Curvas** (`rlm/plots.py` → `reports/phase1_grpo_curves.png`, `phase1_sft_loss.png`,
`phase1_pass1.png`; historiales copiados a `reports/history_{sft,grpo}.json` para que las
figuras se regeneren sin los pesos):

- *SFT:* la pérdida baja de 0,82 a 0,45 y el acierto por token sube de 75 a 84 %. El escalón del
  paso ~115 es el cambio de época (2 épocas, 230 pasos): el modelo vuelve a ver las mismas
  trazas. No hay conjunto de validación, así que la curva no dice nada de sobreajuste; lo mide la
  evaluación.
- *GRPO:* recompensa 1,55 → ~1,80. El formato se satura en ~40 pasos (0,90 → 0,98); etiqueta
  (0,75 → 0,89) y cita (0,69 → 0,84) suben más despacio y juntas. **La longitud baja** (786 → ~600
  tokens), al revés que en R1-Zero: partimos de un SFT con trazas largas y una respuesta truncada
  vale 0 en las tres recompensas, así que GRPO castiga directamente los bucles (truncadas 9 % → 2-3 %).
  La reanudación del paso 175 no deja ningún salto visible.
- *Señal que se agota:* la fracción de grupos con las 8 recompensas iguales (ventaja 0, sin
  gradiente) sube de 0,33 a 0,55 por tramos de 50 pasos, y `reward_std` baja de 0,60 a 0,23. Al
  final la mitad de los problemas ya no enseñan nada; más pasos con estos datos darían poco. Es el
  argumento para parar en 250 y no alargar el entrenamiento.
- *El recorte nunca actúa:* `clip_ratio` es 0 en los 250 pasos. Con `num_iterations = 1` cada
  tanda se usa en un solo paso de optimización, así que π_θ = π_θ_old al calcular la pérdida y
  ρ = 1. El recorte solo importa al reutilizar las mismas muestras en varios pasos (μ > 1). La
  entropía se mantiene plana (~0,50): sin KL (β = 0) no vimos colapso.

**Análisis de fallos** (GRPO, `reports/eval_test*_grpo.json`; 42 fallos de etiqueta o cita en
test y 27 en OOD). Por tipo:

1. **Bucles hasta el truncamiento (16/200 en test, 12/100 en OOD).** Test #8
   (`derogation_profiling`, seguros que *perfilan* personas, esperado alto riesgo, III.5.c): el
   modelo alterna «es tarea preparatoria → 6.3.d» y «pero mejora la redacción, no prepara una
   evaluación» hasta agotar los 1536 tokens. El 43 % de las frases de las respuestas truncadas está
   repetido, frente al 2 % en las demás. Son casos donde dos reglas chocan (excepción del 6.3 frente
   a la cláusula de perfilado del último párrafo del 6.3).
2. **Excepción del 6.3 aplicada pese al perfilado.** Test #17: revisa decisiones pasadas sin
   sustituirlas y además «elabora un perfil de cada persona con predicciones sobre su
   comportamiento». El modelo cita bien la letra c del 6.3 y concluye `riesgo_minimo (Art. 6.3.c)`
   sin aplicar que el perfilado anula cualquier excepción. Es el sesgo que heredamos del profesor
   Qwen3-8B y que GRPO solo corrige en parte (18 → 29 %).
3. **Una obligación de transparencia tapa el alto riesgo.** Test #19: polígrafo en interrogatorios
   policiales (Anexo III.6.b) con un chatbot que no se identifica. Responde `riesgo_limitado
   (Art. 50.1)`: «the chatbot is the main interaction». Las obligaciones del Art. 50 se suman al
   alto riesgo, no lo sustituyen. Pasa en 5 de los 19 casos `annex_iii` de test.
4. **No ve la excepción del Art. 5.1.h.** Test #0: reconocimiento facial en tiempo real para
   «pequeños hurtos». No es uno de los delitos graves del Anexo II, así que está prohibido, pero el
   modelo duda entre 5.1.h y el Anexo III.1.a y elige alto riesgo. Lo mismo en 3 de 12
   `police_rbi_prohibited`.
5. **Contexto educativo fuera de distribución.** OOD #2: análisis de voz de alumnos en tutorías
   para detectar frustración. El 5.1.f prohíbe el reconocimiento de emociones en educación, pero el
   modelo razona «education → Annex III.1.c» y responde alto riesgo. En el caso médico (OOD #35) se
   equivoca al revés: lee la excepción médica del 5.1.f y aun así concluye prohibido. El modelo
   conoce la regla y su excepción, pero no siempre las aplica en el orden correcto.
6. **Etiqueta bien, cita mal.** Test #23: herramienta que propone el fallo a los jueces. Acierta
   alto riesgo pero cita III.5.a (servicios públicos) en lugar de III.8.a (administración de
   justicia). La recompensa de cita existe para esto: este caso cuenta como acierto en pass@1 y como
   fallo en etiqueta + cita.

Además, el razonamiento sale en inglés en ~2/3 de las respuestas, aunque la pregunta y la respuesta
están en español. Viene de las trazas del profesor y ninguna recompensa lo penaliza. Si importara
al usuario, sería un candidato natural para la `domain_reward`.

**Endpoint `/reasoning`.** `.env` (plantilla en `.env.example`): `ARCA_RLM_ADAPTER =
rlm/weights/final_rlm_lora`, `ARCA_RLM_VERIFIER = aiact` y la nueva `ARCA_RLM_TEMPERATURE = 0`
(greedy, como `evaluate.py`; antes el endpoint muestreaba siempre a 0,6). `docker-compose.yml`
pasa `.env` al contenedor y `arca-api` lo lee al arrancar. El `max_new_tokens` por defecto sube de
1024 a 1536, el presupuesto con el que se entrenó y evaluó: con 1024 se truncaría bastante más
que el 8 %. Prueba con `uv run arca-api` (puerto 8077, porque el 8000 estaba ocupado en la DGX):
`/health` marca `reasoning: ready`. En 8 casos de test, el veredicto coincide con `evaluate.py`
en 7, y las 8 respuestas traen `thinking` y `answer` separados. La que difiere (#61) es un caso
límite: generar en lote con padding a la izquierda y en bf16 no da exactamente los mismos tokens
que generar de uno en uno.

## Fase 2 · Herramientas

### 2026-11-03 · Fase 2 · Herramientas del AI Act, bucle a mano y banco de 34 casos

**Herramientas** (`tool_use/tools.py`), una de cada tipo, como se prometió en la propuesta:

- *Consulta externa*, `lookup_ai_act`: texto oficial del Reglamento. EUR-Lex responde 202 con el
  cuerpo vacío a peticiones automáticas (es un desafío anti-bots), así que usamos la API CELLAR de
  la Oficina de Publicaciones: `GET publications.europa.eu/resource/celex/32024R1689` con
  `Accept: application/xhtml+xml` y `Accept-Language: spa`. Son 1,4 MB de XHTML bien formado en
  ~0,4 s, con `id` para cada artículo, apartado y anexo. `tool_use/ai_act_text.py` lo convierte en
  un árbol y resuelve referencias como `Art. 5.1.f`, `Art. 3.60` o `Anexo III.4.a` (113 artículos y
  13 anexos). Se cachea por proceso: la primera llamada cuesta ~400 ms y las siguientes ~0 ms.
- *Cálculo*, `classify_risk`: el motor de reglas de la fase 1. El modelo traduce el texto a hechos
  y el motor aplica la precedencia legal. Su esquema tiene 42 campos, y se construye desde
  `SystemProfile` con una descripción por campo; los `Literal` llegan como `enum`. También
  `compute_max_fine` (Art. 99: el mayor de dos importes, o el menor si es pyme según el 99.6) y la
  `calculator` de clase.
- *Acción*, `write_assessment_report`: escribe un PDF en `reports/assessments/` con fpdf2. Se
  registra con `requires_confirmation=True`. En la fase 2 se ejecuta (`ARCA_TOOLS_AUTO_CONFIRM=1`),
  y con `0` el bucle no la ejecuta y le devuelve al modelo «requiere confirmación». La política
  definitiva se decide en la fase 4.

**Bucle** (`tool_use/executor.py`): son los cinco pasos sin ningún framework. Cada generación
puede incluir varios `<tool_call>`, que se ejecutan en el mismo turno (llamadas paralelas). Los
resultados vuelven como mensajes `tool`, y el bucle se repite hasta que el modelo responde sin
pedir herramientas o se agota `max_turns` (multi-step). El JSON mal formado, las herramientas
desconocidas y los argumentos inválidos vuelven al modelo como observación. Al llegar a
`max_turns`, la respuesta sale con `max_turns_reached: true`, un campo opcional nuevo. Modelo:
**Qwen3-4B**, con generación greedy y sin razonamiento. Elegimos el 4B en lugar del 1.7B de la
plantilla por la comparación de abajo.

**Banco** (`tool_use/bench/build_cases.py` → `cases.jsonl`, 34 casos): 10 de clasificación (la
etiqueta esperada la calcula el propio motor), 6 de consulta, 4 de multas, 2 de informe, 1 de
calculadora, 6 que no necesitan herramienta, 3 encadenados y 2 paralelos. `evaluate.py` mide:

- *selección*: la primera llamada es la herramienta correcta;
- *argumentos*: coinciden tras validar ambos lados con el esquema, de modo que los `false` por
  defecto no cuentan y `artículo 5.1.f` vale lo mismo que `Art. 5.1.f`;
- *cadena*: se llamaron todas las herramientas esperadas;
- *etiqueta*: lo que devolvió `classify_risk`;
- *respuesta*: la respuesta final contiene el dato verificable.

| Configuración | Selección | Argumentos | Cadena (n=5) | Etiqueta (n=10) | Respuesta | Llamadas fallidas | s/caso |
|---|---|---|---|---|---|---|---|
| v1: descripciones iniciales | 94 % | 65 % | 60 % | 60 % | 71 % | 3 | 8,9 |
| v2: + índice de artículos, errores accionables, instrucciones multi-step | 97 % | 88 % | 80 % | 80 % | 82 % | 1 | 6,8 |
| **v3: + frase resuelta en `compute_max_fine`** | **97 %** | **88 %** | 80 % | **80 %** | 89 % | 1 | 6,9 |
| v3 con el prompt de las diapositivas | 91 % | 74 % | 100 % | 80 % | 93 % | 4 | 8,7 |
| v3 con razonamiento (thinking) | 97 % | 68 % | 100 % | 70 % | 86 % | 2 | 24,9 |
| v3 con Qwen3-1.7B | 91 % | 53 % | 60 % | 10 % | 57 % | 5 | 6,1 |
| Qwen3-4B **sin herramientas** | — | — | — | 0 % | 11 % | — | 10,4 |

Son 34 casos, así que un caso son 3 puntos en las métricas globales y 20 en la de cadena: las
diferencias pequeñas entre filas no son significativas. La métrica de respuesta se volvió a puntuar
(`--rescore`) después de añadir variantes como «riesgo alto», sin regenerar las respuestas.

**Qué cambió de v1 a v3 (las descripciones importan más de lo que parece):**

1. *Mezcla de esquemas.* En v1 el modelo rellenó `classify_risk` con los campos de
   `write_assessment_report` (`system_name`, `label`, `justification`…) en 3 de los 10 casos de
   clasificación. Al principio Pydantic los **ignoraba en silencio**. Es peligroso: un hecho mal
   escrito que se ignora convierte un alto riesgo en riesgo mínimo. Pasamos a `extra="forbid"`,
   pero ante el error el modelo se lo explicaba al usuario en vez de reintentar. En v2 la
   descripción dice «no hay argumento name, description ni label», el error lista los nombres
   válidos y termina en «Fix the arguments and call the tool again», y el prompt de sistema pide
   reintentar. Resultado: 0 casos de mezcla y argumentos del 65 al 88 %.
2. *No sabe dónde está cada cosa.* Pedía `Art. 2` para la definición de «sistema de IA» (está en el
   Art. 3.1) y `Art. 1` para las fechas de aplicación (están en el Art. 113). Un índice de dos
   líneas en la descripción de `lookup_ai_act` lo arregla: consulta del 67 al 100 %.
3. *Se para tras la primera herramienta.* La instrucción «haz todas las llamadas necesarias antes
   de responder» sube la cadena del 60 al 80 %. El fallo que queda (multi-03) es que clasifica y
   luego parafrasea el Art. 50.1 de memoria en vez de consultarlo.
4. *Lee mal su propio resultado.* `compute_max_fine` devolvía bien el importe (1,5 M€ para la
   startup), pero el modelo respondía «15 millones», que es el importe fijo. Con el colegio pasaba
   igual: 35 M€ en lugar de 70 M€. En v3 el primer campo es una frase que ya da la respuesta
   («Multa máxima: 1.500.000 EUR (Art. 99.4). Es la menor de…»), y la respuesta pasa del 82 al 89 %.
   La forma de la salida es tan parte de la herramienta como su descripción.

**Sin herramientas** el mismo modelo acierta el 11 % de las respuestas verificables y no acierta ni
una clasificación. Se inventa el texto del Art. 50.1 («el Reglamento se aplica a cualquier sistema
que pueda causar daño grave…»), da porcentajes de multa propios del RGPD (4 %, 10 %) y clasifica el
scoring crediticio por el «artículo 5, apartado 1, letra a)». Este es el argumento central de la
fase: en derecho, una respuesta fluida y segura sin fuente es peor que ninguna.

**Plantilla nativa frente al prompt de las diapositivas.** Con `slides` el modelo encadena mejor
(5/5) y su respuesta final es algo mejor, pero falla más en los argumentos (74 % frente a 88 %):
añade hechos que no se dicen (7 campos de más en el caso de la policía) y en el caso del colegio
empezó consultando el texto en lugar de clasificar. La plantilla nativa es el formato con el que
Qwen3 aprendió a llamar a funciones y produce argumentos más limpios; la dejamos por defecto.

**Thinking** no compensa aquí. Es 3,6 veces más lento y razona hacia hechos que el texto no dice:
marca `workplace_or_education` en el filtro de CV y en el chatbot, y `infers_sensitive_attributes`
en el detector de emociones. En el caso del ayuntamiento pierde `score_causes_detrimental_treatment`
y la etiqueta. Encadena 5/5, pero los argumentos bajan al 68 %.

**Qwen3-1.7B** elige bien la herramienta (91 %), pero no sabe traducir el texto a hechos: llama a
`classify_risk` casi vacío y todo le sale riesgo mínimo (etiqueta 1/10). En la primera pasada
escribía `"annex_iii_use_cases": null` y el esquema lo rechazaba. Recibía 36 errores seguidos y
repetía la misma llamada hasta agotar los turnos. Aceptamos `null` como «sin valor» en todos los
campos de `classify_risk` (las métricas del 1.7B en la tabla son de después del cambio). Lo difícil
no es el formato de la llamada, sino la extracción de hechos, igual que vimos con el extractor del
benchmark en la fase 1.

**Fallos que quedan con v3** (en `reports/phase2_eval.json`):

- *classify-06*: «genera imágenes para campañas publicitarias» → `annex_iii_use_cases: ["4.a"]`
  (empleo). Confunde «campañas» con los «anuncios de empleo» del 4.a y no marca
  `generates_synthetic_content`.
- *classify-09*: «solo traduce los currículums antes de que el reclutador evalúe» → marca 4.a pero
  no la excepción `preparatory_task`, así que sale alto riesgo. Es el mismo punto débil que el
  modelo de la fase 1: la excepción del Art. 6.3.
- *classify-08*: etiqueta correcta, pero añade `rbi_objective: victim_search` y
  `predicts_individual_crime_risk` sin que el texto lo diga. Aquí no cambia el resultado, pero un
  objetivo inventado puede convertir un «prohibido» en un «alto riesgo» autorizado.
- *no_tool-06*: «¿Cuánto es 2 + 2?» → llama a la calculadora. Es el único caso de exceso en los 6
  casos sin herramienta.
- *Paralelo cuando debía ser secuencial* (probado en el endpoint, fuera del banco): «clasifícala y
  dime la multa máxima» → emite las dos llamadas en el mismo turno. Como calcula la multa sin
  conocer la etiqueta, elige `other_obligation`: 30 M€ en vez de 70 M€. En el banco (multi-01) la
  frase «si está prohibida» fuerza el orden y sale bien. Las llamadas paralelas solo sirven cuando
  las llamadas son independientes. Es un caso para la fase 4, donde el agente planifica.

**Endpoint**: `/health` marca `tools: ready`; `POST /tools` devuelve `tool_calls` con el resultado
real y la latencia. El test `test_pending_phase_returns_501_with_phase_name` usaba `/tools`, y al
implementar la fase cargaba un modelo de 8 GB dentro de `pytest`. Ahora usa `/rag`, que sigue
pendiente. Los 17 tests nuevos (`tests/test_ai_act_tools.py`) no necesitan red ni GPU: usan un
fragmento del XHTML como fixture y un bucle con salidas del modelo grabadas.

Ficheros: `reports/phase2_eval.json` (v3, la configuración final), `phase2_eval_v1.json`,
`phase2_eval_v2.json`, `phase2_eval_{slides,thinking,qwen3_1.7b,no_tools}.json`.

## Fase 3 · RAG

### 2026-11-03 · Fase 3 · Corpus oficial, dos chunkings, tres retrievers y respuestas con citas

**Corpus** (`rag/download_corpus.py`, fuentes y licencias en `rag/CORPUS.md`): 149 documentos y
1,76 M de caracteres, todos oficiales.

- El Reglamento en español desde CELLAR: un fichero por cada artículo (113) y anexo (13), y los 180
  considerandos en 18 ficheros.
- Las dos directrices de la Comisión en español: prácticas prohibidas (164 páginas) y definición de
  sistema de IA (15 páginas). Había que buscar la versión española entre las 24 traducciones del
  newsroom (documentos 118659 y 118640).
- El borrador de directrices de alto riesgo de mayo de 2026, que solo existe en inglés (167
  páginas).

Dos defectos que encontramos al leer el corpus: el exponente de `10^25` FLOPs (Art. 51) llegaba como
«1025», porque el DOUE usa `<span class="oj-super">`, y pypdf parte palabras en las directrices
(«recom endaciones»).

**Conjunto dorado** (`rag/gold/gold.jsonl`): 46 preguntas con respuesta y 5 sin respuesta en el
corpus. El reparto es 32 sobre artículos y anexos, 3 sobre considerandos, 7 sobre las directrices
en español y 4 sobre el borrador en inglés (preguntadas en español: recuperación entre idiomas).
**Es un borrador que redactó Claude** leyendo el corpus, igual que la auditoría de la fase 1. El
equipo tiene que revisarlo y reescribir lo que haga falta, porque la rúbrica pide que lo escriba el
equipo. Las preguntas no guardan ids de fragmento, que cambian con cada chunking: guardan la
**evidencia**, es decir, el fichero y una cita literal. Un fragmento es relevante si contiene la cita
o sus primeros o últimos 60 caracteres, por si un corte la parte. Así el mismo conjunto evalúa
cualquier chunking. Las 46 citas se comprobaron automáticamente contra el corpus. Hay 2 preguntas
con dos pasajes igual de válidos (q06: Art. 6.4 y Art. 49.2; q26: los apartados 1 y 2 del Art. 49
repiten la frase).

**Recuperación** (`rag/evaluate.py` → `reports/phase3_eval.json`, Qwen3-Embedding-0.6B, top-k = 5,
n = 46; una pregunta son 2,2 puntos):

| Chunking | Fragmentos | Media (car.) | BM25 hit@5 / MRR | Denso hit@5 / MRR | λ* | Híbrido hit@1 / hit@5 / MRR |
|---|---|---|---|---|---|---|
| recursivo (800/100) | 3262 | 542 | 0,41 / 0,22 | 0,46 / 0,36 | 0,3 | 0,33 / 0,46 / 0,40 |
| semántico 0,50 | 5602 | 312 | 0,39 / 0,26 | 0,48 / 0,39 | 0,4 | 0,35 / 0,52 / 0,42 |
| semántico 0,60 | 4333 | 404 | 0,39 / 0,24 | 0,48 / 0,41 | 0,3 | 0,33 / 0,59 / 0,43 |
| semántico 0,70 | 3261 | 537 | 0,46 / 0,27 | 0,52 / 0,43 | 0,3 | 0,35 / 0,56 / 0,45 |
| semántico 0,80 | 2076 | 844 | 0,39 / 0,24 | 0,54 / 0,42 | 0,2 | 0,35 / 0,63 / 0,46 |
| recursivo + título | 3262 | 611 | 0,41 / 0,27 | 0,56 / 0,41 | 0,3 | 0,33 / 0,56 / 0,44 |
| semántico 0,50 + título | 5602 | 388 | 0,46 / 0,30 | 0,59 / 0,48 | 0,3 | 0,43 / 0,56 / 0,50 |
| semántico 0,60 + título | 4333 | 480 | 0,50 / 0,28 | 0,61 / 0,47 | 0,3 | 0,39 / 0,63 / 0,50 |
| **semántico 0,70 + título** | 3261 | 612 | 0,50 / 0,33 | 0,61 / 0,47 | **0,3** | **0,43 / 0,63 / 0,52** |
| semántico 0,80 + título | 2076 | 914 | 0,48 / 0,29 | 0,56 / 0,42 | 0,3 | 0,37 / 0,59 / 0,47 |

- **Umbral semántico** (`reports/phase3_sentence_distances.png`, Art. 5): la distancia mediana entre
  frases consecutivas del corpus es 0,55 (p25 = 0,41, p75 = 0,71). Con el 0,25 por defecto, casi
  cada frase sería un fragmento. En el Art. 5 se ve bien: las letras de la misma prohibición quedan
  por debajo de 0,5 y los saltos entre prohibiciones superan 0,7. Por eso 0,70 corta donde cambia
  el tema y da fragmentos del mismo tamaño medio que el recursivo (3261 frente a 3262). Así la
  comparación es justa: **a igual tamaño, el semántico gana** (MRR híbrido 0,45 frente a 0,40),
  porque corta entre ideas y no a mitad de una enumeración. Con 0,80 los fragmentos son más largos
  y el MRR deja de subir.
- **El título del documento delante de cada fragmento** («Artículo 14. Supervisión humana — …») es
  la mejora más grande y más barata: entre +0,04 y +0,08 de MRR en todas las variantes. Un apartado
  del Art. 14 no dice que es del Art. 14 ni de qué trata el artículo.
- **Por qué el denso gana a BM25:** las preguntas parafrasean («¿cuántas personas tienen que
  confirmar…?» frente a «al menos dos personas físicas… han verificado y confirmado»). Además, 4
  preguntas son en español sobre texto en inglés, y BM25 ahí no tiene ninguna opción. BM25 sigue
  aportando: el híbrido mejora al denso solo en todos los chunkings.
- **λ:** el óptimo es 0,3 en 8 de las 10 variantes (`reports/phase3_lambda_sweep.png`). La curva es
  plana entre 0,2 y 0,4 y cae deprisa a partir de 0,5. Con el min-max, el último resultado de cada
  lista vale 0, lo mismo que un fragmento que esa lista no recuperó, y eso penaliza las listas de
  BM25 con puntuaciones muy juntas.
- **Por qué falla** (análisis hecho con el semántico 0,70 *sin* título, que deja 20 preguntas sin
  el fragmento relevante entre los 5 primeros; la configuración final deja 17): en 9 de ellas los primeros resultados son de las directrices de prácticas
  prohibidas, que comentan el mismo artículo con más palabras que el propio artículo. Por ejemplo,
  en q08 (verificación por dos personas, Art. 14.5) los tres primeros son de las directrices. Solo
  en un caso (q21) el fragmento recuperado de las directrices contenía la cita literal y el
  conjunto dorado lo contaba como fallo. El problema no es el conjunto dorado: es que el comentario
  le gana a la norma. Lo siguiente que probaríamos es un reranker o filtrar por tipo de fuente
  cuando la pregunta pide «qué dice el reglamento».

**Instrucción de la consulta** (sobre `semántico 0,50 + título`, el mejor denso):

| Instrucción | hit@1 | hit@5 | MRR |
|---|---|---|---|
| por defecto («Given a user question, retrieve the passages… that answer it») | **0,39** | 0,59 | **0,48** |
| del dominio en español («Dada una pregunta sobre el Reglamento… recupera el pasaje…») | 0,17 | 0,61 | 0,36 |
| del dominio en inglés («…retrieve the article, recital or guideline paragraph…») | 0,33 | 0,54 | 0,43 |
| «mismo tema» | 0,33 | 0,61 | 0,46 |
| sin instrucción (consulta embebida como pasaje) | 0,30 | 0,61 | 0,44 |

Fue la mayor sorpresa: la instrucción del dominio en español **hunde el hit@1 de 0,39 a 0,17** y
casi no cambia el hit@5. Pone delante los fragmentos que hablan de «el reglamento» y «las
directrices» en general (preámbulos, considerandos introductorios) frente al pasaje concreto. La
instrucción genérica en inglés, que es como se entrenó Qwen3-Embedding, es la mejor. Más
especificidad no es mejor si nombra el corpus en lugar de la tarea.

**Generación** (`rag/evaluate_answers.py`, híbrido, top-k = 5, Qwen3-4B en greedy):

| | RAG | Sin recuperación (closed book) |
|---|---|---|
| Respuestas con el dato clave (n = 46) | **63 %** | 13 % |
| Respuestas con al menos una cita | 91 % | — |
| Respuestas sin ids inventados (n = 51) | 94 % | — |
| Cita un fragmento relevante | 59 % | — |
| El fragmento relevante estaba entre los recuperados | 63 % | — |
| Negativas en preguntas con respuesta | 9 % | 41 % |
| «No lo sé» en las 5 sin respuesta | 3/5 | 5/5 |

- **La generación está limitada por la recuperación:** el 63 % de respuestas correctas coincide con
  el 63 % de veces que se recuperó el pasaje. Cuando el pasaje está, el modelo casi siempre lo usa.
- **Sin recuperación** se niega en el 41 % de las preguntas y, cuando contesta, inventa definiciones
  («la ultrasuplantación es la representación falsa de una persona…»). El 5/5 en las preguntas sin
  respuesta no tiene mérito: se niega a casi todo.
- **Citas inventadas** (3 respuestas): cita `-0010` cuando solo recuperó `-0009` del mismo
  documento, es decir, se inventa el id del fragmento vecino. `to_response` las descarta (solo pasan
  las que existen) y `invalid_citations` las registra.
- **Lee el pasaje al revés** (q05): recupera el fragmento del borrador en inglés que dice «none of the
  exceptions… apply when the system performs profiling» y responde «Sí, puede acogerse a las
  excepciones». Es el fallo más grave, porque la respuesta cita una fuente real.
- **Rodeos en lugar de «No lo sé»** (q47, q48): sobre la multa del RGPD responde «no se especifica
  directamente en los documentos… sin embargo, se menciona…» y rellena con lo que encuentra. Está
  medio bien, porque reconoce el límite, pero no usa la frase fija.
- La métrica del dato clave es conservadora. En q16 el modelo escribe `$10^{25}$` en LaTeX y no
  cuenta; en q17 y q18 parafrasea bien y cita el fragmento correcto sin usar la palabra clave.

**Endpoint**: `/health` marca `rag: ready`. `POST /rag` funciona con `dense`, `bm25` e `hybrid`, y
las puntuaciones tienen sentido: híbrido en [0, 1], denso como coseno y BM25 sin acotar. En la
pregunta de las 24 horas, el híbrido y el denso aciertan con cita, y BM25 responde «No lo sé»
porque sus 5 fragmentos no la contienen. Ante «¿Cuál es la capital de Australia?» responde «No lo
sé». El chunking semántico embebe 9.288 frases (~190 s), así que los fragmentos se cachean en
`rag/index/chunks-<chunking>.json` con un hash del corpus: el segundo arranque tarda 4 s. Qwen3-4B
se comparte entre `/tools` y `/rag` (`rlm.gpu.shared_causal_lm`): dos copias más el modelo de
embeddings no caben en 16 GB. 11 tests nuevos sin red ni GPU (`tests/test_rag_pipeline.py`).

Ficheros: `reports/phase3_eval.json`, `phase3_answers.json`, `phase3_answers_closed_book.json`,
`phase3_sentence_distances.png`, `phase3_lambda_sweep.png`.
