# Cuaderno de la Fase 4 · Agente

→ **Etapa 1 · ARCA** (hasta el 24): el agente con los mínimos del enunciado.
→ **Etapa 2 · Proyecto final** (24-29): mejoras sobre las limitaciones de ARCA.

---

## Etapa 1 · ARCA

### Prompts · `agent/prompts.py`
- Bucle ReAct en dos versiones: **nativa** (las herramientas las pone la plantilla de Qwen, por
  defecto) y **texto** (pegadas en el prompt, como en clase).
- Reglas del dominio: qué herramienta para qué, «No puedo responder» si no hay datos.
- **Sin ejemplos de argumentos**: los modelos pequeños los copian.
- Experimento pendiente → 4 prompts (nativo/texto × con/sin ejemplo) con cada cerebro: ¿llamadas
  bien formadas? ¿copia el ejemplo «Art. 50.1»?

### Memoria · `agent/memory.py`
- Problema → la conversación crece en cada paso y no cabe o el modelo se pierde.
- Solución → **buffer + resumen**, con presupuesto de 9000 tokens:
  tarea fija → últimos mensajes enteros → recortar resultados viejos → resumir lo viejo.
- Con `session_id` recuerda entre peticiones (multi-turno).
- 7 tests. Experimento pendiente → tareas largas para medirla.

### Banco y evaluación · `agent/bench/`, `agent/evaluate.py`
- **20 tareas** que obligan a usar 2 herramientas: clasificar + multa, clasificar + citar, RAG +
  calcular, clasificar + informe, dos multas, y 2 imposibles.
- Respuestas esperadas **comprobadas** con el motor de reglas y el reglamento.
- Métricas → éxito (con intervalo de confianza), herramientas usadas, pasos, tokens, tiempo.
- 6 tests.

### Bucle ReAct · `agent/react_agent.py`, `toolbox.py`, `brains.py`
- Propio, sin frameworks. **3 cerebros**: `base` (0.6B), `rlm` (0.6B + fase 1), `thinking` (1.7B).
  → `base` vs `rlm` = efecto de nuestro entrenamiento.
- Una acción por paso; la generación se corta en `</tool_call>`.
- **Los errores vuelven al modelo** como observación → se corrige, el bucle no se rompe.
- Tiempo máximo por herramienta, detección de bucles, el PDF solo si se pide.
- 14 tests con un cerebro falso.

### Conexión con las fases 2 y 3 · `agent/connectors.py`
- Las 5 herramientas de Gloria tal cual + `search_knowledge_base` (solo la búsqueda del RAG de
  Victoria; la respuesta la escribe el agente).
- Dentro de la API usa el mismo buscador que `/rag` → los modelos se cargan una vez.
- 4 tests.

### Visor de trazas · `agent/ui.py`
- Página con cada paso en colores: pensamiento → acción → observación → respuesta. Es la demo.
- 2 tests.

---

### Primera prueba real (23 nov)
- Las 4 fases responden (`api/warmup.py`). Caso: banco, modelo de solvencia, 3.000 M.
- Con `thinking` (1.7B) la multa sale bien (90 M, la calcula la herramienta), pero la
  clasificación no:
  - busca un anexo antes de clasificar y se lo inventa (III.4.a, empleo);
  - llama a `classify_risk` con los 40 datos y **escribe él mismo el resultado**;
  - el JSON lleva una `}` de más → error → no reintenta → responde con lo inventado.
- Arreglos en el bucle:
  - la llamada se lee aunque sobre algo detrás;
  - reglas nuevas: clasificar antes de citar, poner solo los datos ciertos, reintentar si
    hay error, nunca responder con algo que no devolvió una herramienta.
- Con eso ya clasifica primero, pero pone todo a `false` → `riesgo_minimo`, que es incorrecto.
  → El límite es el tamaño del modelo: traducir una descripción a 40 datos es demasiado para
  un 1.7B.

---

## Etapa 2 · Proyecto final

### Mejora 1 · Cerebro `remote` (modelo grande de la universidad)
- Limitación → el 1.7B no rellena bien los datos de `classify_risk`.
- Mejora → cuarto cerebro: el modelo grande del servidor vLLM (el mismo que ya usan `/tools` y
  `/rag`, con el cliente de Victoria). No ocupa GPU propia.
- Resultado en el caso del banco, mismo bucle y mismas herramientas:
  `classify_risk` con solo `["5.b"]` → **alto riesgo, Anexo III.5.b** → multa 90 M → respuesta
  completa en 3 pasos.
- Pendiente → medirlo con las 20 tareas frente a `base`, `rlm` y `thinking`.
- Resultado con las 20 tareas: **18/20 (90 %, IC 95 % 70-97 %)**, 3,2 pasos y ~7 s por tarea,
  las 2 imposibles rechazadas, ningún bucle.

### Mejora 2 (pendiente) · Confusión con «reconocimiento de emociones»
- Limitación → los 2 fallos de `remote` tienen la misma causa: el modelo marca
  `emotion_recognition` cuando la tarea habla de vigilar un comportamiento.
  - Examen online que detecta si copian → dice **prohibido** (Art. 5.1.f); es **alto riesgo**
    (Anexo III.3.d) y la multa sería de 15 M, no de 35 M.
  - Detector de mentiras en asilo → dice alto riesgo por III.1.c (emociones); es un
    **polígrafo** (III.7.a).
- Ideas → aclarar la descripción de ese dato en `classify_risk`, o un paso que verifique la
  clasificación antes de responder.

### Limitación conocida · Corpus incompleto (detectada por Victoria, 23 nov)
- 19 ficheros de artículos del corpus tienen el texto incompleto.
- Afecta a las respuestas que se basan en esos artículos (`/rag` y `search_knowledge_base`
  del agente); la clasificación y las multas no, porque salen del motor de reglas.
- Regenerarlo lleva ~3 h con la evaluación → se arregla en el proyecto final.

### Comparación de cerebros (20 tareas, 23 nov)
- remote 90 % (IC 70-97) · thinking 30 % (15-52) · rlm 10 % (3-30) · base 5 % (1-24).
- base casi no usa herramientas (18/20 responde directamente).
- rlm sí las usa pero se atasca (6 bucles detectados, 6 sin pasos): la Fase 1 lo entrenó para
  razonar, no para usar herramientas. Las protecciones del bucle funcionaron.
- base vs rlm: los intervalos se solapan → con 20 tareas no se ve efecto de la Fase 1.
- Idea para el proyecto final → SFT del modelo pequeño con las trazas buenas de remote.
