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
- `connectors.py` → único punto que depende de las fases 2 y 3.

### Visor de trazas · `agent/ui.py`
- Página con cada paso en colores: pensamiento → acción → observación → respuesta. Es la demo.
- 2 tests.

---

## Etapa 2 · Proyecto final

*(A partir del 24.)*