# Fase 4 — El agente: todo junto en un bucle

**Endpoint que se evalúa:** `POST /agent` · **Clase:** The augmented LLM, bloque de agentes

## De qué va

Un agente LLM es un modelo con herramientas que actúa en bucle: piensa, actúa, observa, y
vuelve a pensar hasta que puede responder. Es el patrón ReAct que vimos en clase, y es donde
se juntan las tres fases anteriores: el cerebro es vuestro modelo de razonamiento, las manos
son vuestras herramientas, y la base de conocimiento es una herramienta más. Vimos también la
diferencia entre agentes que emiten JSON y agentes que escriben código, y jugamos con
`smolagents`. Aquí escribís el vuestro.

## Lo que tendréis que hacer

**1. El bucle ReAct.** En `agent/react_agent.py`: Pensamiento, Acción, Observación, hasta que
el modelo llama a `final_answer` o se agota el número de pasos. Con límite de tiempo por
herramienta y recuperación de errores: si una herramienta falla, el agente tiene que
enterarse y decidir, no morir. Reutilizad lo que ya tenéis: el parser y el registro de la
fase 2, y la memoria de `agent/memory.py`. El retriever de la fase 3 entra como la herramienta
`search_knowledge_base`.

**2. Un cerebro intercambiable.** El agente tiene que funcionar con el modelo base
(`brain="base"`), con vuestro modelo de la fase 1 (`brain="rlm"`) y con un modelo thinking
(`brain="thinking"`). Y vais a comparar los tres en el banco de tareas. Es una de las
preguntas más interesantes de la práctica: ¿ayuda de verdad haber entrenado el modelo para
razonar cuando lo pones a actuar?

**3. Memoria.** `ConversationMemory` ya cuenta tokens y sabe comprimir; lo que decidís
vosotros es la política: qué se resume, cuándo, qué se conserva siempre. Recordad el efecto
"perdido en el medio": meter todo en el contexto no es la solución.

**4. Un banco de tareas.** Una veintena en `agent/bench/tasks.jsonl` (formato en
`example_tasks.jsonl`) que requieran encadenar al menos dos herramientas, con una forma
automática de comprobar la respuesta final. `agent/evaluate.py` mide tasa de éxito, pasos y
tokens por cerebro. Y comparad vuestro agente JSON con un agente de código: aquí sí podéis
usar el `CodeAgent` de `smolagents` como contraste, y contarme si se cumple lo que decía el
artículo de CodeAct.

**5. La traza a la vista.** Cada ejecución devuelve todos los pasos, y `agent/ui.py` los
pinta como tarjetas de colores, igual que el notebook de clase. Es vuestra herramienta de
depuración y lo que enseñaréis en la demo.

## Para ir más allá

- **Multi-agente:** un planificador que delega en sub-agentes especializados de vuestro dominio.
- **Cerrar el círculo con la fase 1:** usar las trazas exitosas del agente como datos de SFT y
  medir si el agente mejora.
- **Guardarraíles:** detección de bucles (la misma llamada repetida), presupuesto de tokens
  por tarea, y qué pasa con las herramientas que piden confirmación.

## Cómo se corrige

Llamaré a `POST /agent` con tareas parecidas a las de vuestro banco, con alguna que requiera
la base de conocimiento y una herramienta, y con alguna imposible, y miraré:

- **Implementación (30 %):** el agente completa las tareas, la traza refleja lo que pasó de
  verdad (pensamiento, acción con argumentos, observación real), respeta `max_steps`, y los
  tres cerebros funcionan. Ante una tarea imposible, lo dice en vez de inventarse la respuesta.
- **Completitud (30 %):** bucle propio con `final_answer`, timeouts y recuperación de errores;
  base de conocimiento como herramienta; tres cerebros comparados; memoria con resumen; banco
  de veinte tareas con métricas; comparación JSON vs código; visor de trazas.
- **Interpretación (30 %):** qué cerebro gana y por qué; en qué tareas falla el agente y qué
  hace cuando falla; qué cambió con la memoria; qué visteis en la comparación con el agente
  de código; cómo reaccionó ante una herramienta rota.
- **Limpieza (10 %):** el agente arranca desde `docker compose up api` con las variables de
  `.env.example`, sin rutas absolutas ni secretos.

## Ficheros

| Fichero | Estado | Qué es |
|---|---|---|
| `prompts.py` | hecho + nuestro | Prompt ReAct de clase y `DOMAIN_RULES`: qué herramienta para qué y la frase fija «No puedo responder» |
| `memory.py` | hecho + nuestro | Memoria con presupuesto; nuestra política: tarea fijada, recorte de observaciones y luego resumen |
| `react_agent.py` | hecho | El bucle, los tres cerebros, `final_answer` y `search_knowledge_base`; detrás de `/agent` |
| `code_agent.py` | hecho | El contraste: `CodeAgent` de `smolagents` con los mismos cerebros, herramientas y tareas |
| `evaluate.py` | hecho + ampliado | Éxito, uso de herramientas, pasos, tokens, motivos de parada y éxito por familia |
| `ui.py` | hecho | Visor de trazas en Gradio contra la API |
| `bench/build_tasks.py` | hecho | Genera `bench/tasks.jsonl`: 20 tareas, todas con al menos dos herramientas, 2 imposibles |

## Cómo está hecho

**El bucle** (`react_agent.py`) es el de clase, Pensamiento → Acción → Observación, sin ningún
framework. Reutiliza el parser y el registro de la fase 2. Cada paso es una generación y, como
mucho, una llamada:

- **Una acción por paso.** Si el modelo escribe varios `<tool_call>`, se ejecuta el primero y la
  observación avisa de que los demás se ignoraron. Así `max_steps` cuenta acciones de verdad.
- **La generación se para en `</tool_call>`**, para que el modelo no se invente la observación.
  Se para solo fuera de `<think>` (`StopOutsideThinking`): `stop_strings` de transformers busca la
  cadena también dentro del razonamiento, y un modelo thinking que escriba un borrador de la
  llamada mientras piensa se cortaría antes de actuar.
- **Timeouts por herramienta** (`TOOL_TIMEOUTS`: 30 s para `lookup_ai_act`, que va a la red; 60 s
  para la base de conocimiento; 10 s para el resto). La herramienta corre en un hilo, y si no
  responde a tiempo el modelo recibe `Error: timeout` y decide qué hacer.
- **Recuperación de errores.** Herramienta desconocida, argumentos inválidos, JSON mal formado,
  una excepción dentro de la herramienta, un timeout, un `final_answer` vacío o una «Action» escrita
  en prosa sin `<tool_call>`: todo vuelve como observación que empieza por `Error:`. El bucle nunca
  lanza una excepción por culpa del modelo.
- **Detección de bucles.** Una llamada idéntica a otra anterior (mismo nombre y mismos argumentos)
  no se vuelve a ejecutar: el modelo recibe el resultado de antes y la instrucción de usarlo. A la
  tercera repetición la ejecución se para con `stop_reason="loop_detected"`.
- **Confirmación.** `write_assessment_report` tiene efectos (escribe un PDF). Con
  `ARCA_AGENT_CONFIRM=task` (por defecto) solo se ejecuta si la tarea pide un informe: la petición
  del usuario es la confirmación. `always` y `never` son las otras dos políticas.
- **Traza fiel.** Un bloque `<tool_call>` que no se puede leer queda en la traza como la acción
  `invalid_tool_call`, con el texto que escribió el modelo, junto al error que recibió.
- **Respuesta sin `final_answer`.** Se acepta como respuesta final y se marca con
  `stop_reason="plain_answer"`, para poder contarlas aparte en el banco.

**Los tres cerebros.** `base` es Qwen3-0.6B sin razonamiento, el modelo de partida de la fase 1.
`rlm` es el mismo modelo con nuestro adaptador GRPO (`ARCA_RLM_ADAPTER`), y razona con
`<think>` como en su entrenamiento. `thinking` es Qwen3-1.7B con razonamiento nativo
(`ARCA_THINKING_MODEL`). Los tres reciben el mismo prompt, las mismas herramientas y la misma
decodificación (greedy sin razonamiento; con razonamiento, el muestreo que recomienda la ficha de
Qwen3, porque en greedy se repite). Se cargan la primera vez que se piden.

**La base de conocimiento** es `search_knowledge_base`: la tubería de la fase 3 (híbrido con
λ = 0,3, consulta traducida y reranker), compartida con `/rag`. Devuelve los fragmentos como
`[id] (fuente)` con el texto recortado a 700 caracteres. Si ni el mejor supera el umbral de «No lo
sé» de la fase 3, lo dice al principio, para que el agente pueda responder que no lo sabe.

**La memoria** (`memory.py`) se comprime cuando la conversación supera el presupuesto
(`ARCA_AGENT_TOKEN_BUDGET`, 9000 tokens por defecto). El prompt de sistema solo ya ocupa 4817
(tokenizador de Qwen3), casi todo por las definiciones de las herramientas y sobre todo por los 40
campos de `classify_risk`. Lo que se conserva siempre: el
prompt de sistema, la tarea en curso (fijada) y los seis últimos mensajes, sin dejar nunca una
observación separada de la acción que la produjo. Primero se recortan las observaciones antiguas a
600 caracteres, que son casi todo el peso, y solo si eso no basta se resume la parte vieja con el
propio cerebro (`SUMMARY_PROMPT`). Con `session_id` la memoria sobrevive entre peticiones
(multi-turno), y la tarea anterior deja de estar fijada cuando llega una nueva.

**El banco** (`bench/build_tasks.py` → `tasks.jsonl`) tiene 20 tareas en seis familias:
clasificar + multa (5), clasificar + citar el texto (4), base de conocimiento + cálculo o
clasificación (5), clasificar + informe PDF (2), dos multas + calculadora (2) e imposibles (2: un
dato que el corpus no tiene y un artículo que no existe). Cada respuesta esperada se calculó con las
herramientas reales o se leyó en el corpus. Las comprobaciones son automáticas (`regex`, `number`,
y combinaciones con `all`/`any`), y un test verifica que cada una acepta la respuesta de referencia
y rechaza «No puedo responder».

**El agente de código** (`code_agent.py`) es el contraste: `smolagents.CodeAgent`, que escribe
Python en lugar de JSON, con nuestros cerebros (un `Model` que llama a `ReActAgent.generate`) y
nuestras herramientas envueltas como `smolagents.Tool`, que se ejecutan a través de
`ReActAgent.execute` (misma validación, mismos timeouts, misma confirmación). Solo cambian el
formato de la acción y el prompt, que es el de `smolagents` con nuestras `DOMAIN_RULES`.

## Cómo se ejecuta

```bash
uv run python -m agent.bench.build_tasks                       # regenera tasks.jsonl
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base rlm thinking
uv run python -m agent.code_agent --bench agent/bench/tasks.jsonl --brains thinking
uv run python -m agent.ui                                      # visor, con la API levantada
```

```bash
curl -s -X POST localhost:8000/agent -H 'Content-Type: application/json' -d '{
  "task": "Un banco con 5.000 millones de facturación puntúa la solvencia de quien pide un préstamo. Clasifícalo y calcula la multa máxima.",
  "brain": "thinking", "max_steps": 10}'
```

La respuesta añade tres campos opcionales al contrato: `stop_reason`, `memory_compressions` y, en la
petición, `session_id`. Los resultados y su análisis están en `docs/EXPERIMENTS.md`.
