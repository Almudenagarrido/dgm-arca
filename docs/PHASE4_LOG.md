# Cuaderno de la Fase 4 · Agente

Qué se ha hecho en la fase 4, por qué y qué se ha aprendido, separado en las dos etapas del
proyecto. Sirve de base para la memoria y la presentación.

- **Etapa 1 · ARCA (hasta el 24 de noviembre):** construir el agente y cumplir los mínimos del
  enunciado (bucle ReAct, tres cerebros, memoria, banco de 20 tareas, agente de código, traza).
- **Etapa 2 · Proyecto final (24-29 de noviembre):** mejorar las limitaciones encontradas en ARCA.

---

## Etapa 1 · ARCA

### 2026-11-22 · Prompts del agente (`agent/prompts.py`)

**Qué:** las instrucciones que recibe el modelo. Cuatro textos: el bucle ReAct en versión
"nativa" (las herramientas las añade la plantilla de chat de Qwen), el mismo bucle en versión
"texto" (las herramientas pegadas, como en las diapositivas), las reglas del dominio y el prompt
de resumen de la memoria.

**Decisiones:**
- **Plantilla nativa por defecto.** Qwen3 se entrenó para llamar a herramientas con su propia
  plantilla de chat, así que es de esperar que siga mejor ese formato que uno escrito a mano. Se
  guarda la versión de texto para comprobarlo.
- **Sin ejemplos de argumentos en las reglas.** Los modelos pequeños tienden a copiar los ejemplos
  del prompt en sus llamadas; un ejemplo como «Art. 50.1» podría acabar como argumento de tareas
  que no tienen nada que ver.
- **Frase fija «No puedo responder»** para las tareas imposibles, para que el banco pueda
  comprobar automáticamente que el agente no se inventa la respuesta.

**Experimento previsto: formato del prompt y copia de ejemplos.** Se guardan las dos versiones de
las reglas: `DOMAIN_RULES` (sin ejemplo) y `DOMAIN_RULES_WITH_EXAMPLE` (con «Art. 50.1»). Se
comparan 4 prompts (texto / nativo × con / sin ejemplo) **con cada cerebro**, en el primer paso de
las 20 tareas del banco, midiendo:
- llamadas bien formadas,
- llamadas a la herramienta correcta,
- llamadas que contienen «Art. 50.1» en los argumentos (copia del ejemplo).

Preguntas: ¿el formato nativo da más llamadas bien formadas que el de texto? ¿Copiar el ejemplo
es cosa del modelo pequeño o le pasa también al grande?
*(Resultado: pendiente.)*

### 2026-11-22 · Memoria con presupuesto de tokens (`agent/memory.py`)

**Problema:** en cada paso el agente envía al modelo toda la conversación, que crece con cada
llamada a herramienta. Si se pasa de la ventana de contexto falla, y aunque quepa, con mucho
texto el modelo pierde lo del medio.

**Diseño: buffer + resumen acumulativo.** Mientras la conversación quepa en el presupuesto
(`ARCA_AGENT_TOKEN_BUDGET`, 9000 tokens por defecto), no se toca. Al pasarse, de lo más barato
a lo más caro:
1. La tarea del usuario se fija y nunca se resume.
2. Los últimos mensajes se quedan literales, sin separar nunca un resultado de la llamada que
   lo produjo.
3. **Recorte** de las observaciones viejas largas (artículos, listas de fragmentos). No cuesta
   nada; si basta, no se resume.
4. **Resumen** de la parte vieja con el propio modelo (`SUMMARY_PROMPT`: conservar números,
   resultados de herramientas y decisiones). El resumen se acumula al anterior y va arriba, en
   el prompt de sistema.

Con un `session_id` la memoria se conserva entre peticiones (multi-turno).

**Alternativas descartadas:** ventana deslizante (olvidaría, por ejemplo, la clasificación del
primer paso cuando se necesita al final) y memoria con buscador vectorial (desproporcionada para
conversaciones de pocos pasos).

**Verificación:** 7 tests sin modelo ni GPU (`tests/test_memory.py`, los tokens se cuentan como
palabras): no actúa mientras cabe, nunca resume la tarea, deja literales los últimos mensajes, no
separa llamada y resultado, recorta antes de resumir, el resumen se acumula y va al prompt de
sistema.

**Experimento previsto:** el banco de 20 tareas se resuelve en pocos pasos, así que probablemente
no llene la memoria. Para medirla hacen falta tareas largas o multi-turno, comparando esta
política con no comprimir y con resumir siempre. *(Pendiente.)*

### 2026-11-22 · Banco de 20 tareas y evaluación (`agent/bench/`, `agent/evaluate.py`)

**Qué:** 20 tareas en español que obligan a encadenar al menos dos herramientas distintas, cada
una con una comprobación automática de la respuesta final.

| Familia | Nº | Herramientas | Ejemplo |
|---|---|---|---|
| `classify-fine` | 5 | clasificar + multa | voz de teleoperadores para detectar estrés (800 M€) |
| `classify-lookup` | 4 | clasificar + citar la ley | detector de mentiras en entrevistas de asilo |
| `kb-*` | 5 | RAG + calculadora / multa / clasificar | ¿desde cuándo se aplica el reglamento? → meses |
| `classify-report` | 2 | clasificar + informe PDF | triaje en urgencias |
| `fine-calc` | 2 | dos multas + calculadora | diferencia entre dos multas máximas |
| `impossible` | 2 | — | el agente debe responder «No puedo responder» |

**Decisiones:**
- **Valores esperados comprobados**, no supuestos: las etiquetas y las multas se calcularon con el
  motor de reglas y la herramienta de multas; las fechas y umbrales salen del reglamento (Art. 4,
  6.1, 99 y 113).
- **Comprobaciones tolerantes al formato:** un importe vale escrito como 56.000.000, 56,000,000,
  56000000 o «56 millones», pero no 156.000.000 ni 5.600.000. Así no se penaliza al agente por la
  forma, solo por el contenido.
- **Tareas imposibles** para medir si el agente se inventa respuestas.
- **Métricas** (`evaluate.py`): tasa de éxito con intervalo de Wilson al 95 % (con 20 tareas los
  intervalos son anchos y hay que decirlo al comparar), % de tareas que usan todas las
  herramientas requeridas, pasos, tokens y segundos medios, motivos de parada y éxito por familia.
  Se guarda tras cada tarea y se puede reanudar (`--resume`).

**Verificación:** 6 tests sin modelo (`tests/test_bench.py`): las 20 respuestas de referencia
pasan su comprobación, responder «No puedo responder» a todo no pasa ninguna tarea posible, los
importes aceptan las formas habituales y rechazan las parecidas, y el resumen y el intervalo de
Wilson se calculan bien.

### 2026-11-22 · El bucle ReAct (`agent/react_agent.py`, `toolbox.py`, `brains.py`)

**Qué:** el bucle piensa → actúa → observa, propio y sin frameworks de agentes, repartido en tres
piezas con una responsabilidad cada una:

| Fichero | Responsabilidad |
|---|---|
| `toolbox.py` | Las herramientas: ejecutarlas por nombre con tiempo máximo y leer las llamadas `<tool_call>` del modelo |
| `brains.py` | Los cerebros: un modelo de Hugging Face, con o sin el adaptador LoRA de la fase 1 |
| `react_agent.py` | El bucle, la memoria y las políticas (errores, bucles, confirmación) |
| `connectors.py` | El único punto que depende de las fases 2 y 3: envuelve sus herramientas y su retriever |

**Decisiones:**
- **Los tres cerebros** del enunciado: `base` (Qwen3-0.6B sin entrenar), `rlm` (el mismo modelo con
  nuestro adaptador de la fase 1) y `thinking` (Qwen3-1.7B con razonamiento nativo). `base` y `rlm`
  son el mismo modelo, así que su comparación aísla el efecto del entrenamiento.
- **Una acción por paso:** si el modelo escribe varias llamadas, solo se ejecuta la primera y se le
  avisa. Así cada paso de la traza tiene una acción y `max_steps` cuenta pasos reales.
- **La generación se corta en `</tool_call>`**, para que el modelo no se invente la observación.
- **Los errores son observaciones:** herramienta desconocida, argumentos que faltan o sobran, JSON
  roto, excepción dentro de la herramienta, tiempo agotado o respuesta final vacía. El modelo
  recibe `Error: ...` y puede corregir; el bucle nunca se rompe por un fallo del modelo.
- **Tiempo máximo por herramienta:** cada herramienta se ejecuta en un hilo aparte; si no responde
  a tiempo, el modelo recibe un error de *timeout* y decide qué hacer.
- **Detección de bucles:** una llamada idéntica a otra anterior no se repite (se devuelve el
  resultado anterior con un aviso); tras 3 repeticiones se para con `loop_detected`.
- **Confirmación de efectos secundarios:** el informe PDF solo se genera si la tarea lo pide
  (política `task`; también `always` y `never`).
- **Respuesta sin herramienta:** se acepta como respuesta final, pero marcada (`plain_answer`). Si
  el texto contiene `Action:`, es una llamada mal escrita, no una respuesta, y recibe un error con
  el formato correcto.
- **Independiente de las otras fases:** el bucle solo necesita un cerebro que genere texto y una
  caja de herramientas. Por eso se ha podido construir y probar antes de tener las fases 1-3.

**Extras en la API** (campos opcionales, como permite la plantilla): `session_id` en la petición
(conversación de varios turnos) y `stop_reason` y `memory_compressions` en la respuesta.

**Verificación:** 14 tests con un cerebro falso que devuelve respuestas preparadas y herramientas
falsas (`tests/test_react_agent.py`): encadena dos herramientas y para en `final_answer`; se
recupera de cinco tipos de error; respeta el tiempo máximo y el límite de pasos; no repite
llamadas y detecta bucles; pide confirmación para el informe; acepta y marca las respuestas sin
herramienta; rechaza llamadas en prosa; ejecuta solo la primera de varias llamadas; avisa si el
razonamiento se corta; pasa las herramientas a la plantilla (modo nativo) o al prompt (modo
texto); y comparte la memoria entre peticiones de una misma sesión.

**Pendiente:** conectar en `connectors.py` las herramientas de la fase 2 y el retriever de la
fase 3, y probar con los modelos reales.

---

## Etapa 2 · Proyecto final

*(Pendiente: se rellena a partir del 24.)*
