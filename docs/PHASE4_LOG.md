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

---

## Etapa 2 · Proyecto final

*(Pendiente: se rellena a partir del 24.)*