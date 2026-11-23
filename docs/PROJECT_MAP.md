# Mapa del proyecto ARCA

**ARCA** → la práctica: clasifica sistemas de IA según el AI Act y cita el artículo. Cuatro fases, cada una con su endpoint. **(martes 24)**

**Proyecto final** → ARCA mejorado (limitaciones de ARCA y cómo las mejoramos). **(entrega 29)**

## Cómo encaja todo

```
motor de reglas (rlm/aiact_rules.py)  ← la ley en código
        │
Fase 1 · el cerebro      → piensa y clasifica         → POST /reasoning
Fase 2 · las manos       → herramientas               → POST /tools
Fase 3 · la biblioteca   → busca en el corpus y cita  → POST /rag
        │
Fase 4 · el agente       → usa 1 + 2 + 3 en un bucle  → POST /agent
```

## Las fases

- **Fase 1 · Razonamiento** · `rlm/` → Qwen3-0.6B con SFT + GRPO, 75,5 % en test.
- **Fase 2 · Herramientas** · `tool_use/` → 5 herramientas, Qwen3-4B elige bien el 97 %.
- **Fase 3 · RAG** · `rag/` → búsqueda híbrida + reranker, cita y dice «No lo sé».
- **Fase 4 · Agente** · `agent/` → bucle, memoria, 20 tareas, visor, 4 cerebros.
- **API** · `api/` → arranca siempre; una fase sin terminar responde 501.

Detalle → `README.md` · `docs/EXPERIMENTS.md` (fases 1-2) · `rag/README.md` · `docs/PHASE4_LOG.md`

---

## Día de la evaluación: levantar la API

**1. Sesión en la DGX**
→ code-server con GPU (la de 16 GB basta: `/tools` y `/rag` generan en el servidor de la
  universidad y el agente usa el cerebro `remote`; en la GPU solo van el buscador, el reranker
  y los modelos pequeños, unos 9 GB).

**2. Preparar** (terminal, en la carpeta del proyecto)
```bash
cd ~/clusters/dgx/dgm-arca && source smoke/dgx_env.sh
git checkout main && git pull
uv sync --extra train --extra rag --extra agent
ls rlm/weights                # tiene que estar final_rlm_lora (si no, ver README, «Pesos»)
```
→ En `.env` tienen que estar: `ARCA_API_PORT=8077`, `LLM_API_KEY=...`, `LLM_BACKEND=openai`,
  `ARCA_TOOLS_BACKEND=openai` y `ARCA_API_ROOT_PATH=/rnode/dgx/<SESIÓN>/proxy/8077`.
→ `<SESIÓN>` = el número de la dirección de code-server. **Cambia en cada sesión**:
```bash
sed -i 's|/rnode/dgx/[0-9]*/|/rnode/dgx/<SESIÓN>/|' .env
```

**3. Levantar** (el 8000 lo usa otro equipo: usamos el 8077)
```bash
pkill -f arca-api; sleep 3                       # por si quedaba una API antigua
uv run arca-api > /tmp/arca-api.log 2>&1 &
```

**4. Calentar** (carga todos los modelos; la primera vez tarda varios minutos)
```bash
uv run python -m api.warmup
```
→ Las 5 líneas tienen que empezar por `OK`.

**5. Enseñarla**
→ Navegador: `https://hpc.comillas.edu/rnode/dgx/<SESIÓN>/proxy/8077/docs`
  → endpoint → **Try it out** → pegar el JSON → **Execute**.
→ En `/agent`, usar `"brain": "remote"` (es el que clasifica bien).

**Si algo falla:**
→ `curl -s localhost:8077/health | python3 -m json.tool` → qué fase falla y por qué.
→ `tail -30 /tmp/arca-api.log` → el error completo.
→ `CUBLAS_STATUS_ALLOC_FAILED` = GPU llena → `pkill -f arca-api` y volver a levantar.

---

## Si nos preguntan cómo funciona

**/reasoning · Fase 1**
→ Qwen3-0.6B piensa paso a paso (`<think>`) y da la etiqueta + el artículo.
→ **SFT**: un modelo grande genera razonamientos, el motor de reglas descarta los que fallan y
  entrenamos con los buenos (LoRA: solo una capa pequeña encima, no el modelo entero).
→ **GRPO**: el modelo responde varias veces a la misma pregunta, se premian las respuestas con
  buen formato y la etiqueta correcta, y aprende de comparar unas con otras.

**/tools · Fase 2**
→ El modelo que elige es el grande del servidor de la universidad (vLLM); las herramientas se
  ejecutan aquí.
→ El modelo recibe la lista de herramientas (nombre, para qué sirve, qué argumentos pide).
→ Decide cuál usar y con qué argumentos → **nuestro código la ejecuta** → el resultado vuelve al
  modelo → escribe la respuesta.
→ Herramientas: clasificar, buscar un artículo, multa máxima, calculadora, informe PDF.
→ Por qué: el modelo no se inventa cifras; las cuentas las hace código que no falla.

**/rag · Fase 3**
→ El reglamento está troceado en fragmentos.
→ Búsqueda **híbrida**: por palabras (BM25) + por significado (embeddings) → un **reranker**
  reordena y se queda con los mejores.
→ El modelo grande de la universidad responde **solo con esos fragmentos** y los cita; si no está ahí → «No lo sé».
→ Por qué: evita que se invente artículos.

**/agent · Fase 4**
→ Bucle **ReAct**: piensa → usa una herramienta → mira el resultado → repite hasta responder.
→ Junta las herramientas de la Fase 2 y el buscador de la Fase 3.
→ Si se equivoca al llamar a una herramienta, el error vuelve al modelo y se corrige.
→ Memoria: si la conversación crece demasiado, recorta resultados viejos y resume lo antiguo.
→ Se puede elegir el cerebro: `base` (0.6B), `rlm` (0.6B + Fase 1), `thinking` (1.7B) o
  `remote` (el grande de la universidad).
→ Los pequeños fallan al clasificar (no rellenan bien los datos de la herramienta); el grande
  acierta con el mismo bucle → el límite era el tamaño del modelo, no el agente.

**Preguntas típicas**
- ¿Por qué modelos tan pequeños? → caben en la GPU, entrenan rápido y se ve el efecto del
  entrenamiento (`base` vs `rlm`). Donde no llegan (elegir herramientas, redactar), usamos el
  modelo grande de la universidad.
- ¿Por qué la primera llamada tarda? → carga el modelo en la GPU; por eso calentamos antes.
- ¿Qué pasa si una fase falla? → la API sigue funcionando; ese endpoint devuelve un error claro.
- ¿Cómo sabéis que acierta? → conjunto de test (Fase 1: 75,5 %) y 20 tareas del agente.
