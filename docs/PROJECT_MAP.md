# Mapa del proyecto ARCA

**ARCA** → la práctica: clasifica sistemas de IA según el AI Act y cita el artículo. Cuatro fases, cada una con su endpoint. **(martes 24)**

**Proyecto final** → ARCA mejorado (limitaciones de ARCA y cómo las mejoramos). (entrega 29)**

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
- **Fase 4 · Agente** · `agent/` → bucle, memoria, 20 tareas, visor.
- **API** · `api/` → arranca siempre; una fase sin terminar responde 501.

Detalle → `README.md` · `docs/EXPERIMENTS.md` (fases 1-2) · `rag/README.md` · `docs/PHASE4_LOG.md`

---

## Levantar la API

**1. Sesión en la DGX** (con antelación, puede haber cola)
→ code-server · perfil **Entrenamiento** · GPU **71 GiB** · duración de sobra.
  Las cuatro fases cargan varios modelos a la vez y en 16 GB no caben.

**2. Preparar** (terminal, en la carpeta del proyecto)
```bash
cd ~/clusters/dgx/dgm-arca && source smoke/dgx_env.sh
git checkout main && git pull
uv sync --extra train --extra rag --extra agent
curl -L -o /tmp/rlm-weights.tar.gz https://github.com/Almudenagarrido/dgm-arca/releases/download/rlm-weights/arca-v1-rlm-weights.tar.gz
mkdir -p rlm/weights && tar -xzf /tmp/rlm-weights.tar.gz -C rlm/weights
cp .env.example .env          # comprobar ARCA_API_PORT=8077 y ARCA_RLM_ADAPTER
```

**3. Levantar** (8000 / 8077)
```bash
uv run arca-api &
```

**4. Calentar** (carga todos los modelos; la primera vez tarda varios minutos)
```bash
uv run python -m api.warmup
```
→ Las 5 líneas tienen que empezar por `OK`.

**5. Enseñarla**
→ Navegador: `https://hpc.comillas.edu/rnode/dgx/<SESIÓN>/proxy/8077/docs`
  (`<SESIÓN>` = el número que sale en la dirección de code-server; lanzar la API con
  `uv run uvicorn api.app:app --host 0.0.0.0 --port 8077 --env-file .env --root-path /rnode/dgx/<SESIÓN>/proxy/8077`).
→ Visor del agente: `uv run python -m agent.ui`

**Si algo falla:** `curl -s localhost:8077/health | python3 -m json.tool` → dice qué fase falla y por qué.

---

## Preguntas sobre funcionamiento

**/reasoning · Fase 1**
→ Qwen3-0.6B piensa paso a paso (`<think>`) y da la etiqueta + el artículo.
→ **SFT**: un modelo grande genera razonamientos, el motor de reglas descarta los que fallan y
  entrenamos con los buenos (LoRA: solo una capa pequeña encima, no el modelo entero).
→ **GRPO**: el modelo responde varias veces a la misma pregunta, se premian las respuestas con
  buen formato y la etiqueta correcta, y aprende de comparar unas con otras.

**/tools · Fase 2**
→ El modelo recibe la lista de herramientas (nombre, para qué sirve, qué argumentos pide).
→ Decide cuál usar y con qué argumentos → **nuestro código la ejecuta** → el resultado vuelve al
  modelo → escribe la respuesta.
→ Herramientas: clasificar, buscar un artículo, multa máxima, calculadora, informe PDF.
→ Por qué: el modelo no se inventa cifras; las cuentas las hace código que no falla.

**/rag · Fase 3**
→ El reglamento está troceado en fragmentos.
→ Búsqueda **híbrida**: por palabras (BM25) + por significado (embeddings) → un **reranker**
  reordena y se queda con los mejores.
→ El modelo responde **solo con esos fragmentos** y los cita; si no está ahí → «No lo sé».
→ Por qué: evita que se invente artículos.

**/agent · Fase 4**
→ Bucle **ReAct**: piensa → usa una herramienta → mira el resultado → repite hasta responder.
→ Junta las herramientas de la Fase 2 y el buscador de la Fase 3.
→ Si se equivoca al llamar a una herramienta, el error vuelve al modelo y se corrige.
→ Memoria: si la conversación crece demasiado, recorta resultados viejos y resume lo antiguo.
→ Se puede elegir el cerebro: `base`, `rlm` (el de la Fase 1) o `thinking` (1.7B).

**Preguntas típicas**
- ¿Por qué modelos tan pequeños? → caben en la GPU, entrenan rápido y se ve el efecto del
  entrenamiento (`base` vs `rlm`).
- ¿Por qué la primera llamada tarda? → carga el modelo en la GPU; por eso calentamos antes.
- ¿Qué pasa si una fase falla? → la API sigue funcionando; ese endpoint devuelve un error claro.
- ¿Cómo sabéis que acierta? → conjunto de test (Fase 1: 75,5 %) y 20 tareas del agente.