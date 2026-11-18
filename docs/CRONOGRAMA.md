# Cronograma del reinicio (18-24 de noviembre de 2026)

## Punto de partida y reglas

- **Lo que ya está:** el corpus oficial (`rag/corpus/`, ver [`CORPUS_EXTRACCION.md`](CORPUS_EXTRACCION.md)).
  Todo lo demás se rehace desde cero.
- **Equipo:** tres personas, aquí **A**, **B** y **C**.
- **Fases secuenciales:** 1 Razonamiento → 2 Herramientas → 3 RAG → 4 Agente.
- **Regla de paso de fase:** una fase no empieza hasta que haya **puesta en común** de la anterior
  y **las tres** la entiendan. Ver «Puesta en común» al final.
- **Fecha límite:** todo acabado el **martes 24**. El martes 17 no cuenta. El miércoles 18 por la
  mañana es la reunión de arranque y se empieza a trabajar por la tarde.
- El calendario **incluye el fin de semana (21 y 22)**. Si no se trabaja, no da tiempo
  (ver «Si vamos tarde»).

## Cómo se ha repartido

Cada tarea tiene una dificultad, que cuenta tanto lo difícil que es técnicamente como el tiempo
que lleva:

| Dificultad | Puntos | Qué significa |
|---|---|---|
| 🔴 Alta | 3 | Pieza central de la fase: si falla, la fase no funciona. Requiere entender bien el problema |
| 🟡 Media | 2 | Trabajo técnico normal, con algo de depuración |
| 🟢 Baja | 1 | Mecánico o corto |

Criterios del reparto:

1. **Cada persona suma los mismos puntos: 20.**
2. **Cada persona lleva dos tareas rojas**, en fases distintas, así que la responsabilidad de las
   piezas difíciles rota.
3. **C se encarga de la GPU en la fase 1** (destilación, SFT y GRPO van seguidos y en la misma
   máquina), y a cambio tiene menos carga en la fase 2.
4. Cuando una tarea depende de otra, las dos van a la misma persona o la que espera empieza con
   una interfaz acordada (por ejemplo, el generador de B usa una firma del motor de A antes de
   que esté acabado).

### Resumen por persona

| | Tareas rojas | F1 | F2 | F3 | F4 | Cierre | **Total** |
|---|---|---|---|---|---|---|---|
| **A** | Motor de reglas (F1), bucle ReAct (F4) | 6 | 3 | 5 | 4 | 2 | **20** |
| **B** | Bucle de tool use (F2), conjunto dorado (F3) | 5 | 4 | 6 | 4 | 1 | **20** |
| **C** | GRPO (F1), umbral de «No lo sé» (F3) | 6 | 4 | 5 | 3 | 2 | **20** |

## Reparto por fase

### Fase 1 · Razonamiento (la más difícil: 17 puntos)

| Tarea | Dificultad | Quién | Por qué es así |
|---|---|---|---|
| Motor de reglas / verificador del dominio + tests | 🔴 3 | **A** | Todo lo demás depende de él: etiqueta los datos, verifica las trazas y puntúa en GRPO. Hay que aplicar bien la precedencia legal |
| Funciones de recompensa (formato, etiqueta, cita) | 🟡 2 | **A** | Usan el verificador; las hace quien lo conoce |
| Endpoint `/reasoning` + entrada en el cuaderno | 🟢 1 | **A** | |
| Generador de problemas (train, test, OOD) con varias plantillas | 🟡 2 | **B** | Debe evitar fugas entre train y test desde el diseño |
| Auditoría del dataset + revisión a mano de unos 50 casos | 🟡 2 | **B** | Lo revisa quien lo generó, pero con el motor de A como referencia |
| Script de evaluación (pass@1) + gráficas | 🟢 1 | **B** | |
| Entorno en la DGX + destilación con el profesor y filtrado | 🟡 2 | **C** | |
| SFT con LoRA | 🟡 2 | **C** | |
| Entrenamiento GRPO | 🔴 3 | **C** | El más inestable: hiperparámetros, memoria de GPU, recompensas que no suben. Es lo que más puede retrasar el calendario |

### Fase 2 · Herramientas (11 puntos)

| Tarea | Dificultad | Quién | Por qué es así |
|---|---|---|---|
| Las herramientas (consulta del AI Act en CELLAR, cálculo, generar informe...) | 🟡 2 | **A** | Puede reutilizar el motor de la fase 1 como herramienta de cálculo |
| Banco de casos con la respuesta esperada | 🟢 1 | **A** | |
| Parser de llamadas + bucle de tool use a mano (encadenadas y en paralelo, sin smolagents) | 🔴 3 | **B** | Es lo que el enunciado quiere ver; hay que tratar salidas mal formadas del modelo |
| Endpoint `/tools` + entrada en el cuaderno | 🟢 1 | **B** | |
| Registro: esquemas desde los type hints + validación de argumentos | 🟡 2 | **C** | |
| Evaluación (herramienta, argumentos, respuesta) con y sin herramientas | 🟡 2 | **C** | |

### Fase 3 · RAG (16 puntos; el corpus ya está)

| Tarea | Dificultad | Quién | Por qué es así |
|---|---|---|---|
| Embeddings, índice y retrievers (denso, BM25, híbrido) | 🟡 2 | **A** | |
| Recall@k y MRR por chunking y retriever | 🟢 1 | **A** | |
| Métricas de respuesta (correcta, citada, fiel) + juez LLM | 🟡 2 | **A** | |
| **Conjunto dorado**: preguntas escritas leyendo el corpus, evidencia exacta, preguntas sin respuesta | 🔴 3 | **B** | El enunciado dice que es «el trabajo de verdad» de esta fase. Es lento: hay que leer el corpus |
| Ingesta + dos chunkings (por estructura legal y semántico) | 🟡 2 | **B** | Quien lee el corpus para el dorado es quien mejor conoce su estructura |
| Endpoint `/rag` + entrada en el cuaderno | 🟢 1 | **B** | |
| Reranker + generación con citas | 🟡 2 | **C** | |
| Umbral de «No lo sé» + calibración con un conjunto aparte | 🔴 3 | **C** | Equilibrar no inventar con no negarse a todo es lo más delicado de la fase |

### Fase 4 · Agente (11 puntos)

| Tarea | Dificultad | Quién | Por qué es así |
|---|---|---|---|
| Bucle ReAct con smolagents + las herramientas de F2 y el retriever de F3 como herramientas | 🔴 3 | **A** | Junta todo; hay que entender las tres fases anteriores |
| Endpoint `/agent` + entrada en el cuaderno | 🟢 1 | **A** | |
| Tres «cerebros»: modelo de la F1, modelo base y uno con razonamiento nativo | 🟡 2 | **B** | |
| Memoria y recuperación de errores | 🟡 2 | **B** | |
| Banco de tareas del agente | 🟢 1 | **C** | |
| Evaluación + ablaciones + comparativa «¿ayuda haber entrenado?» | 🟡 2 | **C** | |

### Cierre (6 puntos + tarea común)

| Tarea | Dificultad | Quién |
|---|---|---|
| Informe final + repaso de `EXPERIMENTS.md` | 🟡 2 | **A** |
| README final (diagrama, tabla de métricas de las 4 fases, cómo levantarlo) | 🟢 1 | **B** |
| API completa (`/health` + 4 endpoints) + Docker | 🟡 2 | **C** |
| Prueba en limpio y revisión contra la rúbrica | — | **las tres** |

## Calendario día a día

### Miércoles 18

**Mañana · Reunión de arranque (las tres, ~3 h)**
- Leer juntas el enunciado y la rúbrica; repasar `CORPUS_EXTRACCION.md`.
- Decidir el tema y el usuario objetivo (¿se mantiene ARCA y el AI Act?) y los modelos.
- **A presenta un borrador de la interfaz del motor de reglas** (entradas y salida) para que B
  pueda empezar el generador esa misma tarde.
- Acordar la estructura del repo, el estilo de código y el formato del cuaderno de experimentos.
- Comprobar el acceso a la DGX.

**Tarde · Fase 1**
- A: motor de reglas. B: generador sobre la interfaz acordada. C: entorno en la DGX y piloto de
  destilación con unos pocos problemas.
- Por la noche: dataset v0 hecho y **destilación completa lanzada en la GPU**.

### Jueves 19 · Fase 1

| | Mañana | Tarde |
|---|---|---|
| A | Terminar motor + tests | Recompensas |
| B | Auditoría del dataset + revisión a mano | Script de evaluación |
| C | Filtrar trazas + SFT | Probar GRPO en pequeño; **lanzarlo de noche** |

### Viernes 20

- **Mañana:** C evalúa base, SFT y GRPO. B hace las gráficas. A hace el endpoint y el cuaderno.
- **13:00-14:30 · Puesta en común F1.**
- **Tarde · Fase 2:** A herramientas · B parser y bucle · C registro y validación.

### Sábado 21

- **Mañana · Fase 2:** A banco de casos · B llamadas encadenadas y en paralelo, endpoint · C evaluación con y sin herramientas.
- **16:00-17:30 · Puesta en común F2.**
- **Tarde · Fase 3:** B empieza el **conjunto dorado** y la ingesta + chunking · A embeddings y
  retrievers · C reranker.

### Domingo 22 · Fase 3

| | Mañana | Tarde |
|---|---|---|
| A | Recall@k y MRR | Métricas de respuesta + juez |
| B | Terminar el conjunto dorado | Endpoint `/rag` + cuaderno |
| C | Generación con citas | Umbral de «No lo sé» y calibración |

**19:00-20:30 · Puesta en común F3.**

### Lunes 23 · Fase 4

| | Mañana | Tarde |
|---|---|---|
| A | Bucle ReAct + herramientas | Endpoint `/agent` + cuaderno |
| B | Tres cerebros | Memoria y recuperación de errores |
| C | Banco de tareas | Evaluación y ablaciones |

### Martes 24 · Cierre y entrega

| Hora | Qué |
|---|---|
| 09:00-10:30 | **Puesta en común F4** |
| 10:30-14:00 | A informe · B README · C API y Docker |
| 14:00-16:00 | Prueba en limpio (las tres): clonar, levantar con una orden, llamar a cada endpoint |
| 16:00-18:00 | Revisión contra la rúbrica y últimos arreglos |
| **Antes de las 20:00** | **Entrega** (deja margen por si algo falla) |

## Puesta en común (al acabar cada fase, ~1,5 h)

1. **Cada persona explica a las otras dos lo que ha hecho** (10-15 min): qué hace el código, qué
   decisiones tomó y por qué, qué no funcionó.
2. **Demostración en vivo** del endpoint.
3. **Resultados:** la tabla de métricas y lo que significan.
4. **Preguntas cruzadas:** cada una explica **una parte que no ha hecho ella**, empezando por las
   rojas. Si alguna no puede, se repasa antes de seguir.
5. Confirmar el reparto de la fase siguiente.

Checklist para cerrar una fase:

- [ ] Los mínimos de la fase según el enunciado están hechos
- [ ] El endpoint responde
- [ ] Los tests pasan
- [ ] Hay entrada fechada en `docs/EXPERIMENTS.md` (incluido lo que no funcionó)
- [ ] El código está subido a la rama principal
- [ ] Las tres podemos explicar cualquier parte de la fase

## Si vamos tarde

- **El colchón es el martes 24 por la mañana.** Si la F4 se alarga, la puesta en común se hace en 45 min.
- **El mayor riesgo es el GRPO (C).** Si el viernes a las 13:00 no ha acabado, se hace la puesta en
  común con el SFT y GRPO sigue en segundo plano durante la F2. C tiene menos carga en la F2 para
  esto.
- **El conjunto dorado (B) es el segundo riesgo.** Si el domingo a mediodía no está, A y C escriben
  10 preguntas cada una y B las revisa.
- **Los trabajos de GPU van de noche.**
- Si no se trabaja el fin de semana: F2 y F3 se quedan en un día cada una y se quitan las
  ablaciones opcionales.
