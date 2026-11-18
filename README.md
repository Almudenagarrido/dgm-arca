# ARCA

Agente con Razonamiento, Conocimiento y Acción: un asistente que clasifica sistemas de IA según
el AI Act (Reglamento (UE) 2024/1689). Es la práctica final de Modelos Generativos Profundos
(MIA, ICAI).

El repositorio se ha **reiniciado** el 18-nov-2026. Ahora solo contiene el corpus oficial, y el
proyecto se rehace a partir de él (plan en [`docs/CRONOGRAMA.md`](docs/CRONOGRAMA.md)).

## Qué hay

```
rag/corpus/              149 documentos Markdown (~2 MB): AI Act en español + directrices de la Comisión
rag/CORPUS.md            fuentes y licencias
rag/download_corpus.py   regenera el corpus desde las fuentes oficiales
tool_use/ai_act_text.py  descarga y parsea el XHTML oficial del reglamento (lo usa download_corpus)
docs/CORPUS_EXTRACCION.md  cómo se descargó y limpió el corpus, y los problemas que hubo
```

Para regenerar el corpus:

```bash
uv sync
uv run python -m rag.download_corpus              # sobrescribe rag/corpus/
uv run python -m rag.download_corpus --out /tmp/c # o en otra carpeta, para comparar
```

## La versión anterior (v1)

Todo lo anterior al reinicio (fases 0 a 4: motor de reglas, SFT + GRPO, tool use, RAG y agente)
sigue en git:

| Qué | Dónde |
|---|---|
| Código completo de la v1 | etiqueta `arca-v1` (`git checkout arca-v1`) |
| Variante de fase 4 con el bench de 22 tareas y `docs/RESUMEN.md` | rama `fase4-victoria` |
| Traducción al inglés en curso | rama `english` |
| Pesos LoRA de la v1 (GRPO final y SFT), sin checkpoints intermedios | Hugging Face: `arca-mia/arca-v1-rlm-grpo`, `arca-mia/arca-v1-rlm-sft` (privados, **pendiente de subir**) |
| Logs, informes e índice del RAG que no estaban en git | Hugging Face (dataset): `arca-mia/arca-v1-artefactos` (privado, **pendiente de subir**) |

Los pesos y artefactos de la v2 irán en repos aparte de la misma organización (`arca-mia/arca-v2-*`).
