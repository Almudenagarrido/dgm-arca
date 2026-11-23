"""Plugs the tools of phases 2 and 3 into the agent's toolbox.

Phase 2 provides ``classify_risk``, ``lookup_ai_act``, ``compute_max_fine``,
``write_assessment_report`` and ``calculator`` (``tool_use.tools.default_registry``); phase 3
provides the retriever behind ``search_knowledge_base`` (``rag.generate.RagPipeline``). This
module wraps them as ``ToolSpec`` objects. It is the only part of the agent that depends on how
the other phases are written.
"""

from __future__ import annotations

import sys
from typing import Any

from agent.toolbox import Toolbox, ToolSpec

# The first search loads the embedder and the reranker and indexes the corpus: minutes, not s.
SEARCH_TIMEOUT = 600.0
REPORT_TIMEOUT = 60.0
MAX_FRAGMENT_CHARS = 700

SEARCH_DESCRIPTION = (
    "Search the AI Act and the official guidelines (the knowledge base of the project) and "
    "return the most relevant fragments with their source. Use it for questions about dates, "
    "definitions, obligations or anything the other tools do not compute. Write the query as a "
    "short question in Spanish."
)
SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look for, as a short question."},
        "top_k": {"type": "integer", "description": "Number of fragments (1-5).", "default": 3},
    },
    "required": ["query"],
}


def phase2_specs(registry=None) -> list[ToolSpec]:
    """The phase 2 tools as ``ToolSpec``: same name, description and JSON Schema."""
    if registry is None:
        from tool_use.tools import default_registry

        registry = default_registry()
    specs = []
    for tool in registry.tools.values():
        timeout = REPORT_TIMEOUT if tool.requires_confirmation else 10.0
        specs.append(
            ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=tool.definition["function"]["parameters"],
                # Calling the Tool validates the arguments with its Pydantic model; a
                # ValidationError becomes a failed observation in Toolbox.run.
                fn=tool,
                timeout=timeout,
                requires_confirmation=tool.requires_confirmation,
            )
        )
    return specs


_pipeline = None


def rag_pipeline():
    """The phase 3 pipeline. Inside the API it is the one /rag uses, so models load only once."""
    global _pipeline
    app = sys.modules.get("api.app")
    if app is not None:
        return app.backends.rag()
    if _pipeline is None:
        from rag.generate import build_pipeline

        _pipeline = build_pipeline()
    return _pipeline


def make_search(get_pipeline=rag_pipeline):
    """``search_knowledge_base``: retrieval only (no generation), the agent writes the answer."""

    def search_knowledge_base(query: str, top_k: int = 3) -> list[dict[str, Any]]:
        top_k = max(1, min(int(top_k), 5))
        hits = get_pipeline().retrieve(query, top_k=top_k, retriever="hybrid")
        return [
            {
                "source": hit.chunk.title,
                "text": hit.chunk.text[:MAX_FRAGMENT_CHARS],
                "score": round(float(hit.score), 3),
            }
            for hit in hits
        ]

    return search_knowledge_base


def search_spec(get_pipeline=rag_pipeline) -> ToolSpec:
    return ToolSpec(
        name="search_knowledge_base",
        description=SEARCH_DESCRIPTION,
        parameters=SEARCH_PARAMETERS,
        fn=make_search(get_pipeline),
        timeout=SEARCH_TIMEOUT,
    )


def default_toolbox() -> Toolbox:
    """The toolbox of the API: the five phase 2 tools and the phase 3 search."""
    return Toolbox().add(*phase2_specs(), search_spec())
