"""Plugs the tools of phases 2 and 3 into the agent's toolbox.

Phase 2 provides ``classify_risk``, ``lookup_ai_act``, ``compute_max_fine``,
``write_assessment_report`` and ``calculator``; phase 3 provides the retriever behind
``search_knowledge_base``. This module wraps them as ``ToolSpec`` objects. It is the only part
of the agent that depends on how the other phases are written.
"""

from __future__ import annotations

from agent.toolbox import Toolbox


def default_toolbox() -> Toolbox:
    """The toolbox of the API. Raises NotImplementedError until phases 2 and 3 are in."""
    raise NotImplementedError(
        "phase 4 is waiting for the tools of phase 2 (tool_use) and the retriever of phase 3 "
        "(rag) to be connected in agent/connectors.py"
    )
