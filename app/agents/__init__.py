"""Phase 11: agent routing over the existing RAG pipeline.

The agent decides per query whether to retrieve grounded context
(``retrieve`` route) or answer directly without retrieval (``direct``
route for greetings/smalltalk), then runs the retrieve → rerank →
generate → check_confidence loop with a bounded retry budget.

Public entry point is :func:`run_agent` in ``graph.py``. Importing
this package needs no credentials, no network, and no ``langgraph``
(the graph library is imported lazily on first agent run, and a
manual loop with identical node order is used when it is absent).
"""

from app.agents.config import (
    agent_confidence_threshold,
    agent_enabled,
    agent_max_iterations,
)
from app.agents.graph import run_agent
from app.agents.state import AgentState

__all__ = [
    "AgentState",
    "agent_confidence_threshold",
    "agent_enabled",
    "agent_max_iterations",
    "run_agent",
]
