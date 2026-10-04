"""Phase 11: agent graph state.

``AgentState`` flows through every node (retrieve → rerank → generate →
check_confidence → respond_or_retry). Nodes read what they need and
return partial updates; the runner merges them. ``total=False`` keeps
test/fake states small -- nodes must use ``.get`` with defaults.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    """Mutable scratchpad for one agent turn."""

    # Inputs (set once by the runner from the chat request).
    raw_query: str
    user_id: str
    chat_history: list[dict]
    k: int
    rewrite_query: bool
    distance_threshold: float | None

    # Routing + retrieval.
    route: Literal["retrieve", "direct"]
    retrieval_query: str
    chunks: list[Any]

    # Generation + quality gate.
    answer: str | None
    confidence: float
    iterations: int
    usage: dict

    # Tool seam (Phase 12): selection records + web grounding.
    tool_calls: list[dict]
    tool_step: str
    tools_used: list[str]

    # Best-attempt tracking across retries (best answer wins).
    best_answer: str | None
    best_confidence: float
    best_chunks: list[Any]
    best_usage: dict

    # Terminal metadata (always present on the returned state).
    attempts: int
    gave_up: bool
    error: str | None
    # Runner-local only: the generation exception for the Phase 7
    # degraded-eligibility check. Never leaves the runner.
    _error_obj: Any
