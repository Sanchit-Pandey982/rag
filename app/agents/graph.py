"""Phase 11: LangGraph state machine over the RAG pipeline.

Flow: ``route → [use_tools] → retrieve → generate → check_confidence →
respond_or_retry``. Weak answers (confidence below
``AGENT_CONFIDENCE_THRESHOLD``) re-run retrieve (wider ``k``) +
generate until the answer passes or ``AGENT_MAX_ITERATIONS`` loops
are spent -- then the best answer seen (tracked across attempts) is
returned with ``gave_up=true`` when it never passed.

Phase 12 tools plug in at ``use_tools``: the calculator answers math
directly (no retrieval/LLM), web results become grounding chunks for
generation (retries fall back to the local corpus). Tool failures
fall through to plain retrieval.

Design rules (see AGENTS.md):
- Flag-gated default-off: ``run_agent`` returns ``None`` unless
  ``use_agent=True`` or ``AGENT_ENABLED=true``, so existing callers
  keep byte-identical legacy behavior by default.
- Lazy: ``langgraph`` is imported inside :func:`build_graph`, never
  at module import or app startup. When the import (or compile)
  fails, :func:`_run_manual` executes the same nodes in the same
  order, so the feature degrades to a dependency-free loop instead
  of an error.
- Graceful: node/tool failures are recorded on the state; a terminal
  generation failure still honors the Phase 7 degraded-excerpt path
  when enabled, else the standard refusal. ``run_agent`` itself only
  raises for programmer errors (bad ``k``/query limits, mirroring
  ``RAGSystem.retrieve``); retrieval/generation outages never raise.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from app.agents.config import (
    agent_confidence_threshold,
    agent_enabled,
    agent_max_iterations,
)
from app.agents.nodes import (
    check_confidence_node,
    decide_route,
    direct_answer,
    generate_node,
    retrieve_node,
    should_retry,
    tools_node,
    track_best,
)
from app.agents.state import AgentState

logger = logging.getLogger(__name__)


def _chunk_to_dict(chunk: Any) -> dict:
    if isinstance(chunk, dict):
        return dict(chunk)
    if is_dataclass(chunk):
        try:
            return asdict(chunk)
        except Exception:
            pass
    try:
        return {
            "chunk_id": getattr(chunk, "chunk_id", ""),
            "text": getattr(chunk, "text", ""),
            "distance": getattr(chunk, "distance", 0.0),
            "metadata": getattr(chunk, "metadata", {}),
        }
    except Exception:
        return {"chunk_id": "", "text": "", "distance": 0.0,
                "metadata": {}}


def build_graph(rag_service, use_hybrid=None, use_rerank=None,
                max_iterations: int = 3, threshold: float = 0.5,
                registry=None):
    """Compile the LangGraph state machine (lazy import).

    Raises whatever the ``langgraph`` import/compile raises; callers
    fall back to :func:`_run_manual`. ``registry`` is the resolved
    :class:`ToolRegistry` (Phase 12): ``use_tools`` runs the first
    claiming tool, else plain retrieval.
    """
    from langgraph.graph import END, StateGraph

    def _retrieve(state: AgentState) -> dict:
        update = retrieve_node(
            state, rag_service,
            use_hybrid=use_hybrid, use_rerank=use_rerank)
        update["attempts"] = int(state.get("attempts", 0)) + 1
        return update

    def _generate(state: AgentState) -> dict:
        return generate_node(state, rag_service)

    def _check(state: AgentState) -> dict:
        update = check_confidence_node(state, threshold)
        merged = dict(state)
        merged.update(update)
        update.update(track_best(merged))
        return update

    def _direct(state: AgentState) -> dict:
        return {
            "answer": direct_answer(state.get("raw_query", "")),
            "retrieval_query": state.get("raw_query", ""),
            "chunks": [],
            "confidence": 1.0,
            "attempts": int(state.get("attempts", 0)) + 1,
        }

    def _tools(state: AgentState) -> dict:
        return tools_node(state, registry)

    def _entry(state: AgentState) -> str:
        return state.get("route") or "retrieve"

    def _after_tools(state: AgentState) -> str:
        step = state.get("tool_step") or "retrieve"
        if step not in ("respond", "generate", "retrieve"):
            return "retrieve"
        return step

    def _after_check(state: AgentState) -> str:
        if should_retry(state, threshold, max_iterations):
            return "retry"
        return "respond"

    builder = StateGraph(AgentState)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("generate", _generate)
    builder.add_node("check_confidence", _check)
    builder.add_node("answer_directly", _direct)
    builder.add_node("use_tools", _tools)
    builder.set_conditional_entry_point(_entry, {
        "retrieve": "use_tools",
        "direct": "answer_directly",
    })
    builder.add_conditional_edges("use_tools", _after_tools, {
        "respond": END,
        "generate": "generate",
        "retrieve": "retrieve",
    })
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "check_confidence")
    builder.add_conditional_edges("check_confidence", _after_check, {
        "retry": "retrieve",
        "respond": END,
    })
    builder.add_edge("answer_directly", END)
    return builder.compile()


def _run_manual(initial: dict, rag_service, use_hybrid=None,
                use_rerank=None, max_iterations: int = 3,
                threshold: float = 0.5, registry=None) -> dict:
    """Same node order without ``langgraph`` (fallback engine)."""
    state = dict(initial)
    if (state.get("route") or "retrieve") == "direct":
        state.update({
            "answer": direct_answer(state.get("raw_query", "")),
            "retrieval_query": state.get("raw_query", ""),
            "chunks": [],
            "confidence": 1.0,
            "attempts": int(state.get("attempts", 0)) + 1,
        })
        return state
    state.update(tools_node(state, registry))
    step = state.get("tool_step") or "retrieve"
    if step == "respond":
        return state
    if step == "generate":
        state.update(generate_node(state, rag_service))
        state.update(check_confidence_node(state, threshold))
        state.update(track_best({**state}))
        if not should_retry(state, threshold, max_iterations):
            return state
    while True:
        state.update(retrieve_node(
            state, rag_service,
            use_hybrid=use_hybrid, use_rerank=use_rerank))
        state["attempts"] = int(state.get("attempts", 0)) + 1
        state.update(generate_node(state, rag_service))
        state.update(check_confidence_node(state, threshold))
        state.update(track_best({**state}))
        if not should_retry(state, threshold, max_iterations):
            return state


def _finalize(state: dict, rag_service, use_degraded=None,
              degraded_tracker=None) -> dict:
    """Shape the terminal state like ``RAGSystem.run_once``.

    Legacy keys (``answer``, ``retrieval_query``,
    ``retrieved_document_ids``, ``chunks``) are identical in shape;
    ``agent`` metadata is additive (ignored by ``ChatResponse``).
    When retries ran, the highest-confidence attempt's answer, chunks,
    and usage win over the last attempt's.
    """
    if state.get("best_answer") is not None:
        try:
            best_conf = float(state.get("best_confidence", 0.0))
        except (TypeError, ValueError):
            best_conf = 0.0
        answer = state.get("best_answer")
        chunks = list(state.get("best_chunks") or [])
        usage = dict(state.get("best_usage") or {})
        confidence = best_conf
    else:
        chunks = list(state.get("chunks") or [])
        answer = state.get("answer")
        usage = dict(state.get("usage") or {})
        try:
            confidence = float(state.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
    degraded = False
    if not answer:
        try:
            rag = getattr(rag_service, "rag", rag_service)
            pair = rag._resolve_degraded(degraded_tracker, use_degraded)
            error = state.get("_error_obj")
            if error is None:
                error = RuntimeError(state.get("error") or "generate")
            text = rag._degraded_answer(error, chunks, pair)
        except Exception:
            logger.exception("Agent degraded fallback failed")
            text = None
        if text is not None:
            answer = text
            degraded = True
        else:
            try:
                from phase1 import REFUSAL_MESSAGE
            except Exception:
                REFUSAL_MESSAGE = (
                    "I do not have enough information to answer that.")
            answer = REFUSAL_MESSAGE
    chunk_dicts = [_chunk_to_dict(c) for c in chunks]
    try:
        document_ids = [c["metadata"]["document_id"] for c in chunk_dicts]
    except Exception:
        document_ids = []
    result: dict[str, Any] = {
        "answer": answer,
        "retrieval_query": state.get("retrieval_query")
        or state.get("raw_query", ""),
        "retrieved_document_ids": document_ids,
        "chunks": chunk_dicts,
    }
    if usage:
        result["usage"] = usage
    if degraded:
        result["degraded"] = True
    result["agent"] = {
        "route": state.get("route") or "retrieve",
        "attempts": int(state.get("attempts", 0)),
        "confidence": confidence,
        "gave_up": bool(state.get("gave_up", False)),
        "tools_used": list(state.get("tools_used") or []),
    }
    return result


def run_agent(
    rag_service,
    *,
    raw_query: str,
    user_id: str,
    chat_history: list[dict] | None = None,
    k: int = 3,
    rewrite_query: bool = False,
    distance_threshold: float | None = None,
    use_agent: bool | None = None,
    use_hybrid: bool | None = None,
    use_rerank: bool | None = None,
    use_degraded: bool | None = None,
    degraded_tracker=None,
    max_iterations: int | None = None,
    confidence_threshold: float | None = None,
    tools: list | None = None,
) -> dict | None:
    """Run the agent loop; ``None`` when the agent path is disabled.

    ``use_agent=None`` follows ``AGENT_ENABLED`` (default off).
    ``max_iterations``/``confidence_threshold`` default to their env
    helpers. ``tools=None`` uses the shipped registry (calculator +
    web search); a list/registry overrides it (empty disables tools).
    On success returns a ``run_once``-shaped dict with an extra
    ``agent`` metadata key (now including ``tools_used``).
    """
    enabled = agent_enabled() if use_agent is None else bool(use_agent)
    if not enabled:
        return None
    # Same input contract as RAGSystem.retrieve (programmer errors).
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")
    if not raw_query or len(raw_query) > 4_000:
        raise ValueError("query must contain 1-4000 characters")
    if (distance_threshold is not None
            and not 0 <= distance_threshold <= 2):
        raise ValueError("distance_threshold must be between 0 and 2")
    try:
        from app.tools.registry import resolve_registry
        registry = resolve_registry(tools)
    except Exception:
        logger.exception("Tool registry unavailable; tools disabled")
        registry = None
    limit = agent_max_iterations() if max_iterations is None \
        else max(1, min(10, int(max_iterations)))
    threshold = agent_confidence_threshold() \
        if confidence_threshold is None else float(confidence_threshold)
    initial: dict[str, Any] = {
        "raw_query": raw_query,
        "user_id": user_id,
        "chat_history": list(chat_history or []),
        "k": k,
        "rewrite_query": bool(rewrite_query),
        "distance_threshold": distance_threshold,
        "route": decide_route(raw_query),
        "retrieval_query": raw_query,
        "chunks": [],
        "answer": None,
        "confidence": 0.0,
        "iterations": 0,
        "usage": {},
        "tool_calls": [],
        "tool_step": "",
        "tools_used": [],
        "best_answer": None,
        "best_confidence": -1.0,
        "best_chunks": [],
        "best_usage": {},
        "attempts": 0,
        "gave_up": False,
        "error": None,
    }
    try:
        graph = build_graph(
            rag_service, use_hybrid=use_hybrid, use_rerank=use_rerank,
            max_iterations=limit, threshold=threshold, registry=registry)
        final = graph.invoke(initial)
        state = dict(final)
    except Exception:
        logger.exception(
            "LangGraph unavailable; running manual agent loop")
        try:
            state = _run_manual(
                initial, rag_service, use_hybrid=use_hybrid,
                use_rerank=use_rerank, max_iterations=limit,
                threshold=threshold, registry=registry)
        except Exception:
            logger.exception("Agent loop failed")
            state = dict(initial)
            state["error"] = "agent-failed"
    if int(state.get("attempts", 0)) >= limit and float(
            state.get("confidence", 0.0)) < threshold \
            and (state.get("route") or "retrieve") == "retrieve":
        state["gave_up"] = True
    return _finalize(state, rag_service, use_degraded=use_degraded,
                     degraded_tracker=degraded_tracker)
