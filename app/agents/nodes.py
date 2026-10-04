"""Phase 11: individual agent node functions.

Nodes are small, dependency-light, and LangGraph-compatible: each
takes the shared ``AgentState`` and returns a partial update dict.
They never raise into the chat path -- retrieval/generation failures
are recorded on the state so the runner can retry, degrade, or fall
back to the legacy pipeline.

Routing is a cheap deterministic heuristic (no LLM call): greetings
and smalltalk answer directly; everything else retrieves grounded
context first. Phase 12 runs registered tools between routing and
retrieval: the calculator answers math directly, and web results
become grounding chunks for generation (retries fall back to the
local corpus, best answer wins).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Greeting/smalltalk answered without retrieval or an LLM call. Full
# match only: anything carrying real content falls through to retrieval.
_DIRECT_RE = re.compile(
    r"(hi+|hii+|hello+|hey+|yo|good\s*morning|good\s*afternoon|"
    r"good\s*evening|thanks?|thank\s*you|thx|bye+|goodbye|"
    r"good\s*night|ok|okay|yes|no|please\s*help)\s*[!?.]*",
    re.IGNORECASE,
)

_DIRECT_ANSWER = (
    "Hello! I'm your learning assistant. "
    "Ask me anything about your documents and I'll look it up."
)


def decide_route(query: str) -> str:
    """``direct`` for smalltalk, ``retrieve`` for everything else."""
    text = (query or "").strip()
    if not text:
        return "retrieve"
    if _DIRECT_RE.fullmatch(text) is not None:
        return "direct"
    return "retrieve"


def direct_answer(query: str) -> str:
    """Conversational reply for the ``direct`` route (no LLM needed)."""
    _ = query
    return _DIRECT_ANSWER


def confidence_of(answer: str | None, chunks: list,
                  route: str) -> float:
    """Quality score in [0, 1] for the check_confidence node.

    - ``direct`` route with a non-empty answer: 1.0 (nothing to ground).
    - Refusal / empty / missing answer: 0.0.
    - ``retrieve`` route with no chunks: 0.0 (ungrounded).
    - Otherwise from mean cosine distance: ``1 - mean/2``, so a mean
      distance of 1.0 scores exactly 0.5 (the default threshold) and
      worse retrieval retries.
    """
    if not answer or not answer.strip():
        return 0.0
    try:
        from phase1 import REFUSAL_MESSAGE
    except Exception:
        REFUSAL_MESSAGE = "I do not have enough information to answer that."
    if answer.strip() == REFUSAL_MESSAGE:
        return 0.0
    if route == "direct":
        return 1.0
    items = list(chunks or [])
    if not items:
        return 0.0
    try:
        mean = sum(float(getattr(c, "distance", 1.0)) for c in items) / len(items)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, 1.0 - mean / 2.0))


def should_retry(state: dict, threshold: float, max_iterations: int) -> bool:
    """True when the answer is weak and budget remains."""
    if state.get("route") != "retrieve":
        return False
    if state.get("gave_up"):
        return False
    try:
        confidence = float(state.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return (
        confidence < threshold
        and int(state.get("attempts", 0)) < int(max_iterations)
    )


def retrieve_node(state: dict, rag_service,
                  use_hybrid: bool | None = None,
                  use_rerank: bool | None = None) -> dict[str, Any]:
    """Fill ``retrieval_query`` + ``chunks`` via the shared pipeline.

    Reuses ``RAGSystem._retrieve_chunks`` (the exact helper
    ``run_once`` uses), so tenant filtering (``where={"user_id"}``),
    hybrid RRF fusion, and cross-encoder reranking behave identically
    under the agent. The long-lived retriever/reranker cached on
    ``RAGService`` are preferred; per-attempt ``k`` expansion widens
    the candidate window on retries.
    """
    raw_query = state.get("raw_query", "")
    user_id = state.get("user_id", "")
    attempt = int(state.get("attempts", 0))
    base_k = int(state.get("k") or 3)
    # Retry with a wider window (capped at Chroma's 10-result limit).
    k = max(1, min(10, base_k + 2 * attempt))
    query = raw_query
    if state.get("rewrite_query") and state.get("chat_history"):
        try:
            from phase1 import condense_question
            query = condense_question(state["chat_history"], raw_query)
        except Exception:
            logger.exception("Agent query rewrite failed; using raw query")
            query = raw_query
    rag = getattr(rag_service, "rag", rag_service)
    try:
        chunks = rag._retrieve_chunks(
            query=query,
            user_id=user_id,
            k=k,
            distance_threshold=state.get("distance_threshold"),
            use_hybrid=use_hybrid,
            hybrid_retriever=getattr(rag_service, "hybrid_retriever", None),
            use_rerank=use_rerank,
            reranker=getattr(rag_service, "reranker", None),
        )
    except Exception as error:
        logger.exception("Agent retrieval failed")
        return {"retrieval_query": query, "chunks": [],
                "error": f"retrieval: {type(error).__name__}"}
    return {"retrieval_query": query, "chunks": list(chunks or [])}


def generate_node(state: dict, rag_service) -> dict[str, Any]:
    """Fill ``answer`` (+ ``usage``) from the retrieved chunks."""
    rag = getattr(rag_service, "rag", rag_service)
    chunks = list(state.get("chunks") or [])
    usage_report: dict = {}
    try:
        answer = rag.generate_answer(
            question=state.get("raw_query", ""),
            chunks=chunks,
            chat_history=state.get("chat_history") or [],
            usage=usage_report,
        )
    except Exception as error:
        logger.exception("Agent generation failed")
        # ``_error_obj`` stays inside the runner (never serialized to
        # the result): the finalize step needs the real exception for
        # the Phase 7 degraded-eligibility check.
        return {"answer": None, "_error_obj": error,
                "error": f"generate: {type(error).__name__}"}
    update: dict[str, Any] = {"answer": answer}
    if usage_report:
        update["usage"] = dict(usage_report)
    return update


def check_confidence_node(state: dict, threshold: float) -> dict[str, Any]:
    """Score the current answer; the runner decides retry vs respond."""
    confidence = confidence_of(
        state.get("answer"), state.get("chunks") or [],
        state.get("route") or "retrieve",
    )
    return {"confidence": confidence}


def track_best(state: dict) -> dict[str, Any]:
    """Remember the highest-confidence attempt (best answer wins).

    Retries strictly improve on the past: the final response uses the
    best attempt's answer, chunks, and usage instead of the last one.
    """
    try:
        confidence = float(state.get("confidence", 0.0))
    except (TypeError, ValueError):
        return {}
    try:
        best = float(state.get("best_confidence", -1.0))
    except (TypeError, ValueError):
        best = -1.0
    if confidence <= best:
        return {}
    update: dict[str, Any] = {
        "best_answer": state.get("answer"),
        "best_confidence": confidence,
        "best_chunks": list(state.get("chunks") or []),
    }
    if state.get("usage"):
        update["best_usage"] = dict(state["usage"])
    return update


def web_results_to_chunks(results: list) -> list:
    """Grounding chunks from web results (external, clearly labeled).

    Chunks carry ``document_id="web_search"`` with the article URL as
    source, so answers stay attributable through the unchanged
    ``retrieved_document_ids``/sources shape. Neutral distance 0.5:
    externally grounded, not corpus-ranked.
    """
    try:
        from phase1 import RetrievedChunk
    except Exception:
        RetrievedChunk = None
    chunks = []
    for index, item in enumerate(results or []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Web result").strip()
        snippet = str(item.get("snippet") or "").strip()
        url = str(item.get("url") or "").strip()
        text = f"{title}\n{snippet}".strip() if snippet else title
        metadata = {"user_id": "", "document_id": "web_search",
                    "source": url or "web", "title": title,
                    "chunk_index": index}
        chunk_id = f"web_search:{index}"
        if RetrievedChunk is not None:
            chunks.append(RetrievedChunk(
                chunk_id=chunk_id, text=text, distance=0.5,
                metadata=metadata))
        else:
            chunks.append({"chunk_id": chunk_id, "text": text,
                           "distance": 0.5, "metadata": metadata})
    return chunks


def tools_node(state: dict, registry) -> dict[str, Any]:
    """Run the first registered tool claiming the query, if any.

    Sets ``tool_step`` for the runner: ``respond`` (calculator
    answered directly), ``generate`` (web chunks ready for the LLM),
    or ``retrieve`` (no tool matched, or the tool failed/empty --
    fall back to local retrieval). Tool failures are recorded on
    ``tool_calls`` and never raise.
    """
    query = state.get("raw_query", "")
    calls = list(state.get("tool_calls") or [])
    used = list(state.get("tools_used") or [])
    selection = None
    try:
        selection = registry.select(query) if registry is not None else None
    except Exception:
        logger.exception("Tool selection failed; using retrieval")
    if selection is None:
        return {"tool_step": "retrieve", "tool_calls": calls,
                "tools_used": used}
    tool, tool_input = selection
    try:
        output = tool.execute(tool_input)
    except Exception as error:
        logger.exception("Tool execution failed: %s", tool.name)
        output = {"ok": False, "error": f"{type(error).__name__}"}
    if not isinstance(output, dict):
        output = {"ok": False, "error": "tool returned no result"}
    calls.append({"tool": tool.name, "input": tool_input,
                  "output": output})
    if not output.get("ok"):
        return {"tool_step": "retrieve", "tool_calls": calls,
                "tools_used": used,
                "error": f"tool:{tool.name}"}
    used.append(tool.name)
    if tool.name == "calculator":
        expression = str(output.get("expression", tool_input))
        display = str(output.get("display", output.get("result", "")))
        return {
            "answer": f"The result of {expression} is {display}.",
            "retrieval_query": query,
            "chunks": [],
            "confidence": 1.0,
            "attempts": int(state.get("attempts", 0)) + 1,
            "tool_step": "respond",
            "tool_calls": calls,
            "tools_used": used,
        }
    if tool.name == "web_search":
        chunks = web_results_to_chunks(output.get("results") or [])
        if not chunks:
            return {"tool_step": "retrieve", "tool_calls": calls,
                    "tools_used": used}
        return {
            "retrieval_query": query,
            "chunks": chunks,
            "attempts": int(state.get("attempts", 0)) + 1,
            "tool_step": "generate",
            "tool_calls": calls,
            "tools_used": used,
        }
    # Future tools default to "answered, respond" when they return ok;
    # unknown payloads fall back to retrieval below via tool_step.
    return {"tool_step": "retrieve", "tool_calls": calls,
            "tools_used": used}
