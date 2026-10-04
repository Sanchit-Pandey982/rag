"""Phase 3.5 orchestration: persistent conversation turns around RAG.

Boundary rule: MongoDB writes live here, never inside
``RAGSystem.run_once*``. This layer loads server-side history, calls the
unchanged RAG pipeline, forwards its events/items untouched, accumulates
the assistant answer, and persists exactly one logical message per turn.

Chat routes are synchronous, while MongoDB access is async. The sync routes
run in a worker thread without a running event loop, so ``asyncio.run``
bridges the few small persistence calls without changing the routes' shape.
"""

import asyncio
import logging

from fastapi import HTTPException, Request

from app.schemas.chat import ChatMessage, ChatRequest
from app.services.conversation_service import (
    CANCELLED,
    FAILED,
    SUMMARY_KEEP_RECENT,
    ConversationNotFound,
    ConversationService,
    ConversationStoreUnavailable,
    summary_max_age,
    summary_threshold,
)

logger = logging.getLogger(__name__)


def _await(coro):
    return asyncio.run(coro)


def default_summarize_fn(messages, previous_summary=None):
    # Lazy import: keeps this module importable without Gemini credentials,
    # and lets tests inject a fake summarize function instead.
    from phase1 import summarize_conversation_history
    return summarize_conversation_history(messages, previous_summary)


def conversation_service_of(request: Request) -> ConversationService | None:
    return getattr(request.app.state, "conversation_service", None)


def usage_service_of(request: Request):
    """Usage ledger, if lifespan wired one; None keeps chat working."""
    return getattr(request.app.state, "usage_service", None)


def _history_messages(history: list[dict]) -> list[ChatMessage]:
    # Stored content was produced server-side; truncation mirrors the
    # frontend history builder so reloaded context always fits the schema.
    return [
        ChatMessage(role=item["role"], content=item["content"][:8000])
        for item in history
    ]


async def build_context_history(
    service: ConversationService,
    conversation_id: str,
    user_id: str,
    summarize_fn=None,
) -> list[dict]:
    """History for RAG: summary + recent messages once past the threshold.

    Short conversations return plain recent history. Long ones return one
    leading ``system`` message holding the stored summary, followed by the
    most recent ``SUMMARY_KEEP_RECENT`` completed messages verbatim. The
    summary refreshes only after ``SUMMARY_MAX_AGE`` new messages; a failed
    summarization keeps the old summary (or plain history when none exists).
    """
    threshold = summary_threshold()
    total = await service.count_completed_messages(conversation_id, user_id)
    if total <= threshold:
        return await service.load_chat_history(
            conversation_id, user_id
        )

    state = await service.get_summary_state(conversation_id, user_id)
    previous_summary = state["summary"]
    covered = state["summary_message_count"]
    recent = await service.load_chat_history(
        conversation_id, user_id, limit=SUMMARY_KEEP_RECENT
    )
    summarizable_count = max(total - len(recent), 0)

    if previous_summary and total - covered < summary_max_age():
        return (
            [{"role": "system", "content": previous_summary}]
            + recent
        )

    if summarize_fn is None:
        summarize_fn = default_summarize_fn
    older = await service.load_oldest_completed(
        conversation_id, user_id, limit=summarizable_count
    )
    try:
        fresh_summary = summarize_fn(older, previous_summary)
    except Exception:
        logger.exception("Conversation summarization failed")
        fresh_summary = ""

    if fresh_summary:
        await service.save_summary(
            conversation_id, user_id, fresh_summary, summarizable_count
        )
        return (
            [{"role": "system", "content": fresh_summary}]
            + recent
        )
    if previous_summary:
        return (
            [{"role": "system", "content": previous_summary}]
            + recent
        )
    return recent


def prepare_chat(
    authorized_payload: ChatRequest,
    service: ConversationService | None,
    summarize_fn=None,
) -> tuple[ChatRequest, str | None]:
    """Resolve RAG history and open a persistence turn.

    Returns ``(rag_payload, turn_id)``. ``turn_id`` is None on the legacy
    path (no ``conversation_id``): the client-supplied ``chat_history`` is
    used unchanged and nothing is persisted. With a ``conversation_id`` the
    server history is authoritative and the client field is replaced.

    Long conversations send a stored summary plus the most recent messages
    instead of raw truncated history (see ``build_context_history``).
    """
    conversation_id = authorized_payload.conversation_id or None
    if conversation_id is None:
        return authorized_payload, None
    if service is None:
        # Fail closed: never silently demote a persistent conversation.
        raise HTTPException(
            status_code=503,
            detail="Conversation service is unavailable. Please retry.",
        )
    try:
        history = _await(build_context_history(
            service, conversation_id, authorized_payload.user_id,
            summarize_fn=summarize_fn,
        ))
        turn_id = _await(service.start_turn(
            conversation_id, authorized_payload.user_id,
            authorized_payload.raw_query,
        ))
    except ConversationNotFound as error:
        # Unknown id and another user's id look identical (no existence leak).
        raise HTTPException(status_code=404, detail="Conversation not found.") from error
    except ConversationStoreUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail="Conversation service is unavailable. Please retry.",
        ) from error
    rag_payload = authorized_payload.model_copy(
        update={"chat_history": _history_messages(history)}
    )
    return rag_payload, turn_id


def _safe_fail(service, conversation_id, user_id, turn_id, status, partial):
    try:
        _await(service.fail_turn(
            conversation_id, user_id, turn_id,
            status=status, partial_content=partial,
        ))
    except ConversationStoreUnavailable:
        logger.exception("Could not record failed turn")


def _safe_record_usage(usage_service, user_id: str, usage: dict | None) -> None:
    """Persist one request's token ledger row, fail-open.

    Usage is observability, not conversation truth: a store outage is
    logged and the (already correct) answer still goes out. Malformed
    payloads are dropped the same way -- never raise into the chat path.
    """
    if usage_service is None or not usage:
        return
    try:
        model = usage.get("model")
        _await(usage_service.record_usage(
            user_id=user_id,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            model=model if isinstance(model, str) and model else "unknown",
        ))
    except Exception:
        logger.exception("Could not record token usage")


def _safe_complete(service, conversation_id, user_id, turn_id, answer):
    _await(service.complete_turn(conversation_id, user_id, turn_id, answer))


def run_chat_once(rag_service, service, authorized_payload: ChatRequest,
                   use_cache: bool | None = None,
                   use_degraded: bool | None = None,
                   use_agent: bool | None = None,
                   usage_service=None) -> dict:
    """Shared logic for the non-streaming ``POST /chat`` route.

    ``use_cache=None`` follows the ``CACHE_ENABLED`` env flag; the
    response cache itself lives on ``rag_service`` (wired in lifespan)
    and is consulted after retrieval, before the LLM call.
    ``use_degraded=None`` follows ``DEGRADED_MODE_ENABLED`` the same
    way: an unavailable LLM serves excerpts instead of an error.
    ``use_agent=None`` follows ``AGENT_ENABLED`` (default off): the
    LangGraph retrieve → rerank → generate → check_confidence loop
    with bounded retries; disabled callers keep the legacy pipeline.
    ``usage_service=None`` skips token ledger recording.
    """
    rag_payload, turn_id = prepare_chat(authorized_payload, service)
    try:
        result = rag_service.run_once(
            rag_payload, use_cache=use_cache, use_degraded=use_degraded,
            use_agent=use_agent)
    except Exception:
        if turn_id is not None:
            _safe_fail(
                service, authorized_payload.conversation_id,
                authorized_payload.user_id, turn_id, FAILED, "",
            )
        raise
    if turn_id is not None:
        try:
            _safe_complete(
                service, authorized_payload.conversation_id,
                authorized_payload.user_id, turn_id, result["answer"],
            )
        except ConversationStoreUnavailable as error:
            # Answer exists but history would lie if returned as success.
            raise HTTPException(
                status_code=503,
                detail="Conversation service is unavailable. Please retry.",
            ) from error
    _safe_record_usage(
        usage_service, authorized_payload.user_id, result.get("usage"))
    return result


def wrap_event_stream(stream, service, authorized_payload, turn_id,
                      usage_service=None):
    """Forward SSE items unchanged while persisting one assistant message.

    Tokens are accumulated, never inserted per-token. ``done`` persists the
    completed turn *before* the item is forwarded, then records the token
    ledger row from ``done`` metadata (fail-open); ``error`` (or any
    abnormal end: exception, cancellation, close without ``done``) records a
    failed/cancelled turn that future history loads exclude. No new event
    names are introduced.
    """
    if turn_id is None:
        if usage_service is None:
            yield from stream
            return
        # Legacy path (no persistence turn) still meters usage: nothing
        # here touches MongoDB history, so a bare forward with ledger
        # recording is the honest shape.
        for item in stream:
            if item["event"] == "done":
                _safe_record_usage(
                    usage_service, authorized_payload.user_id,
                    item["data"].get("metadata", {}).get("usage"),
                )
            yield item
        return
    conversation_id = authorized_payload.conversation_id
    user_id = authorized_payload.user_id
    parts: list[str] = []
    completed = False
    try:
        for item in stream:
            event = item["event"]
            if event == "token":
                parts.append(item["data"].get("text", ""))
            elif event == "done":
                try:
                    _safe_complete(service, conversation_id, user_id, turn_id, "".join(parts))
                except ConversationStoreUnavailable:
                    logger.exception("Could not save completed turn")
                    yield {
                        "event": "error",
                        "data": {
                            "stage": "done",
                            "message": "The response could not be completed.",
                        },
                    }
                    return
                _safe_record_usage(
                    usage_service, user_id,
                    item["data"].get("metadata", {}).get("usage"),
                )
                completed = True
            elif event == "error":
                _safe_fail(service, conversation_id, user_id, turn_id, FAILED, "".join(parts))
            yield item
        if not completed:
            # Exhausted without "done" (and without "error", which returns
            # via the branch above): must not become successful history.
            # RAGSystem always ends with one of them; this guards the shape.
            _safe_fail(service, conversation_id, user_id, turn_id, FAILED, "".join(parts))
    except GeneratorExit:
        _safe_fail(service, conversation_id, user_id, turn_id, CANCELLED, "".join(parts))
        raise
    except asyncio.CancelledError:
        _safe_fail(service, conversation_id, user_id, turn_id, CANCELLED, "".join(parts))
        raise
    except Exception:
        _safe_fail(service, conversation_id, user_id, turn_id, FAILED, "".join(parts))
        raise


def wrap_text_stream(stream, service, authorized_payload, turn_id,
                     usage_service=None, usage=None):
    """Same turn semantics for ``POST /chat/stream`` (plain-text tokens).

    The plain-text transport has no error channel, so a persistence failure
    after the full body was delivered is only logged; the turn stays
    pending and is therefore excluded from future history (safe direction).
    Token counts arrive out-of-band via the caller-owned ``usage``
    collector (filled by the pipeline during iteration) and are recorded
    on successful exhaustion, fail-open.
    """
    if turn_id is None:
        if usage_service is None or usage is None:
            yield from stream
            return
        for text in stream:
            yield text
        _safe_record_usage(usage_service, authorized_payload.user_id, usage)
        return
    conversation_id = authorized_payload.conversation_id
    user_id = authorized_payload.user_id
    parts: list[str] = []
    try:
        for text in stream:
            parts.append(text)
            yield text
    except (GeneratorExit, asyncio.CancelledError):
        _safe_fail(service, conversation_id, user_id, turn_id, CANCELLED, "".join(parts))
        raise
    except Exception:
        _safe_fail(service, conversation_id, user_id, turn_id, FAILED, "".join(parts))
        raise
    else:
        try:
            _safe_complete(service, conversation_id, user_id, turn_id, "".join(parts))
        except ConversationStoreUnavailable:
            logger.exception("Could not save completed turn")
        _safe_record_usage(usage_service, user_id, usage)
