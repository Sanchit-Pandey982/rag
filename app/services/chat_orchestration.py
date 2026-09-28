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
    ConversationNotFound,
    ConversationService,
    ConversationStoreUnavailable,
)

logger = logging.getLogger(__name__)


def _await(coro):
    return asyncio.run(coro)


def conversation_service_of(request: Request) -> ConversationService | None:
    return getattr(request.app.state, "conversation_service", None)


def _history_messages(history: list[dict]) -> list[ChatMessage]:
    # Stored content was produced server-side; truncation mirrors the
    # frontend history builder so reloaded context always fits the schema.
    return [
        ChatMessage(role=item["role"], content=item["content"][:8000])
        for item in history
    ]


def prepare_chat(
    authorized_payload: ChatRequest,
    service: ConversationService | None,
) -> tuple[ChatRequest, str | None]:
    """Resolve RAG history and open a persistence turn.

    Returns ``(rag_payload, turn_id)``. ``turn_id`` is None on the legacy
    path (no ``conversation_id``): the client-supplied ``chat_history`` is
    used unchanged and nothing is persisted. With a ``conversation_id`` the
    server history is authoritative and the client field is replaced.
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
        history = _await(service.load_chat_history(
            conversation_id, authorized_payload.user_id
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


def _safe_complete(service, conversation_id, user_id, turn_id, answer):
    _await(service.complete_turn(conversation_id, user_id, turn_id, answer))


def run_chat_once(rag_service, service, authorized_payload: ChatRequest) -> dict:
    """Shared logic for the non-streaming ``POST /chat`` route."""
    rag_payload, turn_id = prepare_chat(authorized_payload, service)
    try:
        result = rag_service.run_once(rag_payload)
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
    return result


def wrap_event_stream(stream, service, authorized_payload, turn_id):
    """Forward SSE items unchanged while persisting one assistant message.

    Tokens are accumulated, never inserted per-token. ``done`` persists the
    completed turn *before* the item is forwarded; ``error`` (or any
    abnormal end: exception, cancellation, close without ``done``) records a
    failed/cancelled turn that future history loads exclude. No new event
    names are introduced.
    """
    if turn_id is None:
        yield from stream
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


def wrap_text_stream(stream, service, authorized_payload, turn_id):
    """Same turn semantics for ``POST /chat/stream`` (plain-text tokens).

    The plain-text transport has no error channel, so a persistence failure
    after the full body was delivered is only logged; the turn stays
    pending and is therefore excluded from future history (safe direction).
    """
    if turn_id is None:
        yield from stream
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
