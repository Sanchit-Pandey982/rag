from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.dependencies.chat import authorize_chat_request
from app.dependencies.rate_limit import enforce_chat_rate_limit
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import chat_orchestration as orchestration


router = APIRouter(prefix="/api/v1", tags=["chat"])


def prepare_sse_chat(
    authorized_payload: Annotated[ChatRequest, Depends(authorize_chat_request)],
    request: Request,
    _: Annotated[None, Depends(enforce_chat_rate_limit)] = None,
) -> tuple[ChatRequest, ChatRequest, str | None]:
    """Resolve conversation history before SSE headers are sent.

    FastAPI 0.141 invokes SSE endpoints in the event loop. Keeping this
    synchronous work in a sync dependency makes FastAPI run it in the
    threadpool, so the existing ``asyncio.run`` bridge is safe and an
    unknown conversation still returns a 404 instead of a broken stream.
    """

    service = orchestration.conversation_service_of(request)
    rag_payload, turn_id = orchestration.prepare_chat(authorized_payload, service)
    return authorized_payload, rag_payload, turn_id


@router.post("/chat", response_model=ChatResponse)
def chat(
    authorized_payload: Annotated[ChatRequest, Depends(authorize_chat_request)],
    request: Request,
    _: Annotated[None, Depends(enforce_chat_rate_limit)] = None,
) -> ChatResponse:
    rag_service = request.app.state.rag_service
    service = orchestration.conversation_service_of(request)
    usage_service = orchestration.usage_service_of(request)

    result = orchestration.run_chat_once(
        rag_service, service, authorized_payload,
        usage_service=usage_service,
    )
    return ChatResponse(**result)


@router.post("/chat/stream", response_class=StreamingResponse)
def chat_stream(
    authorized_payload: Annotated[ChatRequest, Depends(authorize_chat_request)],
    request: Request,
    _: Annotated[None, Depends(enforce_chat_rate_limit)] = None,
) -> StreamingResponse:
    rag_service = request.app.state.rag_service
    service = orchestration.conversation_service_of(request)
    usage_service = orchestration.usage_service_of(request)

    rag_payload, turn_id = orchestration.prepare_chat(authorized_payload, service)
    # Caller-owned usage collector: the pipeline fills it while the
    # response streams, and the wrapper below records it on completion.
    usage: dict = {}
    stream = rag_service.run_once_stream(rag_payload, usage=usage)
    wrapped = orchestration.wrap_text_stream(
        stream, service, authorized_payload, turn_id,
        usage_service=usage_service, usage=usage,
    )
    return StreamingResponse(content=wrapped, media_type="text/plain")


@router.post("/chat/sse", response_class=EventSourceResponse)
def chat_sse(
    prepared: Annotated[
        tuple[ChatRequest, ChatRequest, str | None],
        Depends(prepare_sse_chat),
    ],
    request: Request,
):
    authorized_payload, rag_payload, turn_id = prepared
    rag_service = request.app.state.rag_service
    service = orchestration.conversation_service_of(request)
    usage_service = orchestration.usage_service_of(request)

    event_stream = rag_service.run_once_event_stream(rag_payload)
    wrapped = orchestration.wrap_event_stream(
        event_stream, service, authorized_payload, turn_id,
        usage_service=usage_service,
    )

    # FastAPI 0.141's EventSourceResponse is a marker response class: yield
    # ServerSentEvent items instead of returning an EventSourceResponse.
    for item in wrapped:
        yield ServerSentEvent(event=item["event"], data=item["data"])
