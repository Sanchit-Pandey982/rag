from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.dependencies.chat import authorize_chat_request
from app.dependencies.rate_limit import enforce_chat_rate_limit
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import chat_orchestration as orchestration


router = APIRouter(prefix="/api/v1", tags=["chat"])


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
    authorized_payload: Annotated[ChatRequest, Depends(authorize_chat_request)],
    request: Request,
    _: Annotated[None, Depends(enforce_chat_rate_limit)] = None,
):
    rag_service = request.app.state.rag_service
    service = orchestration.conversation_service_of(request)
    usage_service = orchestration.usage_service_of(request)

    # Runs before the 200/stream starts: unknown or foreign conversation_id
    # is a 404 here, never a broken stream.
    rag_payload, turn_id = orchestration.prepare_chat(authorized_payload, service)

    event_stream = rag_service.run_once_event_stream(rag_payload)
    wrapped = orchestration.wrap_event_stream(
        event_stream, service, authorized_payload, turn_id,
        usage_service=usage_service,
    )

    return EventSourceResponse(
        content=(
            ServerSentEvent(event=item["event"], data=item["data"])
            for item in wrapped
        )
    )
