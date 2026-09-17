from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.dependencies.chat import authorize_chat_request
from app.schemas.chat import ChatRequest, ChatResponse


router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    authorized_payload: Annotated[ChatRequest, Depends(authorize_chat_request)],
    request: Request,
) -> ChatResponse:
    rag_service = request.app.state.rag_service

    result = rag_service.run_once(authorized_payload)
    return ChatResponse(**result)


@router.post("/chat/stream", response_class=StreamingResponse)
def chat_stream(
    authorized_payload: Annotated[ChatRequest, Depends(authorize_chat_request)],
    request: Request,
) -> StreamingResponse:
    rag_service = request.app.state.rag_service

    stream = rag_service.run_once_stream(authorized_payload)
    return StreamingResponse(content=stream, media_type="text/plain")


@router.post("/chat/sse", response_class=EventSourceResponse)
def chat_sse(
    authorized_payload: Annotated[ChatRequest, Depends(authorize_chat_request)],
    request: Request,
):

    # Dependencies run before this generator starts and before HTTP 200 is sent.
    rag_service = request.app.state.rag_service

    event_stream = rag_service.run_once_event_stream(authorized_payload)

    for item in event_stream:

        yield ServerSentEvent(
            event=item["event"],
            data=item["data"],
        )
