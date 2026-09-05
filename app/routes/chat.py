from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import (
    ChatRequest,
    ChatResponse
)


router = APIRouter(
    prefix="/api/v1",
    tags=["chat"]
)


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(
    payload: ChatRequest,
    request: Request
):

    rag_service = request.app.state.rag_service

    result = rag_service.run_once(
        payload
    )

    return ChatResponse(
        **result
    )


@router.post(
    "/chat/stream",
    response_class=StreamingResponse
)
def chat_stream(
    payload: ChatRequest,
    request: Request
):

    rag_service = request.app.state.rag_service

    stream = rag_service.run_once_stream(
        payload
    )

    return StreamingResponse(
        content=stream,
        media_type="text/plain"
    )