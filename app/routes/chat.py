from fastapi import APIRouter, Request

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