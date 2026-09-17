from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.dependencies.auth import get_current_user
from app.schemas.auth import UserResponse
from app.schemas.chat import ChatRequest


def authorize_chat_request(
    payload: ChatRequest,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> ChatRequest:
    """Verify tenant ownership before any route starts RAG work or streaming."""
    authenticated_user_id = current_user.user_id
    if payload.user_id != authenticated_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot access another user's chat data.",
        )

    # Keep the public request model, but use the server's identity downstream.
    return payload.model_copy(update={"user_id": authenticated_user_id})
