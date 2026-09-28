"""Phase 3.5: minimal persistent-conversation API.

Only the endpoints chat persistence needs: create, list mine, read one,
read its messages. Every lookup is scoped by the authenticated user id --
a valid conversation_id alone never grants access, and unknown vs.
foreign ids both return 404 so ownership cannot be probed.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.dependencies.auth import get_current_user
from app.schemas.auth import UserResponse
from app.schemas.conversation import (
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
)
from app.services.conversation_service import (
    ConversationService,
    ConversationStoreUnavailable,
)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


def _service(request: Request) -> ConversationService:
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is unavailable. Please retry.",
        )
    return service


def _unavailable() -> HTTPException:
    # Internal MongoDB details stay in backend logs (see service layer).
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Conversation service is unavailable. Please retry.",
    )


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: CreateConversationRequest,
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> ConversationResponse:
    try:
        document = await _service(request).create_conversation(
            current_user.user_id, title=payload.title
        )
    except ConversationStoreUnavailable as error:
        raise _unavailable() from error
    return ConversationResponse(**document)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[ConversationResponse]:
    try:
        documents = await _service(request).list_conversations(
            current_user.user_id, limit=limit
        )
    except ConversationStoreUnavailable as error:
        raise _unavailable() from error
    return [ConversationResponse(**document) for document in documents]


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> ConversationResponse:
    try:
        document = await _service(request).get_conversation(
            conversation_id, current_user.user_id
        )
    except ConversationStoreUnavailable as error:
        raise _unavailable() from error
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return ConversationResponse(**document)


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def list_messages(
    conversation_id: str,
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[MessageResponse]:
    service = _service(request)
    try:
        conversation = await service.get_conversation(conversation_id, current_user.user_id)
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found."
            )
        documents = await service.list_messages(
            conversation_id, current_user.user_id, limit=limit
        )
    except ConversationStoreUnavailable as error:
        raise _unavailable() from error
    return [MessageResponse(**document) for document in documents]
