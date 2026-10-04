from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


MessageRole = Literal["user", "assistant", "system"]
MessageStatus = Literal["pending", "completed", "failed", "cancelled"]


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=256)


class ConversationResponse(BaseModel):
    conversation_id: str
    user_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    summary: str | None = None
    summary_message_count: int = 0
    summary_updated_at: datetime | None = None


class MessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    user_id: str
    role: MessageRole
    content: str
    status: MessageStatus
    turn_id: str
    created_at: datetime
