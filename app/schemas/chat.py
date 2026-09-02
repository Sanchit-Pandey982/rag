from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    raw_query: str = Field(min_length=1)

    user_id: str = Field(min_length=1)

    chat_history: list[ChatMessage] = Field(
        default_factory=list
    )

    k: int = Field(
        default=3,
        ge=1
    )

    rewrite_query: bool = True

    distance_threshold: float | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str

    retrieval_query: str

    retrieved_document_ids: list[str]

    chunks: list[RetrievedChunkResponse]