from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    # Fix 7: bound message fields and reject arbitrary roles at the API boundary.
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=8_000)


class ChatRequest(BaseModel):
    # Fix 7: enforce practical limits for prompt size and history memory usage.
    raw_query: str = Field(min_length=1, max_length=4_000)

    user_id: str = Field(min_length=1, max_length=256)

    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20
    )

    k: int = Field(
        default=3,
        ge=1,
        le=10
    )

    # Fix 8: avoid an extra Gemini rewrite call unless the caller opts in.
    rewrite_query: bool = False

    # Fix 7: cosine distance is non-negative and bounded to a meaningful range.
    distance_threshold: float | None = Field(default=None, ge=0, le=2)


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
