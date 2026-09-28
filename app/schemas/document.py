from datetime import datetime
from typing import Literal

from pydantic import BaseModel


DocumentStatus = Literal["uploaded", "processing", "ready", "failed"]


class DocumentResponse(BaseModel):
    document_id: str
    user_id: str
    original_filename: str
    content_type: str | None = None
    size_bytes: int
    status: DocumentStatus
    chunk_count: int = 0
    title: str | None = None
    created_at: datetime
    updated_at: datetime
