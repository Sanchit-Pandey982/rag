from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def normalize_username(username: str) -> str:
    return username.strip().lower()


class User(BaseModel):
    """Application-owned fields of a MongoDB users document.

    MongoDB's additional _id field is ignored when reading a document.
    user_id is the stable application identity.
    """

    user_id: str = Field(default_factory=lambda: str(uuid4()))
    username: str = Field(min_length=1, max_length=256)
    password_hash: str = Field(repr=False)
    disabled: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("username", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        # Leave non-string values to Pydantic's normal type validation.
        return normalize_username(value) if isinstance(value, str) else value
