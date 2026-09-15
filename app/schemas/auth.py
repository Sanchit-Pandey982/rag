from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

from app.models.user import normalize_username


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=256)
    password: SecretStr = Field(min_length=1)

    @field_validator("username", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return normalize_username(value) if isinstance(value, str) else value


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=256)
    password: SecretStr = Field(min_length=1)

    @field_validator("username", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return normalize_username(value) if isinstance(value, str) else value


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    user_id: str
    username: str
    created_at: datetime
