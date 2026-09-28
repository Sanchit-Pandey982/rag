from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

from app.models.user import normalize_username


class RegisterRequest(BaseModel):
    # NIST SP 800-63B floor: 8+ chars at enrollment. The 256-char ceiling
    # is not a policy statement -- it bounds what Argon2 will hash per
    # request, so a megabyte-long "password" cannot buy server CPU.
    username: str = Field(min_length=1, max_length=256)
    password: SecretStr = Field(min_length=8, max_length=256)

    @field_validator("username", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return normalize_username(value) if isinstance(value, str) else value


class LoginRequest(BaseModel):
    # Deliberately permissive: login must never become a policy oracle.
    # A short or overlong guess returns the uniform 401, never a 422 that
    # distinguishes "bad format" from "bad credentials". The ceiling only
    # caps Argon2 work per attempt (see RegisterRequest).
    username: str = Field(min_length=1, max_length=256)
    password: SecretStr = Field(min_length=1, max_length=256)

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
