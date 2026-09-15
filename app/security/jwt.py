"""Application-owned JWT configuration, signing, and validation."""

from datetime import datetime, timedelta, timezone
import os
from typing import Any
from uuid import uuid4

import jwt


class AccessTokenError(Exception):
    """An access token could not be authenticated."""


class JWTService:
    """Create once during lifespan startup; never store request identity here."""

    expires_in = 900

    def __init__(
        self,
        secret_key: str,
        issuer: str = "rag-learning-api",
        audience: str = "rag-learning-api-users",
    ):
        if not secret_key.strip() or len(secret_key.encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 UTF-8 bytes and not be blank")
        if not issuer.strip():
            raise ValueError("JWT_ISSUER must not be blank")
        if not audience.strip():
            raise ValueError("JWT_AUDIENCE must not be blank")

        try:
            # Validate key format at startup, including rejection of PEM/SSH
            # asymmetric keys that cannot serve as an HMAC signing secret.
            jwt.get_algorithm_by_name("HS256").prepare_key(secret_key)
        except jwt.InvalidKeyError as error:
            raise ValueError("JWT_SECRET_KEY must be a valid HS256 signing secret") from error

        self._secret_key = secret_key
        self.issuer = issuer
        self.audience = audience

    @classmethod
    def from_environment(cls) -> "JWTService":
        return cls(
            secret_key=os.environ.get("JWT_SECRET_KEY", ""),
            issuer=os.environ.get("JWT_ISSUER", "rag-learning-api"),
            audience=os.environ.get("JWT_AUDIENCE", "rag-learning-api-users"),
        )

    def create_access_token(self, user_id: str) -> str:
        """The caller must supply the identity returned by authentication."""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(seconds=self.expires_in),
            "iss": self.issuer,
            "aud": self.audience,
            "jti": str(uuid4()),
        }
        return jwt.encode(payload, self._secret_key, algorithm="HS256")

    def decode_access_token(self, token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=["HS256"],
                issuer=self.issuer,
                audience=self.audience,
                options={
                    "require": ["sub", "type", "iat", "exp", "iss", "aud", "jti"],
                    "verify_exp": True,
                },
            )
        except (jwt.InvalidTokenError, TypeError, OverflowError) as error:
            # PyJWT's NumericDate conversions can also raise TypeError or
            # OverflowError for signed claims containing objects or infinity.
            raise AccessTokenError("Invalid access token") from error

        if payload["type"] != "access":
            raise AccessTokenError("Invalid access token")
        if not isinstance(payload["sub"], str) or not payload["sub"].strip():
            raise AccessTokenError("Invalid access token")
        if not isinstance(payload["jti"], str) or not payload["jti"].strip():
            raise AccessTokenError("Invalid access token")
        return payload
