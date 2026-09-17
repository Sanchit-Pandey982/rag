"""Application-owned JWT configuration, signing, and validation."""

from datetime import datetime, timedelta, timezone
import os
from typing import Any
from uuid import uuid4

import jwt


class AccessTokenError(Exception):
    """An access token could not be authenticated."""


class RefreshTokenError(Exception):
    """A refresh token could not be authenticated."""


class JWTService:
    """Create once during lifespan startup; never store request identity here."""

    expires_in = 900
    refresh_expires_in = 7 * 24 * 60 * 60

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
        token, _ = self._create_token(user_id, "access", self.expires_in)
        return token

    def create_refresh_token(self, user_id: str) -> tuple[str, str]:
        """Return the signed refresh token and its Redis session identifier."""
        return self._create_token(user_id, "refresh", self.refresh_expires_in)

    def _create_token(self, user_id: str, token_type: str, lifetime: int) -> tuple[str, str]:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "type": token_type,
            "iat": now,
            "exp": now + timedelta(seconds=lifetime),
            "iss": self.issuer,
            "aud": self.audience,
            "jti": str(uuid4()),
        }
        return jwt.encode(payload, self._secret_key, algorithm="HS256"), payload["jti"]

    def decode_access_token(self, token: str) -> dict[str, Any]:
        try:
            return self._decode_token(token, expected_type="access")
        except (jwt.InvalidTokenError, TypeError, OverflowError) as error:
            # PyJWT's NumericDate conversions can also raise TypeError or
            # OverflowError for signed claims containing objects or infinity.
            raise AccessTokenError("Invalid access token") from error

    def decode_refresh_token(self, token: str) -> dict[str, Any]:
        try:
            return self._decode_token(token, expected_type="refresh")
        except (jwt.InvalidTokenError, TypeError, OverflowError) as error:
            raise RefreshTokenError("Invalid refresh token") from error

    def _decode_token(self, token: str, *, expected_type: str) -> dict[str, Any]:
        """Shared verification; each public decoder fixes the required token type."""
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
        if payload["type"] != expected_type:
            raise jwt.InvalidTokenError("Incorrect token type")
        if not isinstance(payload["sub"], str) or not payload["sub"].strip():
            raise jwt.InvalidTokenError("Invalid subject")
        if not isinstance(payload["jti"], str) or not payload["jti"].strip():
            raise jwt.InvalidTokenError("Invalid token identifier")
        # Our issuer creates integer NumericDates, also used for Redis expiry.
        if any(type(payload[name]) is not int for name in ("iat", "exp")):
            raise jwt.InvalidTokenError("Invalid token dates")
        if payload["exp"] <= payload["iat"]:
            raise jwt.InvalidTokenError("Invalid token lifetime")
        return payload
