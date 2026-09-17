"""Redis stores refresh-token permissions, never raw refresh JWTs."""

import redis.asyncio as redis
from redis.exceptions import RedisError


class RefreshSessionUnavailable(Exception):
    """Session state could not be reliably read or changed."""


class RefreshTokenService:
    key_prefix = "auth:refresh:"

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    async def store_refresh_token(self, jti: str, user_id: str, *, expires_at: int) -> None:
        try:
            # Absolute expiry avoids extending the JWT lifetime through I/O delay.
            stored = await self.redis_client.set(
                self.key_prefix + jti, user_id, exat=expires_at, nx=True,
            )
        except RedisError as error:
            raise RefreshSessionUnavailable("Could not store refresh session") from error
        if not stored:
            # Never overwrite an existing session, even on an identifier collision.
            raise RefreshSessionUnavailable("Refresh session already exists")

    async def consume_refresh_token(self, jti: str) -> str | None:
        try:
            # One Redis command makes concurrent use of the same token single-use.
            return await self.redis_client.getdel(self.key_prefix + jti)
        except RedisError as error:
            raise RefreshSessionUnavailable("Could not consume refresh session") from error

    async def revoke_refresh_token(self, jti: str) -> None:
        try:
            await self.redis_client.delete(self.key_prefix + jti)
        except RedisError as error:
            raise RefreshSessionUnavailable("Could not revoke refresh session") from error
