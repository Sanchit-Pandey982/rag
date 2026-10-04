"""Phase 4: JWT access-token blacklisting backed by Redis.

Access tokens are stateless by design, so a logged-out (or rotated)
token would otherwise stay valid until its 15-minute expiry. This
service records revoked access-token JTIs as
``auth:blacklist:<jti> -> 1`` with ``EXAT = token expiry``: entries
vanish on their own, so no cleanup job exists.

``get_current_user`` checks the list before returning a user; logout
(and refresh) blacklist the presented access token on a best-effort
basis -- both endpoints also accept requests without one, preserving
their existing contracts.
"""

import time

import redis.asyncio as redis
from redis.exceptions import RedisError


class BlacklistUnavailable(Exception):
    """The blacklist could not be reliably read or changed."""


class TokenBlacklistService:
    key_prefix = "auth:blacklist:"

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    async def blacklist_jti(self, jti: str, *, expires_at: int) -> None:
        """Record a revoked access token until its natural expiry.

        Idempotent: an already-listed JTI stays listed. JTIs whose
        expiry already passed are skipped (nothing left to protect).
        """
        if expires_at <= int(time.time()):
            return
        try:
            # Absolute expiry keeps the key lifetime tied to the JWT,
            # never extending it through I/O delay. NX keeps concurrent
            # duplicate revocations from erroring.
            await self.redis_client.set(
                self.key_prefix + jti, "1", exat=expires_at, nx=True,
            )
        except RedisError as error:
            raise BlacklistUnavailable(
                "Could not blacklist access token") from error

    async def is_blacklisted(self, jti: str) -> bool:
        try:
            return bool(
                await self.redis_client.exists(self.key_prefix + jti)
            )
        except RedisError as error:
            raise BlacklistUnavailable(
                "Could not check access token blacklist") from error
