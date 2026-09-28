"""Phase 3.9: Redis fixed-window rate limiting.

Namespace: ``rate_limit:<scope>:<kind>:<value>`` (``ip:`` pre-auth,
``user:`` once the JWT identity is known), far from ``auth:refresh:``.

Atomicity comes from a single Lua script (INCR + conditional PEXPIRE +
PTTL in one round trip): separate INCR-then-EXPIRE calls could strand a
TTL-less key on crash and block its bucket forever. Fixed-window was
chosen over sliding-window-log (more memory/commands per request) and
token-bucket (smoother, more machinery); the edge-burst is bounded at 2x
and acceptable at these limits.

A Redis outage fails OPEN (logged), not closed: coupling chat/upload
availability to Redis would contradict the Phase 3.4 invariant that
access-token requests survive a Redis outage. Login/refresh already fail
closed for session reasons, which bounds fail-open abuse to
already-issued (15-minute) access tokens.
"""

import logging
import math

import redis.asyncio as redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)

SCOPE_REGISTER = "register"
SCOPE_LOGIN = "login"
SCOPE_REFRESH = "refresh"
SCOPE_CHAT = "chat"
SCOPE_UPLOAD = "upload"

# (limit, window_seconds) per scope. Login is tight enough to blunt
# brute force and the per-attempt Argon2 cost behind it; registration is
# stricter (account farming); refresh is generous (healthy clients call
# it rarely; concurrent tabs burst); chat/upload are priced by Gemini
# spend behind them.
DEFAULT_LIMITS = {
    SCOPE_REGISTER: (5, 300),
    SCOPE_LOGIN: (10, 60),
    SCOPE_REFRESH: (30, 60),
    SCOPE_CHAT: (30, 60),
    SCOPE_UPLOAD: (10, 60),
}

# One round trip: count, set the window on first hit, report TTL for
# Retry-After. EVAL (not EVALSHA) keeps this dependency-free at our scale.
_LUA_FIXED_WINDOW = """local current = redis.call('INCR', KEYS[1])
if current == 1 then redis.call('PEXPIRE', KEYS[1], ARGV[1]) end
return {current, redis.call('PTTL', KEYS[1])}"""


class RateLimitExceeded(Exception):
    """The bucket is exhausted; carries seconds until retry."""

    def __init__(self, retry_after_seconds: int):
        super().__init__("Rate limit exceeded")
        self.retry_after_seconds = max(1, int(retry_after_seconds))


class RateLimitUnavailable(Exception):
    """Redis could not be read or changed; callers fail open (logged)."""


class RateLimitService:
    key_prefix = "rate_limit:"

    def __init__(
        self,
        redis_client: redis.Redis,
        limits: dict[str, tuple[int, int]] | None = None,
    ):
        self.redis_client = redis_client
        self.limits = dict(DEFAULT_LIMITS)
        if limits:
            for scope, (limit, window_seconds) in limits.items():
                if scope not in self.limits:
                    raise ValueError(f"Unknown rate limit scope: {scope}")
                if limit <= 0 or window_seconds <= 0:
                    raise ValueError(
                        f"Invalid limit for scope {scope}: "
                        f"({limit}, {window_seconds})"
                    )
                self.limits[scope] = (limit, window_seconds)

    def bucket(self, scope: str, key: str) -> str:
        return f"{self.key_prefix}{scope}:{key}"

    async def check(self, *, scope: str, key: str) -> None:
        """Consume one token; raise on exhaustion or Redis outage."""
        if scope not in self.limits:
            raise ValueError(f"Unknown rate limit scope: {scope}")
        limit, window_seconds = self.limits[scope]
        try:
            count, ttl_ms = await self.redis_client.eval(
                _LUA_FIXED_WINDOW,
                1,
                self.bucket(scope, key),
                window_seconds * 1000,
            )
        except RedisError as error:
            raise RateLimitUnavailable(
                "Rate limiter unavailable"
            ) from error
        if int(count) > limit:
            raise RateLimitExceeded(math.ceil(int(ttl_ms) / 1000))
