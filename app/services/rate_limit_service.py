"""Phase 3.9: Redis fixed-window rate limiting + Wave 1.3 daily quota.

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

Wave 1.3 adds a per-user daily chat quota on top of the burst limiter.
The counter key carries the UTC date and expires at the next UTC
midnight, so "tomorrow" starts with a fresh bucket without a cron job.
The quota is an abuse guard, not a billing meter; exempt tenants (the
corpus/eval identity by default) skip it.
"""

import logging
import math
import os
from datetime import datetime, time, timedelta, timezone

import redis.asyncio as redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)

SCOPE_REGISTER = "register"
SCOPE_LOGIN = "login"
SCOPE_REFRESH = "refresh"
SCOPE_CHAT = "chat"
SCOPE_UPLOAD = "upload"
# Not a fixed-window scope: the daily quota has its own counter/expiry.
SCOPE_CHAT_DAILY = "chat_daily"

DEFAULT_CHAT_DAILY_QUOTA = 200
DEFAULT_CHAT_QUOTA_EXEMPT_USERS = frozenset({"eval_user"})

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

# Daily quota: count, expire at the next UTC midnight (TTL seconds supplied
# by the caller), report the remaining TTL for Retry-After. The TTL check
# repairs a key that somehow lost its expiry instead of stranding it.
_LUA_DAILY_QUOTA = """local current = redis.call('INCR', KEYS[1])
if redis.call('TTL', KEYS[1]) < 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return {current, redis.call('TTL', KEYS[1])}"""


def get_chat_daily_quota() -> int:
    """Read ``CHAT_DAILY_QUOTA``; invalid/non-positive values use the default."""
    raw = os.getenv(
        "CHAT_DAILY_QUOTA", str(DEFAULT_CHAT_DAILY_QUOTA)
    ).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CHAT_DAILY_QUOTA
    return value if value > 0 else DEFAULT_CHAT_DAILY_QUOTA


def get_chat_quota_exempt_users() -> frozenset[str]:
    """Read ``CHAT_QUOTA_EXEMPT_USERS`` (comma-separated ids).

    Unset keeps the default eval tenant; set-but-empty means nobody is
    exempt, which is useful for testing the guard itself.
    """
    raw = os.getenv("CHAT_QUOTA_EXEMPT_USERS")
    if raw is None:
        return DEFAULT_CHAT_QUOTA_EXEMPT_USERS
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def seconds_until_utc_midnight(now: datetime | None = None) -> int:
    current = now or datetime.now(timezone.utc)
    tomorrow = (current + timedelta(days=1)).date()
    midnight = datetime.combine(tomorrow, time.min, tzinfo=timezone.utc)
    return max(1, math.ceil((midnight - current).total_seconds()))


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
        daily_quota: int | None = None,
        quota_exempt_users: frozenset[str] | set[str] | None = None,
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
        self.daily_quota = (
            DEFAULT_CHAT_DAILY_QUOTA if daily_quota is None else daily_quota
        )
        if self.daily_quota <= 0:
            raise ValueError("daily_quota must be a positive integer")
        self.quota_exempt_users = frozenset(quota_exempt_users or ())

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

    async def check_daily_quota(self, *, user_id: str) -> None:
        """Consume one daily chat turn; raise when the quota is spent.

        Exempt users (the eval corpus tenant by default) never touch Redis
        here. The key includes the UTC date, so a late-night turn and an
        early-morning turn cannot share a bucket even if the previous key
        somehow outlived its expiry.
        """
        if user_id in self.quota_exempt_users:
            return
        now = datetime.now(timezone.utc)
        key = self.bucket(
            SCOPE_CHAT_DAILY, f"user:{user_id}:{now.strftime('%Y-%m-%d')}"
        )
        try:
            count, ttl_seconds = await self.redis_client.eval(
                _LUA_DAILY_QUOTA,
                1,
                key,
                seconds_until_utc_midnight(now),
            )
        except RedisError as error:
            raise RateLimitUnavailable("Rate limiter unavailable") from error
        if int(count) > self.daily_quota:
            raise RateLimitExceeded(max(1, int(ttl_seconds)))
