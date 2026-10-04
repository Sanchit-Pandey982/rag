"""Phase 7: Redis counter tracking degraded LLM responses.

When Gemini is unavailable (after Phase 6 retries are exhausted), the
chat path serves document excerpts instead of an error. This service
counts those fallbacks in Redis so outages are visible in monitoring.
One global total key, bumped with a single atomic ``INCR`` at the
moment a degraded answer is produced.

Fail-open like the response cache (never raises into chat): if Redis
is down, the degraded answer is still served, just uncounted.
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


def _run_sync(coro):
    """Bridge an async Redis call from the synchronous RAG pipeline.

    Returns ``None`` when a loop is already running in this thread;
    callers treat that as tracker-unavailable (serve degraded anyway).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    logger.debug("Event loop already running; skipping degraded counter")
    return None


class DegradationTracker:
    """Single Redis total of served degraded responses."""

    key = "llm:degraded:responses"

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    async def aincrement(self) -> None:
        try:
            await self.redis_client.incr(self.key)
        except RedisError:
            logger.warning("Degraded-response counter unwritable",
                           exc_info=True)

    async def acount(self) -> int:
        try:
            value = await self.redis_client.get(self.key)
        except RedisError:
            logger.warning("Degraded-response counter unreadable",
                           exc_info=True)
            return 0
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    def increment(self) -> None:
        _run_sync(self.aincrement())

    def count(self) -> int:
        result = _run_sync(self.acount())
        return result if isinstance(result, int) else 0
