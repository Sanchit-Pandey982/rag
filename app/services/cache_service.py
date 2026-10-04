"""Phase 5: Redis-backed LLM response caching.

Identical queries against unchanged documents get the stored answer
instead of a fresh Gemini call. Keyed by
``sha256(user_id + raw_query + sorted chunk ids)`` under a per-user
generation counter, so any document ingest/delete for that user
orphans every prior entry in O(1) (a single ``INCR``); orphans expire
on their own via the entry TTL, so no cleanup job exists.

Design rules (see AGENTS.md):
- Flag-gated: ``CACHE_ENABLED`` defaults to false, so default behavior
  is byte-identical to uncached (no extra result keys, ``done`` stays
  ``{}``). ``CACHE_TTL_SECONDS`` defaults to 3600.
- Fail-open: any Redis failure degrades to a transparent miss (lookup)
  or a skipped store -- caching must never break or delay a chat turn.
  Unlike auth blacklisting, there is no exception type here on purpose.
- Only successful, grounded answers are stored: errors propagate before
  any store, and refusals (``REFUSAL_MESSAGE``) are skipped by the
  caller. Empty answers are skipped too.
- Sync-first: the RAG pipeline (``phase1.run_once*``) is synchronous,
  so this service exposes sync ``get``/``store``/``invalidate_user``
  wrappers that bridge the async Redis client (same convention as
  ``chat_orchestration._await``). Async ``aget``/``astore``/
  ``ainvalidate_user`` variants exist for future async callers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os

import redis.asyncio as redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600

# Oversized LLM outputs are never cached; Redis stays a fast small-value
# store and a runaway generation cannot evict unrelated keys.
MAX_PAYLOAD_BYTES = 256 * 1024


def cache_enabled() -> bool:
    return os.getenv("CACHE_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def cache_ttl_seconds() -> int:
    try:
        value = int(os.getenv("CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
    except ValueError:
        return DEFAULT_TTL_SECONDS
    return value if value > 0 else DEFAULT_TTL_SECONDS


def build_cache_key(
    user_id: str,
    raw_query: str,
    chunk_ids: list[str],
    generation: str | int,
) -> str:
    """Deterministic entry key: user + query + sorted chunk ids + gen.

    Chunk ids already capture every retrieval variable (k, threshold,
    hybrid/rerank outcome), so the key needs nothing else. Chunk order
    is normalized: the same chunk set in a different order is one entry.
    """
    digest = hashlib.sha256()
    digest.update(user_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(raw_query.encode("utf-8"))
    digest.update(b"\x00")
    for chunk_id in sorted(chunk_ids):
        digest.update(str(chunk_id).encode("utf-8"))
        digest.update(b"\x00")
    return f"response_cache:{user_id}:{generation}:{digest.hexdigest()}"


def _run_sync(coro):
    """Bridge an async Redis call from sync pipeline code.

    Returns ``None`` when the bridge itself cannot run (a loop is
    already running in this thread): callers treat that as
    cache-unavailable, i.e. a transparent miss / skipped store.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    logger.debug("Event loop already running; skipping response cache")
    return None


class CacheService:
    """Small JSON values in Redis: one entry per answered query."""

    key_prefix = "response_cache:"
    gen_prefix = "response_cache:gen:"

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    # ----------------------------------------------------------
    # Async primitives (fail-open: Redis errors -> miss/no-op)
    # ----------------------------------------------------------

    async def current_generation(self, user_id: str) -> str:
        """Per-user corpus version; ``"0"`` until the first invalidation."""
        try:
            generation = await self.redis_client.get(
                self.gen_prefix + user_id
            )
        except RedisError:
            logger.warning("Response cache generation unreadable",
                           exc_info=True)
            return "0"
        return str(generation) if generation is not None else "0"

    async def aget(
        self,
        user_id: str,
        raw_query: str,
        chunk_ids: list[str],
    ) -> dict | None:
        """Stored answer payload, or None on miss/unreadable cache."""
        try:
            generation = await self.current_generation(user_id)
            raw = await self.redis_client.get(
                build_cache_key(user_id, raw_query, chunk_ids, generation)
            )
        except RedisError:
            logger.warning("Response cache lookup failed", exc_info=True)
            return None
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("Response cache entry is not valid JSON")
            return None
        if not isinstance(payload, dict) or not payload.get("answer"):
            return None
        return payload

    async def astore(
        self,
        user_id: str,
        raw_query: str,
        chunk_ids: list[str],
        payload: dict,
    ) -> bool:
        """Store an answer payload; False when skipped or unwritable."""
        if not isinstance(payload, dict) or not payload.get("answer"):
            return False
        try:
            encoded = json.dumps(payload)
        except (ValueError, TypeError):
            logger.warning("Response cache payload is not JSON-serializable")
            return False
        if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            logger.debug("Response cache payload too large; skipping store")
            return False
        try:
            generation = await self.current_generation(user_id)
            await self.redis_client.set(
                build_cache_key(user_id, raw_query, chunk_ids, generation),
                encoded,
                ex=cache_ttl_seconds(),
            )
        except RedisError:
            logger.warning("Response cache store failed", exc_info=True)
            return False
        return True

    async def ainvalidate_user(self, user_id: str) -> None:
        """Bump the user's corpus generation; prior entries go unread.

        Single atomic ``INCR`` -- O(1) no matter how many entries the
        user has. Orphaned entries keep their TTL and vanish on their
        own; worst-case staleness after an upload is one TTL window,
        and only when this call itself failed.
        """
        try:
            await self.redis_client.incr(self.gen_prefix + user_id)
        except RedisError:
            logger.warning("Response cache invalidation failed",
                           exc_info=True)

    # ----------------------------------------------------------
    # Sync wrappers for the synchronous RAG pipeline
    # ----------------------------------------------------------

    def get(
        self,
        user_id: str,
        raw_query: str,
        chunk_ids: list[str],
    ) -> dict | None:
        result = _run_sync(self.aget(user_id, raw_query, chunk_ids))
        return result if isinstance(result, dict) else None

    def store(
        self,
        user_id: str,
        raw_query: str,
        chunk_ids: list[str],
        payload: dict,
    ) -> bool:
        return bool(
            _run_sync(self.astore(user_id, raw_query, chunk_ids, payload))
        )

    def invalidate_user(self, user_id: str) -> None:
        _run_sync(self.ainvalidate_user(user_id))
