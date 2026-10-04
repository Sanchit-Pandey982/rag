"""Phase 8: token usage + cost tracking backed by MongoDB.

Every answered generation reports its Gemini token counts
(``response.usage_metadata``); this service persists one row per
request and aggregates daily totals for the usage endpoint:

    usage_logs: usage_id, user_id, timestamp, prompt_tokens,
                completion_tokens, total_tokens, model, cost_usd

Cost is estimated at ``COST_PER_1K_INPUT`` USD per 1K prompt tokens and
``COST_PER_1K_OUTPUT`` USD per 1K completion tokens (configurable).

Scope note: only generation calls are tracked (the per-request LLM
cost). Query-rewrite (``condense_question``) and conversation-summary
calls carry no user context at their call sites, so they stay out of
the ledger by design, not by accident.

Recording is retried like every other Mongo op (``circuit="mongodb"``);
callers treat logging as observability and stay fail-open, so a store
outage never breaks a chat turn.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo.errors import PyMongoError

from app.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

DEFAULT_COST_PER_1K_INPUT = 0.0001
DEFAULT_COST_PER_1K_OUTPUT = 0.0004


def cost_per_1k_input() -> float:
    try:
        value = float(os.getenv(
            "COST_PER_1K_INPUT", str(DEFAULT_COST_PER_1K_INPUT)))
    except ValueError:
        return DEFAULT_COST_PER_1K_INPUT
    return value if value >= 0 else DEFAULT_COST_PER_1K_INPUT


def cost_per_1k_output() -> float:
    try:
        value = float(os.getenv(
            "COST_PER_1K_OUTPUT", str(DEFAULT_COST_PER_1K_OUTPUT)))
    except ValueError:
        return DEFAULT_COST_PER_1K_OUTPUT
    return value if value >= 0 else DEFAULT_COST_PER_1K_OUTPUT


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    input_rate: float | None = None,
    output_rate: float | None = None,
) -> float:
    """USD estimate for one request at the configured per-1K rates."""
    if input_rate is None:
        input_rate = cost_per_1k_input()
    if output_rate is None:
        output_rate = cost_per_1k_output()
    return prompt_tokens / 1000 * input_rate + completion_tokens / 1000 * output_rate


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UsageStoreUnavailable(Exception):
    """MongoDB could not be read or changed; never surfaces internals."""


def _checked_count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


class UsageService:
    def __init__(self, usage_logs):
        self.usage_logs = usage_logs

    @retry_with_backoff(circuit="mongodb")
    async def ensure_indexes(self) -> None:
        await self.usage_logs.create_index("usage_id", unique=True)
        # Supports the ownership-scoped daily aggregation
        # (match user + recent window) from a single index.
        await self.usage_logs.create_index([("user_id", 1), ("timestamp", -1)])

    @retry_with_backoff(circuit="mongodb")
    async def record_usage(
        self,
        *,
        user_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: str,
        cost_usd: float | None = None,
    ) -> dict[str, Any]:
        """Persist one request's token ledger row."""
        prompt_tokens = _checked_count("prompt_tokens", prompt_tokens)
        completion_tokens = _checked_count(
            "completion_tokens", completion_tokens)
        total_tokens = _checked_count("total_tokens", total_tokens)
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        if cost_usd is None:
            cost_usd = estimate_cost(prompt_tokens, completion_tokens)
        if isinstance(cost_usd, bool) or not isinstance(
                cost_usd, (int, float)) or cost_usd < 0:
            raise ValueError("cost_usd must be a non-negative number")
        document = {
            "usage_id": str(uuid4()),
            "user_id": user_id,
            "timestamp": _utcnow(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model": model,
            "cost_usd": float(cost_usd),
        }
        try:
            await self.usage_logs.insert_one(document)
        except PyMongoError as error:
            logger.exception("Could not record token usage")
            raise UsageStoreUnavailable(
                "Usage store unavailable") from error
        return document

    @retry_with_backoff(circuit="mongodb")
    async def get_usage_summary(
        self, user_id: str, days: int = 7
    ) -> dict[str, Any]:
        """Daily totals plus rolled-up totals for the trailing window.

        ``days`` counts back from now (``7`` ≈ weekly, ``1`` ≈ daily).
        Dates are UTC calendar days (``%Y-%m-%d``), matching MongoDB's
        ``$dateToString`` default timezone.
        """
        if isinstance(days, bool) or not isinstance(days, int):
            raise ValueError("days must be an int")
        if not 1 <= days <= 31:
            raise ValueError("days must be between 1 and 31")
        cutoff = _utcnow() - timedelta(days=days)
        pipeline = [
            {"$match": {
                "user_id": user_id,
                "timestamp": {"$gte": cutoff},
            }},
            {"$group": {
                "_id": {"$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": "$timestamp",
                }},
                "prompt_tokens": {"$sum": "$prompt_tokens"},
                "completion_tokens": {"$sum": "$completion_tokens"},
                "total_tokens": {"$sum": "$total_tokens"},
                "cost_usd": {"$sum": "$cost_usd"},
                "requests": {"$sum": 1},
            }},
            {"$sort": {"_id": 1}},
        ]
        try:
            cursor = self.usage_logs.aggregate(pipeline)
            rows = await cursor.to_list(length=days + 1)
        except PyMongoError as error:
            logger.exception("Could not aggregate token usage")
            raise UsageStoreUnavailable(
                "Usage store unavailable") from error
        daily = [
            {
                "date": row["_id"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
                "cost_usd": round(row["cost_usd"], 6),
                "requests": row["requests"],
            }
            for row in rows
        ]
        total = {
            "prompt_tokens": sum(day["prompt_tokens"] for day in daily),
            "completion_tokens": sum(
                day["completion_tokens"] for day in daily),
            "total_tokens": sum(day["total_tokens"] for day in daily),
            "cost_usd": round(sum(day["cost_usd"] for day in daily), 6),
            "requests": sum(day["requests"] for day in daily),
        }
        return {
            "user_id": user_id,
            "days": days,
            "total": total,
            "daily": daily,
        }
