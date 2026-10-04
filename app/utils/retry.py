"""Phase 6: retries with exponential backoff plus a circuit breaker.

External calls (Gemini, MongoDB) fail transiently: timeouts, dropped
connections, 5xx/429. This module retries exactly those, with
exponential backoff and jitter, and stops calling a dependency that
keeps failing (circuit breaker) so sustained outages fail fast instead
of stacking latency on every request.

Design rules (see AGENTS.md):
- Retry only transient failures: timeouts, connection errors, and
  status codes 429 / 5xx (duck-typed via ``code``/``status_code``/
  ``status`` attributes, so google-genai ``APIError`` and httpx errors
  work without importing either SDK). Never retry other 4xx, duplicate
  keys, auth errors, validation bugs, or cancellations
  (``CancelledError``/``GeneratorExit`` are ``BaseException`` and are
  never caught here at all).
- Sync-first like the rest of the pipeline, but async functions are
  supported too (``asyncio.sleep`` between attempts). Generators retry
  only before the first item is produced: once a stream has emitted,
  replaying it would duplicate output, so mid-stream errors propagate.
- The breaker counts consecutive *transient* failures per named
  dependency (``circuit="gemini"`` / ``circuit="mongodb"``); after
  ``CIRCUIT_BREAKER_THRESHOLD`` (default 5) it fails fast for
  ``CIRCUIT_BREAKER_COOLDOWN_SECONDS`` (default 30), then allows one
  probe: success closes it, failure re-opens it. Thread-safe.
- Stdlib-only: importing this module needs no credentials, network,
  or third-party packages.

Write path vs retry: retried inserts use fresh backend uuids per
attempt (existing service code generates them inside the method), so a
retry after a lost reply can theoretically persist a duplicate row.
That at-least-once tradeoff is accepted here the same way MongoDB's
own retryable writes accept it: losing chat turns on every network
blip is the worse failure mode, and the breaker bounds sustained
outages to fast failures.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import random
import threading
import time

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 8.0
DEFAULT_CB_THRESHOLD = 5
DEFAULT_CB_COOLDOWN = 30.0


def retry_max_attempts() -> int:
    try:
        value = int(os.getenv("RETRY_MAX_ATTEMPTS", str(DEFAULT_MAX_ATTEMPTS)))
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS
    return value if value >= 1 else DEFAULT_MAX_ATTEMPTS


def retry_base_delay() -> float:
    try:
        value = float(os.getenv("RETRY_BASE_DELAY", str(DEFAULT_BASE_DELAY)))
    except ValueError:
        return DEFAULT_BASE_DELAY
    return value if value >= 0 else DEFAULT_BASE_DELAY


def retry_max_delay() -> float:
    try:
        value = float(os.getenv("RETRY_MAX_DELAY", str(DEFAULT_MAX_DELAY)))
    except ValueError:
        return DEFAULT_MAX_DELAY
    return value if value > 0 else DEFAULT_MAX_DELAY


def circuit_threshold() -> int:
    try:
        value = int(os.getenv(
            "CIRCUIT_BREAKER_THRESHOLD", str(DEFAULT_CB_THRESHOLD)))
    except ValueError:
        return DEFAULT_CB_THRESHOLD
    return value if value >= 1 else DEFAULT_CB_THRESHOLD


def circuit_cooldown() -> float:
    try:
        value = float(os.getenv(
            "CIRCUIT_BREAKER_COOLDOWN_SECONDS", str(DEFAULT_CB_COOLDOWN)))
    except ValueError:
        return DEFAULT_CB_COOLDOWN
    return value if value > 0 else DEFAULT_CB_COOLDOWN


class CircuitBreakerOpen(Exception):
    """A named dependency is circuit-open; the call was not attempted."""


class CircuitBreaker:
    """Consecutive-transient-failure breaker for one dependency.

    Closed until ``threshold`` transient failures in a row, then open
    for ``cooldown`` seconds (fail fast), then half-open: the next call
    is a probe that closes the breaker on success or re-opens it on
    failure. Any success resets the failure streak. Thread-safe.
    """

    def __init__(
        self,
        threshold: int | None = None,
        cooldown: float | None = None,
    ):
        self.threshold = threshold if threshold is not None else 5
        self.cooldown = cooldown if cooldown is not None else 30.0
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            return (
                time.monotonic() - self._opened_at < self.cooldown
            )

    def allow(self) -> bool:
        """Whether a call may proceed (closed, or a half-open probe)."""
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= self.cooldown:
                logger.debug("Circuit half-open: allowing probe call")
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures < self.threshold:
                return
            now = time.monotonic()
            # Always (re)start the cooldown on a threshold-meeting
            # failure: this both opens a closed breaker and re-opens it
            # after a failed half-open probe (whose timer already
            # expired, so without the refresh the breaker would stay
            # permissive). Calls can only reach here when allowed
            # (closed or probing), so concurrent re-opens just agree.
            if self._opened_at is None or now - self._opened_at >= self.cooldown:
                self._opened_at = now
                logger.warning(
                    "Circuit opened after %d consecutive failures; "
                    "failing fast for %.0fs",
                    self._failures, self.cooldown,
                )

    def reset(self) -> None:
        """Test seam: return to the closed, zero-failure state."""
        with self._lock:
            self._failures = 0
            self._opened_at = None


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """Shared breaker per dependency name (env-configured at creation)."""
    with _breakers_lock:
        breaker = _breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                threshold=circuit_threshold(),
                cooldown=circuit_cooldown(),
            )
            _breakers[name] = breaker
        return breaker


def reset_breakers() -> None:
    """Test seam: drop all shared breakers so tests start closed."""
    with _breakers_lock:
        _breakers.clear()


def _status_code(error: BaseException) -> int | None:
    """HTTP-ish status from SDK-agnostic attributes, else None.

    Only real ints count: Mock auto-attributes and bools must never
    steer retry decisions.
    """
    for attribute in ("code", "status_code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _is_pymongo_transient(error: BaseException) -> bool:
    """Connection-level PyMongo failures (import-guarded, no hard dep)."""
    try:
        from pymongo.errors import (
            AutoReconnect,
            ConnectionFailure,
            NetworkTimeout,
            ServerSelectionTimeoutError,
        )
    except Exception:
        return False
    return isinstance(error, (
        AutoReconnect,
        ConnectionFailure,
        NetworkTimeout,
        ServerSelectionTimeoutError,
    ))


def is_transient_error(error: BaseException) -> bool:
    """Whether this failure is worth one more attempt.

    Timeouts, connection errors, and 429/5xx status codes retry. A
    fail-fast ``CircuitBreakerOpen`` never retries (that would defeat
    the breaker), and anything else -- other 4xx, duplicate keys, auth
    errors, programming bugs -- raises immediately.

    The ``__cause__`` chain is followed: services translate driver
    errors into domain exceptions (``raise StoreUnavailable from
    error``), and the retry decision must see through that wrapper to
    the original failure. Cycles are guarded.
    """
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, CircuitBreakerOpen):
            return False
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if _is_pymongo_transient(current):
            return True
        status = _status_code(current)
        if status is not None:
            return status == 429 or 500 <= status <= 599
        # getattr: Mock objects raise AttributeError on dunder access.
        cause = getattr(current, "__cause__", None)
        current = cause if isinstance(cause, BaseException) else None
    return False


def backoff_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
) -> float:
    """Exponential backoff with jitter for a just-failed attempt.

    ``attempt`` is 1-based: the first retry waits ~``base_delay``.
    Jitter (50-100% of the exponential value) keeps fleet-wide retries
    from stampeding the recovering dependency.
    """
    capped = min(max_delay, base_delay * (2 ** max(attempt - 1, 0)))
    return capped * random.uniform(0.5, 1.0)


def _resolve_breaker(circuit, breaker):
    if breaker is not None:
        return breaker
    if circuit:
        return get_breaker(circuit)
    return None


def _log_retry(action, error, attempt, max_attempts, delay):
    logger.warning(
        "%s failed with %s (attempt %d/%d); retrying in %.2fs",
        action, type(error).__name__, attempt, max_attempts, delay,
        extra={
            "retry_attempt": attempt,
            "retry_max_attempts": max_attempts,
            "retry_delay": round(delay, 3),
            "retry_error": type(error).__name__,
        },
    )


def retry_with_backoff(_fn=None, *, attempts=None, base_delay=None,
                       max_delay=None, retry_on=None, circuit=None,
                       breaker=None, sleep=None, action=None):
    """Retry transient failures with backoff; optionally circuit-break.

    Works on sync functions, async functions, and (sync) generators.
    Usable bare (``@retry_with_backoff``) or parametrized.
    ``attempts``/``base_delay``/``max_delay`` default to the
    ``RETRY_*`` env config resolved per call. ``retry_on`` defaults to
    :func:`is_transient_error`. ``circuit="name"`` shares a fail-fast
    breaker across all functions guarding that dependency; pass an
    explicit ``breaker`` instance instead in tests.
    """

    def decorate(fn):
        name = action or getattr(fn, "__qualname__", "operation")
        predicate = retry_on or is_transient_error

        def located_breaker():
            return _resolve_breaker(circuit, breaker)

        if inspect.isasyncgenfunction(fn) or inspect.isgeneratorfunction(fn):
            if inspect.isasyncgenfunction(fn):
                raise TypeError(
                    "retry_with_backoff does not support async generators"
                )

            @functools.wraps(fn)
            def generator_wrapper(*args, **kwargs):
                wait = sleep or time.sleep
                attempt = 0
                while True:
                    attempt += 1
                    located = located_breaker()
                    if located is not None and not located.allow():
                        raise CircuitBreakerOpen(
                            f"Circuit breaker open for {circuit or 'dependency'}; "
                            "failing fast"
                        )
                    stream = fn(*args, **kwargs)
                    try:
                        first = next(stream)
                    except StopIteration:
                        if located is not None:
                            located.record_success()
                        return
                    except Exception as error:
                        max_tries = attempts if attempts is not None else retry_max_attempts()
                        if predicate(error) and attempt < max_tries:
                            if located is not None:
                                located.record_failure()
                            delay = backoff_delay(
                                attempt,
                                base_delay if base_delay is not None else retry_base_delay(),
                                max_delay if max_delay is not None else retry_max_delay(),
                            )
                            _log_retry(name, error, attempt, max_tries, delay)
                            wait(delay)
                            continue
                        if predicate(error) and located is not None:
                            located.record_failure()
                        raise
                    if located is not None:
                        located.record_success()
                    # The stream has started: from here errors propagate
                    # without retry so output is never duplicated.
                    yield first
                    yield from stream
                    return

            return generator_wrapper

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                max_tries = attempts if attempts is not None else retry_max_attempts()
                base = base_delay if base_delay is not None else retry_base_delay()
                ceiling = max_delay if max_delay is not None else retry_max_delay()
                wait = sleep or asyncio.sleep
                attempt = 0
                while True:
                    attempt += 1
                    located = located_breaker()
                    if located is not None and not located.allow():
                        raise CircuitBreakerOpen(
                            f"Circuit breaker open for {circuit or 'dependency'}; "
                            "failing fast"
                        )
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as error:
                        if predicate(error) and attempt < max_tries:
                            if located is not None:
                                located.record_failure()
                            delay = backoff_delay(attempt, base, ceiling)
                            _log_retry(name, error, attempt, max_tries, delay)
                            await wait(delay)
                            continue
                        if predicate(error) and located is not None:
                            located.record_failure()
                        raise
                    if located is not None:
                        located.record_success()
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            max_tries = attempts if attempts is not None else retry_max_attempts()
            base = base_delay if base_delay is not None else retry_base_delay()
            ceiling = max_delay if max_delay is not None else retry_max_delay()
            wait = sleep or time.sleep
            attempt = 0
            while True:
                attempt += 1
                located = located_breaker()
                if located is not None and not located.allow():
                    raise CircuitBreakerOpen(
                        f"Circuit breaker open for {circuit or 'dependency'}; "
                        "failing fast"
                    )
                try:
                    result = fn(*args, **kwargs)
                except Exception as error:
                    if predicate(error) and attempt < max_tries:
                        if located is not None:
                            located.record_failure()
                        delay = backoff_delay(attempt, base, ceiling)
                        _log_retry(name, error, attempt, max_tries, delay)
                        wait(delay)
                        continue
                    if predicate(error) and located is not None:
                        located.record_failure()
                    raise
                if located is not None:
                    located.record_success()
                return result

        return sync_wrapper

    if _fn is not None:
        return decorate(_fn)
    return decorate
