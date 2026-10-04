"""Phase 12: tool configuration, env-driven like every other phase.

- ``SEARCH_BACKEND`` (default ``none``): ``tavily`` | ``serpapi`` |
  ``none``. Anything else behaves as ``none`` (no network call).
- ``TAVILY_API_KEY`` / ``SERPAPI_API_KEY``: provider keys. Absent keys
  make the web tool report "not configured" instead of calling out.
  Keys are never logged.
- ``SEARCH_MAX_RESULTS`` (default 5, clamped 1-10).
- ``SEARCH_TIMEOUT_SECONDS`` (default 10, clamped 1-60).
- ``CALCULATOR_MAX_EXPR_LEN`` (default 200, clamped 1-1000): bounds
  the math input the agent will evaluate.

No central ``app/config.py`` exists by design; each phase exposes
small ``os.getenv`` helpers, and this module follows that pattern.
"""

from __future__ import annotations

import os

DEFAULT_SEARCH_MAX_RESULTS = 5
DEFAULT_SEARCH_TIMEOUT_SECONDS = 10
DEFAULT_CALCULATOR_MAX_EXPR_LEN = 200


def search_backend() -> str:
    value = os.getenv("SEARCH_BACKEND", "none").strip().lower()
    if value in ("tavily", "serpapi", "none"):
        return value
    return "none"


def tavily_api_key() -> str:
    return os.getenv("TAVILY_API_KEY", "").strip()


def serpapi_api_key() -> str:
    return os.getenv("SERPAPI_API_KEY", "").strip()


def search_max_results() -> int:
    try:
        value = int(os.getenv(
            "SEARCH_MAX_RESULTS", str(DEFAULT_SEARCH_MAX_RESULTS)))
    except ValueError:
        return DEFAULT_SEARCH_MAX_RESULTS
    return max(1, min(10, value))


def search_timeout_seconds() -> int:
    try:
        value = int(os.getenv(
            "SEARCH_TIMEOUT_SECONDS",
            str(DEFAULT_SEARCH_TIMEOUT_SECONDS)))
    except ValueError:
        return DEFAULT_SEARCH_TIMEOUT_SECONDS
    return max(1, min(60, value))


def calculator_max_expr_len() -> int:
    try:
        value = int(os.getenv(
            "CALCULATOR_MAX_EXPR_LEN",
            str(DEFAULT_CALCULATOR_MAX_EXPR_LEN)))
    except ValueError:
        return DEFAULT_CALCULATOR_MAX_EXPR_LEN
    return max(1, min(1000, value))
