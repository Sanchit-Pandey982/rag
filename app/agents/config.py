"""Phase 11: agent configuration, env-driven like every other phase.

- ``AGENT_ENABLED`` (default false): master switch. ``None`` overrides
  in code follow this flag, so default behavior is byte-identical to
  the legacy pipeline until opted in.
- ``AGENT_MAX_ITERATIONS`` (default 3): hard cap on
  retrieve → generate → check_confidence loops per request.
- ``AGENT_CONFIDENCE_THRESHOLD`` (default 0.5): answers scoring below
  this retry (up to the iteration cap), then the best-effort answer
  (or the refusal) is returned.

No central ``app/config.py`` exists by design; each phase exposes
small ``os.getenv`` helpers, and this module follows that pattern.
"""

from __future__ import annotations

import os

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def agent_enabled() -> bool:
    return os.getenv("AGENT_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def agent_max_iterations() -> int:
    try:
        value = int(os.getenv(
            "AGENT_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
    except ValueError:
        return DEFAULT_MAX_ITERATIONS
    if value < 1:
        return 1
    if value > 10:
        return 10
    return value


def agent_confidence_threshold() -> float:
    try:
        value = float(os.getenv(
            "AGENT_CONFIDENCE_THRESHOLD",
            str(DEFAULT_CONFIDENCE_THRESHOLD)))
    except ValueError:
        return DEFAULT_CONFIDENCE_THRESHOLD
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value
