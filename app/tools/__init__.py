"""Phase 12: agent tool ecosystem (web search + calculator).

Every tool exposes the same minimal surface -- ``name``,
``description``, ``execute`` -- and is held by the
:class:`ToolRegistry`. The agent selects tools by asking each one
whether it handles the query (``match``), using the tool's own
description for logging and future prompt-based routing.

Design rules (see AGENTS.md):
- Config-driven: backends, keys, and limits come from env helpers in
  ``app/tools/config.py``; unset keys degrade to honest error dicts.
- Lazy: stdlib-only transports, imported inside ``execute`` paths, so
  importing this package needs no credentials or network.
- Graceful: ``execute`` never raises for input/transport problems --
  it returns ``{"ok": False, "error": ...}``. Only programmer errors
  (wrong Python types) raise.
"""

from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry, default_registry
from app.tools.web_search import WebSearchTool

__all__ = [
    "BaseTool",
    "CalculatorTool",
    "ToolRegistry",
    "WebSearchTool",
    "default_registry",
]
