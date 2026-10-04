"""Phase 12: shared tool interface.

``BaseTool`` fixes the registry contract: every tool has a ``name``,
a human/LLM-readable ``description`` (used by the agent to decide
which tool to call), an ``execute`` entry point, and a ``match``
selector that returns the tool input for a query or ``None`` when the
tool does not apply. New tools subclass this and register -- no agent
changes needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Minimal surface every agent tool must implement."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def match(self, query: str) -> Any | None:
        """Return the tool input for ``query``, or ``None`` to pass."""

    @abstractmethod
    def execute(self, tool_input: Any) -> dict:
        """Run the tool; always a dict, never raises on soft failure.

        Success: ``{"ok": True, ...tool-specific payload}``.
        Soft failure (bad input, missing key, network trouble):
        ``{"ok": False, "error": "<human-readable reason>"}``.
        """

    def describe(self) -> dict:
        """Registry card the agent reads when choosing tools."""
        return {"name": self.name, "description": self.description}
