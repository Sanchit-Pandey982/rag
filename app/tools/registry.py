"""Phase 12c: tool registry.

Holds every tool the agent may call, keyed by name. The default
registry wires the shipped tools (calculator first: a calculation
question must never pay for a web call); tests and future phases
register fakes or new tools without touching the agent. Selection is
data-driven -- the agent asks each tool, in registration order,
whether it handles the query (``match``), so adding a tool only
means writing its class and registering it.
"""

from __future__ import annotations

import logging

from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Name-keyed tool set with first-match selection."""

    def __init__(self, tools: list[BaseTool] | None = None):
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> BaseTool:
        """Add (or replace) a tool; returns it for fluent wiring."""
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must subclass BaseTool")
        if not tool.name:
            raise ValueError("tool must have a name")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> BaseTool | None:
        """Fetch one tool by name, or ``None`` when unknown."""
        return self._tools.get(name)

    def all(self) -> list[BaseTool]:
        """Every registered tool, in registration order."""
        return list(self._tools.values())

    def describe(self) -> list[dict]:
        """Name + description cards the agent reads to choose tools."""
        return [tool.describe() for tool in self._tools.values()]

    def select(self, query: str) -> tuple[BaseTool, object] | None:
        """First tool whose ``match`` claims ``query`` + its input.

        Registration order is priority order. Tools that raise inside
        ``match`` are skipped (logged), never fatal.
        """
        for tool in self._tools.values():
            try:
                tool_input = tool.match(query)
            except Exception:
                logger.exception("Tool match failed: %s", tool.name)
                continue
            if tool_input is not None:
                return tool, tool_input
        return None


def default_registry() -> ToolRegistry:
    """Shipped tool set (lazy imports keep startup dependency-free)."""
    from app.tools.calculator import CalculatorTool
    from app.tools.web_search import WebSearchTool

    return ToolRegistry([CalculatorTool(), WebSearchTool()])


def resolve_registry(tools: list | ToolRegistry | None) -> ToolRegistry:
    """Normalize the ``run_agent(tools=...)`` argument.

    ``None`` → shipped defaults; a ``ToolRegistry`` passes through;
    a list of tools becomes a registry (empty list disables tools).
    """
    if tools is None:
        return default_registry()
    if isinstance(tools, ToolRegistry):
        return tools
    return ToolRegistry(list(tools))
