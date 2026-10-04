"""Phase 12a: web search tool (Tavily or SerpAPI, env-selected).

Returns the top-N results as ``[{title, snippet, url}]``. The stdlib
``urllib`` transport is imported inside the fetch helpers, so module
import (and app startup) never touches the network. Missing backend
selection, missing API keys, timeouts, and malformed responses all
become ``{"ok": False, "error": ...}`` -- the agent falls back to
local retrieval instead of failing the turn.

``match`` claims queries with recency/current-events intent (latest,
news, today, year mentions, ...); durable knowledge questions stay on
local retrieval.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse

from app.tools.base import BaseTool
from app.tools.config import (
    search_backend,
    search_max_results,
    search_timeout_seconds,
    serpapi_api_key,
    tavily_api_key,
)

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

_RECENCY_RE = re.compile(
    r"\b(latest|newest|current|currently|today|yesterday|this\s+week|"
    r"this\s+month|breaking|news|update[sd]?|trending|price\s+of|"
    r"stock\s+price|election|who\s+won|what\s+happened|"
    r"up\s*to\s*date)\b"
    r"|20(?:2[4-9]|[3-9][0-9])",
    re.IGNORECASE,
)


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    """POST JSON and decode the JSON body (stdlib, lazy import)."""
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "User-Agent": "rag-agent/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if getattr(response, "status", 200) != 200:
            raise ValueError(f"search HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, params: dict, timeout: int) -> dict:
    """GET with query params and decode the JSON body."""
    import urllib.request

    full = url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        full, headers={"User-Agent": "rag-agent/1.0"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if getattr(response, "status", 200) != 200:
            raise ValueError(f"search HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _normalize(items: list, limit: int) -> list[dict]:
    results = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("content")
                      or item.get("description") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        if not title and not snippet:
            continue
        results.append({"title": title, "snippet": snippet, "url": url})
        if len(results) >= limit:
            break
    return results


def _tavily_search(query: str, limit: int, timeout: int) -> list[dict]:
    key = tavily_api_key()
    if not key:
        raise ValueError("TAVILY_API_KEY is not set")
    body = _post_json(TAVILY_ENDPOINT, {
        "api_key": key,
        "query": query,
        "max_results": limit,
        "include_answer": False,
    }, timeout)
    return _normalize(body.get("results"), limit)


def _serpapi_search(query: str, limit: int, timeout: int) -> list[dict]:
    key = serpapi_api_key()
    if not key:
        raise ValueError("SERPAPI_API_KEY is not set")
    body = _get_json(SERPAPI_ENDPOINT, {
        "q": query,
        "api_key": key,
        "num": limit,
    }, timeout)
    return _normalize(
        [{**r, "url": r.get("link", "")} for r in
         body.get("organic_results", [])],
        limit,
    )


class WebSearchTool(BaseTool):
    """Fresh/external knowledge the local corpus cannot have."""

    name = "web_search"
    description = (
        "Searches the public web and returns the top results with "
        "title, snippet, and URL. "
        "Use for current events, latest news, recent releases, prices, "
        "and anything time-sensitive the local documents may not cover."
    )

    def match(self, query: str) -> str | None:
        text = (query or "").strip()
        if not text or len(text) > 1000:
            return None
        if _RECENCY_RE.search(text) is None:
            return None
        return text

    def execute(self, tool_input: str) -> dict:
        if not isinstance(tool_input, str):
            raise TypeError("query must be a string")
        query = tool_input.strip()
        if not query:
            return {"ok": False, "query": tool_input,
                    "error": "empty search query"}
        backend = search_backend()
        if backend == "none":
            return {"ok": False, "query": query,
                    "error": "search backend not configured "
                             "(set SEARCH_BACKEND=tavily|serpapi "
                             "with an API key)"}
        limit = search_max_results()
        timeout = search_timeout_seconds()
        try:
            if backend == "tavily":
                results = _tavily_search(query, limit, timeout)
            else:
                results = _serpapi_search(query, limit, timeout)
        except ValueError as error:
            # Includes missing-key and HTTP-status failures.
            return {"ok": False, "query": query, "error": str(error)}
        except Exception:
            logger.exception("Web search request failed")
            return {"ok": False, "query": query,
                    "error": "search request failed"}
        return {"ok": True, "query": query, "backend": backend,
                "results": results}
