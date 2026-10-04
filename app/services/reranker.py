"""Phase 3a: cross-encoder reranking of retrieved chunks.

Hybrid/vector retrieval returns up to 10 candidate chunks (the Chroma
single-query cap). This module rescores those candidates against the
original query with a cross-encoder (query + chunk scored jointly, which
is more precise than the bi-encoder distance used for retrieval) and
keeps the top-k.

Design rules (see AGENTS.md):
- Flag-gated: ``RERANKER_ENABLED`` defaults to false, so default
  behavior is byte-identical to retrieval-only.
- Lazy: ``sentence_transformers`` (and its ``torch`` dependency) is
  imported on first ``rerank()`` call, never at module import or app
  startup. Importing this module needs no credentials, GPU, or network.
- Graceful: any model/import/scoring failure returns the input order
  truncated to top-k -- never raises into the chat path.
- Stateless: no cache; input chunks are already tenant-filtered, so no
  ownership logic lives here.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def reranker_enabled() -> bool:
    return os.getenv("RERANKER_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def reranker_model_name() -> str:
    return os.getenv("RERANKER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def reranker_top_k() -> int:
    try:
        value = int(os.getenv("RERANKER_TOP_K", "5"))
    except ValueError:
        return 5
    return value if value > 0 else 5


class CrossEncoderReranker:
    """Joint query-chunk scorer with a lazily loaded cross-encoder.

    The heavy model is constructed once, on first use, under a lock so
    concurrent chat requests cannot build it twice. ``model`` may be
    injected (tests pass a fake); otherwise it is created from
    ``sentence_transformers.CrossEncoder`` on demand.
    """

    def __init__(self, model_name: str | None = None, model=None):
        self._model_name = model_name or reranker_model_name()
        self._model = model
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, chunks, top_k: int | None = None):
        """Return the top-k chunks by cross-encoder score, best first.

        ``top_k=None`` follows ``RERANKER_TOP_K``. Empty input returns
        empty output without touching the model (so disabled/empty
        retrieval never pays the load cost). Any failure falls back to
        the input order truncated to top-k.
        """
        items = list(chunks)
        if not items:
            return []
        limit = top_k if top_k is not None else reranker_top_k()
        limit = max(0, min(limit, len(items)))
        if limit == 0:
            return []
        try:
            model = self._get_model()
            pairs = [(query, chunk.text) for chunk in items]
            scores = model.predict(pairs)
        except Exception:
            logger.exception("Cross-encoder reranking failed; "
                             "keeping retrieval order")
            return items[:limit]
        order = sorted(range(len(items)),
                       key=lambda i: (scores[i], -i),
                       reverse=True)
        return [items[i] for i in order[:limit]]

    def rerank_fused(self, query: str, chunks, k: int,
                     top_k: int | None = None,
                     use_rerank: bool | None = None):
        """Apply reranking when enabled; otherwise truncate to k.

        ``use_rerank=None`` follows ``RERANKER_ENABLED``; ``True``/``False``
        forces or disables (per-request override / tests).
        """
        enabled = reranker_enabled() if use_rerank is None else bool(use_rerank)
        if not enabled:
            return list(chunks)[: max(k, 0)]
        limit = top_k if top_k is not None else reranker_top_k()
        return self.rerank(query, chunks, top_k=min(limit, max(k, 0)))
