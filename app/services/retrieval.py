"""Phase 2: hybrid BM25 + vector retrieval with Reciprocal Rank Fusion.

Dense-only retrieval misses keyword-heavy queries (exact error codes,
product names, rare terms). This module adds a per-user in-memory BM25
index built from the same Chroma chunks the vector leg already uses,
then merges both rankings with RRF:

    score(d) = 1/(rrf_k + rank_bm25(d)) + 1/(rrf_k + rank_vector(d))

Design rules (see AGENTS.md):
- Retrieval stays tenant-scoped: the BM25 corpus is loaded with
  ``where={"user_id": ...}`` and chunk ids remain ``{user}:{doc}:{idx}``.
- Flag-gated: ``HYBRID_ENABLED`` defaults to false, so existing
  vector-only behavior is unchanged unless opted in.
- Lazy: ``rank_bm25`` is imported on first index build, never at module
  import, so startup and vector-only tests need no new dependency.
- Graceful: any BM25 failure (missing package, empty corpus, Chroma
  error) falls back to pure vector results, never raises to the caller.
"""

from __future__ import annotations

import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")

# Chroma caps a single query at 10 results (RAGSystem.retrieve enforces
# k 1-10), so the over-fetch window is clamped to 10 as well.
_VECTOR_LIMIT = 10


def hybrid_enabled() -> bool:
    return os.getenv("HYBRID_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def hybrid_rrf_k() -> int:
    try:
        value = int(os.getenv("HYBRID_RRF_K", "60"))
    except ValueError:
        return 60
    return value if value > 0 else 60


def hybrid_overfetch_min() -> int:
    try:
        value = int(os.getenv("HYBRID_OVERFETCH_MIN", "20"))
    except ValueError:
        return 20
    return value if value > 0 else 20


def tokenize(text: str) -> list[str]:
    """Shared BM25 tokenizer: lowercase alphanumeric words."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    vector_chunks,
    bm25_chunks,
    rrf_k: int = 60,
    k: int = 3,
):
    """Merge two ranked chunk lists with RRF; return top-k fused chunks.

    Ranks are 1-based in list order. Chunks appearing in only one list
    still score (single term). Ties break deterministically by vector
    rank, then BM25 rank, then chunk id, so results are stable.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, object] = {}
    vector_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    for rank, chunk in enumerate(vector_chunks, start=1):
        cid = chunk.chunk_id
        by_id[cid] = chunk
        vector_rank[cid] = rank
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    for rank, chunk in enumerate(bm25_chunks, start=1):
        cid = chunk.chunk_id
        if cid not in by_id:
            by_id[cid] = chunk
        bm25_rank[cid] = rank
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            vector_rank.get(item[0], 10**9),
            bm25_rank.get(item[0], 10**9),
            item[0],
        ),
    )
    return [by_id[cid] for cid, _ in ranked[: max(k, 0)]]


class HybridRetriever:
    """Per-user cached BM25 + vector fusion over a ``RAGSystem``.

    The BM25 corpus is the user's full Chroma chunk set
    (``collection.get(where={"user_id": ...})``). Entries are cached
    per user and dropped via :meth:`invalidate_user`, which
    ``RAGService`` calls after every ingest/delete.
    """

    def __init__(self, rag, rrf_k: int | None = None,
                 overfetch_min: int | None = None):
        self._rag = rag
        self._rrf_k = rrf_k if rrf_k is not None else hybrid_rrf_k()
        self._overfetch_min = (
            overfetch_min if overfetch_min is not None
            else hybrid_overfetch_min()
        )
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------
    # Cache management
    # ----------------------------------------------------------

    def invalidate_user(self, user_id: str) -> None:
        with self._lock:
            self._cache.pop(user_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # ----------------------------------------------------------
    # Corpus loading
    # ----------------------------------------------------------

    def _load_user_corpus(self, user_id: str) -> dict:
        """Paginated ``collection.get`` so large tenants never truncate."""
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        offset = 0
        page = 512
        while True:
            try:
                stored = self._rag.collection.get(
                    where={"user_id": user_id},
                    include=["documents", "metadatas"],
                    limit=page,
                    offset=offset,
                )
            except TypeError:
                # Older Chroma doubles in tests may not accept
                # limit/offset; fall back to a single full fetch.
                stored = self._rag.collection.get(
                    where={"user_id": user_id},
                    include=["documents", "metadatas"],
                )
                offset = -1  # mark single-shot below
            batch_ids = stored.get("ids", []) or []
            batch_docs = stored.get("documents", []) or []
            batch_meta = stored.get("metadatas", []) or []
            ids.extend(batch_ids)
            documents.extend(batch_docs)
            metadatas.extend(
                [m if isinstance(m, dict) else {} for m in batch_meta]
            )
            if offset == -1 or len(batch_ids) < page:
                break
            offset += len(batch_ids)
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def _get_or_build_index(self, user_id: str):
        """Return cached (bm25, corpus) or rebuild; None when unusable."""
        with self._lock:
            cached = self._cache.get(user_id)
        if cached is not None:
            return cached
        try:
            from rank_bm25 import BM25Okapi
        except Exception:
            logger.debug("rank_bm25 unavailable; hybrid falls back to vector")
            return None
        try:
            corpus = self._load_user_corpus(user_id)
        except Exception:
            logger.exception("Hybrid BM25 corpus load failed")
            return None
        if not corpus["ids"]:
            return None
        try:
            bm25 = BM25Okapi([tokenize(t) for t in corpus["documents"]])
        except Exception:
            logger.exception("Hybrid BM25 index build failed")
            return None
        entry = {"bm25": bm25, **corpus}
        with self._lock:
            self._cache[user_id] = entry
        return entry

    # ----------------------------------------------------------
    # Search legs
    # ----------------------------------------------------------

    def bm25_search(self, user_id: str, query: str, n: int):
        """Top-n BM25 chunks for a user; empty list when unavailable."""
        from phase1 import RetrievedChunk

        entry = self._get_or_build_index(user_id)
        if entry is None:
            return []
        try:
            scores = entry["bm25"].get_scores(tokenize(query))
        except Exception:
            logger.exception("Hybrid BM25 scoring failed")
            return []
        order = sorted(range(len(scores)), key=lambda i: scores[i],
                       reverse=True)[: max(n, 0)]
        results = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            results.append(RetrievedChunk(
                chunk_id=entry["ids"][idx],
                text=entry["documents"][idx],
                # Map BM25 score to a distance-like value so lower still
                # means more similar, matching cosine-distance semantics.
                distance=1.0 / (1.0 + float(scores[idx])),
                metadata=entry["metadatas"][idx]
                if idx < len(entry["metadatas"]) else {},
            ))
        return results

    def overfetch_n(self, k: int) -> int:
        """Candidates per leg: max(2*k, overfetch_min), clamped to 10."""
        return max(1, min(_VECTOR_LIMIT,
                          max(2 * k, self._overfetch_min)))

    # ----------------------------------------------------------
    # Fused entry point
    # ----------------------------------------------------------

    def hybrid_retrieve(self, query: str, user_id: str, k: int = 3,
                        distance_threshold=None,
                        use_hybrid: bool | None = None):
        """RRF-fused retrieval; falls back to vector-only on any issue.

        ``use_hybrid=None`` follows the ``HYBRID_ENABLED`` env flag;
        ``True``/``False`` forces or disables fusion (used by tests and
        per-request overrides without touching global env).
        """
        enabled = hybrid_enabled() if use_hybrid is None else bool(use_hybrid)
        if not enabled:
            return self._rag.retrieve(
                query=query, user_id=user_id, k=k,
                distance_threshold=distance_threshold,
            )
        if not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10")
        if not query or len(query) > 4_000:
            raise ValueError("query must contain 1-4000 characters")
        if (distance_threshold is not None
                and not 0 <= distance_threshold <= 2):
            raise ValueError("distance_threshold must be between 0 and 2")

        n = self.overfetch_n(k)
        try:
            vector_chunks = self._rag.retrieve(
                query=query, user_id=user_id, k=n,
                distance_threshold=distance_threshold,
            )
        except Exception:
            # Vector leg is authoritative; never mask its errors with an
            # empty fusion result.
            raise
        try:
            bm25_chunks = self.bm25_search(user_id, query, n)
        except Exception:
            logger.exception("Hybrid BM25 leg failed; using vector results")
            return vector_chunks[:k]
        if not bm25_chunks:
            return vector_chunks[:k]
        if not vector_chunks:
            return bm25_chunks[:k]
        return reciprocal_rank_fusion(
            vector_chunks, bm25_chunks, rrf_k=self._rrf_k, k=k,
        )
