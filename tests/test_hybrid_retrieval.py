"""Phase 2 tests: hybrid BM25 + vector retrieval with RRF fusion.

Fake RAGSystem + fake Chroma collection double; no live Redis/Mongo/AI.
Covers RRF math, keyword-heavy win over vector-only, per-user isolation,
per-user cache + invalidation on ingest/delete, flag-gated default-off,
graceful fallback, and limit validation.
"""

import os
import unittest
from unittest.mock import Mock, patch

from phase1 import RetrievedChunk

from app.services.retrieval import (
    HybridRetriever,
    hybrid_enabled,
    reciprocal_rank_fusion,
    tokenize,
)


def make_chunk(chunk_id, text="text", distance=0.5, document_id="doc"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        distance=distance,
        metadata={"user_id": "alice", "document_id": document_id,
                  "source": "s.txt", "title": "T", "chunk_index": 0},
    )


class FakeChromaCollection:
    """Minimal collection double recording where filters."""

    def __init__(self, rows):
        # rows: list of (chunk_id, text, metadata)
        self.rows = list(rows)
        self.get_calls = []
        self.deleted = []

    def get(self, where=None, include=None, limit=None, offset=None):
        self.get_calls.append({"where": where, "include": include,
                               "limit": limit, "offset": offset})
        user_id = (where or {}).get("user_id")
        matched = [(cid, text, meta) for cid, text, meta in self.rows
                   if meta.get("user_id") == user_id]
        if offset is not None and limit is not None:
            matched = matched[offset:offset + limit]
        return {
            "ids": [cid for cid, _, _ in matched],
            "documents": [text for _, text, _ in matched],
            "metadatas": [meta for _, _, meta in matched],
        }

    def delete(self, where=None):
        self.deleted.append(where)


class FakeRAG:
    def __init__(self, collection, vector_results=None):
        self.collection = collection
        self.vector_results = list(vector_results or [])
        self.retrieve_calls = []

    def retrieve(self, query, user_id, k=3, distance_threshold=None):
        if not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10")
        if not query or len(query) > 4_000:
            raise ValueError("query must contain 1-4000 characters")
        if (distance_threshold is not None
                and not 0 <= distance_threshold <= 2):
            raise ValueError("distance_threshold must be between 0 and 2")
        self.retrieve_calls.append({"query": query, "user_id": user_id,
                                    "k": k,
                                    "distance_threshold": distance_threshold})
        return list(self.vector_results[:k])


def rows_for(user_id, texts):
    rows = []
    for i, text in enumerate(texts):
        rows.append((
            f"{user_id}:doc:{i}", text,
            {"user_id": user_id, "document_id": "doc",
             "source": "s.txt", "title": "T", "chunk_index": i},
        ))
    return rows


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_splits_words(self):
        self.assertEqual(tokenize("Hello, Error CODE-42!"),
                         ["hello", "error", "code", "42"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class RRFFusionTests(unittest.TestCase):
    def test_common_doc_outranks_single_list_docs(self):
        a = make_chunk("a")
        b = make_chunk("b")
        c = make_chunk("c")
        # a is #1 in both lists -> must win.
        fused = reciprocal_rank_fusion([a, b], [a, c], rrf_k=60, k=3)
        self.assertEqual([ch.chunk_id for ch in fused], ["a", "b", "c"])

    def test_bm25_can_rescue_low_vector_rank(self):
        # kw is last in the vector leg but first in BM25 while the
        # vector top-1 is absent from BM25: fusion must promote kw.
        v1 = make_chunk("v1")
        v2 = make_chunk("v2")
        kw = make_chunk("kw")
        fused = reciprocal_rank_fusion([v1, v2, kw], [kw],
                                       rrf_k=60, k=1)
        self.assertEqual(fused[0].chunk_id, "kw")

    def test_deduplicates_by_chunk_id(self):
        a = make_chunk("a")
        fused = reciprocal_rank_fusion([a], [a], rrf_k=60, k=5)
        self.assertEqual(len(fused), 1)

    def test_k_limits_output(self):
        chunks = [make_chunk(f"c{i}") for i in range(5)]
        fused = reciprocal_rank_fusion(chunks, list(reversed(chunks)),
                                       rrf_k=60, k=2)
        self.assertEqual(len(fused), 2)


class HybridEnabledTests(unittest.TestCase):
    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HYBRID_ENABLED", None)
            self.assertFalse(hybrid_enabled())

    def test_truthy_values(self):
        for val in ("1", "true", "TRUE", "yes", "on", " True "):
            with patch.dict(os.environ, {"HYBRID_ENABLED": val}):
                self.assertTrue(hybrid_enabled(), val)


class HybridRetrieveTests(unittest.TestCase):
    def test_disabled_flag_delegates_to_vector_only(self):
        collection = FakeChromaCollection(rows_for("alice", ["hello world"]))
        rag = FakeRAG(collection, [make_chunk("alice:doc:0")])
        retriever = HybridRetriever(rag)
        with patch.dict(os.environ, {"HYBRID_ENABLED": "false"}):
            out = retriever.hybrid_retrieve("hello", "alice", k=1)
        self.assertEqual([c.chunk_id for c in out], ["alice:doc:0"])
        self.assertEqual(collection.get_calls, [])  # no BM25 corpus load

    def test_keyword_heavy_query_beats_vector_only(self):
        texts = [
            "generic introduction to databases and storage engines",
            "generic overview of query planning and caching layers",
            "zebracorn error code XZ-9919 restart sequence and fix",
        ]
        collection = FakeChromaCollection(rows_for("alice", texts))
        # Vector leg ranks the keyword doc last (simulates dense miss).
        vector = [make_chunk("alice:doc:0", texts[0]),
                  make_chunk("alice:doc:1", texts[1]),
                  make_chunk("alice:doc:2", texts[2])]
        rag = FakeRAG(collection, vector)
        retriever = HybridRetriever(rag)
        out = retriever.hybrid_retrieve("XZ-9919 zebracorn", "alice", k=1,
                                        use_hybrid=True)
        self.assertEqual(out[0].chunk_id, "alice:doc:2")

    def test_tenant_isolation_in_corpus_load(self):
        collection = FakeChromaCollection(
            rows_for("alice", ["alice secret token alpha"]) +
            rows_for("bob", ["bob secret token alpha"]))
        rag = FakeRAG(collection, [make_chunk("alice:doc:0")])
        retriever = HybridRetriever(rag)
        retriever.bm25_search("alice", "secret", 5)
        self.assertTrue(collection.get_calls)
        for call in collection.get_calls:
            self.assertEqual(call["where"], {"user_id": "alice"})
        entry = retriever._cache["alice"]
        for meta in entry["metadatas"]:
            self.assertEqual(meta["user_id"], "alice")

    def test_cache_reused_and_invalidated(self):
        collection = FakeChromaCollection(rows_for("alice", ["hello world"]))
        rag = FakeRAG(collection, [make_chunk("alice:doc:0")])
        retriever = HybridRetriever(rag)
        retriever.bm25_search("alice", "hello", 5)
        calls_after_first = len(collection.get_calls)
        retriever.bm25_search("alice", "hello", 5)
        self.assertEqual(len(collection.get_calls), calls_after_first)
        retriever.invalidate_user("alice")
        retriever.bm25_search("alice", "hello", 5)
        self.assertGreater(len(collection.get_calls), calls_after_first)

    def test_empty_corpus_falls_back_to_vector(self):
        collection = FakeChromaCollection([])
        rag = FakeRAG(collection, [make_chunk("alice:doc:0")])
        retriever = HybridRetriever(rag)
        out = retriever.hybrid_retrieve("anything", "alice", k=1,
                                        use_hybrid=True)
        self.assertEqual([c.chunk_id for c in out], ["alice:doc:0"])

    def test_bm25_failure_falls_back_to_vector(self):
        collection = FakeChromaCollection(rows_for("alice", ["hello world"]))
        rag = FakeRAG(collection, [make_chunk("alice:doc:0")])
        retriever = HybridRetriever(rag)
        with patch.object(HybridRetriever, "bm25_search",
                           side_effect=RuntimeError("boom")):
            out = retriever.hybrid_retrieve("hello", "alice", k=1,
                                            use_hybrid=True)
        self.assertEqual([c.chunk_id for c in out], ["alice:doc:0"])

    def test_overfetch_clamped_to_vector_limit(self):
        collection = FakeChromaCollection(rows_for("alice", ["x y z"]))
        rag = FakeRAG(collection, [make_chunk("alice:doc:0")])
        retriever = HybridRetriever(rag, overfetch_min=20)
        self.assertEqual(retriever.overfetch_n(3), 10)
        self.assertEqual(retriever.overfetch_n(1), 10)
        out = retriever.hybrid_retrieve("x", "alice", k=3, use_hybrid=True)
        self.assertLessEqual(len(out), 3)
        self.assertEqual(rag.retrieve_calls[0]["k"], 10)

    def test_validation_errors(self):
        collection = FakeChromaCollection(rows_for("alice", ["hi"]))
        rag = FakeRAG(collection)
        retriever = HybridRetriever(rag)
        with self.assertRaises(ValueError):
            retriever.hybrid_retrieve("hi", "alice", k=0, use_hybrid=True)
        with self.assertRaises(ValueError):
            retriever.hybrid_retrieve("", "alice", k=1, use_hybrid=True)
        with self.assertRaises(ValueError):
            retriever.hybrid_retrieve("hi", "alice", k=1,
                                      distance_threshold=5,
                                      use_hybrid=True)


class RAGServiceInvalidationTests(unittest.TestCase):
    def test_ingest_and_delete_invalidate_hybrid_cache(self):
        from app.services.rag_services import RAGService

        collection = FakeChromaCollection(
            rows_for("alice", ["hybrid cache invalidation probe"]))
        rag = FakeRAG(collection, [make_chunk("alice:doc:0")])
        # Give FakeRAG the ingest/delete surface RAGService expects.
        rag.ingest_documents = Mock()
        rag.collection = collection
        service = RAGService(rag)
        self.assertIsNotNone(service.hybrid_retriever)
        service.hybrid_retriever.bm25_search("alice", "probe", 5)
        self.assertIn("alice", service.hybrid_retriever._cache)
        service.hybrid_retriever._cache["alice"] = {"seeded": True}
        service.delete_user_document("doc", "alice")
        self.assertNotIn("alice", service.hybrid_retriever._cache)
        # Upload path invalidates too: seed, ingest, expect a rebuild.
        service.hybrid_retriever._cache["alice"] = {"seeded": True}
        from phase1 import Document as RAGDocument
        service.ingest_user_document(
            RAGDocument(document_id="doc", source="s.txt",
                        text="new text", title="T"),
            "alice",
        )
        self.assertNotIn("alice", service.hybrid_retriever._cache)


if __name__ == "__main__":
    unittest.main()
