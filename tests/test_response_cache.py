"""Phase 5 tests: Redis-backed LLM response caching, no live services.

Real CacheService + real RAG pipeline against the InMemoryRedis double;
LLM/chroma legs are fakes. Covers config parsing, key stability, the
miss->store->hit cycle, TTL expiry, generation-bump invalidation,
fail-open Redis behavior, refusal/error exclusion, all three transports
(non-streaming, plain-text stream, SSE event stream with done metadata),
and RAGService invalidation/threading on ingest/delete.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from redis.exceptions import RedisError

from redis_double import InMemoryRedis

import phase1
from app.schemas.chat import ChatRequest
from app.services import chat_orchestration as orchestration
from app.services.cache_service import (
    CacheService,
    build_cache_key,
    cache_enabled,
    cache_ttl_seconds,
)
from app.services.rag_services import RAGService
from phase1 import REFUSAL_MESSAGE, RAGSystem, RetrievedChunk


def cached_env(**overrides):
    base = {
        "CACHE_ENABLED": "true",
        "CACHE_TTL_SECONDS": "3600",
    }
    base.update(overrides)
    return patch.dict(os.environ, base, clear=False)


def uncached_env():
    env = {
        key: value for key, value in os.environ.items()
        if key not in ("CACHE_ENABLED", "CACHE_TTL_SECONDS")
    }
    return patch.dict(os.environ, env, clear=True)


def make_chunk(chunk_id="chunk-1"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Cached context about RAG.",
        distance=0.2,
        metadata={
            "document_id": "rag_basics",
            "source": "rag_basics.txt",
            "title": "RAG basics",
            "chunk_index": 0,
        },
    )


class ConfigTests(unittest.TestCase):
    def test_defaults_are_disabled_with_one_hour_ttl(self):
        with uncached_env():
            self.assertFalse(cache_enabled())
            self.assertEqual(cache_ttl_seconds(), 3600)

    def test_enabled_flag_variants(self):
        for raw in ("1", "true", "yes", "on", "TRUE", " On "):
            with cached_env(CACHE_ENABLED=raw):
                self.assertTrue(cache_enabled(), raw)
        for raw in ("0", "false", "no", "off", ""):
            with cached_env(CACHE_ENABLED=raw):
                self.assertFalse(cache_enabled(), raw)

    def test_ttl_parsing_and_fallback(self):
        with cached_env(CACHE_TTL_SECONDS="120"):
            self.assertEqual(cache_ttl_seconds(), 120)
        for raw in ("bogus", "0", "-5"):
            with cached_env(CACHE_TTL_SECONDS=raw):
                self.assertEqual(cache_ttl_seconds(), 3600, raw)


class KeyTests(unittest.TestCase):
    def test_deterministic_and_order_insensitive(self):
        first = build_cache_key("u", "q", ["b", "a"], "0")
        self.assertEqual(first, build_cache_key("u", "q", ["a", "b"], "0"))

    def test_varies_by_user_query_chunks_generation(self):
        base = build_cache_key("u", "q", ["a"], "0")
        self.assertNotEqual(base, build_cache_key("other", "q", ["a"], "0"))
        self.assertNotEqual(base, build_cache_key("u", "other", ["a"], "0"))
        self.assertNotEqual(base, build_cache_key("u", "q", ["b"], "0"))
        self.assertNotEqual(base, build_cache_key("u", "q", ["a"], "1"))


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.redis = InMemoryRedis()
        self.cache = CacheService(self.redis)

    def test_miss_store_hit_roundtrip(self):
        with cached_env():
            self.assertIsNone(self.cache.get("u", "q", ["a"]))
            payload = {"answer": "grounded", "retrieval_query": "q",
                       "retrieved_document_ids": ["d"], "chunks": [],
                       "sources": []}
            self.assertTrue(self.cache.store("u", "q", ["a"], payload))
            self.assertEqual(
                self.cache.get("u", "q", ["a"])["answer"], "grounded")

    def test_ttl_expiry_returns_to_miss(self):
        with cached_env(CACHE_TTL_SECONDS="100"):
            self.cache.store("u", "q", ["a"], {"answer": "x"})
            self.redis.now += 101
            self.assertIsNone(self.cache.get("u", "q", ["a"]))

    def test_invalidation_orphans_prior_entries(self):
        with cached_env():
            self.cache.store("u", "q", ["a"], {"answer": "stale"})
            self.cache.invalidate_user("u")
            self.assertIsNone(self.cache.get("u", "q", ["a"]))
            # A fresh store after invalidation works under the new generation.
            self.cache.store("u", "q", ["a"], {"answer": "fresh"})
            self.assertEqual(
                self.cache.get("u", "q", ["a"])["answer"], "fresh")

    def test_other_users_unaffected_by_invalidation(self):
        with cached_env():
            self.cache.store("u", "q", ["a"], {"answer": "kept"})
            self.cache.invalidate_user("someone-else")
            self.assertEqual(
                self.cache.get("u", "q", ["a"])["answer"], "kept")

    def test_malformed_entry_is_a_miss(self):
        with cached_env():
            key = build_cache_key("u", "q", ["a"], "0")
            asyncio.run(self.redis.set(key, "not-json", ex=60))
            self.assertIsNone(self.cache.get("u", "q", ["a"]))

    def test_empty_and_unserializable_payloads_rejected(self):
        with cached_env():
            self.assertFalse(self.cache.store("u", "q", ["a"], {}))
            self.assertFalse(
                self.cache.store("u", "q", ["a"], {"answer": ""}))
            self.assertFalse(
                self.cache.store("u", "q", ["a"], {"answer": object()}))

    def test_oversized_payload_skipped(self):
        with cached_env(), patch(
                "app.services.cache_service.MAX_PAYLOAD_BYTES", 16):
            self.assertFalse(
                self.cache.store("u", "q", ["a"], {"answer": "way-too-long"}))
            self.assertIsNone(self.cache.get("u", "q", ["a"]))

    def test_redis_outage_is_fail_open(self):
        self.redis.get = AsyncMock(side_effect=RedisError("down"))
        self.redis.set = AsyncMock(side_effect=RedisError("down"))
        self.redis.incr = AsyncMock(side_effect=RedisError("down"))
        with cached_env(), self.assertLogs(
                "app.services.cache_service", level="WARNING"):
            self.assertIsNone(self.cache.get("u", "q", ["a"]))
            self.assertFalse(
                self.cache.store("u", "q", ["a"], {"answer": "x"}))
            # Must not raise: worst case is bounded staleness, not a 500.
            self.cache.invalidate_user("u")


class RunOnceCacheTests(unittest.TestCase):
    def setUp(self):
        self.chunk = make_chunk()
        self.rag = RAGSystem.__new__(RAGSystem)
        self.rag._retrieve_chunks = Mock(return_value=[self.chunk])
        self.rag.generate_answer = Mock(return_value="A grounded answer.")
        self.redis = InMemoryRedis()
        self.cache = CacheService(self.redis)
        self.arguments = {
            "raw_query": "What is RAG?",
            "user_id": "alice",
            "rewrite_query": False,
            "use_cache": True,
            "response_cache": self.cache,
        }

    def test_miss_calls_llm_stores_and_reports_miss(self):
        with cached_env():
            result = self.rag.run_once(**self.arguments)
        self.rag.generate_answer.assert_called_once()
        self.assertEqual(result["answer"], "A grounded answer.")
        self.assertFalse(result["cache_hit"])
        self.assertEqual(
            result["retrieved_document_ids"], ["rag_basics"])
        self.assertEqual(result["chunks"][0]["chunk_id"], "chunk-1")

    def test_hit_skips_llm_and_returns_stored_shapes(self):
        with cached_env():
            first = self.rag.run_once(**self.arguments)
            second = self.rag.run_once(**self.arguments)
        self.rag.generate_answer.assert_called_once()
        self.assertTrue(second["cache_hit"])
        for key in ("answer", "retrieval_query",
                    "retrieved_document_ids", "chunks"):
            self.assertEqual(second[key], first[key])

    def test_refusal_is_never_cached(self):
        self.rag.generate_answer = Mock(return_value=REFUSAL_MESSAGE)
        with cached_env():
            self.rag.run_once(**self.arguments)
            result = self.rag.run_once(**self.arguments)
        self.assertEqual(self.rag.generate_answer.call_count, 2)
        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["answer"], REFUSAL_MESSAGE)

    def test_generation_error_propagates_and_stores_nothing(self):
        self.rag.generate_answer = Mock(
            side_effect=RuntimeError("gemini down"))
        with cached_env(), self.assertRaises(RuntimeError):
            self.rag.run_once(**self.arguments)
        with cached_env():
            self.assertIsNone(self.cache.get("alice", "What is RAG?",
                                             ["chunk-1"]))

    def test_disabled_by_default_keeps_historical_shape(self):
        # use_cache=None follows the env flag: explicitly forcing True
        # must win over a disabled env (same as use_hybrid/use_rerank).
        arguments = dict(self.arguments, use_cache=None)
        with uncached_env():
            result = self.rag.run_once(**arguments)
            second = self.rag.run_once(**arguments)
        self.assertEqual(self.rag.generate_answer.call_count, 2)
        self.assertNotIn("cache_hit", result)
        self.assertNotIn("cache_hit", second)

    def test_use_cache_false_forces_off(self):
        with cached_env():
            arguments = dict(self.arguments, use_cache=False)
            result = self.rag.run_once(**arguments)
            second = self.rag.run_once(**arguments)
        self.assertEqual(self.rag.generate_answer.call_count, 2)
        self.assertNotIn("cache_hit", result)
        self.assertNotIn("cache_hit", second)
        self.assertIsNone(self.cache.get("alice", "What is RAG?",
                                         ["chunk-1"]))

    def test_use_cache_true_forces_on_despite_disabled_env(self):
        with uncached_env():
            first = self.rag.run_once(**self.arguments)
            second = self.rag.run_once(**self.arguments)
        self.rag.generate_answer.assert_called_once()
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])

    def test_no_cache_object_means_uncached(self):
        arguments = dict(self.arguments, response_cache=None)
        with cached_env():
            result = self.rag.run_once(**arguments)
        self.rag.generate_answer.assert_called_once()
        self.assertNotIn("cache_hit", result)

    def test_different_chunks_produce_different_entries(self):
        other = make_chunk(chunk_id="chunk-2")
        with cached_env():
            self.rag.run_once(**self.arguments)
            self.rag._retrieve_chunks = Mock(return_value=[other])
            result = self.rag.run_once(**self.arguments)
        self.assertEqual(self.rag.generate_answer.call_count, 2)
        self.assertFalse(result["cache_hit"])


class StreamCacheTests(unittest.TestCase):
    def setUp(self):
        self.chunk = make_chunk()
        self.rag = RAGSystem.__new__(RAGSystem)
        self.rag._retrieve_chunks = Mock(return_value=[self.chunk])
        self.rag.generate_answer_stream = Mock(
            side_effect=lambda **kwargs: iter(["A grounded ", "answer."]))
        self.redis = InMemoryRedis()
        self.cache = CacheService(self.redis)
        self.arguments = {
            "raw_query": "What is RAG?",
            "user_id": "alice",
            "rewrite_query": False,
            "use_cache": True,
            "response_cache": self.cache,
        }

    def test_plain_stream_hit_replays_answer_without_llm(self):
        with cached_env():
            first = list(self.rag.run_once_stream(**self.arguments))
            second = list(self.rag.run_once_stream(**self.arguments))
        self.rag.generate_answer_stream.assert_called_once()
        self.assertEqual("".join(first), "A grounded answer.")
        self.assertEqual(second, ["A grounded answer."])

    def test_plain_stream_refusal_not_stored(self):
        self.rag.generate_answer_stream = Mock(
            side_effect=lambda **kwargs: iter([REFUSAL_MESSAGE]))
        with cached_env():
            list(self.rag.run_once_stream(**self.arguments))
            list(self.rag.run_once_stream(**self.arguments))
        self.assertEqual(self.rag.generate_answer_stream.call_count, 2)

    def test_event_stream_hit_preserves_event_shape(self):
        with cached_env():
            miss_events = list(
                self.rag.run_once_event_stream(**self.arguments))
            hit_events = list(
                self.rag.run_once_event_stream(**self.arguments))
        self.rag.generate_answer_stream.assert_called_once()
        self.assertEqual(
            [item["event"] for item in hit_events],
            ["start", "retrieval", "token", "sources", "done"])
        self.assertEqual(
            hit_events[1]["data"],
            {"retrieval_query": "What is RAG?",
             "retrieved_document_ids": ["rag_basics"]})
        self.assertEqual(hit_events[2]["data"],
                         {"text": "A grounded answer."})
        self.assertEqual(
            hit_events[3]["data"]["sources"][0]["chunk_id"], "chunk-1")
        self.assertEqual(hit_events[4]["data"],
                         {"metadata": {"cache": "hit"}})
        self.assertEqual(miss_events[5]["data"],
                         {"metadata": {"cache": "miss"}})

    def test_event_stream_error_stores_nothing(self):
        def failed(**kwargs):
            yield "Partial answer"
            raise RuntimeError("gemini down")

        self.rag.generate_answer_stream = Mock(side_effect=failed)
        with cached_env(), self.assertLogs("phase1", level="ERROR"):
            events = list(self.rag.run_once_event_stream(**self.arguments))
        self.assertEqual(events[-1]["event"], "error")
        self.assertIsNone(self.cache.get("alice", "What is RAG?",
                                         ["chunk-1"]))

    def test_event_stream_disabled_keeps_done_empty(self):
        arguments = dict(self.arguments, use_cache=None)
        with uncached_env():
            events = list(self.rag.run_once_event_stream(**arguments))
        self.rag.generate_answer_stream.assert_called_once()
        self.assertEqual(
            [item["event"] for item in events],
            ["start", "retrieval", "token", "token", "sources", "done"])
        self.assertEqual(events[-1]["data"], {})


class RAGServiceCacheTests(unittest.TestCase):
    def test_run_once_threads_cache_arguments(self):
        fake_rag = Mock()
        fake_rag.run_once = Mock(return_value={"answer": "ok"})
        fake_cache = Mock()
        service = RAGService(rag=fake_rag, response_cache=fake_cache)
        request = ChatRequest(raw_query="q", user_id="u")
        service.run_once(request, use_cache=True)
        fake_rag.run_once.assert_called_once()
        _, kwargs = fake_rag.run_once.call_args
        self.assertIs(kwargs["response_cache"], fake_cache)
        self.assertTrue(kwargs["use_cache"])

    def test_ingest_and_delete_invalidate_response_cache(self):
        fake_rag = Mock()
        fake_rag.collection.get = Mock(return_value={"ids": ["u:d:0"]})
        fake_cache = Mock()
        service = RAGService(rag=fake_rag, response_cache=fake_cache)
        document = Mock()
        document.document_id = "doc-1"
        service.ingest_user_document(document, "alice")
        service.delete_user_document("doc-1", "alice")
        self.assertEqual(
            fake_cache.invalidate_user.call_count, 2)
        fake_cache.invalidate_user.assert_called_with("alice")

    def test_orchestration_passes_use_cache_through(self):
        fake_service = Mock()
        fake_service.run_once = Mock(return_value={"answer": "ok"})
        payload = ChatRequest(raw_query="q", user_id="u")
        orchestration.run_chat_once(
            fake_service, None, payload, use_cache=True)
        fake_service.run_once.assert_called_once()
        _, kwargs = fake_service.run_once.call_args
        self.assertTrue(kwargs["use_cache"])

    def test_invalidation_failure_never_breaks_uploads(self):
        fake_rag = Mock()
        fake_rag.collection.get = Mock(return_value={"ids": []})
        fake_cache = Mock()
        fake_cache.invalidate_user = Mock(
            side_effect=RuntimeError("redis down"))
        service = RAGService(rag=fake_rag, response_cache=fake_cache)
        document = Mock()
        document.document_id = "doc-1"
        with self.assertLogs("app.services.rag_services", level="ERROR"):
            service.ingest_user_document(document, "alice")


if __name__ == "__main__":
    unittest.main()
