"""Phase 7 tests: degraded LLM responses, no live services.

Real RAG pipeline with fake generation legs, a real DegradationTracker
over the InMemoryRedis double. Covers config parsing, excerpt building,
all three transports serving excerpts after retries are exhausted,
refusal/error-path preservation, max-chunk limits, Redis counting
(including fail-open), circuit-open fallbacks, and the RAGService /
orchestration threading seams.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from redis.exceptions import RedisError

from redis_double import InMemoryRedis

from app.services import chat_orchestration as orchestration
from app.services.cache_service import CacheService
from app.services.degradation_service import DegradationTracker
from app.services.rag_services import RAGService
from app.utils.retry import get_breaker, reset_breakers
from phase1 import (
    DEGRADED_PREFIX,
    REFUSAL_MESSAGE,
    RAGSystem,
    RetrievedChunk,
    build_degraded_answer,
    degraded_max_chunks,
    degraded_mode_enabled,
    is_llm_unavailable,
)


def degraded_env(**overrides):
    base = {
        "DEGRADED_MODE_ENABLED": "true",
        "DEGRADED_MAX_CHUNKS": "3",
        "RETRY_BASE_DELAY": "0",
    }
    base.update(overrides)
    return patch.dict(os.environ, base, clear=False)


def plain_env():
    env = {
        key: value for key, value in os.environ.items()
        if key not in ("DEGRADED_MODE_ENABLED", "DEGRADED_MAX_CHUNKS")
    }
    return patch.dict(os.environ, env, clear=True)


def make_chunk(chunk_id="chunk-1", text="Excerpt context about RAG."):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        distance=0.2,
        metadata={
            "document_id": "rag_basics",
            "source": "rag_basics.txt",
            "title": "RAG basics",
            "chunk_index": 0,
        },
    )


class ConfigTests(unittest.TestCase):
    def test_defaults_are_disabled_with_three_chunks(self):
        with plain_env():
            self.assertFalse(degraded_mode_enabled())
            self.assertEqual(degraded_max_chunks(), 3)

    def test_enabled_flag_variants(self):
        for raw in ("1", "true", "yes", "on", "TRUE"):
            with degraded_env(DEGRADED_MODE_ENABLED=raw):
                self.assertTrue(degraded_mode_enabled(), raw)
        for raw in ("0", "false", "no", "off", ""):
            with degraded_env(DEGRADED_MODE_ENABLED=raw):
                self.assertFalse(degraded_mode_enabled(), raw)

    def test_max_chunks_parsing_and_fallback(self):
        with degraded_env(DEGRADED_MAX_CHUNKS="5"):
            self.assertEqual(degraded_max_chunks(), 5)
        for raw in ("bogus", "0", "-2"):
            with degraded_env(DEGRADED_MAX_CHUNKS=raw):
                self.assertEqual(degraded_max_chunks(), 3, raw)

    def test_unavailability_predicate(self):
        from app.utils.retry import CircuitBreakerOpen
        from google.genai.errors import APIError
        self.assertTrue(is_llm_unavailable(TimeoutError("t")))
        self.assertTrue(is_llm_unavailable(
            APIError(code=503, response_json={})))
        self.assertTrue(is_llm_unavailable(CircuitBreakerOpen("open")))
        self.assertFalse(is_llm_unavailable(ValueError("bug")))
        self.assertFalse(is_llm_unavailable(
            APIError(code=400, response_json={})))


class ExcerptTests(unittest.TestCase):
    def test_prefix_and_top_chunks(self):
        chunks = [make_chunk(f"chunk-{i}", f"Text {i}.") for i in range(5)]
        answer = build_degraded_answer(chunks)
        self.assertTrue(answer.startswith(DEGRADED_PREFIX))
        self.assertIn("Text 0.", answer)
        self.assertIn("Text 2.", answer)
        self.assertNotIn("Text 3.", answer)
        self.assertIn("[Source 1]", answer)
        self.assertIn("rag_basics", answer)

    def test_max_chunks_override(self):
        chunks = [make_chunk(f"chunk-{i}", f"Text {i}.") for i in range(3)]
        answer = build_degraded_answer(chunks, max_chunks=1)
        self.assertIn("Text 0.", answer)
        self.assertNotIn("Text 1.", answer)

    def test_empty_chunks_yield_prefix_only(self):
        self.assertEqual(build_degraded_answer([]), DEGRADED_PREFIX)


class TrackerTests(unittest.TestCase):
    def test_count_starts_at_zero_and_increments(self):
        tracker = DegradationTracker(InMemoryRedis())
        self.assertEqual(tracker.count(), 0)
        tracker.increment()
        tracker.increment()
        self.assertEqual(tracker.count(), 2)

    def test_redis_outage_is_fail_open(self):
        redis = InMemoryRedis()
        redis.incr = AsyncMock(side_effect=RedisError("down"))
        redis.get = AsyncMock(side_effect=RedisError("down"))
        tracker = DegradationTracker(redis)
        with self.assertLogs(
                "app.services.degradation_service", level="WARNING"):
            tracker.increment()
            self.assertEqual(tracker.count(), 0)


class RunOnceDegradedTests(unittest.TestCase):
    def setUp(self):
        reset_breakers()
        self.addCleanup(reset_breakers)
        self.chunk = make_chunk()
        self.rag = RAGSystem.__new__(RAGSystem)
        self.rag._retrieve_chunks = Mock(return_value=[self.chunk])
        self.rag.generate_answer = Mock(
            side_effect=TimeoutError("gemini down"))
        self.tracker = DegradationTracker(InMemoryRedis())
        self.arguments = {
            "raw_query": "What is RAG?",
            "user_id": "alice",
            "rewrite_query": False,
            "use_degraded": True,
            "degraded_tracker": self.tracker,
        }

    def test_unavailable_llm_serves_excerpts_after_retries(self):
        with degraded_env():
            result = self.rag.run_once(**self.arguments)
        # The Mock stands in for the decorated method, so no retry
        # happens here; exhaustion-then-fallback is covered in
        # test_retry.py, and end to end below with a real client.
        self.assertEqual(self.rag.generate_answer.call_count, 1)
        self.assertTrue(result["answer"].startswith(DEGRADED_PREFIX))
        self.assertIn("Excerpt context about RAG.", result["answer"])
        self.assertTrue(result["degraded"])
        self.assertEqual(result["retrieved_document_ids"], ["rag_basics"])
        self.assertEqual(result["chunks"][0]["chunk_id"], "chunk-1")
        self.assertEqual(self.tracker.count(), 1)

    def test_retries_exhaust_then_degraded_end_to_end(self):
        # Real decorated generate_answer against a always-failing fake
        # client: Phase 6 burns all attempts, Phase 7 serves excerpts.
        rag = RAGSystem.__new__(RAGSystem)
        rag._retrieve_chunks = Mock(return_value=[self.chunk])
        fake_client = Mock()
        fake_client.models.generate_content = Mock(
            side_effect=TimeoutError("gemini down"))
        arguments = dict(self.arguments)
        with degraded_env(RETRY_BASE_DELAY="0"), patch(
                "phase1.get_gemini_client", return_value=fake_client):
            result = rag.run_once(**arguments)
        self.assertEqual(
            fake_client.models.generate_content.call_count, 3)
        self.assertTrue(result["degraded"])
        self.assertTrue(result["answer"].startswith(DEGRADED_PREFIX))
        self.assertEqual(self.tracker.count(), 1)

    def test_disabled_by_default_raises(self):
        with plain_env(), self.assertRaises(TimeoutError):
            self.rag.run_once(**dict(self.arguments, use_degraded=None))
        self.assertEqual(self.tracker.count(), 0)

    def test_use_degraded_true_forces_on_despite_disabled_env(self):
        with plain_env():
            result = self.rag.run_once(**self.arguments)
        self.assertTrue(result["degraded"])
        self.assertEqual(self.tracker.count(), 1)

    def test_non_transient_error_stays_an_error(self):
        self.rag.generate_answer = Mock(
            side_effect=ValueError("programming bug"))
        with degraded_env(), self.assertRaises(ValueError):
            self.rag.run_once(**self.arguments)
        self.assertEqual(self.tracker.count(), 0)

    def test_empty_retrieval_keeps_refusal(self):
        self.rag._retrieve_chunks = Mock(return_value=[])
        self.rag.generate_answer = Mock(return_value=REFUSAL_MESSAGE)
        with degraded_env():
            result = self.rag.run_once(**self.arguments)
        self.assertEqual(result["answer"], REFUSAL_MESSAGE)
        self.assertNotIn("degraded", result)
        self.assertEqual(self.tracker.count(), 0)

    def test_degraded_answers_are_never_cached(self):
        cache = CacheService(InMemoryRedis())
        arguments = dict(
            self.arguments, use_cache=True, response_cache=cache)
        with degraded_env():
            first = self.rag.run_once(**arguments)
            second = self.rag.run_once(**arguments)
        self.assertTrue(first["degraded"])
        self.assertTrue(second["degraded"])
        # The LLM is consulted again: nothing was stored.
        self.assertEqual(self.rag.generate_answer.call_count, 2)
        # Cache was consulted (miss both times) alongside degraded.
        self.assertFalse(first["cache_hit"])
        self.assertFalse(second["cache_hit"])
        self.assertIsNone(cache.get("alice", "What is RAG?", ["chunk-1"]))
        self.assertEqual(self.tracker.count(), 2)

    def test_served_without_tracker_when_redis_missing(self):
        arguments = dict(self.arguments, degraded_tracker=None)
        with degraded_env():
            result = self.rag.run_once(**arguments)
        self.assertTrue(result["degraded"])
        self.assertTrue(result["answer"].startswith(DEGRADED_PREFIX))

    def test_open_circuit_serves_excerpts_without_llm_call(self):
        # Real decorated generate_answer against an open shared
        # breaker: the Gemini client must never be touched (fail-fast),
        # and the failure still qualifies as LLM-unavailable.
        fake_client = Mock()
        fake_client.models.generate_content = Mock(
            return_value=Mock(text="Should never be used."))
        breaker = get_breaker("gemini")
        for _ in range(5):
            breaker.record_failure()
        self.assertFalse(breaker.allow())
        rag = RAGSystem.__new__(RAGSystem)
        rag._retrieve_chunks = Mock(return_value=[self.chunk])
        arguments = dict(self.arguments)
        with degraded_env(), patch(
                "phase1.get_gemini_client", return_value=fake_client):
            result = rag.run_once(**arguments)
        fake_client.models.generate_content.assert_not_called()
        self.assertTrue(result["degraded"])
        self.assertTrue(result["answer"].startswith(DEGRADED_PREFIX))
        self.assertEqual(self.tracker.count(), 1)


class StreamDegradedTests(unittest.TestCase):
    def setUp(self):
        reset_breakers()
        self.addCleanup(reset_breakers)
        self.chunk = make_chunk()
        self.rag = RAGSystem.__new__(RAGSystem)
        self.rag._retrieve_chunks = Mock(return_value=[self.chunk])
        self.rag.generate_answer_stream = Mock(
            side_effect=TimeoutError("gemini down"))
        self.tracker = DegradationTracker(InMemoryRedis())
        self.arguments = {
            "raw_query": "What is RAG?",
            "user_id": "alice",
            "rewrite_query": False,
            "use_degraded": True,
            "degraded_tracker": self.tracker,
        }

    def test_plain_stream_replays_excerpts_as_tokens(self):
        with degraded_env():
            tokens = list(self.rag.run_once_stream(**self.arguments))
        self.assertEqual(len(tokens), 1)
        self.assertTrue(tokens[0].startswith(DEGRADED_PREFIX))
        self.assertEqual(self.tracker.count(), 1)

    def test_mid_stream_failure_stays_an_error(self):
        def partial_then_down(**kwargs):
            yield "Partial answer"
            raise TimeoutError("mid-stream")

        self.rag.generate_answer_stream = Mock(side_effect=partial_then_down)
        with degraded_env():
            stream = self.rag.run_once_stream(**self.arguments)
            self.assertEqual(next(stream), "Partial answer")
            with self.assertRaises(TimeoutError):
                list(stream)
        self.assertEqual(self.tracker.count(), 0)

    def test_event_stream_degraded_shape(self):
        cache = CacheService(InMemoryRedis())
        arguments = dict(
            self.arguments, use_cache=True, response_cache=cache)
        with degraded_env():
            events = list(self.rag.run_once_event_stream(**arguments))
        self.assertEqual(
            [item["event"] for item in events],
            ["start", "retrieval", "token", "sources", "done"])
        self.assertTrue(
            events[2]["data"]["text"].startswith(DEGRADED_PREFIX))
        self.assertEqual(
            events[3]["data"]["sources"][0]["chunk_id"], "chunk-1")
        self.assertEqual(events[4]["data"], {
            "metadata": {"cache": "miss", "degraded": True},
        })
        self.assertEqual(self.tracker.count(), 1)

    def test_event_stream_degraded_without_cache(self):
        arguments = dict(self.arguments, use_degraded=True)
        with degraded_env():
            events = list(self.rag.run_once_event_stream(**arguments))
        self.assertEqual(events[-1]["data"], {
            "metadata": {"degraded": True},
        })

    def test_event_stream_disabled_keeps_error_path(self):
        with plain_env(), self.assertLogs("phase1", level="ERROR"):
            events = list(self.rag.run_once_event_stream(**dict(
                self.arguments, use_degraded=None)))
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(self.tracker.count(), 0)

    def test_event_stream_non_transient_stays_an_error(self):
        self.rag.generate_answer_stream = Mock(
            side_effect=ValueError("bug"))
        with degraded_env(), self.assertLogs("phase1", level="ERROR"):
            events = list(self.rag.run_once_event_stream(**self.arguments))
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(self.tracker.count(), 0)


class ThreadingTests(unittest.TestCase):
    def test_rag_service_threads_degraded_arguments(self):
        from app.schemas.chat import ChatRequest
        fake_rag = Mock()
        fake_rag.run_once = Mock(return_value={"answer": "ok"})
        fake_tracker = Mock()
        service = RAGService(
            rag=fake_rag, degradation_tracker=fake_tracker)
        service.run_once(
            ChatRequest(raw_query="q", user_id="u"), use_degraded=True)
        _, kwargs = fake_rag.run_once.call_args
        self.assertIs(kwargs["degraded_tracker"], fake_tracker)
        self.assertTrue(kwargs["use_degraded"])

    def test_orchestration_passes_use_degraded_through(self):
        from app.schemas.chat import ChatRequest
        fake_service = Mock()
        fake_service.run_once = Mock(return_value={"answer": "ok"})
        orchestration.run_chat_once(
            fake_service, None, ChatRequest(raw_query="q", user_id="u"),
            use_degraded=True)
        _, kwargs = fake_service.run_once.call_args
        self.assertTrue(kwargs["use_degraded"])


if __name__ == "__main__":
    unittest.main()
