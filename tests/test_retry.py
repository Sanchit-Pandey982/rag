"""Phase 6 tests: retries with backoff + circuit breaker, no live services.

Unit-cover the predicate (transient vs permanent), the sync/async/
generator wrappers (attempt counting, delay recording, no-retry
mid-stream, no-retry on cancellation), and the breaker lifecycle
(open after consecutive failures, fail-fast, half-open probe, reset).
Integration-cover one Gemini call (phase1.generate_answer with a flaky
fake client) and one Mongo read (ConversationService with a flaky
fake collection).
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    DuplicateKeyError,
    NetworkTimeout,
    OperationFailure,
    ServerSelectionTimeoutError,
)

from app.utils.retry import (
    CircuitBreaker,
    CircuitBreakerOpen,
    backoff_delay,
    circuit_cooldown,
    circuit_threshold,
    get_breaker,
    is_transient_error,
    reset_breakers,
    retry_base_delay,
    retry_max_attempts,
    retry_max_delay,
    retry_with_backoff,
)


def no_retry_env():
    return patch.dict(os.environ, {
        "RETRY_MAX_ATTEMPTS": "3",
        "RETRY_BASE_DELAY": "0",
        "RETRY_MAX_DELAY": "0",
    }, clear=False)


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        env = {k: v for k, v in os.environ.items()
               if not (k.startswith("RETRY_") or k.startswith("CIRCUIT_"))}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(retry_max_attempts(), 3)
            self.assertEqual(retry_base_delay(), 0.5)
            self.assertEqual(retry_max_delay(), 8.0)
            self.assertEqual(circuit_threshold(), 5)
            self.assertEqual(circuit_cooldown(), 30.0)

    def test_invalid_values_fall_back(self):
        with patch.dict(os.environ, {
            "RETRY_MAX_ATTEMPTS": "bogus",
            "RETRY_BASE_DELAY": "-1",
            "RETRY_MAX_DELAY": "0",
            "CIRCUIT_BREAKER_THRESHOLD": "0",
            "CIRCUIT_BREAKER_COOLDOWN_SECONDS": "nope",
        }, clear=False):
            self.assertEqual(retry_max_attempts(), 3)
            self.assertEqual(retry_base_delay(), 0.5)
            self.assertEqual(retry_max_delay(), 8.0)
            self.assertEqual(circuit_threshold(), 5)
            self.assertEqual(circuit_cooldown(), 30.0)


class FakeStatusError(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.code = code


class PredicateTests(unittest.TestCase):
    def test_timeouts_and_connections_retry(self):
        self.assertTrue(is_transient_error(TimeoutError("t")))
        self.assertTrue(is_transient_error(ConnectionError("c")))
        self.assertTrue(is_transient_error(asyncio.TimeoutError()))

    def test_rate_limit_and_server_errors_retry(self):
        from google.genai.errors import APIError
        self.assertTrue(is_transient_error(
            APIError(code=429, response_json={})))
        self.assertTrue(is_transient_error(
            APIError(code=500, response_json={})))
        self.assertTrue(is_transient_error(
            APIError(code=503, response_json={})))
        self.assertTrue(is_transient_error(FakeStatusError(429)))
        self.assertTrue(is_transient_error(FakeStatusError(599)))

    def test_client_errors_never_retry(self):
        from google.genai.errors import APIError
        for code in (400, 401, 403, 404):
            self.assertFalse(is_transient_error(
                APIError(code=code, response_json={})), code)
            self.assertFalse(is_transient_error(FakeStatusError(code)), code)
        self.assertFalse(is_transient_error(ValueError("bug")))
        self.assertFalse(is_transient_error(RuntimeError("bug")))
        self.assertFalse(is_transient_error(
            OperationFailure("auth failed")))
        self.assertFalse(is_transient_error(
            DuplicateKeyError("dup")))
        self.assertFalse(is_transient_error(CircuitBreakerOpen("open")))

    def test_pymongo_connection_failures_retry(self):
        self.assertTrue(is_transient_error(AutoReconnect("gone")))
        self.assertTrue(is_transient_error(ConnectionFailure("gone")))
        self.assertTrue(is_transient_error(NetworkTimeout("slow")))
        self.assertTrue(
            is_transient_error(ServerSelectionTimeoutError("none")))

    def test_mock_attributes_do_not_crash_or_retry(self):
        # Mock auto-attributes are not ints: must be ignored, not retried.
        self.assertFalse(is_transient_error(Mock()))
        self.assertFalse(is_transient_error(FakeStatusError(True)))


class SyncRetryTests(unittest.TestCase):
    def test_transient_then_success(self):
        calls = []
        delays = []

        @retry_with_backoff(attempts=3, base_delay=1.0, max_delay=30.0,
                            sleep=delays.append)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError("blip")
            return "ok"

        with self.assertLogs("app.utils.retry", level="WARNING") as logs:
            self.assertEqual(flaky(), "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(delays), 2)
        # Exponential growth with 50-100% jitter bands.
        self.assertGreaterEqual(delays[0], 0.5)
        self.assertLessEqual(delays[0], 1.0)
        self.assertGreaterEqual(delays[1], 1.0)
        self.assertLessEqual(delays[1], 2.0)
        self.assertIn("attempt 1/3", logs.output[0])
        self.assertIn("TimeoutError", logs.output[0])

    def test_permanent_error_raises_immediately(self):
        calls = []
        delays = []

        @retry_with_backoff(attempts=3, sleep=delays.append)
        def broken():
            calls.append(1)
            raise ValueError("bug")

        with self.assertRaises(ValueError):
            broken()
        self.assertEqual(len(calls), 1)
        self.assertEqual(delays, [])

    def test_exhaustion_reraises_last_error(self):
        calls = []

        @retry_with_backoff(attempts=2, sleep=lambda delay: None)
        def down():
            calls.append(1)
            raise ConnectionError("down")

        with self.assertRaises(ConnectionError):
            down()
        self.assertEqual(len(calls), 2)

    def test_bare_decorator_uses_env_config(self):
        calls = []

        @retry_with_backoff
        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("blip")
            return "ok"

        with no_retry_env():
            self.assertEqual(flaky(), "ok")
        self.assertEqual(len(calls), 2)

    def test_429_retries_but_404_does_not(self):
        from google.genai.errors import APIError
        limited_calls = []

        @retry_with_backoff(attempts=3, sleep=lambda delay: None)
        def limited():
            limited_calls.append(1)
            raise APIError(code=429, response_json={})

        with self.assertRaises(APIError):
            limited()
        self.assertEqual(len(limited_calls), 3)

        calls = []

        @retry_with_backoff(attempts=3, sleep=calls.append)
        def missing_counted():
            raise APIError(code=404, response_json={})

        with self.assertRaises(APIError):
            missing_counted()
        self.assertEqual(calls, [])

    def test_delay_capped_by_max(self):
        with patch("app.utils.retry.random.uniform", return_value=1.0):
            self.assertEqual(
                backoff_delay(10, base_delay=1.0, max_delay=5.0), 5.0)
            self.assertEqual(
                backoff_delay(1, base_delay=2.0, max_delay=100.0), 2.0)
            self.assertEqual(
                backoff_delay(3, base_delay=2.0, max_delay=100.0), 8.0)


class AsyncRetryTests(unittest.TestCase):
    def test_async_transient_then_success(self):
        calls = []
        delays = []

        async def record(delay):
            delays.append(delay)

        @retry_with_backoff(attempts=3, base_delay=1.0, sleep=record)
        async def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("blip")
            return "ok"

        self.assertEqual(asyncio.run(flaky()), "ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(delays), 1)

    def test_async_cancellation_never_retried(self):
        calls = []

        @retry_with_backoff(attempts=3, sleep=lambda delay: None)
        async def cancelled():
            calls.append(1)
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(cancelled())
        self.assertEqual(len(calls), 1)


class GeneratorRetryTests(unittest.TestCase):
    def test_retry_before_first_item(self):
        calls = []
        delays = []

        @retry_with_backoff(attempts=2, sleep=delays.append)
        def flaky_stream():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("no stream yet")
            yield "a"
            yield "b"

        self.assertEqual(list(flaky_stream()), ["a", "b"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(delays), 1)

    def test_no_retry_after_stream_started(self):
        calls = []
        delays = []

        @retry_with_backoff(attempts=3, sleep=delays.append)
        def failing_stream():
            calls.append(1)
            yield "partial"
            raise TimeoutError("mid-stream")

        with self.assertRaises(TimeoutError):
            list(failing_stream())
        self.assertEqual(len(calls), 1)
        self.assertEqual(delays, [])

    def test_empty_stream_succeeds_without_retry(self):
        @retry_with_backoff(attempts=3, sleep=Mock())
        def empty():
            if False:
                yield "never"
        self.assertEqual(list(empty()), [])


class CircuitBreakerTests(unittest.TestCase):
    def test_opens_after_consecutive_failures(self):
        breaker = CircuitBreaker(threshold=3, cooldown=60.0)
        self.assertTrue(breaker.allow())
        breaker.record_failure()
        breaker.record_failure()
        self.assertTrue(breaker.allow())
        breaker.record_failure()
        self.assertTrue(breaker.is_open)
        self.assertFalse(breaker.allow())

    def test_success_resets_streak(self):
        breaker = CircuitBreaker(threshold=2, cooldown=60.0)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        self.assertTrue(breaker.allow())
        self.assertEqual(breaker.failures, 1)

    def test_half_open_probe(self):
        breaker = CircuitBreaker(threshold=1, cooldown=0.01)
        breaker.record_failure()
        self.assertFalse(breaker.allow())
        import time
        time.sleep(0.02)
        self.assertTrue(breaker.allow())
        breaker.record_failure()
        self.assertFalse(breaker.allow())
        time.sleep(0.02)
        self.assertTrue(breaker.allow())
        breaker.record_success()
        self.assertTrue(breaker.allow())
        self.assertEqual(breaker.failures, 0)

    def test_permanent_errors_do_not_trip_breaker(self):
        breaker = CircuitBreaker(threshold=2, cooldown=60.0)

        @retry_with_backoff(attempts=3, breaker=breaker,
                            sleep=lambda delay: None)
        def broken():
            raise ValueError("bug")

        with self.assertRaises(ValueError):
            broken()
        with self.assertRaises(ValueError):
            broken()
        self.assertEqual(breaker.failures, 0)
        self.assertTrue(breaker.allow())

    def test_open_breaker_fails_fast_without_calling(self):
        calls = []
        breaker = CircuitBreaker(threshold=3, cooldown=60.0)

        @retry_with_backoff(attempts=3, breaker=breaker,
                            sleep=lambda delay: None)
        def down():
            calls.append(1)
            raise TimeoutError("down")

        with self.assertRaises(TimeoutError):
            down()
        self.assertEqual(len(calls), 3)
        self.assertEqual(breaker.failures, 3)
        with self.assertRaises(CircuitBreakerOpen):
            down()
        # Fail-fast: no further attempts, no sleeps, no calls.
        self.assertEqual(len(calls), 3)

    def test_registry_shares_breakers_by_name(self):
        self.addCleanup(reset_breakers)
        reset_breakers()
        first = get_breaker("gemini")
        self.assertIs(get_breaker("gemini"), first)
        self.assertIsNot(get_breaker("mongodb"), first)
        reset_breakers()
        self.assertIsNot(get_breaker("gemini"), first)


class IntegrationTests(unittest.TestCase):
    def tearDown(self):
        # Gemini-failure paths record on the shared registry breaker;
        # never leak an open circuit into other modules.
        reset_breakers()

    def test_gemini_transient_failure_retries_then_succeeds(self):
        from phase1 import RAGSystem
        rag = RAGSystem.__new__(RAGSystem)
        response = Mock(text="  A grounded answer.  ")
        fake_client = Mock()
        fake_client.models.generate_content = Mock(side_effect=[
            TimeoutError("blip"), response,
        ])
        chunks = [Mock(chunk_id="c1", text="t", distance=0.1,
                       metadata={"document_id": "d", "source": "d.txt",
                                 "title": "D", "chunk_index": 0})]
        with no_retry_env(), patch(
                "phase1.get_gemini_client", return_value=fake_client):
            answer = rag.generate_answer(
                question="q?", chunks=chunks, chat_history=[])
        self.assertEqual(answer, "A grounded answer.")
        self.assertEqual(
            fake_client.models.generate_content.call_count, 2)

    def test_gemini_404_does_not_retry(self):
        from google.genai.errors import APIError
        from phase1 import RAGSystem
        rag = RAGSystem.__new__(RAGSystem)
        fake_client = Mock()
        fake_client.models.generate_content = Mock(
            side_effect=APIError(code=404, response_json={}))
        chunks = [Mock(chunk_id="c1", text="t", distance=0.1,
                       metadata={"document_id": "d", "source": "d.txt",
                                 "title": "D", "chunk_index": 0})]
        with no_retry_env(), patch(
                "phase1.get_gemini_client", return_value=fake_client):
            with self.assertRaises(APIError):
                rag.generate_answer(
                    question="q?", chunks=chunks, chat_history=[])
        self.assertEqual(
            fake_client.models.generate_content.call_count, 1)

    def test_mongo_transient_failure_retries(self):
        from app.services.conversation_service import ConversationService
        service = ConversationService(Mock(), Mock())
        document = {"conversation_id": "c", "user_id": "u"}
        service.conversations.find_one = AsyncMock(side_effect=[
            AutoReconnect("blip"),
            AutoReconnect("blip"),
            document,
        ])
        with no_retry_env():
            result = asyncio.run(
                service.get_conversation("c", "u"))
        self.assertEqual(result, document)
        self.assertEqual(service.conversations.find_one.call_count, 3)

    def test_mongo_permanent_failure_raises_at_once(self):
        from pymongo.errors import OperationFailure
        from app.services.conversation_service import (
            ConversationService,
            ConversationStoreUnavailable,
        )
        service = ConversationService(Mock(), Mock())
        service.conversations.find_one = AsyncMock(
            side_effect=OperationFailure("bad query"))
        with no_retry_env():
            with self.assertLogs(
                    "app.services.conversation_service", level="ERROR"):
                with self.assertRaises(ConversationStoreUnavailable):
                    asyncio.run(service.get_conversation("c", "u"))
        self.assertEqual(service.conversations.find_one.call_count, 1)


if __name__ == "__main__":
    unittest.main()
