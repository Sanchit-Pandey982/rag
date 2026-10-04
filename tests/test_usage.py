"""Phase 8 tests: token usage + cost tracking, no live services.

Real UsageService + real phase1 capture against fakes (fake Gemini
client with usage_metadata, in-memory usage_logs double with a small
aggregation interpreter). Covers config parsing, cost math, metadata
extraction strictness, all three transports reporting usage, ledger
persistence + daily aggregation + cutoff filtering, the /api/v1/usage
endpoint (auth scoping, 503 without service), and orchestration
recording (skip when absent, fail-open on store outage).
"""

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies.auth import get_current_user
from app.routes.usage import router as usage_router
from app.schemas.auth import UserResponse
from app.schemas.chat import ChatRequest
from app.services import chat_orchestration as orchestration
from app.services.usage_service import (
    UsageService,
    UsageStoreUnavailable,
    cost_per_1k_input,
    cost_per_1k_output,
    estimate_cost,
)
from phase1 import (
    RAGSystem,
    RetrievedChunk,
    extract_usage_metadata,
    usage_from_metadata,
)


def plain_env():
    import os
    env = {
        key: value for key, value in os.environ.items()
        if key not in ("COST_PER_1K_INPUT", "COST_PER_1K_OUTPUT")
    }
    return patch.dict(os.environ, env, clear=True)


def cost_env(**overrides):
    import os
    base = {
        "COST_PER_1K_INPUT": "0.0001",
        "COST_PER_1K_OUTPUT": "0.0004",
    }
    base.update(overrides)
    return patch.dict(os.environ, base, clear=False)


def make_chunk(chunk_id="chunk-1"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Metered context about RAG.",
        distance=0.2,
        metadata={
            "document_id": "rag_basics",
            "source": "rag_basics.txt",
            "title": "RAG basics",
            "chunk_index": 0,
        },
    )


def make_metadata(prompt=100, completion=50, total=150):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=completion,
        total_token_count=total,
    )


class FakeAggCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    async def to_list(self, length=None):
        if length is None:
            return list(self._rows)
        return self._rows[:length]


class FakeUsageCollection:
    """Usage-log double: exact inserts + a $match/$group/$sort interpreter."""

    def __init__(self):
        self.documents = []

    async def create_index(self, *args, **kwargs):
        return "index"

    async def insert_one(self, document):
        self.documents.append(dict(document))
        return SimpleNamespace(inserted_id=document.get("usage_id"))

    def aggregate(self, pipeline):
        documents = list(self.documents)
        for stage in pipeline:
            if "$match" in stage:
                documents = [
                    doc for doc in documents
                    if self._matches(doc, stage["$match"])
                ]
            elif "$group" in stage:
                documents = self._group(documents, stage["$group"])
            elif "$sort" in stage:
                key, direction = next(iter(stage["$sort"].items()))
                documents.sort(
                    key=lambda doc: doc.get(key),
                    reverse=(direction < 0),
                )
        return FakeAggCursor(documents)

    def _matches(self, document, spec):
        for key, condition in spec.items():
            if isinstance(condition, dict) and "$gte" in condition:
                if not document.get(key) >= condition["$gte"]:
                    return False
            elif document.get(key) != condition:
                return False
        return True

    def _group(self, documents, spec):
        groups = {}
        for document in documents:
            date_spec = spec["_id"]["$dateToString"]
            field = date_spec["date"].lstrip("$")
            day = document[field].strftime(date_spec["format"])
            group = groups.setdefault(day, {
                "_id": day,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
            })
            for output, expression in spec.items():
                if output == "_id":
                    continue
                operator, value = next(iter(expression.items()))
                if operator == "$sum":
                    group[output] += (
                        1 if value == 1
                        else document[value.lstrip("$")]
                    )
        return list(groups.values())


class ConfigTests(unittest.TestCase):
    def test_defaults_match_roadmap_rates(self):
        with plain_env():
            self.assertEqual(cost_per_1k_input(), 0.0001)
            self.assertEqual(cost_per_1k_output(), 0.0004)

    def test_custom_rates_and_zero_allowed(self):
        with cost_env(COST_PER_1K_INPUT="0.002",
                      COST_PER_1K_OUTPUT="0"):
            self.assertEqual(cost_per_1k_input(), 0.002)
            self.assertEqual(cost_per_1k_output(), 0.0)

    def test_invalid_and_negative_fall_back(self):
        for raw in ("bogus", "-1"):
            with cost_env(COST_PER_1K_INPUT=raw, COST_PER_1K_OUTPUT=raw):
                self.assertEqual(cost_per_1k_input(), 0.0001, raw)
                self.assertEqual(cost_per_1k_output(), 0.0004, raw)


class CostTests(unittest.TestCase):
    def test_estimate_is_accurate(self):
        with cost_env():
            # 1000 in @ .0001 + 2000 out @ .0004 = .0009 exactly.
            self.assertAlmostEqual(
                estimate_cost(1000, 2000), 0.0009, places=9)
            self.assertEqual(estimate_cost(0, 0), 0.0)

    def test_estimate_honors_explicit_rates(self):
        self.assertAlmostEqual(
            estimate_cost(1000, 1000, input_rate=1.0, output_rate=2.0),
            3.0, places=9)


class ExtractTests(unittest.TestCase):
    def test_full_metadata_extracted(self):
        report = extract_usage_metadata(
            SimpleNamespace(text="hi", usage_metadata=make_metadata()))
        self.assertEqual(report, {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "model": "gemini-3.6-flash",
        })

    def test_absent_or_partial_metadata_rejected(self):
        self.assertIsNone(extract_usage_metadata(SimpleNamespace(text="hi")))
        self.assertIsNone(extract_usage_metadata(
            SimpleNamespace(usage_metadata=None)))
        self.assertIsNone(usage_from_metadata(
            SimpleNamespace(prompt_token_count=1)))
        self.assertIsNone(usage_from_metadata(
            SimpleNamespace(prompt_token_count=1,
                            candidates_token_count=2,
                            total_token_count="3")))

    def test_negative_bool_and_mock_values_rejected(self):
        self.assertIsNone(usage_from_metadata(
            SimpleNamespace(prompt_token_count=-1,
                            candidates_token_count=0,
                            total_token_count=0)))
        self.assertIsNone(usage_from_metadata(
            SimpleNamespace(prompt_token_count=True,
                            candidates_token_count=0,
                            total_token_count=1)))
        # Mock auto-attributes are never ints: must not fabricate rows.
        self.assertIsNone(extract_usage_metadata(Mock()))


class GenerateCaptureTests(unittest.TestCase):
    def setUp(self):
        self.rag = RAGSystem.__new__(RAGSystem)
        self.chunk = make_chunk()

    def test_generate_answer_fills_collector(self):
        response = SimpleNamespace(
            text="  A grounded answer.  ",
            usage_metadata=make_metadata())
        fake_client = Mock()
        fake_client.models.generate_content = Mock(return_value=response)
        usage: dict = {}
        with patch("phase1.get_gemini_client", return_value=fake_client):
            answer = self.rag.generate_answer(
                question="q?", chunks=[self.chunk], chat_history=[],
                usage=usage)
        self.assertEqual(answer, "A grounded answer.")
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 150)

    def test_generate_answer_without_metadata_leaves_collector(self):
        fake_client = Mock()
        fake_client.models.generate_content = Mock(
            return_value=SimpleNamespace(text="hi"))
        usage: dict = {}
        with patch("phase1.get_gemini_client", return_value=fake_client):
            self.rag.generate_answer(
                question="q?", chunks=[self.chunk], chat_history=[],
                usage=usage)
        self.assertEqual(usage, {})

    def test_stream_takes_last_chunk_metadata(self):
        chunks = [
            SimpleNamespace(text="a", usage_metadata=None),
            SimpleNamespace(text="b", usage_metadata=make_metadata(10, 5, 15)),
        ]
        fake_client = Mock()
        fake_client.models.generate_content_stream = Mock(
            return_value=iter(chunks))
        usage: dict = {}
        with patch("phase1.get_gemini_client", return_value=fake_client):
            tokens = list(self.rag.generate_answer_stream(
                question="q?", chunks=[self.chunk], chat_history=[],
                usage=usage))
        self.assertEqual(tokens, ["a", "b"])
        self.assertEqual(usage, {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "model": "gemini-3.6-flash",
        })

    def test_stream_without_metadata_leaves_collector(self):
        fake_client = Mock()
        fake_client.models.generate_content_stream = Mock(
            return_value=iter([SimpleNamespace(text="a")]))
        usage: dict = {}
        with patch("phase1.get_gemini_client", return_value=fake_client):
            self.assertEqual(list(self.rag.generate_answer_stream(
                question="q?", chunks=[self.chunk], chat_history=[],
                usage=usage)), ["a"])
        self.assertEqual(usage, {})


class RunOnceUsageTests(unittest.TestCase):
    def setUp(self):
        self.chunk = make_chunk()
        self.rag = RAGSystem.__new__(RAGSystem)
        self.rag._retrieve_chunks = Mock(return_value=[self.chunk])
        self.arguments = {
            "raw_query": "What is RAG?",
            "user_id": "alice",
            "rewrite_query": False,
        }

    def _client(self, metadata=None, text="A grounded answer."):
        response = SimpleNamespace(text=text, usage_metadata=metadata)
        fake_client = Mock()
        fake_client.models.generate_content = Mock(return_value=response)
        fake_client.models.generate_content_stream = Mock(
            return_value=iter([SimpleNamespace(
                text=text, usage_metadata=metadata)]))
        return fake_client

    def test_run_once_attaches_usage(self):
        with patch("phase1.get_gemini_client",
                   return_value=self._client(make_metadata())):
            result = self.rag.run_once(**self.arguments)
        self.assertEqual(result["usage"]["prompt_tokens"], 100)
        self.assertEqual(result["usage"]["total_tokens"], 150)
        self.assertEqual(result["usage"]["model"], "gemini-3.6-flash")

    def test_run_once_without_metadata_keeps_shape(self):
        with patch("phase1.get_gemini_client",
                   return_value=self._client(None)):
            result = self.rag.run_once(**self.arguments)
        self.assertNotIn("usage", result)

    def test_event_stream_done_carries_usage(self):
        with patch("phase1.get_gemini_client",
                   return_value=self._client(make_metadata(7, 8, 15))):
            events = list(self.rag.run_once_event_stream(**self.arguments))
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["data"]["metadata"]["usage"], {
            "prompt_tokens": 7,
            "completion_tokens": 8,
            "total_tokens": 15,
            "model": "gemini-3.6-flash",
        })

    def test_event_stream_without_metadata_keeps_done_empty(self):
        with patch("phase1.get_gemini_client",
                   return_value=self._client(None)):
            events = list(self.rag.run_once_event_stream(**self.arguments))
        self.assertEqual(events[-1]["data"], {})

    def test_plain_stream_fills_caller_collector(self):
        usage: dict = {}
        with patch("phase1.get_gemini_client",
                   return_value=self._client(make_metadata(3, 4, 7))):
            tokens = list(self.rag.run_once_stream(
                **dict(self.arguments, usage=usage)))
        self.assertEqual("".join(tokens), "A grounded answer.")
        self.assertEqual(usage["completion_tokens"], 4)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeUsageCollection()
        self.service = UsageService(self.collection)

    def test_record_persists_estimated_cost(self):
        with cost_env():
            document = asyncio.run(self.service.record_usage(
                user_id="alice", prompt_tokens=1000, completion_tokens=2000,
                total_tokens=3000, model="gemini-3.6-flash"))
        self.assertEqual(document["user_id"], "alice")
        self.assertAlmostEqual(document["cost_usd"], 0.0009, places=9)
        self.assertIn("usage_id", document)
        self.assertIsNotNone(document["timestamp"].tzinfo)
        self.assertEqual(len(self.collection.documents), 1)

    def test_record_rejects_bad_rows(self):
        with self.assertRaises(ValueError):
            asyncio.run(self.service.record_usage(
                user_id="alice", prompt_tokens=-1, completion_tokens=0,
                total_tokens=0, model="m"))
        with self.assertRaises(ValueError):
            asyncio.run(self.service.record_usage(
                user_id="alice", prompt_tokens=1, completion_tokens=0,
                total_tokens=1, model=""))
        self.assertEqual(self.collection.documents, [])

    def test_summary_groups_days_and_cuts_off(self):
        now = datetime.now(timezone.utc)
        self.collection.documents.extend([
            {"usage_id": "a", "user_id": "alice", "timestamp": now,
             "prompt_tokens": 100, "completion_tokens": 50,
             "total_tokens": 150, "model": "m", "cost_usd": 0.001},
            {"usage_id": "b", "user_id": "alice",
             "timestamp": now - timedelta(days=1, hours=1),
             "prompt_tokens": 200, "completion_tokens": 0,
             "total_tokens": 200, "model": "m", "cost_usd": 0.002},
            {"usage_id": "c", "user_id": "bob", "timestamp": now,
             "prompt_tokens": 999, "completion_tokens": 999,
             "total_tokens": 1998, "model": "m", "cost_usd": 9.0},
            {"usage_id": "d", "user_id": "alice",
             "timestamp": now - timedelta(days=40),
             "prompt_tokens": 500, "completion_tokens": 500,
             "total_tokens": 1000, "model": "m", "cost_usd": 1.0},
        ])
        summary = asyncio.run(
            self.service.get_usage_summary("alice", days=7))
        self.assertEqual(summary["user_id"], "alice")
        self.assertEqual(summary["days"], 7)
        # Bob's row and the 40-day-old row are excluded.
        self.assertEqual(summary["total"]["prompt_tokens"], 300)
        self.assertEqual(summary["total"]["completion_tokens"], 50)
        self.assertEqual(summary["total"]["total_tokens"], 350)
        self.assertAlmostEqual(
            summary["total"]["cost_usd"], 0.003, places=9)
        self.assertEqual(summary["total"]["requests"], 2)
        self.assertEqual(len(summary["daily"]), 2)
        self.assertLess(summary["daily"][0]["date"],
                        summary["daily"][1]["date"])

    def test_summary_empty_window(self):
        summary = asyncio.run(
            self.service.get_usage_summary("nobody", days=7))
        self.assertEqual(summary["total"]["requests"], 0)
        self.assertEqual(summary["daily"], [])

    def test_summary_rejects_bad_days(self):
        for days in (0, 32, "7", True):
            with self.assertRaises(ValueError):
                asyncio.run(
                    self.service.get_usage_summary("alice", days=days))

    def test_store_outage_maps_to_domain_error(self):
        class DownCollection(FakeUsageCollection):
            async def insert_one(self, document):
                from pymongo.errors import ConnectionFailure
                raise ConnectionFailure("mongo down")

        service = UsageService(DownCollection())
        with self.assertRaises(UsageStoreUnavailable):
            asyncio.run(service.record_usage(
                user_id="alice", prompt_tokens=1, completion_tokens=1,
                total_tokens=2, model="m"))


class RouteTests(unittest.TestCase):
    def _app(self, usage_service):
        async def fake_current_user():
            return UserResponse(
                user_id="alice",
                username="alice",
                created_at=datetime.now(timezone.utc),
            )

        app = FastAPI()
        app.dependency_overrides[get_current_user] = fake_current_user
        app.include_router(usage_router)
        if usage_service is not None:
            app.state.usage_service = usage_service
        return app

    def test_usage_endpoint_returns_totals(self):
        collection = FakeUsageCollection()
        service = UsageService(collection)
        now = datetime.now(timezone.utc)
        collection.documents.append({
            "usage_id": "a", "user_id": "alice", "timestamp": now,
            "prompt_tokens": 100, "completion_tokens": 50,
            "total_tokens": 150, "model": "m", "cost_usd": 0.001,
        })
        with TestClient(self._app(service)) as client:
            response = client.get("/api/v1/usage?days=7")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user_id"], "alice")
        self.assertEqual(body["days"], 7)
        self.assertEqual(body["total"]["prompt_tokens"], 100)
        self.assertEqual(len(body["daily"]), 1)

    def test_usage_endpoint_rejects_bad_days(self):
        service = UsageService(FakeUsageCollection())
        with TestClient(self._app(service)) as client:
            self.assertEqual(
                client.get("/api/v1/usage?days=99").status_code, 422)

    def test_usage_endpoint_503_without_service(self):
        with TestClient(self._app(None)) as client:
            response = client.get("/api/v1/usage")
        self.assertEqual(response.status_code, 503)


class OrchestrationTests(unittest.TestCase):
    def test_run_chat_once_records_usage(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5,
                 "total_tokens": 15, "model": "gemini-3.6-flash"}
        fake_rag = Mock()
        fake_rag.run_once = Mock(
            return_value={"answer": "hi", "usage": usage})
        usage_service = Mock()
        usage_service.record_usage = AsyncMock(return_value={})
        result = orchestration.run_chat_once(
            fake_rag, None, ChatRequest(raw_query="q", user_id="alice"),
            usage_service=usage_service)
        self.assertEqual(result["answer"], "hi")
        usage_service.record_usage.assert_called_once()
        _, kwargs = usage_service.record_usage.call_args
        self.assertEqual(kwargs["user_id"], "alice")
        self.assertEqual(kwargs["prompt_tokens"], 10)
        self.assertEqual(kwargs["completion_tokens"], 5)
        self.assertEqual(kwargs["total_tokens"], 15)
        self.assertEqual(kwargs["model"], "gemini-3.6-flash")

    def test_run_chat_once_skips_without_usage(self):
        fake_rag = Mock()
        fake_rag.run_once = Mock(return_value={"answer": "hi"})
        usage_service = Mock()
        usage_service.record_usage = AsyncMock()
        orchestration.run_chat_once(
            fake_rag, None, ChatRequest(raw_query="q", user_id="alice"),
            usage_service=usage_service)
        usage_service.record_usage.assert_not_called()

    def test_run_chat_once_fail_open_on_store_outage(self):
        fake_rag = Mock()
        fake_rag.run_once = Mock(return_value={
            "answer": "hi",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2, "model": "m"},
        })
        usage_service = Mock()
        usage_service.record_usage = AsyncMock(
            side_effect=RuntimeError("mongo down"))
        with self.assertLogs("app.services.chat_orchestration",
                             level="ERROR") as logs:
            result = orchestration.run_chat_once(
                fake_rag, None,
                ChatRequest(raw_query="q", user_id="alice"),
                usage_service=usage_service)
        # The outage is logged, but the answer still goes out: only the
        # ledger write was lost.
        self.assertEqual(result["answer"], "hi")
        self.assertIn("Could not record token usage", logs.output[0])

    def test_event_stream_records_on_done(self):
        usage = {"prompt_tokens": 4, "completion_tokens": 2,
                 "total_tokens": 6, "model": "m"}
        events = [
            {"event": "token", "data": {"text": "hi"}},
            {"event": "done", "data": {"metadata": {"usage": usage}}},
        ]
        usage_service = Mock()
        usage_service.record_usage = AsyncMock(return_value={})
        payload = ChatRequest(raw_query="q", user_id="alice")
        forwarded = list(orchestration.wrap_event_stream(
            iter(events), None, payload, None,
            usage_service=usage_service))
        self.assertEqual(forwarded, events)
        usage_service.record_usage.assert_called_once()

    def test_event_stream_skips_without_usage(self):
        events = [{"event": "done", "data": {}}]
        usage_service = Mock()
        usage_service.record_usage = AsyncMock()
        payload = ChatRequest(raw_query="q", user_id="alice")
        list(orchestration.wrap_event_stream(
            iter(events), None, payload, None,
            usage_service=usage_service))
        usage_service.record_usage.assert_not_called()

    def test_text_stream_records_collector_on_completion(self):
        usage_service = Mock()
        usage_service.record_usage = AsyncMock(return_value={})
        payload = ChatRequest(raw_query="q", user_id="alice")
        usage = {"prompt_tokens": 9, "completion_tokens": 9,
                 "total_tokens": 18, "model": "m"}
        out = list(orchestration.wrap_text_stream(
            iter(["a", "b"]), None, payload, None,
            usage_service=usage_service, usage=usage))
        self.assertEqual(out, ["a", "b"])
        usage_service.record_usage.assert_called_once()


if __name__ == "__main__":
    unittest.main()
