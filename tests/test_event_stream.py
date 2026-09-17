"""Exercise the real RAG pipeline and SSE route without Gemini or Chroma calls."""

import asyncio
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import phase1
from app.dependencies.auth import get_current_user
from app.routes.chat import router
from app.schemas.auth import UserResponse
from app.services.rag_services import RAGService
from phase1 import RAGSystem, RetrievedChunk


class EventStreamTests(unittest.TestCase):
    def setUp(self):
        # Skip persistent-store construction while keeping the real pipeline.
        self.rag = RAGSystem.__new__(RAGSystem)
        self.chunk = RetrievedChunk(
            chunk_id="chunk-1",
            text="RAG retrieves context before generation.",
            distance=0.2,
            metadata={
                "document_id": "rag_basics",
                "source": "rag_basics.txt",
                "title": "RAG basics",
                "chunk_index": 0,
            },
        )
        self.rag.retrieve = Mock(return_value=[self.chunk])
        self.rag.generate_answer_stream = Mock(
            side_effect=lambda **kwargs: iter(["A grounded ", "answer."])
        )
        self.arguments = {
            "raw_query": "What is RAG?",
            "user_id": "test-user",
            "rewrite_query": False,
        }

    def events(self):
        return list(self.rag.run_once_event_stream(**self.arguments))

    def assert_failure(self, stage, expected_names):
        with self.assertLogs("phase1", level="ERROR") as logs:
            events = self.events()
        self.assertEqual([item["event"] for item in events], expected_names)
        self.assertEqual(events[-1]["data"], {
            "stage": stage,
            "message": "The response could not be completed.",
        })
        self.assertNotIn("private failure detail", json.dumps(events))
        self.assertIn("private failure detail", logs.output[0])
        self.assertIn("Traceback", logs.output[0])
        self.assertIn(stage, logs.output[0])
        return events

    def test_normal_completion_preserves_event_contract(self):
        events = self.events()
        self.assertEqual([item["event"] for item in events], [
            "start", "retrieval", "token", "token", "sources", "done",
        ])
        self.assertEqual(events[1]["data"], {
            "retrieval_query": "What is RAG?",
            "retrieved_document_ids": ["rag_basics"],
        })
        self.assertEqual(events[2]["data"], {"text": "A grounded "})
        self.assertEqual(events[-2]["data"]["sources"][0]["chunk_id"], "chunk-1")
        self.assertEqual(events[-1]["data"], {})

    def test_retrieval_failure(self):
        self.rag.retrieve.side_effect = RuntimeError("private failure detail")
        self.assert_failure("retrieval", ["start", "error"])
        self.rag.generate_answer_stream.assert_not_called()

    def test_generation_failure_after_partial_answer(self):
        def failed_generation(**kwargs):
            yield "Partial answer"
            raise RuntimeError("private failure detail")

        self.rag.generate_answer_stream.side_effect = failed_generation
        events = self.assert_failure("generation", ["start", "retrieval", "token", "error"])
        self.assertEqual(events[2]["data"], {"text": "Partial answer"})

    def test_generation_timeout_before_first_token(self):
        self.rag.generate_answer_stream.side_effect = TimeoutError("private failure detail")
        self.assert_failure("generation", ["start", "retrieval", "error"])

    def test_question_rewrite_failure(self):
        self.arguments["rewrite_query"] = True
        with patch("phase1.condense_question", side_effect=RuntimeError("private failure detail")):
            self.assert_failure("query_rewrite", ["start", "error"])
        self.rag.retrieve.assert_not_called()

    def test_source_failure(self):
        del self.chunk.metadata["title"]
        with self.assertLogs("phase1", level="ERROR"):
            events = self.events()
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["data"]["stage"], "sources")
        self.assertNotIn("done", [item["event"] for item in events])

    def test_cancellation_is_not_an_application_error(self):
        self.rag.retrieve.side_effect = asyncio.CancelledError()
        stream = self.rag.run_once_event_stream(**self.arguments)
        self.assertEqual(next(stream)["event"], "start")
        with self.assertNoLogs("phase1", level="ERROR"):
            with self.assertRaises(asyncio.CancelledError):
                next(stream)

    def test_closing_generator_does_not_emit_error_or_continue_work(self):
        stream = self.rag.run_once_event_stream(**self.arguments)
        self.assertEqual(next(stream)["event"], "start")
        with self.assertNoLogs("phase1", level="ERROR"):
            stream.close()
        self.rag.retrieve.assert_not_called()
        self.assertEqual(list(stream), [])

    def test_fastapi_forwards_normal_and_failure_events_as_sse(self):
        async def fake_current_user():
            return UserResponse(
                user_id=self.arguments["user_id"],
                username="test-user",
                created_at=datetime.now(timezone.utc),
            )

        app = FastAPI()
        app.dependency_overrides[get_current_user] = fake_current_user
        app.include_router(router)
        app.state.rag_service = RAGService(self.rag)
        with TestClient(app) as client:
            response = client.post("/api/v1/chat/sse", json=self.arguments)
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            self.assertIn("event: sources", response.text)
            self.assertIn("event: done", response.text)

            self.rag.retrieve.side_effect = RuntimeError("private failure detail")
            with self.assertLogs("phase1", level="ERROR"):
                response = client.post("/api/v1/chat/sse", json=self.arguments)
            self.assertEqual(response.status_code, 200)
            self.assertIn("event: error", response.text)
            self.assertIn("The response could not be completed.", response.text)
            self.assertNotIn("private failure detail", response.text)
            self.assertNotIn("event: done", response.text)

    def test_gemini_client_configures_timeout_and_is_reused(self):
        with patch.object(phase1, "gemini_client", None):
            with patch.object(phase1.genai, "Client") as client_class:
                client = phase1.get_gemini_client()
                self.assertIs(phase1.get_gemini_client(), client)
                client_class.assert_called_once()
                self.assertEqual(client_class.call_args.kwargs["http_options"].timeout, 60_000)


if __name__ == "__main__":
    unittest.main()
