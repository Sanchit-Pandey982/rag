"""Test the HTTP tenant boundary without MongoDB, Gemini, or a Chroma server."""

from datetime import datetime, timezone
from secrets import token_urlsafe
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.dependencies.auth import get_current_user
from app.dependencies.chat import authorize_chat_request
from app.models.user import User
from app.routes.chat import router
from app.schemas.auth import UserResponse
from app.schemas.chat import ChatRequest
from app.security.jwt import JWTService
from app.services.rag_services import RAGService
from phase1 import RAGSystem


CHAT_ROUTES = {
    "/api/v1/chat": "run_once",
    "/api/v1/chat/stream": "run_once_stream",
    "/api/v1/chat/sse": "run_once_event_stream",
}


class ChatAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.current_user = UserResponse(
            user_id="alice", username="alice", created_at=datetime.now(timezone.utc),
        )
        self.payload = {
            "raw_query": "What is RAG?",
            "user_id": "alice",
            "chat_history": [{"role": "user", "content": "Explain retrieval."}],
            "k": 4,
            "rewrite_query": False,
            "distance_threshold": 0.8,
            # Phase 3.5 additive field; None keeps the legacy history behavior.
            "conversation_id": None,
        }
        self.rag_service = Mock(spec=RAGService)
        self.rag_service.run_once.return_value = {
            "answer": "An answer", "retrieval_query": "What is RAG?",
            "retrieved_document_ids": [], "chunks": [],
        }
        self.rag_service.run_once_stream.side_effect = lambda payload, **kwargs: iter(["An answer"])
        self.rag_service.run_once_event_stream.side_effect = lambda payload, **kwargs: iter([
            {"event": "start", "data": {"raw_query": payload.raw_query}},
            {"event": "done", "data": {}},
        ])
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.state.rag_service = self.rag_service
        self.client = self.enterContext(TestClient(self.app))

    def authenticate_as_alice(self):
        async def fake_current_user():
            return self.current_user

        self.app.dependency_overrides[get_current_user] = fake_current_user

    def assert_blocked(self, response, status_code):
        self.assertEqual(response.status_code, status_code)
        self.assertIn("application/json", response.headers["content-type"])
        self.assertNotIn("event:", response.text)
        self.assertEqual(self.rag_service.mock_calls, [])

    def test_missing_authentication_on_every_route(self):
        for path in CHAT_ROUTES:
            with self.subTest(path=path):
                response = self.client.post(path, json=self.payload)
                self.assert_blocked(response, 401)
                self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_invalid_token_on_every_route(self):
        self.app.state.jwt_service = JWTService(token_urlsafe(32))
        for path in CHAT_ROUTES:
            with self.subTest(path=path):
                response = self.client.post(path, json=self.payload, headers={
                    "Authorization": "Bearer invalid-token",
                })
                self.assert_blocked(response, 401)

    def test_matching_tenant_reaches_only_the_expected_service_method(self):
        self.authenticate_as_alice()
        for path, method_name in CHAT_ROUTES.items():
            with self.subTest(path=path):
                self.rag_service.reset_mock()
                response = self.client.post(path, json=self.payload)
                self.assertEqual(response.status_code, 200)
                method = getattr(self.rag_service, method_name)
                method.assert_called_once()
                trusted_request = method.call_args.args[0]
                self.assertIsInstance(trusted_request, ChatRequest)
                self.assertEqual(trusted_request.model_dump(), self.payload)
                self.assertIs(trusted_request.user_id, self.current_user.user_id)
                self.assertEqual(len(self.rag_service.mock_calls), 1)
                if path.endswith("/sse"):
                    self.assertIn("text/event-stream", response.headers["content-type"])
                    self.assertIn("event: done", response.text)

    def test_cross_tenant_requests_are_forbidden_before_rag_on_every_route(self):
        self.authenticate_as_alice()
        for path in CHAT_ROUTES:
            with self.subTest(path=path):
                response = self.client.post(path, json={**self.payload, "user_id": "bob"})
                self.assert_blocked(response, 403)

    def test_disabled_user_is_forbidden_before_rag_on_every_route(self):
        # Exercise the existing disabled-user check with a database-free lookup.
        jwt_service = JWTService(token_urlsafe(32))
        self.app.state.jwt_service = jwt_service
        self.app.state.auth_service = Mock(get_user_by_id=AsyncMock(return_value=User(
            user_id="alice", username="alice", password_hash="unused", disabled=True,
        )))
        token = jwt_service.create_access_token("alice")
        for path in CHAT_ROUTES:
            with self.subTest(path=path):
                response = self.client.post(path, json=self.payload, headers={
                    "Authorization": f"Bearer {token}",
                })
                self.assert_blocked(response, 403)
                self.assertEqual(response.json(), {"detail": "Account is disabled."})

    def test_helper_copies_the_request_without_mutating_it(self):
        payload = ChatRequest(**self.payload)
        original = payload.model_dump()
        trusted_request = authorize_chat_request(payload, self.current_user)
        self.assertIsNot(trusted_request, payload)
        self.assertEqual(payload.model_dump(), original)
        self.assertEqual(trusted_request.model_dump(), original)
        self.assertIs(trusted_request.user_id, self.current_user.user_id)

    def test_helper_rejects_mismatch_instead_of_silently_replacing_it(self):
        payload = ChatRequest(**{**self.payload, "user_id": "bob"})
        with self.assertRaises(HTTPException) as raised:
            authorize_chat_request(payload, self.current_user)
        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(payload.user_id, "bob")

    def test_invalid_schema_still_returns_422_before_rag(self):
        self.authenticate_as_alice()
        for path in CHAT_ROUTES:
            with self.subTest(path=path):
                response = self.client.post(path, json={**self.payload, "k": 0})
                self.assert_blocked(response, 422)

    def test_authorized_identity_reaches_chroma_filter_on_every_route(self):
        self.authenticate_as_alice()
        # Keep real service/pipeline/retrieval code, replacing only external I/O.
        rag = RAGSystem.__new__(RAGSystem)
        rag.collection = Mock()
        rag.collection.query.return_value = {
            "documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]],
        }
        rag.generate_answer = Mock(return_value="An answer")
        rag.generate_answer_stream = Mock(side_effect=lambda **kwargs: iter(["An answer"]))
        service = RAGService(rag)
        self.app.state.rag_service = service
        with patch("phase1.embed_query", return_value=[0.1, 0.2]):
            for path, method_name in CHAT_ROUTES.items():
                with self.subTest(path=path):
                    rag.collection.reset_mock()
                    with patch.object(service, method_name, wraps=getattr(service, method_name)) as method:
                        response = self.client.post(path, json=self.payload)
                        self.assertEqual(response.status_code, 200)
                        method.assert_called_once()
                        rag.collection.query.assert_called_once()
                        self.assertEqual(rag.collection.query.call_args.kwargs["where"], {
                            "user_id": self.current_user.user_id,
                        })

                        method.reset_mock()
                        rag.collection.reset_mock()
                        response = self.client.post(path, json={**self.payload, "user_id": "bob"})
                        self.assertEqual(response.status_code, 403)
                        method.assert_not_called()
                        rag.collection.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
