"""Phase 3.9 tests: Redis fixed-window rate limiting, no live services.

FakeRedis emulates the service Lua script's contract (INCR, expire-on-
first-write, PTTL) with a controllable millisecond clock -- it mirrors
fixed-window semantics, not Lua itself. Covers per-scope enforcement on
all five real routes (register/login/refresh/chat/upload), user/IP
isolation, window reset, Retry-After, limiter-before-credential-check
ordering on login, and fail-open on Redis outage.
"""

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_urlsafe
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from app.dependencies.auth import get_current_user
from app.models.user import User
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.routes.documents import router as documents_router
from app.schemas.auth import UserResponse
from app.security.cookies import RefreshCookieSettings
from app.security.jwt import JWTService
from app.services.document_service import DocumentLocks, DocumentService
from app.services.rate_limit_service import (
    SCOPE_CHAT,
    SCOPE_LOGIN,
    SCOPE_REFRESH,
    SCOPE_REGISTER,
    SCOPE_UPLOAD,
    RateLimitExceeded,
    RateLimitService,
    RateLimitUnavailable,
)


class FakeRedis:
    """Fixed-window counter with controllable time (milliseconds)."""

    def __init__(self):
        self.counts = {}
        self.deadlines = {}
        self.now_ms = 0
        self.eval_calls = []

    def _expired(self, key):
        return key in self.deadlines and self.now_ms >= self.deadlines[key]

    def _reap(self, key):
        if key not in self.counts or self._expired(key):
            self.counts[key] = 0
            self.deadlines.pop(key, None)

    async def eval(self, script, numkeys, *args):
        key, window_ms = args[0], args[1]
        self.eval_calls.append((script, key, window_ms))
        self._reap(key)
        self.counts[key] += 1
        if self.counts[key] == 1:
            self.deadlines[key] = self.now_ms + window_ms
        return [self.counts[key], self.deadlines[key] - self.now_ms]

    def advance(self, ms):
        self.now_ms += ms


class FailingRedis(FakeRedis):
    async def eval(self, script, numkeys, *args):
        raise RedisError("private redis detail")


class FakeCollection:
    def __init__(self):
        self.documents = []

    async def create_index(self, keys, **kwargs):
        return "index"

    async def insert_one(self, document):
        self.documents.append(dict(document))
        return SimpleNamespace(inserted_id=document.get("document_id"))

    async def find_one(self, filt):
        for document in self.documents:
            if all(document.get(k) == v for k, v in filt.items()):
                return dict(document)
        return None

    async def update_one(self, filt, update):
        for document in self.documents:
            if all(document.get(k) == v for k, v in filt.items()):
                document.update(update.get("$set", {}))
                return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)


def run(coro):
    return asyncio.run(coro)


def user_response(user_id):
    return UserResponse(
        user_id=user_id, username=user_id, created_at=datetime.now(timezone.utc)
    )


def service(redis, **limits):
    return RateLimitService(redis, limits=limits or None)


class RateLimitServiceTests(unittest.TestCase):
    def test_allows_up_to_limit_then_raises_with_retry_after(self):
        redis = FakeRedis()
        limiter = service(redis, **{SCOPE_LOGIN: (2, 60)})
        run(limiter.check(scope=SCOPE_LOGIN, key="ip:testclient"))
        run(limiter.check(scope=SCOPE_LOGIN, key="ip:testclient"))
        with self.assertRaises(RateLimitExceeded) as raised:
            run(limiter.check(scope=SCOPE_LOGIN, key="ip:testclient"))
        self.assertGreaterEqual(raised.exception.retry_after_seconds, 1)
        self.assertLessEqual(raised.exception.retry_after_seconds, 60)

    def test_window_reset_allows_again(self):
        redis = FakeRedis()
        limiter = service(redis, **{SCOPE_LOGIN: (1, 60)})
        run(limiter.check(scope=SCOPE_LOGIN, key="ip:x"))
        with self.assertRaises(RateLimitExceeded):
            run(limiter.check(scope=SCOPE_LOGIN, key="ip:x"))
        redis.advance(61_000)
        run(limiter.check(scope=SCOPE_LOGIN, key="ip:x"))

    def test_buckets_isolate_scopes_and_keys(self):
        redis = FakeRedis()
        limiter = service(redis, **{SCOPE_CHAT: (1, 60)})
        run(limiter.check(scope=SCOPE_CHAT, key="user:alice"))
        run(limiter.check(scope=SCOPE_CHAT, key="user:bob"))
        with self.assertRaises(RateLimitExceeded):
            run(limiter.check(scope=SCOPE_CHAT, key="user:alice"))
        self.assertEqual(
            limiter.bucket(SCOPE_CHAT, "user:alice"),
            "rate_limit:chat:user:alice",
        )
        keys = [call[1] for call in redis.eval_calls]
        self.assertIn("rate_limit:chat:user:alice", keys)
        self.assertIn("rate_limit:chat:user:bob", keys)
        self.assertTrue(all("auth:refresh:" not in k for k in keys))

    def test_redis_errors_become_unavailable_not_exceeded(self):
        limiter = service(FailingRedis())
        with self.assertRaises(RateLimitUnavailable):
            run(limiter.check(scope=SCOPE_CHAT, key="user:alice"))

    def test_unknown_scope_and_bad_config_fail_fast(self):
        limiter = service(FakeRedis())
        with self.assertRaises(ValueError):
            run(limiter.check(scope="nope", key="x"))
        with self.assertRaises(ValueError):
            service(FakeRedis(), **{"nope": (1, 60)})
        with self.assertRaises(ValueError):
            service(FakeRedis(), **{SCOPE_CHAT: (0, 60)})


class RateLimitRouteTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()

    def limiter(self, scope, limit=2, window=60):
        return RateLimitService(self.redis, limits={scope: (limit, window)})

    def retry_after(self, response):
        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json(),
            {"detail": "Too many requests. Please slow down and retry."},
        )
        value = int(response.headers["retry-after"])
        self.assertGreaterEqual(value, 1)
        return value

    # -- register ----------------------------------------------------

    def auth_app(self, scope, **route_states):
        app = FastAPI()
        app.include_router(auth_router)
        app.state.rate_limit_service = self.limiter(scope)
        for name, value in route_states.items():
            setattr(app.state, name, value)
        return TestClient(app)

    def test_register_is_limited_per_ip_before_user_creation(self):
        auth_service = Mock()
        auth_service.register_user = AsyncMock(
            side_effect=lambda username, password: User(
                username=username, password_hash="hashed"
            )
        )
        client = self.auth_app(SCOPE_REGISTER, auth_service=auth_service)
        for _ in range(2):
            response = client.post("/api/v1/auth/register", json={
                "username": "sanchit", "password": "long-enough-password",
            })
            self.assertEqual(response.status_code, 201)
        self.retry_after(client.post("/api/v1/auth/register", json={
            "username": "sanchit", "password": "long-enough-password",
        }))
        self.assertEqual(auth_service.register_user.await_count, 2)

    def test_register_window_reset_allows_again(self):
        auth_service = Mock()
        auth_service.register_user = AsyncMock(
            side_effect=lambda username, password: User(
                username=username, password_hash="hashed"
            )
        )
        client = self.auth_app(SCOPE_REGISTER, auth_service=auth_service)
        payload = {"username": "sanchit", "password": "long-enough-password"}
        client.post("/api/v1/auth/register", json=payload)
        client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(
            client.post("/api/v1/auth/register", json=payload).status_code, 429
        )
        self.redis.advance(61_000)
        # Register window is 60s in this test limiter (not the 300s default).
        self.assertEqual(
            client.post("/api/v1/auth/register", json=payload).status_code, 201
        )

    # -- login -------------------------------------------------------

    def test_login_is_limited_before_credential_check(self):
        auth_service = Mock()
        auth_service.authenticate_user = AsyncMock(return_value=None)
        client = self.auth_app(SCOPE_LOGIN, auth_service=auth_service)
        payload = {"username": "sanchit", "password": "wrong-password"}
        for _ in range(2):
            self.assertEqual(
                client.post("/api/v1/auth/login", json=payload).status_code, 401
            )
        self.retry_after(client.post("/api/v1/auth/login", json=payload))
        # The third attempt never reached password verification.
        self.assertEqual(auth_service.authenticate_user.await_count, 2)

    # -- refresh -----------------------------------------------------

    def test_refresh_without_cookie_counts_then_limits(self):
        client = self.auth_app(
            SCOPE_REFRESH,
            jwt_service=JWTService(token_urlsafe(32)),
            refresh_token_service=Mock(),
            refresh_cookie_settings=RefreshCookieSettings(secure=False),
            auth_service=Mock(),
        )
        for _ in range(2):
            self.assertEqual(client.post("/api/v1/auth/refresh").status_code, 401)
        self.retry_after(client.post("/api/v1/auth/refresh"))

    # -- chat --------------------------------------------------------

    def chat_client(self, user_id="alice"):
        app = FastAPI()
        app.include_router(chat_router)
        rag_service = Mock()
        rag_service.run_once.return_value = {
            "answer": "An answer", "retrieval_query": "q",
            "retrieved_document_ids": [], "chunks": [],
        }
        app.state.rag_service = rag_service
        app.state.rate_limit_service = self.limiter(SCOPE_CHAT)

        async def fake_current_user():
            return user_response(user_id)

        app.dependency_overrides[get_current_user] = fake_current_user
        return TestClient(app), rag_service

    def chat_payload(self, user_id="alice"):
        return {
            "raw_query": "What is RAG?", "user_id": user_id,
            "chat_history": [], "rewrite_query": False,
        }

    def test_chat_is_limited_per_user(self):
        client, rag_service = self.chat_client()
        for _ in range(2):
            self.assertEqual(
                client.post("/api/v1/chat", json=self.chat_payload()).status_code,
                200,
            )
        self.retry_after(
            client.post("/api/v1/chat", json=self.chat_payload())
        )
        self.assertEqual(rag_service.run_once.call_count, 2)

    def test_chat_limits_isolate_users(self):
        alice, _ = self.chat_client("alice")
        bob, _ = self.chat_client("bob")
        # Separate apps, shared Redis: each identity gets its own bucket.
        for _ in range(2):
            alice.post("/api/v1/chat", json=self.chat_payload("alice"))
        self.assertEqual(
            alice.post("/api/v1/chat", json=self.chat_payload("alice")).status_code,
            429,
        )
        self.assertEqual(
            bob.post("/api/v1/chat", json=self.chat_payload("bob")).status_code, 200
        )

    # -- upload ------------------------------------------------------

    def upload_client(self, user_id="alice"):
        app = FastAPI()
        app.include_router(documents_router)
        collection = FakeCollection()
        app.state.document_service = DocumentService(collection)
        run(app.state.document_service.ensure_indexes())
        rag_service = Mock()
        rag_service.ingest_user_document.return_value = 2
        app.state.rag_service = rag_service
        tmp = tempfile.TemporaryDirectory()
        app.state.upload_dir = Path(tmp.name)
        app.state.max_upload_bytes = 5 * 1024 * 1024
        app.state.document_locks = DocumentLocks()
        app.state.rate_limit_service = self.limiter(SCOPE_UPLOAD)

        async def fake_current_user():
            return user_response(user_id)

        app.dependency_overrides[get_current_user] = fake_current_user
        client = TestClient(app)
        client._tmp = tmp  # keep the directory alive for the test
        return client, collection

    def upload_file(self, client, filename="notes.txt"):
        return client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, b"Tenant-scoped upload content.", "text/plain")},
        )

    def test_upload_is_limited_per_user(self):
        client, collection = self.upload_client()
        for _ in range(2):
            self.assertEqual(self.upload_file(client).status_code, 202)
        self.retry_after(self.upload_file(client))
        self.assertEqual(len(collection.documents), 2)

    # -- fail-open ---------------------------------------------------

    def test_redis_outage_allows_requests(self):
        app = FastAPI()
        app.include_router(chat_router)
        rag_service = Mock()
        rag_service.run_once.return_value = {
            "answer": "An answer", "retrieval_query": "q",
            "retrieved_document_ids": [], "chunks": [],
        }
        app.state.rag_service = rag_service
        app.state.rate_limit_service = RateLimitService(FailingRedis())

        async def fake_current_user():
            return user_response("alice")

        app.dependency_overrides[get_current_user] = fake_current_user
        with self.assertLogs("app.dependencies.rate_limit", level="WARNING"):
            response = TestClient(app).post(
                "/api/v1/chat", json=self.chat_payload()
            )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
