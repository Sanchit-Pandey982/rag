"""Test authentication with real Argon2 and isolated MongoDB test doubles.

Run with: .venv/Scripts/python.exe -m unittest discover -s tests -v
No running MongoDB, Chroma, or AI service is required.
"""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import os
from secrets import token_urlsafe
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch
from uuid import UUID

from bson import BSON, ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError

from app.models.user import User
from app.routes.auth import router
from app.schemas.auth import UserResponse
from app.security.passwords import DUMMY_PASSWORD_HASH, hash_password, verify_password
from app.security.jwt import JWTService
from app.services.auth_service import AuthService, UsernameAlreadyExistsError
from redis_double import InMemoryRedis


class InMemoryUsers:
    """Small collection double; duplicate enforcement depends on created indexes."""

    def __init__(self):
        self.documents = []
        self.unique_indexes = set()
        self.find_one = AsyncMock(side_effect=self.find_document)
        self.insert_one = AsyncMock(side_effect=self.insert_document)
        self.create_index = AsyncMock(side_effect=self.add_index)

    async def add_index(self, field, *, unique=False):
        if unique:
            self.unique_indexes.add(field)
        return f"{field}_1"

    async def find_document(self, query):
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                return deepcopy(document)
        return None

    async def insert_document(self, document):
        # Let competing registrations reach insertion before checking uniqueness.
        await asyncio.sleep(0)
        for field in self.unique_indexes:
            # The lifespan test shares one double across collections, so an
            # index from another collection may be absent here. MongoDB does
            # not treat a missing field as equal to a present value.
            if field not in document:
                continue
            if any(field in saved and saved[field] == document[field] for saved in self.documents):
                raise DuplicateKeyError(
                    "duplicate key", 11000, {"keyPattern": {field: 1}}
                )
        document["_id"] = ObjectId()
        self.documents.append(deepcopy(document))


class PasswordTests(unittest.TestCase):
    def test_password_hash_is_salted_argon2_and_verifies_exact_input(self):
        password = "  Exact Password  "
        password_hash = hash_password(password)
        self.assertNotEqual(password_hash, password)
        self.assertTrue(password_hash.startswith("$argon2id$"))
        self.assertNotEqual(hash_password(password), password_hash)
        self.assertTrue(verify_password(password, password_hash))
        for incorrect in ("wrong", password.strip(), password.lower()):
            with self.subTest(password=incorrect):
                self.assertFalse(verify_password(incorrect, password_hash))

    def test_unsupported_hash_fails_closed(self):
        self.assertFalse(verify_password("password", "not-a-supported-hash"))


class AuthServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.users = InMemoryUsers()
        self.service = AuthService(self.users)
        await self.service.ensure_indexes()

    async def test_both_identity_indexes_are_unique(self):
        self.users.create_index.assert_any_await("username", unique=True)
        self.users.create_index.assert_any_await("user_id", unique=True)
        self.assertEqual(self.users.unique_indexes, {"username", "user_id"})

    async def test_registration_persists_user_and_bson_datetime(self):
        user = await self.service.register_user("  Sanchit  ", "  Exact Password  ")
        document = self.users.documents[0]
        self.assertEqual(user.username, "sanchit")
        self.assertEqual(UUID(user.user_id).version, 4)
        self.assertEqual(document["user_id"], user.user_id)
        self.assertNotEqual(str(document["_id"]), user.user_id)
        self.assertFalse(document["disabled"])
        self.assertNotIn("password", document)
        self.assertNotEqual(document["password_hash"], "  Exact Password  ")
        self.assertTrue(verify_password("  Exact Password  ", document["password_hash"]))
        self.assertIsInstance(document["created_at"], datetime)
        self.assertEqual(document["created_at"].tzinfo, timezone.utc)
        self.assertEqual(BSON(BSON.encode(document)).decode()["user_id"], user.user_id)
        self.users.find_one.assert_not_awaited()

    async def test_registered_users_get_different_server_ids(self):
        first = await self.service.register_user("first", "password")
        second = await self.service.register_user("second", "password")
        self.assertNotEqual(first.user_id, second.user_id)

    async def test_normalized_duplicate_rejected_by_insert_without_precheck(self):
        await self.service.register_user("Sanchit", "password")
        with self.assertRaises(UsernameAlreadyExistsError):
            await self.service.register_user("  SANCHIT  ", "another password")
        self.assertEqual(self.users.insert_one.await_count, 2)
        self.users.find_one.assert_not_awaited()
        self.assertEqual(len(self.users.documents), 1)

    async def test_concurrent_registration_has_one_winner(self):
        results = await asyncio.gather(
            self.service.register_user("Sanchit", "first password"),
            self.service.register_user("  SANCHIT  ", "second password"),
            return_exceptions=True,
        )
        self.assertEqual(sum(isinstance(result, User) for result in results), 1)
        self.assertEqual(
            sum(isinstance(result, UsernameAlreadyExistsError) for result in results), 1
        )
        self.assertEqual(len(self.users.documents), 1)
        self.users.find_one.assert_not_awaited()

    async def test_user_id_collision_is_not_reported_as_username_conflict(self):
        collision = DuplicateKeyError("duplicate key", 11000, {"keyPattern": {"user_id": 1}})
        self.users.insert_one.side_effect = collision
        with self.assertRaises(DuplicateKeyError):
            await self.service.register_user("new-user", "password")

    async def test_duplicate_without_key_pattern_checks_username_after_insert(self):
        await self.service.register_user("sanchit", "password")
        self.users.insert_one.side_effect = DuplicateKeyError("duplicate key", 11000)
        with self.assertRaises(UsernameAlreadyExistsError):
            await self.service.register_user("SANCHIT", "password")
        self.users.find_one.assert_awaited_once_with({"username": "sanchit"})

    async def test_unidentified_duplicate_is_not_mislabeled(self):
        self.users.insert_one.side_effect = DuplicateKeyError("duplicate key", 11000)
        with self.assertRaises(DuplicateKeyError):
            await self.service.register_user("new-user", "password")

    async def test_active_user_authenticates_with_stable_identity(self):
        registered = await self.service.register_user("Sanchit", "  Exact Password  ")
        # A new service instance retrieves the persisted identity, not a new UUID.
        service = AuthService(self.users)
        authenticated = await service.authenticate_user("  SANCHIT  ", "  Exact Password  ")
        self.assertIsInstance(authenticated, User)
        self.assertEqual(authenticated.user_id, registered.user_id)
        self.users.find_one.assert_awaited_once_with({"username": "sanchit"})

    async def test_wrong_password_returns_failure(self):
        await self.service.register_user("sanchit", "  Exact Password  ")
        for password in ("wrong", "Exact Password", "  exact password  "):
            with self.subTest(password=password):
                self.assertIsNone(await self.service.authenticate_user("sanchit", password))

    async def test_get_user_by_id_uses_persisted_identity(self):
        registered = await self.service.register_user("sanchit", "password")
        user = await self.service.get_user_by_id(registered.user_id)
        self.assertEqual(user.user_id, registered.user_id)
        self.users.find_one.assert_awaited_once_with({"user_id": registered.user_id})

    async def test_get_unknown_user_by_id_returns_none(self):
        self.assertIsNone(await self.service.get_user_by_id("deleted-user"))

    async def test_missing_user_verifies_precreated_dummy_hash(self):
        with patch("app.services.auth_service.verify_password", wraps=verify_password) as verify:
            self.assertIsNone(await self.service.authenticate_user("  Missing  ", "password"))
        verify.assert_called_once_with("password", DUMMY_PASSWORD_HASH)
        self.users.find_one.assert_awaited_once_with({"username": "missing"})

    async def test_dummy_verification_can_never_authenticate_missing_user(self):
        with patch("app.services.auth_service.verify_password", return_value=True):
            self.assertIsNone(await self.service.authenticate_user("missing", "password"))

    async def test_disabled_user_rejected_after_password_verification(self):
        user = await self.service.register_user("sanchit", "password")
        self.users.documents[0]["disabled"] = True
        with patch("app.services.auth_service.verify_password", wraps=verify_password) as verify:
            self.assertIsNone(await self.service.authenticate_user("sanchit", "password"))
        verify.assert_called_once_with("password", user.password_hash)
        self.assertEqual(len(self.users.documents), 1)
        self.assertEqual(self.users.documents[0]["user_id"], user.user_id)


class RegistrationRouteTests(unittest.TestCase):
    def setUp(self):
        self.users = InMemoryUsers()
        service = AuthService(self.users)
        asyncio.run(service.ensure_indexes())
        app = FastAPI()
        app.state.auth_service = service
        app.include_router(router)
        self.client = self.enterContext(TestClient(app))

    def test_created_response_has_only_public_fields(self):
        response = self.client.post("/api/v1/auth/register", json={
            "username": "  Sanchit  ",
            "password": "  Exact Password  ",
            "user_id": "client-chosen-id",
            "disabled": True,
        })
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(set(body), {"user_id", "username", "created_at"})
        self.assertEqual(body["username"], "sanchit")
        self.assertEqual(UUID(body["user_id"]).version, 4)
        self.assertNotEqual(body["user_id"], "client-chosen-id")
        self.assertFalse(self.users.documents[0]["disabled"])
        self.assertTrue(verify_password(
            "  Exact Password  ", self.users.documents[0]["password_hash"]
        ))
        self.assertNotIn("password", UserResponse.model_fields)
        self.assertNotIn("password_hash", UserResponse.model_fields)

    def test_duplicate_username_returns_conflict(self):
        self.client.post("/api/v1/auth/register", json={
            "username": "sanchit", "password": "password",
        })
        response = self.client.post("/api/v1/auth/register", json={
            "username": "  SANCHIT  ", "password": "password",
        })
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json(), {"detail": "Username already exists"})

    def test_invalid_requests_do_not_insert_users(self):
        for payload in (
            {"username": "   ", "password": "password"},
            {"username": "sanchit", "password": ""},
            {"username": 123, "password": "password"},
            {"username": "sanchit"},
        ):
            with self.subTest(payload=payload):
                response = self.client.post("/api/v1/auth/register", json=payload)
                self.assertEqual(response.status_code, 422)
        self.users.insert_one.assert_not_awaited()

    def test_registration_rejects_weak_or_absurd_passwords(self):
        # NIST floor (8) and Argon2 work bound (256); neither inserts.
        for password in ("short7!", "x" * 257):
            with self.subTest(password=password[:8] + "..."):
                response = self.client.post("/api/v1/auth/register", json={
                    "username": "sanchit", "password": password,
                })
                self.assertEqual(response.status_code, 422)
        self.users.insert_one.assert_not_awaited()

    def test_login_never_reports_policy_only_bad_credentials(self):
        # A 5-char guess fails closed with the uniform 401, not a 422 that
        # would turn login into a password-policy oracle.
        response = self.client.post("/api/v1/auth/login", json={
            "username": "sanchit", "password": "wrong",
        })
        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json(), {"detail": "Invalid username or password."}
        )

    def test_validation_errors_never_echo_submitted_credentials(self):
        for payload in (
            {"password": "private-password", "password_hash": "private-hash"},
            {"username": "sanchit", "password": {"secret": "private-password"}},
        ):
            with self.subTest(payload=payload):
                response = self.client.post("/api/v1/auth/register", json=payload)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("private-password", response.text)
                self.assertNotIn("private-hash", response.text)
                for error in response.json()["detail"]:
                    self.assertEqual(set(error), {"type", "loc", "msg"})
        self.users.insert_one.assert_not_awaited()


class AuthLifespanTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from app import main

        self.main = main
        self.redis = InMemoryRedis()
        self.redis_factory = self.enterContext(patch.object(main.redis, "from_url", return_value=self.redis))
        self.users = InMemoryUsers()
        self.mongo = MagicMock()
        self.mongo.admin.command = AsyncMock()
        self.mongo.close = AsyncMock()
        self.mongo.__getitem__.return_value.__getitem__.return_value = self.users
        self.mongo_class = self.enterContext(patch.object(main, "AsyncMongoClient", return_value=self.mongo))
        self.rag = Mock()
        self.rag.collection.count.return_value = 1
        self.rag_class = self.enterContext(patch.object(main, "RAGSystem", return_value=self.rag))
        self.enterContext(patch.object(main, "get_chroma_path", return_value="unused-test-path"))
        self.documents = self.enterContext(patch.object(main, "load_txt_documents", return_value=["document"]))
        self.enterContext(patch.dict("os.environ", {
            "MONGODB_URI": "mongodb://test-host:27017",
            "MONGODB_DATABASE": "test_auth",
            "JWT_SECRET_KEY": token_urlsafe(32),
            "JWT_ISSUER": "test-issuer",
            "JWT_AUDIENCE": "test-audience",
            "REDIS_URL": "redis://test-redis:6379/0",
            "REFRESH_COOKIE_SECURE": "false",
        }))
        self.app = FastAPI()

    def test_application_register_route_reuses_lifespan_service(self):
        with TestClient(self.main.app) as client:
            for username in ("first", "second"):
                response = client.post("/api/v1/auth/register", json={
                    "username": username, "password": "password",
                })
                self.assertEqual(response.status_code, 201)
            self.mongo_class.assert_called_once()
            self.assertEqual(len(self.users.documents), 2)
        self.mongo.close.assert_awaited_once()

    async def test_startup_initializes_services_and_shutdown_closes_clients(self):
        async with self.main.lifespan(self.app):
            self.redis_factory.assert_called_once()
            self.assertEqual(self.redis_factory.call_args.args, ("redis://test-redis:6379/0",))
            self.assertTrue(self.redis_factory.call_args.kwargs["decode_responses"])
            self.assertEqual(self.redis_factory.call_args.kwargs["retry"].get_retries(), 0)
            self.redis.ping.assert_awaited_once()
            self.assertIs(self.app.state.redis_client, self.redis)
            self.assertIs(self.app.state.refresh_token_service.redis_client, self.redis)
            self.redis.aclose.assert_not_awaited()
            self.mongo_class.assert_called_once_with(
                "mongodb://test-host:27017", serverSelectionTimeoutMS=5_000, tz_aware=True
            )
            self.mongo.admin.command.assert_awaited_once_with("ping")
            self.mongo.__getitem__.assert_called_once_with("test_auth")
            self.assertEqual(
                self.mongo.__getitem__.return_value.__getitem__.call_args_list,
                [call("users"), call("conversations"),
                 call("messages"), call("documents")],
            )
            # The lifespan shares one double across collections, so other
            # services' unique fields accumulate here; auth only needs its
            # own identity indexes present.
            self.assertLessEqual({"username", "user_id"}, self.users.unique_indexes)
            self.assertIs(self.app.state.auth_service.users, self.users)
            self.assertIsInstance(self.app.state.jwt_service, JWTService)
            self.assertEqual(self.app.state.jwt_service.issuer, "test-issuer")
            self.assertEqual(self.app.state.jwt_service.audience, "test-audience")
            self.assertIs(self.app.state.rag_service.rag, self.rag)
            self.assertTrue(self.app.state.ready)
            self.mongo.close.assert_not_awaited()
        self.mongo.close.assert_awaited_once()
        self.rag.client.close.assert_called_once()
        self.redis.aclose.assert_awaited_once()
        self.assertFalse(self.app.state.ready)

    async def test_existing_empty_corpus_is_still_ingested(self):
        self.rag.collection.count.side_effect = [0, 1]
        async with self.main.lifespan(self.app):
            self.documents.assert_called_once()
            self.rag.ingest_documents.assert_called_once_with(["document"], user_id="eval_user")
            self.assertTrue(self.app.state.ready)

    async def test_missing_jwt_secret_fails_before_external_resources_open(self):
        del os.environ["JWT_SECRET_KEY"]
        with self.assertRaisesRegex(ValueError, "JWT_SECRET_KEY"):
            async with self.main.lifespan(self.app):
                self.fail("Startup must fail")
        self.mongo_class.assert_not_called()
        self.rag_class.assert_not_called()
        self.assertFalse(self.app.state.ready)

    async def test_invalid_jwt_configuration_fails_startup_without_exposing_value(self):
        for name, value in (
            ("JWT_SECRET_KEY", "private-short-key"),
            ("JWT_SECRET_KEY", " " * 40),
            ("JWT_SECRET_KEY", "ssh-rsa " + "A" * 64),
            ("JWT_ISSUER", ""),
            ("JWT_AUDIENCE", "   "),
        ):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                with self.assertRaisesRegex(ValueError, name) as error:
                    async with self.main.lifespan(self.app):
                        self.fail("Startup must fail")
                if value.strip():
                    self.assertNotIn(value, str(error.exception))
        self.mongo_class.assert_not_called()
        self.rag_class.assert_not_called()

    def test_application_login_and_me_use_shared_services(self):
        with TestClient(self.main.app) as client:
            registered = client.post("/api/v1/auth/register", json={
                "username": "sanchit", "password": "password",
            }).json()
            login = client.post("/api/v1/auth/login", json={
                "username": "SANCHIT", "password": "password",
            })
            self.assertEqual(login.status_code, 200)
            response = client.get("/api/v1/auth/me", headers={
                "Authorization": "Bearer " + login.json()["access_token"],
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), registered)
            self.mongo_class.assert_called_once()
        self.mongo.close.assert_awaited_once()

    async def test_ping_failure_closes_mongo_and_prevents_serving(self):
        self.mongo.admin.command.side_effect = RuntimeError("ping failed")
        with self.assertRaisesRegex(RuntimeError, "ping failed"):
            async with self.main.lifespan(self.app):
                self.fail("Startup must fail")
        self.mongo.close.assert_awaited_once()
        self.rag_class.assert_not_called()

    async def test_index_failure_closes_mongo_and_prevents_serving(self):
        self.users.create_index.side_effect = RuntimeError("index failed")
        with self.assertRaisesRegex(RuntimeError, "index failed"):
            async with self.main.lifespan(self.app):
                self.fail("Startup must fail")
        self.mongo.close.assert_awaited_once()
        self.rag_class.assert_not_called()

    async def test_rag_startup_failure_closes_both_clients(self):
        self.rag.collection.count.side_effect = RuntimeError("rag failed")
        with self.assertRaisesRegex(RuntimeError, "rag failed"):
            async with self.main.lifespan(self.app):
                self.fail("Startup must fail")
        self.mongo.close.assert_awaited_once()
        self.rag.client.close.assert_called_once()

    async def test_rag_cleanup_failure_still_closes_mongo(self):
        self.rag.client.close.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            async with self.main.lifespan(self.app):
                pass
        self.mongo.close.assert_awaited_once()
        self.redis.aclose.assert_awaited_once()

    async def test_redis_ping_failure_closes_clients_and_prevents_serving(self):
        self.redis.ping.side_effect = RuntimeError("redis unavailable")
        with self.assertRaisesRegex(RuntimeError, "redis unavailable"):
            async with self.main.lifespan(self.app):
                self.fail("Startup must fail")
        self.redis.aclose.assert_awaited_once()
        self.mongo.close.assert_awaited_once()
        self.rag_class.assert_not_called()
        self.assertFalse(self.app.state.ready)

    async def test_redis_close_failure_still_closes_mongo(self):
        self.redis.aclose.side_effect = RuntimeError("redis close failed")
        with self.assertRaisesRegex(RuntimeError, "redis close failed"):
            async with self.main.lifespan(self.app):
                pass
        self.mongo.close.assert_awaited_once()
        self.rag.client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
