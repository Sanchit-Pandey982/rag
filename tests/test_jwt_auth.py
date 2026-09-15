"""JWT and auth HTTP tests: no MongoDB, model, or RAG execution required.

Run: .venv/Scripts/python.exe -m unittest discover -s tests -p test_jwt_auth.py -v
"""

from datetime import datetime, timezone
import os
from secrets import token_urlsafe
import unittest
from unittest.mock import patch
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
import jwt

from app.routes.auth import router
from app.security.jwt import AccessTokenError, JWTService
from app.security.passwords import hash_password
from app.services.auth_service import AuthService
from test_auth import InMemoryUsers


class JWTTests(unittest.TestCase):
    def setUp(self):
        self.secret = token_urlsafe(32)
        self.service = JWTService(self.secret)

    def test_token_contains_only_expected_claims_and_expires_in_15_minutes(self):
        before = int(datetime.now(timezone.utc).timestamp())
        token = self.service.create_access_token("server-user-id")
        claims = self.service.decode_access_token(token)
        self.assertEqual(set(claims), {"sub", "type", "iat", "exp", "iss", "aud", "jti"})
        self.assertEqual(jwt.get_unverified_header(token)["alg"], "HS256")
        self.assertEqual(claims["sub"], "server-user-id")
        self.assertEqual(claims["type"], "access")
        self.assertEqual(claims["iss"], self.service.issuer)
        self.assertEqual(claims["aud"], self.service.audience)
        self.assertGreaterEqual(claims["iat"], before)
        self.assertLessEqual(claims["iat"], int(datetime.now(timezone.utc).timestamp()))
        self.assertEqual(claims["exp"] - claims["iat"], 900)
        self.assertEqual(UUID(claims["jti"]).version, 4)
        next_claims = self.service.decode_access_token(self.service.create_access_token("server-user-id"))
        self.assertNotEqual(claims["jti"], next_claims["jti"])

    def test_configuration_defaults_and_overrides(self):
        with patch.dict(os.environ, {"JWT_SECRET_KEY": self.secret}, clear=True):
            service = JWTService.from_environment()
            self.assertEqual(service.issuer, "rag-learning-api")
            self.assertEqual(service.audience, "rag-learning-api-users")
            self.assertEqual(service.expires_in, 900)
        with patch.dict(os.environ, {
            "JWT_SECRET_KEY": self.secret,
            "JWT_ISSUER": "my-issuer",
            "JWT_AUDIENCE": "my-audience",
        }, clear=True):
            service = JWTService.from_environment()
            claims = service.decode_access_token(service.create_access_token("user-id"))
            self.assertEqual(claims["iss"], "my-issuer")
            self.assertEqual(claims["aud"], "my-audience")

    def test_creation_rejects_empty_or_non_string_identity(self):
        for user_id in ("", "   ", None, 123):
            with self.subTest(user_id=user_id), self.assertRaises(ValueError):
                self.service.create_access_token(user_id)

    def test_library_error_is_wrapped_in_application_exception(self):
        with self.assertRaises(AccessTokenError):
            self.service.decode_access_token("malformed")


class AccessTokenRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password = "  Exact Password  "
        cls.password_hash = hash_password(cls.password)

    def setUp(self):
        self.secret = token_urlsafe(32)
        self.jwt_service = JWTService(self.secret)
        self.users = InMemoryUsers()
        self.users.documents.append({
            "user_id": "server-generated-user-id",
            "username": "sanchit",
            "password_hash": self.password_hash,
            "disabled": False,
            "created_at": datetime.now(timezone.utc),
        })
        self.auth_service = AuthService(self.users)
        self.app = FastAPI()
        self.app.state.auth_service = self.auth_service
        self.app.state.jwt_service = self.jwt_service
        self.app.include_router(router)
        self.client = self.enterContext(TestClient(self.app))
        self.token = self.jwt_service.create_access_token(self.users.documents[0]["user_id"])

    def get_me(self, token):
        return self.client.get("/api/v1/auth/me", headers={"Authorization": "Bearer " + token})

    def assert_unauthorized(self, response):
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")
        self.assertEqual(response.json(), {"detail": "Invalid or missing access token."})
        self.assertNotIn(self.secret, response.text)
        self.assertNotIn(self.password_hash, response.text)

    def signed_claims(self, **changes):
        claims = self.jwt_service.decode_access_token(self.token)
        claims.update(changes)
        return jwt.encode(claims, self.secret, algorithm="HS256")

    def test_successful_login_then_me_uses_server_identity_and_exact_password(self):
        response = self.client.post("/api/v1/auth/login", json={
            "username": "  SANCHIT  ",
            "password": self.password,
            "user_id": "attacker-chosen-id",
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"access_token", "token_type", "expires_in"})
        self.assertEqual(body["token_type"], "bearer")
        self.assertEqual(body["expires_in"], 900)
        claims = self.jwt_service.decode_access_token(body["access_token"])
        self.assertEqual(claims["sub"], "server-generated-user-id")
        self.assertNotIn("password", claims)
        self.assertNotIn("password_hash", claims)
        me = self.get_me(body["access_token"])
        self.assertEqual(me.status_code, 200)
        self.assertEqual(set(me.json()), {"user_id", "username", "created_at"})
        self.assertEqual(me.json()["user_id"], claims["sub"])
        self.assertEqual(me.json()["username"], "sanchit")

    def test_wrong_password_and_unknown_username_have_identical_failures(self):
        for username, password in (
            ("sanchit", "wrong"),
            ("sanchit", self.password.strip()),
            ("unknown", self.password),
        ):
            with self.subTest(username=username, password=password):
                response = self.client.post("/api/v1/auth/login", json={
                    "username": username, "password": password,
                })
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json(), {"detail": "Invalid username or password."})
                self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_disabled_user_cannot_login(self):
        self.users.documents[0]["disabled"] = True
        response = self.client.post("/api/v1/auth/login", json={
            "username": "sanchit", "password": self.password,
        })
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Invalid username or password."})
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_login_validation_does_not_echo_credentials(self):
        for payload in (
            {"password": "private-password", "password_hash": "private-hash"},
            {"username": "sanchit", "password": {"secret": "private-password"}},
            {"username": "   ", "password": "private-password"},
        ):
            with self.subTest(payload=payload):
                response = self.client.post("/api/v1/auth/login", json=payload)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("private-password", response.text)
                self.assertNotIn("private-hash", response.text)

    def test_valid_token_resolves_database_user_on_every_request(self):
        response = self.get_me(self.token)
        self.assertEqual(response.status_code, 200)
        self.users.find_one.assert_awaited_once_with({"user_id": "server-generated-user-id"})
        self.users.documents[0]["username"] = "renamed"
        self.assertEqual(self.get_me(self.token).json()["username"], "renamed")
        self.assertEqual(self.users.find_one.await_count, 2)
        self.assertFalse(hasattr(self.app.state, "current_user"))

    def test_missing_and_malformed_bearer_headers(self):
        for headers in ({}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer"},
                        {"Authorization": "Bearer "}):
            with self.subTest(headers=headers):
                self.assert_unauthorized(self.client.get("/api/v1/auth/me", headers=headers))
        self.users.find_one.assert_not_awaited()

    def test_malformed_tokens(self):
        for token in ("not-a-jwt", "a.b.c", "too.many.jwt.parts", "null"):
            with self.subTest(token=token):
                self.assert_unauthorized(self.get_me(token))
        self.users.find_one.assert_not_awaited()

    def test_modified_signature(self):
        header, payload, signature = self.token.split(".")
        signature = ("A" if signature[0] != "A" else "B") + signature[1:]
        self.assert_unauthorized(self.get_me(".".join((header, payload, signature))))
        self.users.find_one.assert_not_awaited()

    def test_token_signed_by_another_key(self):
        token = JWTService(token_urlsafe(32)).create_access_token("server-generated-user-id")
        self.assert_unauthorized(self.get_me(token))

    def test_expired_token(self):
        now = int(datetime.now(timezone.utc).timestamp())
        self.assert_unauthorized(self.get_me(self.signed_claims(iat=now - 1000, exp=now - 1)))
        self.users.find_one.assert_not_awaited()

    def test_wrong_type_issuer_audience_subject_and_jti(self):
        for changes in (
            {"type": "refresh"}, {"iss": "other-issuer"}, {"aud": "other-audience"},
            {"sub": ""}, {"sub": "   "}, {"sub": 123}, {"jti": ""}, {"jti": 123},
        ):
            with self.subTest(changes=changes):
                self.assert_unauthorized(self.get_me(self.signed_claims(**changes)))
        self.users.find_one.assert_not_awaited()

    def test_every_expected_claim_is_required(self):
        original = self.jwt_service.decode_access_token(self.token)
        for name in ("sub", "type", "iat", "exp", "iss", "aud", "jti"):
            with self.subTest(claim=name):
                claims = dict(original)
                del claims[name]
                token = jwt.encode(claims, self.secret, algorithm="HS256")
                self.assert_unauthorized(self.get_me(token))
        self.users.find_one.assert_not_awaited()

    def test_invalid_dates_and_future_issued_at_are_rejected(self):
        for name in ("iat", "exp"):
            for value in (None, {}, [], "not-a-date", float("inf")):
                with self.subTest(claim=name, value=value):
                    self.assert_unauthorized(self.get_me(self.signed_claims(**{name: value})))
        future = int(datetime.now(timezone.utc).timestamp()) + 3600
        self.assert_unauthorized(self.get_me(self.signed_claims(iat=future, exp=future + 900)))
        self.users.find_one.assert_not_awaited()

    def test_other_algorithms_and_unsigned_tokens_are_rejected(self):
        claims = self.jwt_service.decode_access_token(self.token)
        for algorithm, key in (("HS384", token_urlsafe(64)), ("none", None)):
            with self.subTest(algorithm=algorithm):
                self.assert_unauthorized(self.get_me(jwt.encode(claims, key, algorithm=algorithm)))
        self.users.find_one.assert_not_awaited()

    def test_unknown_and_deleted_users(self):
        self.assert_unauthorized(self.get_me(self.signed_claims(sub="unknown-user")))
        self.users.documents.clear()
        self.assert_unauthorized(self.get_me(self.token))

    def test_disabling_user_after_token_issuance_returns_forbidden(self):
        self.users.documents[0]["disabled"] = True
        response = self.get_me(self.token)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"detail": "Account is disabled."})
        self.assertNotIn(self.password_hash, response.text)


if __name__ == "__main__":
    unittest.main()
