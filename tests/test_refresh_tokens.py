"""Real JWTs and HTTP routes with isolated Redis and user lookups."""

import asyncio
from http.cookies import SimpleCookie
import os
from secrets import token_urlsafe
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import jwt
from redis.exceptions import ConnectionError as RedisConnectionError

from app.models.user import User
from app.routes.auth import router
from app.security.cookies import REFRESH_COOKIE_NAME, RefreshCookieSettings
from app.security.jwt import AccessTokenError, JWTService, RefreshTokenError
from app.services.refresh_token_service import RefreshSessionUnavailable, RefreshTokenService
from redis_double import InMemoryRedis


class RefreshJWTTests(unittest.TestCase):
    def setUp(self):
        self.secret = token_urlsafe(32)
        self.jwt_service = JWTService(self.secret)

    def test_refresh_claims_and_lifetime(self):
        token, jti = self.jwt_service.create_refresh_token("alice")
        claims = self.jwt_service.decode_refresh_token(token)
        self.assertEqual(set(claims), {"sub", "type", "iat", "exp", "iss", "aud", "jti"})
        self.assertEqual(claims["sub"], "alice")
        self.assertEqual(claims["jti"], jti)
        self.assertEqual(claims["type"], "refresh")
        self.assertEqual(claims["exp"] - claims["iat"], 604800)
        self.assertNotEqual(jti, self.jwt_service.create_refresh_token("alice")[1])

    def test_access_and_refresh_decoders_reject_each_others_tokens(self):
        access = self.jwt_service.create_access_token("alice")
        refresh, _ = self.jwt_service.create_refresh_token("alice")
        with self.assertRaises(AccessTokenError):
            self.jwt_service.decode_access_token(refresh)
        with self.assertRaises(RefreshTokenError):
            self.jwt_service.decode_refresh_token(access)

    def test_refresh_decoder_requires_and_validates_all_claims(self):
        token, _ = self.jwt_service.create_refresh_token("alice")
        original = self.jwt_service.decode_refresh_token(token)
        for name in original:
            with self.subTest(missing=name):
                claims = dict(original)
                del claims[name]
                invalid = jwt.encode(claims, self.secret, algorithm="HS256")
                with self.assertRaises(RefreshTokenError):
                    self.jwt_service.decode_refresh_token(invalid)
        for changes in (
            {"iss": "wrong"}, {"aud": "wrong"}, {"sub": ""}, {"jti": " "},
            {"type": "access"}, {"exp": 0}, {"iat": int(time.time()) + 600},
            {"exp": None}, {"iat": {}}, {"exp": float("inf")}, {"exp": True},
        ):
            with self.subTest(changes=changes):
                invalid = jwt.encode({**original, **changes}, self.secret, algorithm="HS256")
                with self.assertRaises(RefreshTokenError):
                    self.jwt_service.decode_refresh_token(invalid)

    def test_wrong_key_algorithm_and_malformed_tokens(self):
        token, _ = self.jwt_service.create_refresh_token("alice")
        claims = self.jwt_service.decode_refresh_token(token)
        for invalid in (
            "invalid-token",
            jwt.encode(claims, token_urlsafe(32), algorithm="HS256"),
            jwt.encode(claims, token_urlsafe(64), algorithm="HS384"),
            jwt.encode(claims, None, algorithm="none"),
        ):
            with self.subTest(token=invalid), self.assertRaises(RefreshTokenError):
                self.jwt_service.decode_refresh_token(invalid)


class RefreshServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = InMemoryRedis()
        self.service = RefreshTokenService(self.redis)

    async def test_store_uses_jwt_expiry_and_expired_sessions_cannot_be_consumed(self):
        expires_at = self.redis.now + 30
        await self.service.store_refresh_token("jti", "alice", expires_at=expires_at)
        self.redis.set.assert_awaited_once_with("auth:refresh:jti", "alice", exat=expires_at, nx=True)
        self.redis.now = expires_at
        self.assertIsNone(await self.service.consume_refresh_token("jti"))
        self.assertEqual(self.redis.entries, {})

    async def test_concurrent_consumption_has_one_winner_and_uses_getdel(self):
        await self.service.store_refresh_token("jti", "alice", expires_at=self.redis.now + 30)
        results = await asyncio.gather(*[self.service.consume_refresh_token("jti") for _ in range(2)])
        self.assertCountEqual(results, ["alice", None])
        self.assertEqual(self.redis.getdel.await_count, 2)
        self.redis.getdel.assert_awaited_with("auth:refresh:jti")
        self.redis.delete.assert_not_awaited()

    async def test_identifier_collision_does_not_overwrite_a_session(self):
        await self.service.store_refresh_token("jti", "alice", expires_at=self.redis.now + 30)
        with self.assertRaises(RefreshSessionUnavailable):
            await self.service.store_refresh_token("jti", "bob", expires_at=self.redis.now + 60)
        self.assertEqual(await self.service.consume_refresh_token("jti"), "alice")


class RefreshRouteTests(unittest.TestCase):
    def setUp(self):
        self.secret = token_urlsafe(32)
        self.jwt_service = JWTService(self.secret)
        self.redis = InMemoryRedis()
        self.user = User(user_id="alice", username="alice", password_hash="unused")
        self.auth_service = Mock(
            authenticate_user=AsyncMock(return_value=self.user),
            get_user_by_id=AsyncMock(return_value=self.user),
        )
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.state.jwt_service = self.jwt_service
        self.app.state.auth_service = self.auth_service
        self.app.state.refresh_token_service = RefreshTokenService(self.redis)
        self.app.state.refresh_cookie_settings = RefreshCookieSettings()
        self.client = self.enterContext(TestClient(self.app, base_url="https://testserver"))

    def login(self):
        response = self.client.post("/api/v1/auth/login", json={
            "username": "alice", "password": "password",
        })
        self.assertEqual(response.status_code, 200)
        return response

    def post_with_token(self, path, token):
        # Explicit cookie avoids any implicit jar updates in replay/concurrency tests.
        return self.client.post(path, headers={"Cookie": f"{REFRESH_COOKIE_NAME}={token}"})

    def assert_cookie_cleared(self, response):
        cookie = SimpleCookie(response.headers["set-cookie"])[REFRESH_COOKIE_NAME]
        self.assertEqual(cookie["max-age"], "0")
        self.assertEqual(cookie["path"], "/api/v1/auth")
        self.assertTrue(cookie["httponly"])
        self.assertTrue(cookie["secure"])

    def test_login_returns_only_access_token_and_sets_secure_cookie_and_redis_expiry(self):
        response = self.login()
        self.assertEqual(set(response.json()), {"access_token", "token_type", "expires_in"})
        self.assertEqual(response.json()["expires_in"], 900)
        self.assertEqual(response.headers["cache-control"], "no-store")
        cookie = SimpleCookie(response.headers["set-cookie"])[REFRESH_COOKIE_NAME]
        self.assertTrue(cookie["httponly"])
        self.assertTrue(cookie["secure"])
        self.assertEqual(cookie["samesite"], "strict")
        self.assertEqual(cookie["path"], "/api/v1/auth")
        self.assertEqual(cookie["domain"], "")
        self.assertGreater(int(cookie["max-age"]), 604790)
        self.assertLessEqual(int(cookie["max-age"]), 604800)
        claims = self.jwt_service.decode_refresh_token(cookie.value)
        self.assertEqual(self.redis.entries, {f'auth:refresh:{claims["jti"]}': ("alice", claims["exp"])})
        self.assertNotIn(cookie.value, response.text)

    def test_successful_refresh_rotates_both_tokens_and_rejects_old_refresh_token(self):
        access_before = self.login().json()["access_token"]
        old_token = self.client.cookies[REFRESH_COOKIE_NAME]
        old_claims = self.jwt_service.decode_refresh_token(old_token)
        refreshed = self.client.post("/api/v1/auth/refresh")
        self.assertEqual(refreshed.status_code, 200)
        new_token = self.client.cookies[REFRESH_COOKIE_NAME]
        new_claims = self.jwt_service.decode_refresh_token(new_token)
        self.assertNotEqual(old_token, new_token)
        self.assertNotEqual(old_claims["jti"], new_claims["jti"])
        self.assertNotEqual(access_before, refreshed.json()["access_token"])
        self.assertEqual(self.redis.entries, {f'auth:refresh:{new_claims["jti"]}': ("alice", new_claims["exp"])})
        replay = self.post_with_token("/api/v1/auth/refresh", old_token)
        self.assertEqual(replay.status_code, 401)
        self.assert_cookie_cleared(replay)
        # Rejecting R1 does not revoke its successor in this per-token design.
        self.assertEqual(self.post_with_token("/api/v1/auth/refresh", new_token).status_code, 200)

    def test_concurrent_http_refresh_requests_have_exactly_one_success(self):
        self.login()
        token = self.client.cookies[REFRESH_COOKIE_NAME]

        async def refresh_twice():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="https://testserver") as client:
                headers = {"Cookie": f"{REFRESH_COOKIE_NAME}={token}"}
                return await asyncio.gather(
                    client.post("/api/v1/auth/refresh", headers=headers),
                    client.post("/api/v1/auth/refresh", headers=headers),
                )

        responses = asyncio.run(refresh_twice())
        self.assertCountEqual([response.status_code for response in responses], [200, 401])
        self.assertEqual(len(self.redis.entries), 1)
        self.assertEqual(self.redis.set.await_count, 2)  # Login plus one rotation.

    def test_missing_cookie_is_401_even_with_a_bearer_refresh_token(self):
        token, _ = self.jwt_service.create_refresh_token("alice")
        response = self.client.post("/api/v1/auth/refresh", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 401)
        self.assert_cookie_cleared(response)
        self.redis.getdel.assert_not_awaited()

    def test_invalid_expired_and_access_jwts_are_rejected_before_redis(self):
        token, _ = self.jwt_service.create_refresh_token("alice")
        claims = self.jwt_service.decode_refresh_token(token)
        expired = jwt.encode({**claims, "iat": int(time.time()) - 120, "exp": int(time.time()) - 60}, self.secret, algorithm="HS256")
        for invalid in ("bad-token", expired, self.jwt_service.create_access_token("alice")):
            with self.subTest(token=invalid):
                response = self.post_with_token("/api/v1/auth/refresh", invalid)
                self.assertEqual(response.status_code, 401)
                self.assert_cookie_cleared(response)
        self.redis.getdel.assert_not_awaited()
        self.auth_service.get_user_by_id.assert_not_awaited()

    def test_missing_redis_session_is_401(self):
        token, _ = self.jwt_service.create_refresh_token("alice")
        response = self.post_with_token("/api/v1/auth/refresh", token)
        self.assertEqual(response.status_code, 401)
        self.assert_cookie_cleared(response)
        self.auth_service.get_user_by_id.assert_not_awaited()

    def test_redis_subject_mismatch_is_rejected_and_consumed(self):
        self.login()
        key, (_, expires_at) = next(iter(self.redis.entries.items()))
        self.redis.entries[key] = ("bob", expires_at)
        response = self.client.post("/api/v1/auth/refresh")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.redis.entries, {})
        self.auth_service.get_user_by_id.assert_not_awaited()

    def test_disabled_or_deleted_user_is_rejected_after_consumption(self):
        for deleted in (False, True):
            with self.subTest(deleted=deleted):
                self.login()
                self.user.disabled = not deleted
                self.auth_service.get_user_by_id.return_value = None if deleted else self.user
                response = self.client.post("/api/v1/auth/refresh")
                self.assertEqual(response.status_code, 401)
                self.assert_cookie_cleared(response)
                self.assertEqual(self.redis.entries, {})
                self.user.disabled = False

    def test_logout_revokes_refresh_but_existing_access_token_still_works(self):
        access = self.login().json()["access_token"]
        refresh = self.client.cookies[REFRESH_COOKIE_NAME]
        response = self.client.post("/api/v1/auth/logout")
        self.assertEqual(response.status_code, 204)
        self.assert_cookie_cleared(response)
        self.assertEqual(self.redis.entries, {})
        self.assertEqual(self.post_with_token("/api/v1/auth/refresh", refresh).status_code, 401)
        me = self.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"})
        self.assertEqual(me.status_code, 200)

    def test_logout_is_idempotent_with_missing_invalid_expired_and_revoked_cookies(self):
        token, _ = self.jwt_service.create_refresh_token("alice")
        claims = self.jwt_service.decode_refresh_token(token)
        expired = jwt.encode({**claims, "iat": 1, "exp": 2}, self.secret, algorithm="HS256")
        for value in ("", "invalid", expired, token, token):
            with self.subTest(cookie=value):
                response = self.post_with_token("/api/v1/auth/logout", value)
                self.assertEqual(response.status_code, 204)
                self.assert_cookie_cleared(response)

    def test_refresh_token_cannot_be_used_as_access_bearer(self):
        self.login()
        token = self.client.cookies[REFRESH_COOKIE_NAME]
        response = self.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 401)

    def test_relogin_revokes_previous_cookie_session(self):
        self.login()
        old_token = self.client.cookies[REFRESH_COOKIE_NAME]
        self.login()
        self.assertEqual(len(self.redis.entries), 1)
        self.assertEqual(self.post_with_token("/api/v1/auth/refresh", old_token).status_code, 401)

    def test_login_redis_failure_does_not_return_tokens_or_set_cookie(self):
        self.redis.set.side_effect = RedisConnectionError("private Redis details")
        response = self.client.post("/api/v1/auth/login", json={"username": "alice", "password": "password"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("access_token", response.json())
        self.assertNotIn("set-cookie", response.headers)
        self.assertNotIn("private Redis details", response.text)

    def test_redis_consume_failure_fails_closed_without_user_lookup(self):
        self.login()
        self.redis.getdel.side_effect = RedisConnectionError("unavailable")
        response = self.client.post("/api/v1/auth/refresh")
        self.assertEqual(response.status_code, 503)
        self.auth_service.get_user_by_id.assert_not_awaited()
        self.assertNotIn("access_token", response.json())
        self.assertNotIn("set-cookie", response.headers)

    def test_replacement_write_failure_does_not_restore_consumed_token(self):
        self.login()
        old_token = self.client.cookies[REFRESH_COOKIE_NAME]
        self.redis.set.side_effect = RedisConnectionError("unavailable")
        response = self.client.post("/api/v1/auth/refresh")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.redis.entries, {})
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(self.post_with_token("/api/v1/auth/refresh", old_token).status_code, 401)

    def test_logout_redis_failure_reports_503_and_retains_cookie_for_retry(self):
        access = self.login().json()["access_token"]
        self.redis.delete.side_effect = RedisConnectionError("unavailable")
        response = self.client.post("/api/v1/auth/logout")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(len(self.redis.entries), 1)
        # Redis is not consulted for a normal protected access-token request.
        me = self.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"})
        self.assertEqual(me.status_code, 200)
        self.redis.delete.side_effect = self.redis.remove
        self.assertEqual(self.client.post("/api/v1/auth/logout").status_code, 204)
        self.assertEqual(self.redis.entries, {})

    def test_local_http_cookie_setting_and_secure_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(RefreshCookieSettings.from_environment().secure)
        with patch.dict(os.environ, {"REFRESH_COOKIE_SECURE": "false"}):
            self.app.state.refresh_cookie_settings = RefreshCookieSettings.from_environment()
        response = self.login()
        self.assertNotIn("Secure", response.headers["set-cookie"])
        with patch.dict(os.environ, {"REFRESH_COOKIE_SECURE": "typo"}):
            with self.assertRaises(ValueError):
                RefreshCookieSettings.from_environment()


if __name__ == "__main__":
    unittest.main()
