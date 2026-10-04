"""Phase 4 tests: JWT access-token blacklisting on logout/rotation.

No live Redis/Mongo/AI: real JWTs + real auth routes against an
InMemoryRedis double (extended with EXISTS) and a mocked AuthService.
Covers blacklist service semantics (EXAT expiry, idempotency,
auto-expiry), logout revoking the presented access token, refresh
retiring the old access token, 401-on-reuse, unchanged behavior without
a Bearer token, and 503 on Redis outage.
"""

import time
import unittest
from secrets import token_urlsafe
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from app.models.user import User
from app.routes.auth import router
from app.security.cookies import RefreshCookieSettings
from app.security.jwt import JWTService
from app.services.refresh_token_service import RefreshTokenService
from app.services.token_blacklist_service import (
    BlacklistUnavailable,
    TokenBlacklistService,
)
from redis_double import InMemoryRedis


class BlacklistServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = InMemoryRedis()
        self.service = TokenBlacklistService(self.redis)

    async def test_blacklist_uses_jti_key_with_token_expiry(self):
        expires_at = int(time.time()) + 900
        await self.service.blacklist_jti("jti-1", expires_at=expires_at)
        self.redis.set.assert_awaited_once_with(
            "auth:blacklist:jti-1", "1", exat=expires_at, nx=True)
        self.assertTrue(await self.service.is_blacklisted("jti-1"))
        self.assertFalse(await self.service.is_blacklisted("other"))
        self.redis.exists.assert_awaited_with("auth:blacklist:other")

    async def test_blacklist_is_idempotent(self):
        expires_at = int(time.time()) + 900
        await self.service.blacklist_jti("jti-1", expires_at=expires_at)
        await self.service.blacklist_jti("jti-1", expires_at=expires_at)
        self.assertTrue(await self.service.is_blacklisted("jti-1"))

    async def test_already_expired_token_is_skipped(self):
        await self.service.blacklist_jti("jti-old",
                                         expires_at=int(time.time()) - 10)
        self.redis.set.assert_not_awaited()
        self.assertFalse(await self.service.is_blacklisted("jti-old"))

    async def test_entry_auto_expires_with_token(self):
        expires_at = self.redis.now + 30
        await self.service.blacklist_jti("jti-1", expires_at=expires_at)
        self.assertTrue(await self.service.is_blacklisted("jti-1"))
        self.redis.now = expires_at
        self.assertFalse(await self.service.is_blacklisted("jti-1"))
        self.assertEqual(self.redis.entries, {})

    async def test_redis_errors_raise_blacklist_unavailable(self):
        self.redis.exists = AsyncMock(
            side_effect=RedisError("down"))
        with self.assertRaises(BlacklistUnavailable):
            await self.service.is_blacklisted("jti-1")
        failing_set = InMemoryRedis()
        failing_set.set = AsyncMock(side_effect=RedisError("down"))
        with self.assertRaises(BlacklistUnavailable):
            await TokenBlacklistService(failing_set).blacklist_jti(
                "jti-1", expires_at=int(time.time()) + 60)


class BlacklistRouteTests(unittest.TestCase):
    def setUp(self):
        self.secret = token_urlsafe(32)
        self.jwt_service = JWTService(self.secret)
        self.redis = InMemoryRedis()
        self.user = User(user_id="alice", username="alice",
                         password_hash="unused")
        self.auth_service = Mock(
            authenticate_user=AsyncMock(return_value=self.user),
            get_user_by_id=AsyncMock(return_value=self.user),
        )
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.state.jwt_service = self.jwt_service
        self.app.state.auth_service = self.auth_service
        self.app.state.refresh_token_service = RefreshTokenService(
            self.redis)
        self.app.state.token_blacklist_service = TokenBlacklistService(
            self.redis)
        self.app.state.refresh_cookie_settings = RefreshCookieSettings(
            secure=False)
        self.client = self.enterContext(TestClient(self.app))

    def login(self):
        response = self.client.post("/api/v1/auth/login", json={
            "username": "alice", "password": "password",
        })
        self.assertEqual(response.status_code, 200)
        return response.json()["access_token"]

    def get_me(self, token):
        return self.client.get("/api/v1/auth/me",
                               headers={"Authorization": f"Bearer {token}"})

    def test_valid_token_passes_before_logout(self):
        access = self.login()
        self.assertEqual(self.get_me(access).status_code, 200)

    def test_logout_with_bearer_revokes_access_token(self):
        access = self.login()
        claims = self.jwt_service.decode_access_token(access)
        response = self.client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access}"})
        self.assertEqual(response.status_code, 204)
        # JTI recorded with the token's own expiry.
        self.assertEqual(
            self.redis.entries.get(f'auth:blacklist:{claims["jti"]}'),
            ("1", claims["exp"]))
        # Reuse is rejected like any invalid token.
        reuse = self.get_me(access)
        self.assertEqual(reuse.status_code, 401)
        self.assertEqual(reuse.headers["www-authenticate"], "Bearer")
        self.assertEqual(reuse.json(),
                         {"detail": "Invalid or missing access token."})

    def test_logout_without_bearer_keeps_access_valid(self):
        access = self.login()
        response = self.client.post("/api/v1/auth/logout")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.get_me(access).status_code, 200)
        self.assertEqual(
            [key for key in self.redis.entries
             if key.startswith("auth:blacklist:")], [])

    def test_logout_with_garbage_bearer_still_204(self):
        access = self.login()
        response = self.client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": "Bearer garbage-token"})
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.get_me(access).status_code, 200)

    def test_refresh_with_bearer_retires_old_access_token(self):
        old_access = self.login()
        response = self.client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {old_access}"})
        self.assertEqual(response.status_code, 200)
        new_access = response.json()["access_token"]
        self.assertNotEqual(old_access, new_access)
        self.assertEqual(self.get_me(old_access).status_code, 401)
        self.assertEqual(self.get_me(new_access).status_code, 200)

    def test_refresh_without_bearer_still_rotates(self):
        old_access = self.login()
        response = self.client.post("/api/v1/auth/refresh")
        self.assertEqual(response.status_code, 200)
        # No access token presented, so nothing is blacklisted.
        self.assertEqual(self.get_me(old_access).status_code, 200)

    def test_blacklist_outage_is_503_not_401(self):
        access = self.login()
        original_exists = self.redis.exists
        self.redis.exists = AsyncMock(side_effect=RedisError("down"))
        try:
            response = self.get_me(access)
        finally:
            self.redis.exists = original_exists
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {
            "detail": "Authentication session service is unavailable. "
                      "Please retry."})

    def test_blacklist_failure_on_logout_is_503(self):
        access = self.login()
        original_set = self.redis.set
        self.redis.set = AsyncMock(side_effect=RedisError("down"))
        try:
            response = self.client.post(
                "/api/v1/auth/logout",
                headers={"Authorization": f"Bearer {access}"})
        finally:
            self.redis.set = original_set
        self.assertEqual(response.status_code, 503)
        # Nothing was falsely revoked: the token still validates once
        # Redis recovers.
        self.assertEqual(self.get_me(access).status_code, 200)


if __name__ == "__main__":
    unittest.main()
