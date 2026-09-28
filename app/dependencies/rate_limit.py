"""Phase 3.9 dependencies: rate-limit enforcement as its own concern.

Authentication proves *who* (``get_current_user``), authorization proves
*whose data* (``authorize_chat_request``), and these dependencies prove
*how often* -- they never mint identity and never check ownership.

Keying: IP while no identity exists yet (register/login/refresh), JWT
``sub`` once it does (chat/upload). The IP is ``request.client.host``
only: ``X-Forwarded-For`` is client-controlled and rotates per request,
so honoring it would be a trivial bypass. Proxy deployments need a
trusted-proxy allowlist instead (future work, not silent trust).

FastAPI caches ``Depends`` per request, so nesting ``get_current_user``
/ ``authorize_chat_request`` here costs no second MongoDB lookup.
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.dependencies.auth import get_current_user
from app.dependencies.chat import authorize_chat_request
from app.schemas.auth import UserResponse
from app.schemas.chat import ChatRequest
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


logger = logging.getLogger(__name__)


def _service(request: Request) -> RateLimitService | None:
    # Missing wiring allows the request (logged): like a Redis outage, an
    # unwired limiter must degrade to "unlimited", never to a dead API.
    # The production lifespan always wires it; this path exists for
    # minimal apps and tests that exercise routes without Redis.
    return getattr(request.app.state, "rate_limit_service", None)


def client_ip(request: Request) -> str:
    if request.client is None:
        return "unknown"
    return request.client.host


async def _enforce(request: Request, *, scope: str, key: str) -> None:
    service = _service(request)
    if service is None:
        logger.warning(
            "Rate limit service not configured; allowing %s request", scope
        )
        return
    try:
        await service.check(scope=scope, key=key)
    except RateLimitUnavailable:
        # Fail open (logged): chat/upload must survive a Redis outage per
        # the Phase 3.4 availability invariant; login/refresh fail closed
        # elsewhere for session reasons.
        logger.warning("Rate limiter unavailable; allowing %s request", scope)
        return
    except RateLimitExceeded as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down and retry.",
            headers={"Retry-After": str(error.retry_after_seconds)},
        ) from error


async def enforce_register_rate_limit(request: Request) -> None:
    await _enforce(request, scope=SCOPE_REGISTER, key=f"ip:{client_ip(request)}")


async def enforce_login_rate_limit(request: Request) -> None:
    await _enforce(request, scope=SCOPE_LOGIN, key=f"ip:{client_ip(request)}")


async def enforce_refresh_rate_limit(request: Request) -> None:
    await _enforce(request, scope=SCOPE_REFRESH, key=f"ip:{client_ip(request)}")


async def enforce_upload_rate_limit(
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> None:
    await _enforce(
        request, scope=SCOPE_UPLOAD, key=f"user:{current_user.user_id}"
    )


async def enforce_chat_rate_limit(
    request: Request,
    authorized: Annotated[ChatRequest, Depends(authorize_chat_request)],
) -> None:
    await _enforce(
        request, scope=SCOPE_CHAT, key=f"user:{authorized.user_id}"
    )
