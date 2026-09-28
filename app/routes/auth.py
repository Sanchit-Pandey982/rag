from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

from app.dependencies.auth import get_current_user
from app.dependencies.rate_limit import (
    enforce_login_rate_limit,
    enforce_refresh_rate_limit,
    enforce_register_rate_limit,
)
from app.schemas.auth import AccessTokenResponse, LoginRequest, RegisterRequest, UserResponse
from app.security.cookies import REFRESH_COOKIE_NAME, clear_refresh_cookie, set_refresh_cookie
from app.security.jwt import JWTService, RefreshTokenError
from app.services.auth_service import AuthService, UsernameAlreadyExistsError
from app.services.refresh_token_service import RefreshSessionUnavailable, RefreshTokenService


class AuthRoute(APIRoute):
    """Keep submitted credentials out of validation error responses too."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        route_handler = super().get_route_handler()

        async def handle_request(request: Request) -> Response:
            try:
                response = await route_handler(request)
            except RequestValidationError as error:
                # FastAPI's default errors can include the entire submitted body.
                # Retain useful field errors, but omit input and context values.
                details = [
                    {"type": item["type"], "loc": item["loc"], "msg": item["msg"]}
                    for item in error.errors()
                ]
                response = JSONResponse(status_code=422, content={"detail": details})
            except RefreshSessionUnavailable:
                # Keep the cookie so logout can be retried; never claim revocation
                # succeeded when Redis did not confirm it.
                response = JSONResponse(
                    status_code=503,
                    content={"detail": "Authentication session service is unavailable. Please retry."},
                )

            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            return response

        return handle_request


router = APIRouter(prefix="/api/v1/auth", tags=["auth"], route_class=AuthRoute)


async def issue_session(request: Request, response: Response, user_id: str) -> AccessTokenResponse:
    """Publish tokens only after Redis confirms the new refresh session exists."""
    jwt_service: JWTService = request.app.state.jwt_service
    refresh_service: RefreshTokenService = request.app.state.refresh_token_service
    access_token = jwt_service.create_access_token(user_id)
    refresh_token, refresh_jti = jwt_service.create_refresh_token(user_id)
    claims = jwt_service.decode_refresh_token(refresh_token)
    await refresh_service.store_refresh_token(refresh_jti, user_id, expires_at=claims["exp"])
    set_refresh_cookie(
        response, refresh_token,
        expires_at=claims["exp"],
        settings=request.app.state.refresh_cookie_settings,
    )
    return AccessTokenResponse(access_token=access_token, expires_in=jwt_service.expires_in)


async def revoke_cookie_session(request: Request) -> None:
    """Invalid/expired cookies need only clearing; valid ones need Redis revocation."""
    token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not token:
        return
    try:
        claims = request.app.state.jwt_service.decode_refresh_token(token)
    except RefreshTokenError:
        return
    await request.app.state.refresh_token_service.revoke_refresh_token(claims["jti"])


def reject_refresh(request: Request) -> JSONResponse:
    response = JSONResponse(
        status_code=401,
        content={"detail": "Invalid or expired refresh session. Please sign in again."},
    )
    clear_refresh_cookie(response, settings=request.app.state.refresh_cookie_settings)
    return response


@router.post("/login", response_model=AccessTokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    _: Annotated[None, Depends(enforce_login_rate_limit)] = None,
) -> AccessTokenResponse:
    auth_service: AuthService = request.app.state.auth_service
    user = await auth_service.authenticate_user(
        username=payload.username,
        password=payload.password.get_secret_value(),
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # A fresh login replaces this browser's old session instead of orphaning it.
    await revoke_cookie_session(request)
    return await issue_session(request, response, user.user_id)


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh(
    request: Request,
    response: Response,
    _: Annotated[None, Depends(enforce_refresh_rate_limit)] = None,
) -> AccessTokenResponse | Response:
    token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not token:
        return reject_refresh(request)

    jwt_service: JWTService = request.app.state.jwt_service
    try:
        claims = jwt_service.decode_refresh_token(token)
    except RefreshTokenError:
        return reject_refresh(request)

    refresh_service: RefreshTokenService = request.app.state.refresh_token_service
    stored_user_id = await refresh_service.consume_refresh_token(claims["jti"])
    if stored_user_id is None or stored_user_id != claims["sub"]:
        return reject_refresh(request)

    # The token's signature does not prove that the account still exists or is enabled.
    user = await request.app.state.auth_service.get_user_by_id(claims["sub"])
    if user is None or user.disabled:
        return reject_refresh(request)

    return await issue_session(request, response, user.user_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request) -> Response:
    await revoke_cookie_session(request)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_refresh_cookie(response, settings=request.app.state.refresh_cookie_settings)
    return response


@router.get("/me", response_model=UserResponse)
async def me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    return current_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    request: Request,
    _: Annotated[None, Depends(enforce_register_rate_limit)] = None,
) -> UserResponse:
    auth_service: AuthService = request.app.state.auth_service

    try:
        user = await auth_service.register_user(
            username=payload.username,
            password=payload.password.get_secret_value(),
        )
    except UsernameAlreadyExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        ) from error

    # Explicitly select public fields; the persisted model contains credentials.
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        created_at=user.created_at,
    )
