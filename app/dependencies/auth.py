from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.schemas.auth import UserResponse
from app.security.jwt import AccessTokenError, JWTService
from app.services.auth_service import AuthService
from app.services.token_blacklist_service import BlacklistUnavailable


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> UserResponse:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing access token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized

    jwt_service: JWTService = request.app.state.jwt_service
    try:
        claims = jwt_service.decode_access_token(credentials.credentials)
    except AccessTokenError as error:
        raise unauthorized from error

    # Phase 4: logged-out (or rotated) access tokens are revoked by JTI.
    # The service is None-safe so unit tests without lifespan state keep
    # working; a Redis outage is a 503 (retryable infra failure), never a
    # silent acceptance of a possibly-revoked token.
    blacklist_service = getattr(
        request.app.state, "token_blacklist_service", None
    )
    if blacklist_service is not None:
        try:
            listed = await blacklist_service.is_blacklisted(claims["jti"])
        except BlacklistUnavailable as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication session service is unavailable. "
                       "Please retry.",
            ) from error
        if listed:
            raise unauthorized

    auth_service: AuthService = request.app.state.auth_service
    user = await auth_service.get_user_by_id(claims["sub"])
    if user is None:
        raise unauthorized
    if user.disabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled.")

    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        created_at=user.created_at,
    )
