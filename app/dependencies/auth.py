from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.schemas.auth import UserResponse
from app.security.jwt import AccessTokenError, JWTService
from app.services.auth_service import AuthService


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
