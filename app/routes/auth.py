from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

from app.dependencies.auth import get_current_user
from app.schemas.auth import AccessTokenResponse, LoginRequest, RegisterRequest, UserResponse
from app.security.jwt import JWTService
from app.services.auth_service import AuthService, UsernameAlreadyExistsError


class AuthRoute(APIRoute):
    """Keep submitted credentials out of validation error responses too."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        route_handler = super().get_route_handler()

        async def handle_request(request: Request) -> Response:
            try:
                return await route_handler(request)
            except RequestValidationError as error:
                # FastAPI's default errors can include the entire submitted body.
                # Retain useful field errors, but omit input and context values.
                details = [
                    {"type": item["type"], "loc": item["loc"], "msg": item["msg"]}
                    for item in error.errors()
                ]
                return JSONResponse(status_code=422, content={"detail": details})

        return handle_request


router = APIRouter(prefix="/api/v1/auth", tags=["auth"], route_class=AuthRoute)


@router.post("/login", response_model=AccessTokenResponse)
async def login(payload: LoginRequest, request: Request) -> AccessTokenResponse:
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

    jwt_service: JWTService = request.app.state.jwt_service
    return AccessTokenResponse(
        access_token=jwt_service.create_access_token(user.user_id),
        expires_in=jwt_service.expires_in,
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    return current_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, request: Request) -> UserResponse:
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
