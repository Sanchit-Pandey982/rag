"""Refresh-cookie settings for the existing same-origin browser/API setup."""

from dataclasses import dataclass
import os
import time

from fastapi import Response


REFRESH_COOKIE_NAME = "refresh_token"
REFRESH_COOKIE_PATH = "/api/v1/auth"


@dataclass(frozen=True)
class RefreshCookieSettings:
    secure: bool = True

    @classmethod
    def from_environment(cls) -> "RefreshCookieSettings":
        value = os.getenv("REFRESH_COOKIE_SECURE", "true").strip().lower()
        if value not in {"true", "false"}:
            raise ValueError("REFRESH_COOKIE_SECURE must be true or false")
        return cls(secure=value == "true")


def set_refresh_cookie(
    response: Response,
    token: str,
    *,
    expires_at: int,
    settings: RefreshCookieSettings,
) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        max_age=max(0, expires_at - int(time.time())),
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        secure=settings.secure,
        samesite="strict",
    )


def clear_refresh_cookie(response: Response, *, settings: RefreshCookieSettings) -> None:
    # Use the same name, path and domain scope as the original cookie.
    response.delete_cookie(
        key=REFRESH_COOKIE_NAME,
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        secure=settings.secure,
        samesite="strict",
    )
