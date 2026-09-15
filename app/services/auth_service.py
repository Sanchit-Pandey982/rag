from typing import Any

from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import DuplicateKeyError
from starlette.concurrency import run_in_threadpool

from app.models.user import User, normalize_username
from app.security.passwords import (
    DUMMY_PASSWORD_HASH,
    hash_password,
    verify_password,
)


class UsernameAlreadyExistsError(Exception):
    """Registration attempted to reuse a normalized username."""


class AuthService:
    def __init__(self, users: AsyncCollection[dict[str, Any]]):
        self.users = users

    async def ensure_indexes(self) -> None:
        await self.users.create_index("username", unique=True)
        await self.users.create_index("user_id", unique=True)

    async def register_user(self, username: str, password: str) -> User:
        username = normalize_username(username)
        password_hash = await run_in_threadpool(hash_password, password)
        user = User(username=username, password_hash=password_hash)

        try:
            # Insert directly: MongoDB's unique index arbitrates concurrent writes.
            # Python mode preserves datetime as a BSON-compatible datetime value.
            await self.users.insert_one(user.model_dump())
        except DuplicateKeyError as error:
            key_pattern = (error.details or {}).get("keyPattern", {})
            if "username" in key_pattern:
                raise UsernameAlreadyExistsError("Username already exists") from error

            # Some MongoDB deployments omit keyPattern in duplicate-key errors.
            # Only translate those errors after confirming a username conflict.
            if not key_pattern:
                existing_user = await self.users.find_one({"username": username})
                if existing_user is not None:
                    raise UsernameAlreadyExistsError("Username already exists") from error
            raise

        return user

    async def get_user_by_id(self, user_id: str) -> User | None:
        document = await self.users.find_one({"user_id": user_id})
        if document is None:
            return None
        return User.model_validate(document)

    async def authenticate_user(self, username: str, password: str) -> User | None:
        document = await self.users.find_one({"username": normalize_username(username)})

        if document is None:
            await run_in_threadpool(verify_password, password, DUMMY_PASSWORD_HASH)
            return None

        user = User.model_validate(document)
        password_is_valid = await run_in_threadpool(
            verify_password, password, user.password_hash
        )

        # All credential failures have the same internal result. Check account
        # state after verification so disabled accounts still do the hash work.
        if not password_is_valid or user.disabled:
            return None

        return user
