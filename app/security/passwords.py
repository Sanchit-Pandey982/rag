from secrets import token_urlsafe

from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError


password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash the exact password, preserving whitespace and case."""
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return password_hasher.verify(password, password_hash)
    except UnknownHashError:
        # An unsupported stored hash must never authenticate a user.
        return False


# Create once per process, not once per failed authentication attempt.
DUMMY_PASSWORD_HASH = hash_password(token_urlsafe(32))
