"""Phase 3.6: authenticated document metadata + upload validation.

This service owns the MongoDB ``documents`` collection only. It never
embeds, chunks, or touches Chroma -- ingestion stays a ``RAGSystem``
responsibility called by the route. Every stored record carries the
authenticated ``user_id`` from ``get_current_user``; the upload endpoint
accepts no caller-supplied owner id, so cross-tenant claims are
impossible by construction.

Validation helpers are pure functions so routes stay thin and tests can
exercise them without HTTP, MongoDB, or Gemini.
"""

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any
from uuid import uuid4

from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

# v1 supports exactly what ``phase1.load_txt_documents`` already handles.
# More formats arrive only with real loaders, not by widening this list.
ALLOWED_EXTENSIONS = frozenset({".txt"})

# Authoritative check; the client MIME type is recorded, never trusted.
ALLOWED_DECODE = "utf-8"

DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_FILENAME_LENGTH = 255

PROCESSING = "processing"
READY = "ready"
FAILED = "failed"


class DocumentStoreUnavailable(Exception):
    """MongoDB could not be read or changed; never surfaces internals."""


class InvalidUpload(Exception):
    """Carries the safe client-facing (status_code, detail) pair."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def slugify_stem(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.strip().lower()).strip("-")
    return slug[:64] or "document"


def generate_document_id(original_filename: str) -> str:
    """Stable-per-upload id; see module docstring for why folder paths
    cannot serve here: uploads have no shared root to relativize, the
    same name collides across users, and path-derived ids invite
    traversal. Human-readable slug + random suffix keeps ids unique,
    safe, and debuggable while chunk ids stay ``{user}:{doc}:{index}``.
    """
    stem = PurePath(original_filename).stem
    return f"{slugify_stem(stem)}-{uuid4().hex[:8]}"


def derive_title(original_filename: str) -> str:
    stem = PurePath(original_filename).stem.replace("_", " ").strip()
    return stem.title()[:256] or "Untitled"


def validate_filename(filename: str | None) -> str:
    """Return the safe basename or raise; never trust directory parts."""
    if not filename or not filename.strip():
        raise InvalidUpload(400, "A file must be provided.")
    candidate = filename.strip()
    if (
        len(candidate) > MAX_FILENAME_LENGTH
        or PurePath(candidate).name != candidate
        or "/" in candidate
        or "\\" in candidate
        or candidate in {".", ".."}
    ):
        raise InvalidUpload(400, "Invalid filename.")
    stem = PurePath(candidate).stem.strip()
    if not stem or stem in {".", ".."}:
        raise InvalidUpload(400, "Invalid filename.")
    return candidate


def validate_extension(filename: str) -> str:
    extension = PurePath(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidUpload(
            415,
            f"Unsupported file type '{extension or '(none)'}'. "
            "Only .txt files are supported.",
        )
    return extension


def validate_and_decode(data: bytes, max_upload_bytes: int) -> str:
    if len(data) > max_upload_bytes:
        raise InvalidUpload(
            413,
            f"File is too large. Maximum size is {max_upload_bytes} bytes.",
        )
    if not data:
        raise InvalidUpload(400, "File is empty.")
    try:
        text = data.decode(ALLOWED_DECODE)
    except UnicodeDecodeError as error:
        raise InvalidUpload(
            400, "File must be decodable UTF-8 text."
        ) from error
    if not text.strip():
        raise InvalidUpload(400, "File contains no readable text.")
    return text


def stored_filename(document_id: str, extension: str) -> str:
    """Filesystem name derives only from the backend document id."""
    return f"{document_id}{extension}"


def user_directory(upload_dir: Path, user_id: str) -> Path:
    """Per-tenant directory; user ids are backend uuids, never raw input
    used as a path, and the stored filename carries no client content."""
    if not user_id or "/" in user_id or "\\" in user_id or ".." in user_id:
        raise ValueError("Invalid user id for storage path")
    return upload_dir / user_id


def stored_path_for(
    upload_dir: Path, user_id: str, record: dict[str, Any]
) -> Path:
    """Rebuild the stored path from the DB record, never from URL input:
    ``document_id`` is the server-minted id and the suffix comes from the
    validated original filename (exact while v1 is .txt-only)."""
    suffix = PurePath(record.get("original_filename") or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".txt"
    return user_directory(upload_dir, user_id) / stored_filename(
        record["document_id"], suffix
    )


class DocumentLocks:
    """Per-document mutexes serializing delete-vs-ingest on one process.

    Entries are refcounted and removed when idle: no unbounded growth and
    no cross-document contention. The check-and-pop is synchronous (no
    await), hence atomic on a single event loop. Single-process only --
    separate workers would need Redis claiming instead.
    """

    def __init__(self) -> None:
        self._entries: dict[str, list] = {}

    @asynccontextmanager
    async def hold(self, document_id: str):
        entry = self._entries.get(document_id)
        if entry is None:
            entry = [asyncio.Lock(), 0]
            self._entries[document_id] = entry
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] <= 0 and not entry[0].locked():
                self._entries.pop(document_id, None)


class DocumentService:
    def __init__(self, documents):
        self.documents = documents

    async def ensure_indexes(self) -> None:
        await self.documents.create_index("document_id", unique=True)
        await self.documents.create_index([("user_id", 1), ("created_at", -1)])

    async def create_processing_document(
        self,
        *,
        user_id: str,
        document_id: str,
        original_filename: str,
        content_type: str | None,
        size_bytes: int,
        title: str | None,
    ) -> dict[str, Any]:
        now = _utcnow()
        document = {
            "document_id": document_id,
            "user_id": user_id,
            "original_filename": original_filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "status": PROCESSING,
            "chunk_count": 0,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self.documents.insert_one(document)
        except PyMongoError as error:
            logger.exception("Could not record uploaded document")
            raise DocumentStoreUnavailable("Document store unavailable") from error
        return document

    async def mark_ready(
        self, document_id: str, user_id: str, chunk_count: int
    ) -> None:
        try:
            await self.documents.update_one(
                {"document_id": document_id, "user_id": user_id},
                {"$set": {
                    "status": READY,
                    "chunk_count": chunk_count,
                    "updated_at": _utcnow(),
                }},
            )
        except PyMongoError as error:
            logger.exception("Could not mark document ready")
            raise DocumentStoreUnavailable("Document store unavailable") from error

    async def mark_failed(self, document_id: str, user_id: str) -> None:
        try:
            await self.documents.update_one(
                {"document_id": document_id, "user_id": user_id},
                {"$set": {"status": FAILED, "updated_at": _utcnow()}},
            )
        except PyMongoError as error:
            logger.exception("Could not mark document failed")
            raise DocumentStoreUnavailable("Document store unavailable") from error

    async def get_document(
        self, document_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Ownership-scoped lookup: id alone never grants access."""
        try:
            return await self.documents.find_one({
                "document_id": document_id,
                "user_id": user_id,
            })
        except PyMongoError as error:
            logger.exception("Could not read document")
            raise DocumentStoreUnavailable("Document store unavailable") from error

    async def list_documents(
        self, user_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """All statuses, newest first: failed/processing rows stay visible
        so users can clean up crashed or rejected uploads."""
        try:
            cursor = (
                self.documents.find({"user_id": user_id})
                .sort("created_at", -1)
            )
            return await cursor.to_list(length=limit)
        except PyMongoError as error:
            logger.exception("Could not list documents")
            raise DocumentStoreUnavailable("Document store unavailable") from error

    async def delete_document(
        self, document_id: str, user_id: str
    ) -> bool:
        """Ownership-scoped delete; True only when a record was removed."""
        try:
            result = await self.documents.delete_one({
                "document_id": document_id,
                "user_id": user_id,
            })
        except PyMongoError as error:
            logger.exception("Could not delete document")
            raise DocumentStoreUnavailable("Document store unavailable") from error
        return result.deleted_count > 0

    async def list_stale_processing(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Processing rows (any owner) for startup requeue after a crash.

        Internal recovery op, not a user query: the Phase 3.8 worker owns
        what happens next (re-ingest; ingest is idempotent per document).
        """
        try:
            cursor = (
                self.documents.find({"status": PROCESSING})
                .sort("created_at", 1)
            )
            return await cursor.to_list(length=limit)
        except PyMongoError as error:
            logger.exception("Could not list stale uploads")
            raise DocumentStoreUnavailable("Document store unavailable") from error
