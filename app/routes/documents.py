"""Phase 3.6/3.7/3.8: authenticated documents, management, background work.

Upload flow (3.8: only the slow embedding step left the request)::

    Bearer access JWT
      -> get_current_user (401/403 before any work)
      -> stream-read upload with size cap (413 on overflow)
      -> validate filename / extension / UTF-8 decoding / emptiness
      -> MongoDB processing record keyed by backend document id
      -> write bytes to uploads/<user_id>/<document_id>.txt
      -> enqueue background ingestion, return 202 + processing record
      -> worker embeds/indexes, flips ready/failed (client polls GET)

The multipart body carries no owner id: ``user_id`` comes only from the
verified JWT, so a forged owner claim is impossible by construction.
Validation failures are 400/413/415 with safe messages; infrastructure
failures are 503 (MongoDB) or generic 500s without leaking internals.
Only ``.txt`` is accepted -- the one format the existing
``load_txt_documents`` pipeline already handles.

Management (3.7): list mine, read one, delete mine (chunks -> file ->
record; unknown and foreign ids are an identical 404).
"""

import functools
import logging
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from starlette.concurrency import run_in_threadpool

from app.dependencies.auth import get_current_user
from app.dependencies.rate_limit import enforce_upload_rate_limit
from app.schemas.auth import UserResponse
from app.schemas.document import DocumentResponse
from app.services.document_jobs import process_document_upload
from app.services.document_service import (
    DEFAULT_MAX_UPLOAD_BYTES,
    DocumentLocks,
    DocumentService,
    DocumentStoreUnavailable,
    InvalidUpload,
    derive_title,
    generate_document_id,
    stored_filename,
    stored_path_for,
    user_directory,
    validate_and_decode,
    validate_extension,
    validate_filename,
)

logger = logging.getLogger(__name__)

_READ_CHUNK_SIZE = 1024 * 1024

# Backend ids are `slug-8hex`; the guard is defense in depth for the
# filesystem touch in delete (Mongo scoping already gates unknown ids).
_DOCUMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")

_RESPONSE_KEYS = (
    "document_id", "user_id", "original_filename", "content_type",
    "size_bytes", "status", "chunk_count", "title",
    "created_at", "updated_at",
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _service(request: Request) -> DocumentService:
    service = getattr(request.app.state, "document_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is unavailable. Please retry.",
        )
    return service


def _rag_service(request: Request):
    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is unavailable. Please retry.",
        )
    return rag_service


def _upload_dir(request: Request) -> Path:
    upload_dir = getattr(request.app.state, "upload_dir", None)
    if upload_dir is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is unavailable. Please retry.",
        )
    return Path(upload_dir)


def _max_upload_bytes(request: Request) -> int:
    return int(getattr(
        request.app.state, "max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES
    ))


def _locks(request: Request) -> DocumentLocks:
    locks = getattr(request.app.state, "document_locks", None)
    if locks is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is unavailable. Please retry.",
        )
    return locks


def _to_response(record: dict[str, Any]) -> DocumentResponse:
    return DocumentResponse(**{
        key: record[key] for key in _RESPONSE_KEYS
    })


def _checked_document_id(document_id: str) -> str:
    """Reject non-id-shaped path input with the same uniform 404 used for
    unknown or foreign ids, before any store or filesystem is touched."""
    if not _DOCUMENT_ID_RE.fullmatch(document_id or ""):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return document_id


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Document not found.",
    )


def _unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Document service is unavailable. Please retry.",
    )


async def _read_bounded(upload: UploadFile, max_upload_bytes: int) -> bytes:
    """Read at most limit+1 bytes so oversize uploads fail fast (413)
    instead of loading unbounded client input into memory."""
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_upload_bytes:
            raise InvalidUpload(
                413,
                f"File is too large. Maximum size is {max_upload_bytes} bytes.",
            )
        parts.append(chunk)
    return b"".join(parts)


@router.post("/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    request: Request,
    background: BackgroundTasks,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    file: Annotated[UploadFile, File(...)],
    _: Annotated[None, Depends(enforce_upload_rate_limit)] = None,
) -> DocumentResponse:
    """Accept, store, and enqueue; readiness arrives via ``GET /{id}``.

    201 used to promise a ready resource. That promise cannot survive the
    request lifecycle once embedding moves out of it, so 202 + a
    ``processing`` record is the honest contract (client polls until
    ``ready``/``failed``). Everything before enqueueing -- validation,
    record creation, byte storage -- stays synchronous and keeps its
    existing status codes.
    """
    user_id = current_user.user_id
    service = _service(request)
    upload_dir = _upload_dir(request)
    max_upload_bytes = _max_upload_bytes(request)

    try:
        safe_name = validate_filename(file.filename)
        extension = validate_extension(safe_name)
        raw = await _read_bounded(file, max_upload_bytes)
        validate_and_decode(raw, max_upload_bytes)
    except InvalidUpload as error:
        raise HTTPException(
            status_code=error.status_code, detail=error.detail
        ) from error

    document_id = generate_document_id(safe_name)
    title = derive_title(safe_name)

    try:
        record = await service.create_processing_document(
            user_id=user_id,
            document_id=document_id,
            original_filename=safe_name,
            content_type=file.content_type,
            size_bytes=len(raw),
            title=title,
        )
    except DocumentStoreUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is unavailable. Please retry.",
        ) from error

    try:
        directory = user_directory(upload_dir, user_id)
        directory.mkdir(parents=True, exist_ok=True)
        stored_path = directory / stored_filename(document_id, extension)
        # Existence check, not a security check: both segments are
        # backend-generated (uuid user id, slug+hex document id).
        await run_in_threadpool(stored_path.write_bytes, raw)
    except (OSError, ValueError) as error:
        logger.exception("Document file storage failed")
        await _best_effort_fail(service, document_id, user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be stored.",
        ) from error

    background.add_task(
        process_document_upload,
        service=service,
        rag_service=_rag_service(request),
        upload_dir=upload_dir,
        locks=_locks(request),
        document_id=document_id,
        user_id=user_id,
        max_upload_bytes=max_upload_bytes,
    )
    return _to_response(record)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[DocumentResponse]:
    try:
        records = await _service(request).list_documents(
            current_user.user_id, limit=limit
        )
    except DocumentStoreUnavailable as error:
        raise _unavailable() from error
    return [_to_response(record) for record in records]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> DocumentResponse:
    document_id = _checked_document_id(document_id)
    try:
        record = await _service(request).get_document(
            document_id, current_user.user_id
        )
    except DocumentStoreUnavailable as error:
        raise _unavailable() from error
    if record is None:
        raise _not_found()
    return _to_response(record)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> Response:
    """Remove Chroma chunks + stored file + Mongo record, in that order.

    Mongo is the user-visible source of truth, so it goes last: if Chroma
    fails nothing else is touched (retryable, nothing orphaned); if Mongo
    fails the record survives (retryable, chunks already cut from
    retrieval). File removal is best-effort -- a leftover file without
    chunks or a record is inert. Unknown and foreign ids are an identical
    404, checked before any deletion work begins.
    """
    user_id = current_user.user_id
    document_id = _checked_document_id(document_id)
    service = _service(request)

    # Serialized against the background worker on this document: without
    # the lock, an ingest finishing between the chunk delete and the
    # record delete below would strand retrievable orphan chunks.
    async with _locks(request).hold(document_id):
        try:
            record = await service.get_document(document_id, user_id)
        except DocumentStoreUnavailable as error:
            raise _unavailable() from error
        if record is None:
            raise _not_found()

        try:
            await run_in_threadpool(
                _rag_service(request).delete_user_document, document_id, user_id
            )
        except Exception as error:
            logger.exception("Document chunk deletion failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The document could not be deleted.",
            ) from error

        try:
            stored_path = stored_path_for(_upload_dir(request), user_id, record)
            await run_in_threadpool(
                functools.partial(stored_path.unlink, missing_ok=True)
            )
        except (OSError, ValueError):
            # Missing files are fine; anything else is logged and the delete
            # proceeds -- chunks are already gone and the record goes next.
            logger.exception("Could not remove stored file")

        try:
            deleted = await service.delete_document(document_id, user_id)
        except DocumentStoreUnavailable as error:
            raise _unavailable() from error
        if not deleted:  # pragma: no cover - defensive; just read above
            raise _not_found()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _best_effort_fail(
    service: DocumentService, document_id: str, user_id: str
) -> None:
    try:
        await service.mark_failed(document_id, user_id)
    except DocumentStoreUnavailable:
        logger.exception("Could not mark document failed")
