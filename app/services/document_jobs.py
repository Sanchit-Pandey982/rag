"""Phase 3.8: background ingestion worker.

Upload returns 202 after validation + file write + processing record; this
worker does the slow embedding/Chroma step outside the request lifecycle
and flips the record to ready/failed. Clients poll ``GET /{id}`` (3.7).

Execution is in-process via FastAPI ``BackgroundTasks``: zero new
infrastructure, at the price of single-process execution (``DocumentLocks``
are in-memory) and no per-job retry endpoint. Crash durability comes from
two properties, not a broker:

* re-ingest is idempotent per document (``ingest_documents`` deletes prior
  chunks for the same ``user_id`` + ``document_id`` first), so a job is
  safe to run twice -- which is exactly what startup requeue does with
  rows still stuck at ``processing``;
* delete-vs-ingest races close with a per-document lock plus a
  post-ingest existence check that compensates (removes just-indexed
  chunks) when the record vanished mid-flight.
"""

import asyncio
import logging
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from phase1 import Document as RAGDocument

from app.services.document_service import (
    DocumentLocks,
    DocumentService,
    DocumentStoreUnavailable,
    InvalidUpload,
    stored_path_for,
    validate_and_decode,
    validate_extension,
)

logger = logging.getLogger(__name__)


async def _fail(
    service: DocumentService, document_id: str, user_id: str
) -> None:
    try:
        await service.mark_failed(document_id, user_id)
    except DocumentStoreUnavailable:
        logger.exception("Could not mark document failed")


async def process_document_upload(
    *,
    service: DocumentService,
    rag_service,
    upload_dir: Path,
    locks: DocumentLocks,
    document_id: str,
    user_id: str,
    max_upload_bytes: int,
) -> None:
    """Run one enqueued ingestion to ready/failed; never raises except on
    cancellation (which is re-raised after a best-effort failed mark)."""
    try:
        async with locks.hold(document_id):
            await _process(
                service=service,
                rag_service=rag_service,
                upload_dir=Path(upload_dir),
                document_id=document_id,
                user_id=user_id,
                max_upload_bytes=max_upload_bytes,
            )
    except asyncio.CancelledError:
        # Shutdown mid-job: leave an honest state if the store is alive.
        await _fail(service, document_id, user_id)
        raise


async def _process(
    *,
    service: DocumentService,
    rag_service,
    upload_dir: Path,
    document_id: str,
    user_id: str,
    max_upload_bytes: int,
) -> None:
    try:
        record = await service.get_document(document_id, user_id)
    except DocumentStoreUnavailable:
        logger.exception("Stale upload requeue cannot read record")
        return
    if record is None:
        # Deleted before the worker started: nothing to index.
        logger.info("Skipping ingestion for deleted document %s", document_id)
        return

    try:
        stored_path = stored_path_for(upload_dir, user_id, record)
    except ValueError:
        logger.exception("Stored path invalid for document %s", document_id)
        await _fail(service, document_id, user_id)
        return

    try:
        raw = await run_in_threadpool(stored_path.read_bytes)
    except OSError:
        logger.exception("Stored file unreadable for document %s", document_id)
        await _fail(service, document_id, user_id)
        return

    try:
        extension = validate_extension(record.get("original_filename") or "")
        text = validate_and_decode(raw, max_upload_bytes)
    except InvalidUpload as error:
        logger.warning(
            "Stored file rejected for document %s: %s", document_id, error.detail
        )
        await _fail(service, document_id, user_id)
        return

    try:
        chunk_count = await run_in_threadpool(
            rag_service.ingest_user_document,
            RAGDocument(
                document_id=document_id,
                source=stored_path.relative_to(upload_dir).as_posix(),
                title=record.get("title"),
                text=text,
            ),
            user_id,
        )
    except Exception:
        logger.exception("Document ingestion failed for %s", document_id)
        await _fail(service, document_id, user_id)
        return

    # The record may have been deleted while embedding ran: drop the
    # just-indexed chunks instead of stranding them without metadata.
    try:
        still_there = await service.get_document(document_id, user_id)
    except DocumentStoreUnavailable:
        logger.exception("Post-ingest record check failed for %s", document_id)
        return
    if still_there is None:
        try:
            await run_in_threadpool(
                rag_service.delete_user_document, document_id, user_id
            )
        except Exception:
            logger.exception("Could not remove orphaned chunks for %s", document_id)
        return

    try:
        await service.mark_ready(document_id, user_id, chunk_count)
    except DocumentStoreUnavailable:
        # Stays `processing`; startup requeue replays it (idempotent).
        logger.exception("Could not mark document ready for %s", document_id)


async def requeue_stale_uploads(
    *,
    service: DocumentService,
    rag_service,
    upload_dir: Path,
    locks: DocumentLocks,
    max_upload_bytes: int,
    limit: int = 100,
) -> int:
    """Schedule workers for rows stuck at ``processing`` (crash recovery).

    Returns the number requeued; a store outage raises
    ``DocumentStoreUnavailable`` to the caller (lifespan logs it).
    """
    stale = await service.list_stale_processing(limit=limit)
    for record in stale:
        asyncio.create_task(process_document_upload(
            service=service,
            rag_service=rag_service,
            upload_dir=Path(upload_dir),
            locks=locks,
            document_id=record["document_id"],
            user_id=record["user_id"],
            max_upload_bytes=max_upload_bytes,
        ))
    if stale:
        logger.info("Requeued %d stale upload(s)", len(stale))
    return len(stale)
