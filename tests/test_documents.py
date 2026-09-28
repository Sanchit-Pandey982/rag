"""Phase 3.6/3.7 tests: authenticated .txt upload + document management.

Real DocumentService + real documents routes against an in-memory async
Mongo double; RAG ingestion/deletion is a Mock. Covers JWT-derived
ownership (no owner id in the body to forge), file validation, safe
storage paths, tenant-scoped Chroma handoff, list/get/delete ownership,
delete ordering (chunks -> file -> record), failure semantics (failed
record, no leak), and infra 503s.
"""

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError

from app.dependencies.auth import get_current_user
from app.routes.documents import router as documents_router
from app.schemas.auth import UserResponse
from app.services.document_jobs import (
    process_document_upload,
    requeue_stale_uploads,
)
from app.services.document_service import (
    DEFAULT_MAX_UPLOAD_BYTES,
    DocumentLocks,
    DocumentService,
    derive_title,
    generate_document_id,
    stored_path_for,
    validate_and_decode,
    validate_extension,
    validate_filename,
    InvalidUpload,
)


class FakeCursor:
    def __init__(self, documents):
        self._documents = list(documents)

    def sort(self, key, direction=1):
        self._documents.sort(
            key=lambda document: document.get(key),
            reverse=(direction < 0),
        )
        return self

    async def to_list(self, length=None):
        documents = self._documents[:length] if length else list(self._documents)
        return [dict(document) for document in documents]


class FakeCollection:
    """Minimal async collection double: exact-match filters only."""

    def __init__(self):
        self.documents = []
        self.index_calls = []

    async def create_index(self, keys, **kwargs):
        self.index_calls.append((keys, kwargs))
        return "index"

    async def insert_one(self, document):
        self.documents.append(dict(document))
        return SimpleNamespace(inserted_id=document.get("document_id"))

    async def find_one(self, filt):
        for document in self.documents:
            if all(document.get(k) == v for k, v in filt.items()):
                return dict(document)
        return None

    def find(self, filt):
        return FakeCursor([
            document for document in self.documents
            if all(document.get(k) == v for k, v in filt.items())
        ])

    async def update_one(self, filt, update):
        matched = 0
        for document in self.documents:
            if all(document.get(k) == v for k, v in filt.items()):
                document.update(update.get("$set", {}))
                matched = 1
                break
        return SimpleNamespace(matched_count=matched)

    async def delete_one(self, filt):
        for index, document in enumerate(self.documents):
            if all(document.get(k) == v for k, v in filt.items()):
                del self.documents[index]
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)


class FailingCollection(FakeCollection):
    async def insert_one(self, document):
        raise PyMongoError("private mongo detail")

    async def find_one(self, filt):
        raise PyMongoError("private mongo detail")

    async def update_one(self, filt, update):
        raise PyMongoError("private mongo detail")

    def find(self, filt):
        raise PyMongoError("private mongo detail")

    async def delete_one(self, filt):
        raise PyMongoError("private mongo detail")


class FailingDeleteCollection(FakeCollection):
    """Mongo dies only at the final record delete: chunks are already cut
    and the file already unlinked, so the record must survive for retry."""

    async def delete_one(self, filt):
        raise PyMongoError("private mongo detail")


def run(coro):
    return asyncio.run(coro)


def user_response(user_id):
    return UserResponse(
        user_id=user_id, username=user_id, created_at=datetime.now(timezone.utc)
    )


TEXT = b"RAG retrieves tenant-scoped context before generation."


class DocumentUploadTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection()
        self.service = DocumentService(self.collection)
        run(self.service.ensure_indexes())
        self.rag_service = Mock()
        self.rag_service.ingest_user_document.return_value = 2
        self.tmp = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.tmp.name)
        self.locks = DocumentLocks()

    def tearDown(self):
        self.tmp.cleanup()

    def client_as(self, user_id, max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES):
        app = FastAPI()
        app.include_router(documents_router)
        app.state.document_service = self.service
        app.state.rag_service = self.rag_service
        app.state.upload_dir = self.upload_dir
        app.state.max_upload_bytes = max_upload_bytes
        app.state.document_locks = self.locks

        async def fake_current_user():
            return user_response(user_id)

        app.dependency_overrides[get_current_user] = fake_current_user
        return TestClient(app)

    def upload(self, client, filename="notes.txt", content=TEXT,
               content_type="text/plain"):
        return client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, content, content_type)},
        )

    def worker_kwargs(self, document_id, user_id="alice"):
        return {
            "service": self.service,
            "rag_service": self.rag_service,
            "upload_dir": self.upload_dir,
            "locks": self.locks,
            "document_id": document_id,
            "user_id": user_id,
            "max_upload_bytes": DEFAULT_MAX_UPLOAD_BYTES,
        }

    def run_worker(self, document_id, user_id="alice"):
        """Complete the enqueued job deterministically, whatever the test
        client already ran inline (re-ingest is idempotent per document)."""
        return run(process_document_upload(
            **self.worker_kwargs(document_id, user_id)
        ))

    # -- happy path --------------------------------------------------

    def test_upload_accepts_and_schedules_background_ingestion(self):
        with patch(
            "app.routes.documents.process_document_upload", new=AsyncMock()
        ) as worker:
            response = self.upload(self.client_as("alice"))
        # 202, not 201: readiness now arrives via GET polling, not the
        # upload response. Ownership still comes only from the JWT.
        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        self.assertEqual(body["user_id"], "alice")
        self.assertEqual(body["original_filename"], "notes.txt")
        self.assertEqual(body["status"], "processing")
        self.assertEqual(body["chunk_count"], 0)
        self.assertTrue(body["document_id"].startswith("notes-"))
        self.assertEqual(body["title"], "Notes")

        # Slow embedding work was enqueued, not executed inline.
        worker.assert_awaited_once()
        scheduled = worker.await_args.kwargs
        self.assertEqual(scheduled["document_id"], body["document_id"])
        self.assertEqual(scheduled["user_id"], "alice")
        self.assertEqual(
            scheduled["max_upload_bytes"], DEFAULT_MAX_UPLOAD_BYTES
        )
        self.assertIs(scheduled["service"], self.service)
        self.assertIs(scheduled["locks"], self.locks)

        # Bytes + processing record are already durable for the worker.
        stored = list((self.upload_dir / "alice").glob("*.txt"))
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].name, f"{body['document_id']}.txt")
        self.assertEqual(stored[0].read_bytes(), TEXT)
        stored_record = run(
            self.service.get_document(body["document_id"], "alice")
        )
        self.assertEqual(stored_record["status"], "processing")

    def test_background_worker_indexes_and_marks_ready(self):
        response = self.upload(self.client_as("alice"))
        body = response.json()
        self.run_worker(body["document_id"])

        # Chroma handoff preserves the tenant filter invariant.
        self.rag_service.ingest_user_document.assert_called()
        rag_document, user_id = (
            self.rag_service.ingest_user_document.call_args.args
        )
        self.assertEqual(user_id, "alice")
        self.assertEqual(rag_document.document_id, body["document_id"])
        self.assertEqual(rag_document.text, TEXT.decode("utf-8"))

        stored_record = run(
            self.service.get_document(body["document_id"], "alice")
        )
        self.assertEqual(stored_record["status"], "ready")
        self.assertEqual(stored_record["chunk_count"], 2)

    def test_same_filename_by_two_users_stays_isolated(self):
        alice_body = self.upload(self.client_as("alice")).json()
        bob_body = self.upload(self.client_as("bob")).json()
        self.assertNotEqual(alice_body["document_id"], bob_body["document_id"])
        self.assertTrue((self.upload_dir / "alice").exists())
        self.assertTrue((self.upload_dir / "bob").exists())
        # Bob's scoped lookup cannot see Alice's record.
        self.assertIsNone(run(
            self.service.get_document(alice_body["document_id"], "bob")
        ))

    def test_client_mime_type_is_recorded_but_never_trusted(self):
        # Browsers may send octet-stream for .txt; decoding decides.
        response = self.upload(
            self.client_as("alice"), content_type="application/octet-stream"
        )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(
            response.json()["content_type"], "application/octet-stream"
        )

    # -- authentication ----------------------------------------------

    def test_unauthenticated_upload_is_rejected_before_any_work(self):
        app = FastAPI()
        app.include_router(documents_router)
        app.state.document_service = self.service
        app.state.rag_service = self.rag_service
        app.state.upload_dir = self.upload_dir
        app.state.max_upload_bytes = DEFAULT_MAX_UPLOAD_BYTES
        response = TestClient(app).post(
            "/api/v1/documents/upload",
            files={"file": ("notes.txt", TEXT, "text/plain")},
        )
        self.assertEqual(response.status_code, 401)
        self.rag_service.ingest_user_document.assert_not_called()
        self.assertEqual(self.collection.documents, [])
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    # -- validation ---------------------------------------------------

    def test_empty_file_is_rejected(self):
        response = self.upload(self.client_as("alice"), content=b"")
        self.assertEqual(response.status_code, 400)
        self.rag_service.ingest_user_document.assert_not_called()
        self.assertEqual(self.collection.documents, [])

    def test_whitespace_only_file_is_rejected(self):
        response = self.upload(self.client_as("alice"), content=b"  \n\n ")
        self.assertEqual(response.status_code, 400)
        self.rag_service.ingest_user_document.assert_not_called()

    def test_unsupported_extension_is_rejected(self):
        response = self.upload(
            self.client_as("alice"), filename="report.pdf", content=b"%PDF fake"
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn(".txt", response.json()["detail"])
        self.rag_service.ingest_user_document.assert_not_called()

    def test_path_traversal_filename_is_rejected(self):
        for name in ("../evil.txt", "..\\evil.txt", "/abs/evil.txt", ".."):
            with self.subTest(name=name):
                response = self.upload(self.client_as("alice"), filename=name)
                self.assertIn(response.status_code, (400, 415), response.text)
        self.rag_service.ingest_user_document.assert_not_called()
        self.assertFalse((self.upload_dir / "evil.txt").exists())

    def test_non_utf8_bytes_are_rejected(self):
        response = self.upload(
            self.client_as("alice"), content=b"\xff\xfe\x00bad bytes"
        )
        self.assertEqual(response.status_code, 400)
        self.rag_service.ingest_user_document.assert_not_called()

    def test_oversize_upload_is_rejected(self):
        response = self.upload(
            self.client_as("alice", max_upload_bytes=10),
            content=b"0123456789X",
        )
        self.assertEqual(response.status_code, 413)
        self.rag_service.ingest_user_document.assert_not_called()
        self.assertEqual(self.collection.documents, [])

    # -- failure semantics --------------------------------------------

    def test_ingestion_failure_marks_failed_without_leaking(self):
        self.rag_service.ingest_user_document.side_effect = RuntimeError(
            "private embedding detail"
        )
        response = self.upload(self.client_as("alice"))
        self.assertEqual(response.status_code, 202, response.text)
        # The 202 stays honest: failure surfaces on the record the client
        # polls, never as leaked internals (the worker only logs).
        with self.assertLogs("app.services.document_jobs", level="ERROR"):
            self.run_worker(response.json()["document_id"])
        record = run(self.service.get_document(
            response.json()["document_id"], "alice"
        ))
        self.assertEqual(record["status"], "failed")

    def test_mongo_outage_returns_503_without_tokens_or_leak(self):
        self.service = DocumentService(FailingCollection())
        response = self.upload(self.client_as("alice"))
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)
        self.rag_service.ingest_user_document.assert_not_called()

    # -- pure helpers --------------------------------------------------

    def test_indexes_cover_identity_and_owner_listing(self):
        keys = [keys for keys, _ in self.collection.index_calls]
        self.assertIn("document_id", keys)
        self.assertIn([("user_id", 1), ("created_at", -1)], keys)

    def test_document_ids_are_unique_and_filesafe(self):
        first = generate_document_id("My Notes.txt")
        second = generate_document_id("My Notes.txt")
        self.assertNotEqual(first, second)
        for value in (first, second):
            self.assertTrue(value.startswith("my-notes-"))
            self.assertNotIn("/", value)
            self.assertNotIn("..", value)

    def test_filename_validation(self):
        self.assertEqual(validate_filename("notes.txt"), "notes.txt")
        for bad in (None, "", "../x.txt", "a/b.txt", "a\\b.txt", ".."):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidUpload):
                    validate_filename(bad)

    def test_extension_gate(self):
        self.assertEqual(validate_extension("NOTES.TXT"), ".txt")
        with self.assertRaises(InvalidUpload) as raised:
            validate_extension("report.pdf")
        self.assertEqual(raised.exception.status_code, 415)

    def test_decode_gate(self):
        self.assertEqual(validate_and_decode(b"hi", 100), "hi")
        with self.assertRaises(InvalidUpload):
            validate_and_decode(b"", 100)
        with self.assertRaises(InvalidUpload) as raised:
            validate_and_decode(b"x" * 101, 100)
        self.assertEqual(raised.exception.status_code, 413)

    def test_title_derivation(self):
        self.assertEqual(derive_title("my_notes.txt"), "My Notes")
        self.assertEqual(generate_document_id("x.txt")[:2], "x-")

    # -- Phase 3.8 worker ----------------------------------------------

    def write_stored_file(self, document_id, user_id="alice", content=TEXT):
        directory = self.upload_dir / user_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{document_id}.txt"
        path.write_bytes(content)
        return path

    def test_worker_skips_deleted_record_without_ingesting(self):
        record = self.create_record("alice")
        run(self.service.delete_document(record["document_id"], "alice"))
        with self.assertLogs("app.services.document_jobs", level="INFO"):
            self.run_worker(record["document_id"])
        self.rag_service.ingest_user_document.assert_not_called()

    def test_worker_marks_failed_when_file_is_missing(self):
        record = self.create_record("alice")
        self.run_worker(record["document_id"])
        stored = run(
            self.service.get_document(record["document_id"], "alice")
        )
        self.assertEqual(stored["status"], "failed")
        self.rag_service.ingest_user_document.assert_not_called()

    def test_worker_marks_failed_when_stored_bytes_are_rejected(self):
        record = self.create_record("alice")
        self.write_stored_file(record["document_id"], content=b"\xff\xfe bad")
        self.run_worker(record["document_id"])
        stored = run(
            self.service.get_document(record["document_id"], "alice")
        )
        self.assertEqual(stored["status"], "failed")
        self.rag_service.ingest_user_document.assert_not_called()

    def test_worker_compensates_when_record_vanishes_mid_ingest(self):
        record = self.create_record("alice")
        document_id = record["document_id"]
        self.write_stored_file(document_id)

        def ingest_then_delete(document, user_id):
            run(self.service.delete_document(document_id, user_id))
            return 2

        self.rag_service.ingest_user_document.side_effect = ingest_then_delete
        self.run_worker(document_id)
        # Just-indexed chunks are removed instead of orphaned; no ready mark.
        self.rag_service.delete_user_document.assert_called_once_with(
            document_id, "alice"
        )
        self.assertIsNone(run(
            self.service.get_document(document_id, "alice")
        ))

    def test_worker_cancellation_leaves_failed_and_reraises(self):
        record = self.create_record("alice")
        self.write_stored_file(record["document_id"])
        with patch.object(
            self.service, "get_document",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            with self.assertRaises(asyncio.CancelledError):
                self.run_worker(record["document_id"])
        stored = run(
            self.service.get_document(record["document_id"], "alice")
        )
        self.assertEqual(stored["status"], "failed")

    def test_document_locks_serialize_and_release(self):
        async def exercise():
            async with self.locks.hold("doc-1"):
                self.assertIn("doc-1", self.locks._entries)
                async def contender():
                    async with self.locks.hold("doc-1"):
                        return "second"
                task = asyncio.create_task(contender())
                await asyncio.sleep(0.05)
                self.assertFalse(task.done())
            self.assertEqual(await task, "second")
            self.assertNotIn("doc-1", self.locks._entries)

        run(exercise())

    def test_requeue_replays_stale_processing_rows(self):
        stale = self.create_record("alice", "stale.txt")
        self.write_stored_file(stale["document_id"])
        done = self.create_record("alice", "done.txt")
        run(self.service.mark_ready(done["document_id"], "alice", 1))

        async def requeue_and_wait():
            count = await requeue_stale_uploads(
                service=self.service,
                rag_service=self.rag_service,
                upload_dir=self.upload_dir,
                locks=self.locks,
                max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES,
            )
            for _ in range(500):
                record = await self.service.get_document(
                    stale["document_id"], "alice"
                )
                if record is not None and record["status"] == "ready":
                    return count
                await asyncio.sleep(0.01)
            raise AssertionError("requeued worker did not finish")

        self.assertEqual(run(requeue_and_wait()), 1)
        self.assertEqual(
            run(self.service.get_document(done["document_id"], "alice"))["status"],
            "ready",
        )

    def test_stored_path_for_rebuilds_from_record(self):
        record = self.create_record("alice", "My Notes.txt")
        path = stored_path_for(self.upload_dir, "alice", record)
        self.assertEqual(
            path, self.upload_dir / "alice" / f"{record['document_id']}.txt"
        )

    # -- Phase 3.7 helpers ---------------------------------------------

    def create_record(self, user_id, filename="notes.txt", created_at=None):
        record = run(self.service.create_processing_document(
            user_id=user_id,
            document_id=generate_document_id(filename),
            original_filename=filename,
            content_type="text/plain",
            size_bytes=len(TEXT),
            title="Notes",
        ))
        if created_at is not None:
            for stored in self.collection.documents:
                if stored["document_id"] == record["document_id"]:
                    stored["created_at"] = created_at
        return record

    def upload_record(self, user_id="alice", filename="notes.txt"):
        """POST then drive the real worker to completion: the returned
        record is `ready` no matter what the test client ran inline."""
        response = self.upload(self.client_as(user_id), filename=filename)
        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        self.run_worker(body["document_id"], user_id)
        record = run(self.service.get_document(body["document_id"], user_id))
        self.assertEqual(record["status"], "ready")
        return record

    # -- list ----------------------------------------------------------

    def test_list_returns_only_caller_documents_newest_first(self):
        base = datetime.now(timezone.utc)
        older = self.create_record(
            "alice", "older.txt", created_at=base - timedelta(seconds=60)
        )
        newer = self.create_record("alice", "newer.txt", created_at=base)
        self.create_record("bob", "bob.txt", created_at=base)

        response = self.client_as("alice").get("/api/v1/documents")
        self.assertEqual(response.status_code, 200, response.text)
        ids = [item["document_id"] for item in response.json()]
        self.assertEqual(ids, [newer["document_id"], older["document_id"]])

        bob_response = self.client_as("bob").get("/api/v1/documents")
        self.assertEqual(len(bob_response.json()), 1)

    def test_list_limit_is_respected(self):
        for index in range(3):
            self.create_record("alice", f"doc{index}.txt")
        response = self.client_as("alice").get(
            "/api/v1/documents", params={"limit": 2}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 2)

    def test_list_requires_authentication(self):
        app = FastAPI()
        app.include_router(documents_router)
        app.state.document_service = self.service
        response = TestClient(app).get("/api/v1/documents")
        self.assertEqual(response.status_code, 401)

    def test_list_mongo_outage_returns_503(self):
        self.service = DocumentService(FailingCollection())
        response = self.client_as("alice").get("/api/v1/documents")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)

    # -- get -----------------------------------------------------------

    def test_get_returns_own_metadata(self):
        uploaded = self.upload_record()
        response = self.client_as("alice").get(
            f"/api/v1/documents/{uploaded['document_id']}"
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["document_id"], uploaded["document_id"])
        self.assertEqual(body["user_id"], "alice")
        self.assertEqual(body["status"], "ready")

    def test_get_unknown_or_foreign_id_returns_identical_404(self):
        uploaded = self.upload_record()
        for client, document_id in (
            (self.client_as("alice"), "no-such-doc-12345678"),
            (self.client_as("bob"), uploaded["document_id"]),
        ):
            with self.subTest(document_id=document_id):
                response = client.get(f"/api/v1/documents/{document_id}")
                self.assertEqual(response.status_code, 404)
                self.assertEqual(
                    response.json(), {"detail": "Document not found."}
                )

    def test_get_malformed_id_returns_404_without_touching_stores(self):
        uploaded = self.upload_record()
        before = len(self.rag_service.mock_calls)
        for bad in ("../evil.txt", "a/b", "..", "has space", "x" * 200):
            with self.subTest(bad=bad):
                response = self.client_as("alice").get(
                    f"/api/v1/documents/{bad}"
                )
                self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(len(self.rag_service.mock_calls), before)
        # The real record is untouched and still readable.
        self.assertEqual(
            self.client_as("alice").get(
                f"/api/v1/documents/{uploaded['document_id']}"
            ).status_code, 200,
        )

    def test_get_requires_authentication(self):
        uploaded = self.upload_record()
        app = FastAPI()
        app.include_router(documents_router)
        app.state.document_service = self.service
        response = TestClient(app).get(
            f"/api/v1/documents/{uploaded['document_id']}"
        )
        self.assertEqual(response.status_code, 401)

    # -- delete --------------------------------------------------------

    def test_delete_removes_chunks_file_and_record(self):
        uploaded = self.upload_record()
        document_id = uploaded["document_id"]
        stored = self.upload_dir / "alice" / f"{document_id}.txt"
        self.assertTrue(stored.exists())

        response = self.client_as("alice").delete(
            f"/api/v1/documents/{document_id}"
        )
        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(response.content, b"")

        # Tenant-scoped Chroma handoff: exactly this user's document.
        self.rag_service.delete_user_document.assert_called_once_with(
            document_id, "alice"
        )
        self.assertFalse(stored.exists())
        self.assertIsNone(run(
            self.service.get_document(document_id, "alice")
        ))
        # Repeat and read-after-delete agree: gone means 404.
        self.assertEqual(
            self.client_as("alice").delete(
                f"/api/v1/documents/{document_id}"
            ).status_code, 404,
        )
        self.assertEqual(
            self.client_as("alice").get(
                f"/api/v1/documents/{document_id}"
            ).status_code, 404,
        )

    def test_delete_foreign_document_has_no_side_effects(self):
        uploaded = self.upload_record()
        document_id = uploaded["document_id"]
        stored = self.upload_dir / "alice" / f"{document_id}.txt"

        response = self.client_as("bob").delete(
            f"/api/v1/documents/{document_id}"
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Document not found."})
        self.rag_service.delete_user_document.assert_not_called()
        self.assertTrue(stored.exists())
        self.assertIsNotNone(run(
            self.service.get_document(document_id, "alice")
        ))
        # Alice herself is unaffected.
        self.assertEqual(
            self.client_as("alice").get(
                f"/api/v1/documents/{document_id}"
            ).status_code, 200,
        )

    def test_delete_unknown_or_malformed_id_returns_404(self):
        for document_id in ("no-such-doc-12345678", "../evil.txt"):
            with self.subTest(document_id=document_id):
                self.assertEqual(
                    self.client_as("alice").delete(
                        f"/api/v1/documents/{document_id}"
                    ).status_code, 404,
                )
        self.rag_service.delete_user_document.assert_not_called()

    def test_delete_requires_authentication(self):
        uploaded = self.upload_record()
        app = FastAPI()
        app.include_router(documents_router)
        app.state.document_service = self.service
        app.state.rag_service = self.rag_service
        app.state.upload_dir = self.upload_dir
        response = TestClient(app).delete(
            f"/api/v1/documents/{uploaded['document_id']}"
        )
        self.assertEqual(response.status_code, 401)
        self.rag_service.delete_user_document.assert_not_called()

    def test_delete_chroma_failure_keeps_record_and_file(self):
        uploaded = self.upload_record()
        document_id = uploaded["document_id"]
        stored = self.upload_dir / "alice" / f"{document_id}.txt"
        self.rag_service.delete_user_document.side_effect = RuntimeError(
            "private chroma detail"
        )
        with self.assertLogs("app.routes.documents", level="ERROR"):
            response = self.client_as("alice").delete(
                f"/api/v1/documents/{document_id}"
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(), {"detail": "The document could not be deleted."}
        )
        self.assertNotIn("private", response.text)
        # Nothing else touched: retry stays possible.
        self.assertTrue(stored.exists())
        self.assertIsNotNone(run(
            self.service.get_document(document_id, "alice")
        ))

    def test_delete_mongo_failure_reports_503_after_chunks_removed(self):
        uploaded = self.upload_record()
        document_id = uploaded["document_id"]
        rag_calls_before = len(self.rag_service.mock_calls)
        self.service = DocumentService(FailingDeleteCollection())
        # Seed the failing double with the live record (same document_id).
        live = next(
            d for d in self.collection.documents
            if d["document_id"] == document_id
        )
        self.service.documents.documents.append(dict(live))
        response = self.client_as("alice").delete(
            f"/api/v1/documents/{document_id}"
        )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)
        self.assertEqual(
            len(self.rag_service.mock_calls), rag_calls_before + 1
        )
        # Record survives so the delete can be retried.
        self.assertIsNotNone(run(
            self.service.get_document(document_id, "alice")
        ))

    def test_delete_missing_file_still_succeeds(self):
        uploaded = self.upload_record()
        document_id = uploaded["document_id"]
        (self.upload_dir / "alice" / f"{document_id}.txt").unlink()
        response = self.client_as("alice").delete(
            f"/api/v1/documents/{document_id}"
        )
        self.assertEqual(response.status_code, 204, response.text)
        self.assertIsNone(run(
            self.service.get_document(document_id, "alice")
        ))


if __name__ == "__main__":
    unittest.main()
