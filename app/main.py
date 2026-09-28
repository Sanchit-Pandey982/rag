from contextlib import asynccontextmanager
import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pymongo import AsyncMongoClient
import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from phase1 import RAGSystem, load_txt_documents

from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.routes.conversations import router as conversations_router
from app.routes.documents import router as documents_router
from app.security.jwt import JWTService
from app.security.cookies import RefreshCookieSettings
from app.services.auth_service import AuthService
from app.services.conversation_service import ConversationService
from app.services.document_jobs import requeue_stale_uploads
from app.services.document_service import (
    DEFAULT_MAX_UPLOAD_BYTES,
    DocumentLocks,
    DocumentService,
)
from app.services.rag_services import RAGService
from app.services.rate_limit_service import RateLimitService
from app.services.refresh_token_service import RefreshTokenService


logger = logging.getLogger(__name__)


def directory_accepts_writes(path: Path) -> bool:
    probe = path / ".write_probe"

    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        return True
    except OSError:
        return False
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass


def get_chroma_path(project_root: Path) -> Path:
    primary_path = project_root / "chroma_data"

    if directory_accepts_writes(primary_path):
        return primary_path

    fallback_path = project_root / "runtime_chroma_data"
    fallback_path.mkdir(parents=True, exist_ok=True)
    return fallback_path


def get_upload_dir(project_root: Path) -> Path:
    """User-owned upload root; relative UPLOAD_DIR resolves under the
    project like the Chroma path so dev and server layouts agree."""
    configured = Path(os.getenv("UPLOAD_DIR", "uploads")).expanduser()
    if not configured.is_absolute():
        configured = project_root / configured
    configured.mkdir(parents=True, exist_ok=True)
    return configured.resolve()


def get_max_upload_bytes() -> int:
    raw = os.getenv("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("MAX_UPLOAD_BYTES must be a positive integer") from error
    if value <= 0:
        raise ValueError("MAX_UPLOAD_BYTES must be a positive integer")
    return value


@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    # Fail before opening external resources if JWT configuration is invalid.
    app.state.ready = False
    app.state.jwt_service = JWTService.from_environment()
    app.state.refresh_cookie_settings = RefreshCookieSettings.from_environment()
    # One MongoDB client per application lifespan, shared by all auth requests.
    # Configure these in the environment or the existing project .env file.
    mongo_client = AsyncMongoClient(
        os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        serverSelectionTimeoutMS=5_000,
        tz_aware=True,
    )
    rag = None
    redis_client = None

    try:
        redis_client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            # GETDEL cannot be transparently retried after an ambiguous timeout.
            retry=Retry(NoBackoff(), 0),
        )
        await redis_client.ping()
        app.state.redis_client = redis_client
        app.state.refresh_token_service = RefreshTokenService(redis_client)
        # Phase 3.9: same client, separate `rate_limit:` namespace.
        app.state.rate_limit_service = RateLimitService(redis_client)

        await mongo_client.admin.command("ping")
        database = mongo_client[os.getenv("MONGODB_DATABASE", "agent")]
        auth_service = AuthService(database["users"])
        await auth_service.ensure_indexes()
        app.state.auth_service = auth_service

        # Phase 3.5: reuse the same long-lived client/database for chat history.
        conversation_service = ConversationService(
            database["conversations"], database["messages"]
        )
        await conversation_service.ensure_indexes()
        app.state.conversation_service = conversation_service

        # Phase 3.6: tenant-owned document metadata lives beside it.
        document_service = DocumentService(database["documents"])
        await document_service.ensure_indexes()
        app.state.document_service = document_service
        # Phase 3.8: in-process delete-vs-ingest mutexes (single process).
        app.state.document_locks = DocumentLocks()

        project_root = Path(__file__).resolve().parents[1]
        app.state.upload_dir = get_upload_dir(project_root)
        app.state.max_upload_bytes = get_max_upload_bytes()

        # Preserve existing RAG construction, ingestion, and readiness behavior.
        rag = RAGSystem(
            collection_name="learning_rag",
            chroma_path=str(get_chroma_path(project_root)),
            reset=False
        )

        if rag.collection.count() == 0:
            documents = load_txt_documents(project_root / "data")
            rag.ingest_documents(documents, user_id="eval_user")

        app.state.rag_service = RAGService(rag=rag)
        app.state.ready = rag.collection.count() > 0

        # Phase 3.8 crash recovery: background jobs die with the process,
        # so rows still stuck at `processing` are re-enqueued here. Re-ingest
        # is idempotent per document (prior chunks are deleted first), which
        # is what makes a blind replay safe. Recovery is best-effort by
        # design: it must never fail startup -- the rows keep their honest
        # status for the next restart either way.
        try:
            await requeue_stale_uploads(
                service=document_service,
                rag_service=app.state.rag_service,
                upload_dir=app.state.upload_dir,
                locks=app.state.document_locks,
                max_upload_bytes=app.state.max_upload_bytes,
            )
        except Exception:
            logger.exception("Could not requeue stale uploads")

        yield
    finally:
        app.state.ready = False
        try:
            if rag is not None:
                close = getattr(rag.client, "close", None)
                if callable(close):
                    close()
        finally:
            try:
                if redis_client is not None:
                    await redis_client.aclose()
            finally:
                # Close every resource even if another cleanup or startup failed.
                await mongo_client.close()


app = FastAPI(
    title="RAG Learning API",
    version="0.1.0",
    lifespan=lifespan
)


app.include_router(
    chat_router
)
app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(documents_router)


@app.get("/health")
def health():
    # Fix 5: retain a cheap liveness endpoint; readiness is reported separately.
    return {
        "status": "ok"
    }


@app.get("/ready")
def ready():
    # Fix 1/5: prevent load balancers from routing traffic to an empty store.
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="service not ready")
    return {"status": "ready"}
