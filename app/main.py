from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from phase1 import RAGSystem, load_txt_documents

from app.routes.chat import router as chat_router
from app.services.rag_services import RAGService


@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    # Fix 1/6: resolve project data from this file, ingest a missing corpus, and
    # expose readiness state instead of depending on the process working dir.
    project_root = Path(__file__).resolve().parents[1]
    rag = RAGSystem(
        collection_name="learning_rag",
        chroma_path=str(project_root / "chroma_data"),
        reset=False
    )

    if rag.collection.count() == 0:
        documents = load_txt_documents(project_root / "data")
        rag.ingest_documents(documents, user_id="eval_user")

    app.state.rag_service = RAGService(
        rag=rag
    )
    app.state.ready = rag.collection.count() > 0

    try:
        yield
    finally:
        # Fix 11: make lifespan ownership explicit and close clients that expose
        # a supported close API without assuming Chroma has one.
        close = getattr(rag.client, "close", None)
        if callable(close):
            close()


app = FastAPI(
    title="RAG Learning API",
    version="0.1.0",
    lifespan=lifespan
)


app.include_router(
    chat_router
)


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
