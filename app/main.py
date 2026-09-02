from contextlib import asynccontextmanager

from fastapi import FastAPI

from phase1 import RAGSystem

from app.routes.chat import router as chat_router
from app.services.rag_services import RAGService


@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    rag = RAGSystem(
        collection_name="learning_rag",
        chroma_path="./chroma_data",
        reset=False
    )

    app.state.rag_service = RAGService(
        rag=rag
    )

    yield


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

    return {
        "status": "ok"
    }