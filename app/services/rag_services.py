import logging

from phase1 import Document, RAGSystem

from app.schemas.chat import ChatRequest


logger = logging.getLogger(__name__)


class RAGService:

    def __init__(
        self,
        rag: RAGSystem
    ):
        self.rag = rag


    def run_once(
        self,
        request: ChatRequest
    ) -> dict:

        chat_history = [
            message.model_dump()
            for message in request.chat_history
        ]

        return self.rag.run_once(
            raw_query=request.raw_query,
            user_id=request.user_id,
            chat_history=chat_history,
            k=request.k,
            rewrite_query=request.rewrite_query,
            distance_threshold=request.distance_threshold
        )


    def run_once_stream(
        self,
        request: ChatRequest
    ):

        chat_history = [
            message.model_dump()
            for message in request.chat_history
        ]

        return self.rag.run_once_stream(
            raw_query=request.raw_query,
            user_id=request.user_id,
            chat_history=chat_history,
            k=request.k,
            rewrite_query=request.rewrite_query,
            distance_threshold=request.distance_threshold
        )


    def run_once_event_stream(
        self,
        request: ChatRequest
    ):

        chat_history = [
            message.model_dump()
            for message in request.chat_history
        ]

        return self.rag.run_once_event_stream(
            raw_query=request.raw_query,
            user_id=request.user_id,
            chat_history=chat_history,
            k=request.k,
            rewrite_query=request.rewrite_query,
            distance_threshold=request.distance_threshold
        )

    def ingest_user_document(
        self,
        document: Document,
        user_id: str
    ) -> int:
        """Index one tenant-owned upload; returns indexed chunk count.

        Additive Phase 3.6 helper: existing run_once* methods are
        untouched. Ownership flows from the authenticated caller through
        ``user_id`` into every chunk's metadata, preserving the
        ``where={"user_id": ...}`` retrieval isolation.
        """
        self.rag.ingest_documents(documents=[document], user_id=user_id)
        try:
            stored = self.rag.collection.get(
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"document_id": document.document_id}
                    ]
                },
                include=[],
            )
            return len(stored.get("ids", []))
        except Exception:
            logger.exception("Could not count indexed chunks")
            return 0

    def delete_user_document(
        self,
        document_id: str,
        user_id: str
    ) -> None:
        """Remove every chunk of one tenant-owned document.

        Additive Phase 3.7 helper. The filter mirrors ``ingest_documents``
        exactly (``$and`` on ``user_id`` + ``document_id``) so deletion can
        never touch another tenant's chunks -- or another document's.
        """
        self.rag.collection.delete(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"document_id": document_id}
                ]
            }
        )
