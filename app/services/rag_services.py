import logging

from phase1 import Document, RAGSystem

from app.schemas.chat import ChatRequest


logger = logging.getLogger(__name__)


class RAGService:

    def __init__(
        self,
        rag: RAGSystem,
        hybrid_retriever=None,
        reranker=None,
        response_cache=None,
        degradation_tracker=None,
    ):
        self.rag = rag
        # Long-lived per-user BM25 cache lives here (not on the
        # short-lived RAGSystem helper) so repeated chat queries reuse
        # the index. Lazy: importing retrieval needs no credentials and
        # constructing HybridRetriever builds no index until first use.
        if hybrid_retriever is None:
            try:
                from app.services.retrieval import HybridRetriever
                hybrid_retriever = HybridRetriever(rag)
            except Exception:
                logger.exception("Hybrid retriever unavailable")
                hybrid_retriever = None
        self.hybrid_retriever = hybrid_retriever
        # Phase 3a: the cross-encoder model loads once, on first reranked
        # query -- never at startup. Import-guarded so vector-only
        # deployments need no sentence-transformers installed.
        if reranker is None:
            try:
                from app.services.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
            except Exception:
                logger.exception("Reranker unavailable")
                reranker = None
        self.reranker = reranker
        # Phase 5: the response cache is a constructed dependency, like
        # the hybrid retriever above -- but unlike those, it cannot be
        # built here (it needs the Redis client), so None simply means
        # uncached. Lifespan wires the real one; tests inject fakes.
        self.response_cache = response_cache
        # Phase 7: Redis counter for served degraded answers. None means
        # serve-but-don't-count; lifespan wires the real one.
        self.degradation_tracker = degradation_tracker

    def invalidate_hybrid_cache(self, user_id: str) -> None:
        """Drop one user's BM25 index; called after ingest/delete."""
        try:
            if self.hybrid_retriever is not None:
                self.hybrid_retriever.invalidate_user(user_id)
        except Exception:
            logger.exception("Could not invalidate hybrid cache")

    def invalidate_response_cache(self, user_id: str) -> None:
        """Orphan one user's cached answers; called after ingest/delete."""
        try:
            if self.response_cache is not None:
                self.response_cache.invalidate_user(user_id)
        except Exception:
            logger.exception("Could not invalidate response cache")

    def run_once(
        self,
        request: ChatRequest,
        use_hybrid: bool | None = None,
        use_rerank: bool | None = None,
        use_cache: bool | None = None,
        use_degraded: bool | None = None,
        use_agent: bool | None = None,
    ) -> dict:

        chat_history = [
            message.model_dump()
            for message in request.chat_history
        ]

        # Phase 11: opt-in agent loop (flag-gated, default off). Any
        # agent failure falls back to the legacy path below, so enabling
        # the flag can never break a chat turn. Streams keep the legacy
        # path (retries cannot replay mid-stream).
        try:
            from app.agents.graph import run_agent
            agent_result = run_agent(
                self,
                raw_query=request.raw_query,
                user_id=request.user_id,
                chat_history=chat_history,
                k=request.k,
                rewrite_query=request.rewrite_query,
                distance_threshold=request.distance_threshold,
                use_agent=use_agent,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                use_degraded=use_degraded,
                degraded_tracker=self.degradation_tracker,
            )
        except Exception:
            logger.exception("Agent path failed; using legacy pipeline")
            agent_result = None
        if agent_result is not None:
            return agent_result

        return self.rag.run_once(
            raw_query=request.raw_query,
            user_id=request.user_id,
            chat_history=chat_history,
            k=request.k,
            rewrite_query=request.rewrite_query,
            distance_threshold=request.distance_threshold,
            use_hybrid=use_hybrid,
            hybrid_retriever=self.hybrid_retriever,
            use_rerank=use_rerank,
            reranker=self.reranker,
            use_cache=use_cache,
            response_cache=self.response_cache,
            use_degraded=use_degraded,
            degraded_tracker=self.degradation_tracker,
        )


    def run_once_stream(
        self,
        request: ChatRequest,
        use_hybrid: bool | None = None,
        use_rerank: bool | None = None,
        use_cache: bool | None = None,
        use_degraded: bool | None = None,
        usage: dict | None = None,
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
            distance_threshold=request.distance_threshold,
            use_hybrid=use_hybrid,
            hybrid_retriever=self.hybrid_retriever,
            use_rerank=use_rerank,
            reranker=self.reranker,
            use_cache=use_cache,
            response_cache=self.response_cache,
            use_degraded=use_degraded,
            degraded_tracker=self.degradation_tracker,
            usage=usage,
        )


    def run_once_event_stream(
        self,
        request: ChatRequest,
        use_hybrid: bool | None = None,
        use_rerank: bool | None = None,
        use_cache: bool | None = None,
        use_degraded: bool | None = None,
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
            distance_threshold=request.distance_threshold,
            use_hybrid=use_hybrid,
            hybrid_retriever=self.hybrid_retriever,
            use_rerank=use_rerank,
            reranker=self.reranker,
            use_cache=use_cache,
            response_cache=self.response_cache,
            use_degraded=use_degraded,
            degraded_tracker=self.degradation_tracker,
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
        # Phase 2: new chunks change the BM25 corpus, so drop the cached
        # index; it rebuilds lazily on the next hybrid query. Runs even
        # when the chunk count below fails -- the corpus already changed.
        self.invalidate_hybrid_cache(user_id)
        # Phase 5: new chunks change future answers, so orphan this
        # user's cached responses (generation bump, entries expire).
        self.invalidate_response_cache(user_id)
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
        # Phase 2: removals change the BM25 corpus; rebuild on next query.
        # Invalidate even on success only -- a failed delete changed nothing.
        self.invalidate_hybrid_cache(user_id)
        # Phase 5: removals change future answers; orphan cached responses.
        self.invalidate_response_cache(user_id)
