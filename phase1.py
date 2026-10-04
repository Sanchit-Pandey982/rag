from __future__ import annotations

import logging
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import chromadb

from dotenv import load_dotenv

from google import genai
from google.genai import types

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from app.utils.retry import (
    CircuitBreakerOpen,
    is_transient_error,
    retry_with_backoff,
)


# ============================================================
# Configuration
# ============================================================

load_dotenv()
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent

# Fix 10: defer external-client construction until a request/explicit operation
# needs it, so importing this module does not require credentials or networking.
gemini_client = None
GEMINI_TIMEOUT_MS = 60_000


def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        # The SDK expects milliseconds; this applies to all Gemini requests.
        gemini_client = genai.Client(
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)
        )
    return gemini_client

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-3.6-flash"

REFUSAL_MESSAGE = "I do not have enough information to answer that."


# ============================================================
# Data models
# ============================================================

@dataclass
class Document:
    document_id: str
    source: str
    text: str
    title: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    distance: float
    metadata: dict


@dataclass
class EvalCase:
    id: str
    user_id: str
    question: str

    # Is the answer supposed to exist in our documents?
    answerable: bool

    # Documents which should contain the answer
    expected_document_ids: list[str]

    # Simple deterministic answer evaluation
    required_answer_terms: list[str]

    expected_answer: str = ""

    # Useful for testing conversational queries
    history: list[dict] | None = None


# ============================================================
# Embeddings
# ============================================================

@traceable(run_type="embedding", name="embed_documents")
@retry_with_backoff(circuit="gemini")
def embed_documents(texts: list[str]) -> list[list[float]]:

    if not texts:
        return []

    result = get_gemini_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    return [
        embedding.values
        for embedding in result.embeddings
    ]


@traceable(run_type="embedding", name="embed_query")
@retry_with_backoff(circuit="gemini")
def embed_query(text: str) -> list[float]:

    result = get_gemini_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    return result.embeddings[0].values


# ============================================================
# Load documents
# ============================================================

def load_txt_documents(folder: str) -> list[Document]:

    # Fix 6: accept Path values and resolve the caller's configured location
    # once, rather than silently depending on the current working directory.
    root = Path(folder).expanduser().resolve()

    documents = []

    for path in root.rglob("*.txt"):

        relative_path = path.relative_to(root)

        # data/rag_basics.txt
        # becomes:
        # rag_basics
        document_id = (
            relative_path
            .with_suffix("")
            .as_posix()
            .replace("/", "__")
        )

        # Fix 3: normalize the historical file-name typo to the canonical ID
        # used by eval_cases.json and downstream dashboards.
        if document_id == "embeedings":
            document_id = "embeddings"

        text = path.read_text(encoding="utf-8")

        documents.append(
            Document(
                document_id=document_id,
                source=str(relative_path),
                title=path.stem.replace("_", " ").title(),
                text=text
            )
        )

    return documents


# ============================================================
# Question rewriting
# ============================================================

@traceable(run_type="llm", name="condense_question")
@retry_with_backoff(circuit="gemini")
def condense_question(
    chat_history: list[dict],
    latest_query: str
) -> str:

    if not chat_history:
        return latest_query

    # Don't send unlimited history.
    recent_history = chat_history[-8:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
    )

    prompt = f"""
Rewrite the latest user question into a standalone question.

Use the conversation only to resolve references such as:
"it", "that", "they", "its price", "what about that?"

Do NOT answer the question.
Do NOT introduce new information.

Conversation:
{history_text}

Latest question:
{latest_query}

Standalone question:
"""

    response = get_gemini_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    if not response.text:
        return latest_query

    return response.text.strip()


# ============================================================
# Conversation summarization
# ============================================================

@traceable(run_type="llm", name="summarize_conversation")
@retry_with_backoff(circuit="gemini")
def summarize_conversation_history(
    messages: list[dict],
    previous_summary: str | None = None,
) -> str:
    """Condense older chat messages into a short factual summary.

    ``messages`` are the oldest-first ``{"role", "content"}`` dicts to fold
    in; ``previous_summary`` is the existing summary to extend, if any.
    Returns "" when there is nothing to summarize or the LLM call fails, so
    callers can fall back to plain truncated history.
    """

    if not messages and not previous_summary:
        return ""

    transcript = "\n".join(
        f"{message['role']}: {message['content'][:2000]}"
        for message in messages[-60:]
    )

    if previous_summary:
        prompt = f"""
Update the conversation summary below so it also covers the new messages.
Keep it short (under 200 words), factual, and in third person.
Do NOT answer any question. Do NOT invent details.

Existing summary:
{previous_summary[:2000]}

New messages:
{transcript}

Updated summary:
"""
    else:
        prompt = f"""
Summarize this conversation in under 200 words: the user's goals,
key facts established, and any decisions or answers given.
Write in third person. Do NOT answer any question.
Do NOT invent details.

Conversation:
{transcript}

Summary:
"""

    try:
        response = get_gemini_client().models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
    except Exception:
        logger.exception("Could not summarize conversation history")
        return ""

    return response.text.strip() if response.text else ""


# ============================================================
# Degraded responses (Phase 7)
# ============================================================

DEGRADED_PREFIX = (
    "I'm experiencing high load. Here's what I found in your documents:"
)


def degraded_mode_enabled() -> bool:
    return os.getenv("DEGRADED_MODE_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def degraded_max_chunks() -> int:
    try:
        value = int(os.getenv("DEGRADED_MAX_CHUNKS", "3"))
    except ValueError:
        return 3
    return value if value > 0 else 3


def is_llm_unavailable(error: BaseException) -> bool:
    """Whether this failure means the LLM (not the request) is down.

    Transient errors after Phase 6 retries are exhausted, plus an open
    Gemini circuit (fail-fast without attempting). Anything else --
    4xx, programming bugs, cancellations -- stays on the error path.
    """
    return is_transient_error(error) or isinstance(
        error, CircuitBreakerOpen)


def build_degraded_answer(
    chunks: list[RetrievedChunk],
    max_chunks: int | None = None,
) -> str:
    """Honest excerpts: the top retrieved chunks as plain text.

    No generation, no claims beyond what retrieval returned -- the
    prefix says exactly what happened.
    """
    limit = max_chunks if max_chunks is not None else degraded_max_chunks()
    lines = [DEGRADED_PREFIX]
    for number, chunk in enumerate(list(chunks)[:max(limit, 0)], start=1):
        lines.append("")
        lines.append(f"[Source {number}]")
        lines.append(f"Document: {chunk.metadata['document_id']}")
        lines.append(f"File: {chunk.metadata['source']}")
        lines.append("")
        lines.append(chunk.text)
    return "\n".join(lines)


# ============================================================
# Token usage (Phase 8)
# ============================================================

def usage_from_metadata(metadata) -> dict | None:
    """Token counts from a Gemini usage-metadata object, else None.

    Completion tokens are ``candidates_token_count``. Counts must be
    real non-negative ints: Mock attributes (never ints) can never
    fabricate ledger rows, and partial metadata is rejected outright.
    """
    if metadata is None:
        return None
    prompt = getattr(metadata, "prompt_token_count", None)
    completion = getattr(metadata, "candidates_token_count", None)
    total = getattr(metadata, "total_token_count", None)
    for value in (prompt, completion, total):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "model": GENERATION_MODEL,
    }


def extract_usage_metadata(response) -> dict | None:
    """Token counts from a Gemini response, or None when absent."""
    return usage_from_metadata(getattr(response, "usage_metadata", None))


# ============================================================
# RAG system
# ============================================================

class RAGSystem:

    def __init__(
        self,
        collection_name: str = "rag_documents",
        chroma_path: str = "./chroma_data",
        reset: bool = False
    ):

        # Fix 6: normalize the persistent-store path at construction time.
        # Fix 6: relative storage paths are anchored to the project module,
        # so Uvicorn and CLI invocations select the same database.
        chroma_path_value = Path(chroma_path).expanduser()
        if not chroma_path_value.is_absolute():
            chroma_path_value = PROJECT_ROOT / chroma_path_value
        chroma_path = str(chroma_path_value.resolve())
        self.client = chromadb.PersistentClient(
            path=chroma_path
        )

        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,

            # Cosine distance instead of default L2
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    # --------------------------------------------------------
    # INGESTION
    # --------------------------------------------------------

    def ingest_documents(
        self,
        documents: list[Document],
        user_id: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        for document in documents:

            chunks = splitter.split_text(document.text)

            if not chunks:
                continue

            # Delete previous version of the document
            self.collection.delete(
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"document_id": document.document_id} 
                    ]
                }
            )

            embeddings = embed_documents(chunks)

            chunk_ids = []
            metadatas = []

            for index, chunk in enumerate(chunks):

                chunk_id = (
                    f"{user_id}:"
                    f"{document.document_id}:"
                    f"{index}"
                )

                chunk_ids.append(chunk_id)

                metadatas.append({
                    "user_id": user_id,
                    "document_id": document.document_id,
                    "source": document.source,
                    "title": document.title,
                    "chunk_index": index
                })

            self.collection.upsert(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    @traceable(run_type="retriever", name="vector_retrieval")
    def retrieve(
        self,
        query: str,
        user_id: str,
        k: int = 3,
        distance_threshold: float | None = None
    ) -> list[RetrievedChunk]:

        # Fix 7: protect non-HTTP callers (CLI/evaluation) with the same limits.
        if not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10")
        if not query or len(query) > 4_000:
            raise ValueError("query must contain 1-4000 characters")
        if distance_threshold is not None and not 0 <= distance_threshold <= 2:
            raise ValueError("distance_threshold must be between 0 and 2")

        query_vector = embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,

            # VERY IMPORTANT for multiple users
            where={
                "user_id": user_id
            },

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances
        ):

            # Lower cosine distance = more similar
            if (
                distance_threshold is not None
                and distance > distance_threshold
            ):
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    distance=distance,
                    metadata=metadata
                )
            )

        return retrieved

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    def build_context(
        self,
        chunks: list[RetrievedChunk]
    ) -> str:

        parts = []

        for number, chunk in enumerate(chunks, start=1):

            parts.append(
                f"""
[Source {number}]
Document: {chunk.metadata["document_id"]}
File: {chunk.metadata["source"]}
Chunk: {chunk.metadata["chunk_index"]}

{chunk.text}
""".strip()
            )
  
        return "\n\n".join(parts)

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    @traceable(run_type="llm", name="generate_answer")
    @retry_with_backoff(circuit="gemini")
    def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        chat_history: list[dict],
        usage: dict | None = None,
    ) -> str:

        # Don't even waste an LLM request if retrieval
        # produced no accepted context.
        if not chunks:
            return REFUSAL_MESSAGE

        context = self.build_context(chunks)

        recent_history = chat_history[-8:]

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent_history
        )

        prompt = f"""
You are a retrieval-augmented assistant.

RULES:

1. Answer using ONLY information supported by the retrieved context.
2. Chat history may help understand the conversation, but it is NOT
   a factual source.
3. Never invent missing details.
4. If the context does not contain enough information, respond exactly:
   "{REFUSAL_MESSAGE}"
5. When possible, cite the source as [Source 1], [Source 2], etc.

Retrieved context:

{context}

Conversation history:

{history_text}

User question:

{question}

Answer:
"""

        response = get_gemini_client().models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        # Phase 8: caller-owned collector (per-request, so concurrent
        # turns cannot clobber each other). Absent/invalid metadata
        # leaves it untouched -- refusals never reach an LLM call.
        if usage is not None:
            report = extract_usage_metadata(response)
            if report is not None:
                usage.update(report)

        return (
            response.text.strip()
            if response.text
            else REFUSAL_MESSAGE
        )

    @traceable(
    run_type="llm",
    name="generate_answer_stream",
    reduce_fn=lambda chunks: "".join(chunks)
    )
    @retry_with_backoff(circuit="gemini")
    def generate_answer_stream(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        chat_history: list[dict],
        usage: dict | None = None,
    ):

        if not chunks:
            yield REFUSAL_MESSAGE
            return

        context = self.build_context(chunks)

        recent_history = chat_history[-8:]

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent_history
        )

        prompt = f"""
    You are a retrieval-augmented assistant.

    RULES:

    1. Answer using ONLY information supported by the retrieved context.
    2. Chat history may help understand the conversation, but it is NOT
    a factual source.
    3. Never invent missing details.
    4. If the context does not contain enough information, respond exactly:
    "{REFUSAL_MESSAGE}"
    5. When possible, cite the source as [Source 1], [Source 2], etc.

    Retrieved context:

    {context}

    Conversation history:

    {history_text}

    User question:

    {question}

    Answer:
    """

        stream = get_gemini_client().models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=prompt
        )

        # Phase 8: usage metadata rides on the final stream chunk, so
        # the last seen report wins. Fills the caller-owned collector
        # only when the stream completes (mid-stream failures keep the
        # error path with nothing recorded).
        last_metadata = None
        for chunk in stream:

            metadata = getattr(chunk, "usage_metadata", None)
            if metadata is not None:
                last_metadata = metadata
            if chunk.text:
                yield chunk.text

        if usage is not None and last_metadata is not None:
            report = usage_from_metadata(last_metadata)
            if report is not None:
                usage.update(report)

    # --------------------------------------------------------
    # COMPLETE PIPELINE
    # --------------------------------------------------------

    # --------------------------------------------------------
    # HYBRID RETRIEVAL (Phase 2)
    # --------------------------------------------------------
    def _retrieve_chunks(
        self,
        query: str,
        user_id: str,
        k: int = 3,
        distance_threshold: float | None = None,
        use_hybrid: bool | None = None,
        hybrid_retriever=None,
        use_rerank: bool | None = None,
        reranker=None,
    ):
        """Vector retrieval, RRF-fused hybrid, then cross-encoder rerank.

        ``use_hybrid=None`` follows the ``HYBRID_ENABLED`` env flag so
        existing callers keep vector-only behavior by default.
        ``hybrid_retriever`` is an injection seam for tests; when omitted
        a short-lived ``HybridRetriever`` wraps this instance (its
        per-user BM25 cache is best-effort here -- the long-lived cache
        lives on ``RAGService``). Any unexpected wrapper failure falls
        back to plain vector retrieval; vector-leg errors propagate.

        ``use_rerank=None`` follows ``RERANKER_ENABLED`` (default off).
        When reranking, candidates are over-fetched (up to 10, the
        Chroma single-query cap) and rescored to the top-k; any rerank
        failure keeps the retrieval order. ``reranker`` is the injection
        seam (long-lived instance lives on ``RAGService``).
        """
        try:
            from app.services.reranker import reranker_enabled
            do_rerank = (
                reranker_enabled() if use_rerank is None else bool(use_rerank)
            )
        except Exception:
            do_rerank = bool(use_rerank) if use_rerank is not None else False
        # Rerank the full candidate window (up to 10), then cut to k.
        fetch_k = min(10, max(k, 10)) if do_rerank else k
        if use_hybrid is False:
            candidates = self.retrieve(
                query=query,
                user_id=user_id,
                k=fetch_k,
                distance_threshold=distance_threshold,
            )
        else:
            try:
                from app.services.retrieval import HybridRetriever
            except Exception:
                candidates = self.retrieve(
                    query=query,
                    user_id=user_id,
                    k=fetch_k,
                    distance_threshold=distance_threshold,
                )
            else:
                retriever = hybrid_retriever
                if retriever is None:
                    if getattr(self, "_hybrid_retriever", None) is None:
                        try:
                            self._hybrid_retriever = HybridRetriever(self)
                        except Exception:
                            candidates = self.retrieve(
                                query=query,
                                user_id=user_id,
                                k=fetch_k,
                                distance_threshold=distance_threshold,
                            )
                            retriever = None
                        else:
                            retriever = self._hybrid_retriever
                    else:
                        retriever = self._hybrid_retriever
                if retriever is None:
                    pass  # candidates already set by the fallback above
                else:
                    try:
                        candidates = retriever.hybrid_retrieve(
                            query=query,
                            user_id=user_id,
                            k=fetch_k,
                            distance_threshold=distance_threshold,
                            use_hybrid=use_hybrid,
                        )
                    except ValueError:
                        raise
                    except Exception:
                        logger.exception(
                            "Hybrid retrieval failed; using vector results")
                        candidates = self.retrieve(
                            query=query,
                            user_id=user_id,
                            k=fetch_k,
                            distance_threshold=distance_threshold,
                        )
        if not do_rerank:
            return candidates
        try:
            from app.services.reranker import CrossEncoderReranker
        except Exception:
            return candidates[:k]
        reranker_obj = reranker
        if reranker_obj is None:
            if getattr(self, "_reranker", None) is None:
                try:
                    self._reranker = CrossEncoderReranker()
                except Exception:
                    return candidates[:k]
            reranker_obj = self._reranker
        try:
            return reranker_obj.rerank_fused(
                query, candidates, k, use_rerank=True,
            )
        except Exception:
            logger.exception("Reranking failed; keeping retrieval order")
            return candidates[:k]

    # --------------------------------------------------------
    # RESPONSE CACHE (Phase 5)
    # --------------------------------------------------------
    @staticmethod
    def _resolve_response_cache(response_cache, use_cache):
        """Return the cache to consult, or None when disabled/absent.

        ``use_cache=None`` follows the ``CACHE_ENABLED`` env flag so
        existing callers keep uncached behavior by default; ``True`` /
        ``False`` forces or disables per request (tests, overrides).
        """
        if response_cache is None:
            return None
        try:
            from app.services.cache_service import cache_enabled
            enabled = (
                cache_enabled() if use_cache is None else bool(use_cache)
            )
        except Exception:
            logger.exception("Response cache flag unreadable; caching off")
            return None
        return response_cache if enabled else None

    @staticmethod
    def _lookup_cached(cache, user_id, raw_query, chunks):
        """Stored payload for this retrieval, or None. Never raises.

        Malformed payloads (missing keys) are treated as a miss so a
        poisoned entry can never alter the event or result contract.
        """
        try:
            payload = cache.get(
                user_id, raw_query, [chunk.chunk_id for chunk in chunks],
            )
        except Exception:
            logger.exception(
                "Response cache lookup failed; continuing uncached")
            return None
        if not isinstance(payload, dict):
            return None
        required = (
            "answer", "retrieval_query", "retrieved_document_ids",
            "chunks", "sources",
        )
        if any(key not in payload for key in required):
            return None
        if not payload["answer"]:
            return None
        return payload

    def _store_cached(
        self, cache, user_id, raw_query, chunks,
        retrieval_query, answer,
    ):
        """Persist a successful grounded answer. Never raises.

        Refusals and empty answers are never stored: an unanswerable
        query must be re-evaluated against the current corpus, and an
        error must never become a cached "answer".
        """
        if cache is None:
            return
        if not answer or answer == REFUSAL_MESSAGE:
            return
        try:
            cache.store(
                user_id, raw_query,
                [chunk.chunk_id for chunk in chunks],
                {
                    "answer": answer,
                    "retrieval_query": retrieval_query,
                    "retrieved_document_ids": [
                        chunk.metadata["document_id"]
                        for chunk in chunks
                    ],
                    "chunks": [asdict(chunk) for chunk in chunks],
                    "sources": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.metadata["document_id"],
                            "source": chunk.metadata["source"],
                            "title": chunk.metadata["title"],
                            "chunk_index": chunk.metadata["chunk_index"],
                            "distance": chunk.distance,
                        }
                        for chunk in chunks
                    ],
                },
            )
        except Exception:
            logger.exception("Response cache store failed")

    # --------------------------------------------------------
    # DEGRADED RESPONSES (Phase 7)
    # --------------------------------------------------------
    @staticmethod
    def _resolve_degraded(degraded_tracker, use_degraded):
        """Whether to serve excerpts on LLM failure, plus the counter.

        ``use_degraded=None`` follows the ``DEGRADED_MODE_ENABLED`` env
        flag so existing callers keep error-path behavior by default.
        Counting is best-effort: a missing tracker still serves the
        fallback, just uncounted.
        """
        try:
            enabled = (
                degraded_mode_enabled()
                if use_degraded is None else bool(use_degraded)
            )
        except Exception:
            logger.exception("Degraded-mode flag unreadable; degraded off")
            return False, None
        return enabled, degraded_tracker if enabled else None

    def _degraded_answer(self, error, chunks, degraded):
        """Excerpt fallback text, or None when this must stay an error.

        Only LLM-unavailability qualifies (transient error after
        retries, or an open circuit). Bugs and 4xx still take the error
        path, and empty retrieval keeps the refusal -- there is nothing
        to excerpt. Served fallbacks are counted in Redis (fail-open).
        ``degraded`` is the ``(enabled, tracker)`` pair above.
        """
        enabled, tracker = degraded
        if not enabled or not chunks:
            return None
        if not is_llm_unavailable(error):
            return None
        try:
            text = build_degraded_answer(chunks)
        except Exception:
            logger.exception("Could not build degraded answer")
            return None
        if tracker is not None:
            try:
                tracker.increment()
            except Exception:
                logger.exception("Degraded-response counter failed")
        return text

    def run_once(
        self,
        raw_query: str,
        user_id: str,
        chat_history: list[dict] | None = None,
        k: int = 3,
        rewrite_query: bool = False,
        distance_threshold: float | None = None,
        use_hybrid: bool | None = None,
        hybrid_retriever=None,
        use_rerank: bool | None = None,
        reranker=None,
        use_cache: bool | None = None,
        response_cache=None,
        use_degraded: bool | None = None,
        degraded_tracker=None,
    ) -> dict:

        if chat_history is None:
            chat_history = []

        if rewrite_query:
            retrieval_query = condense_question(
                chat_history,
                raw_query
            )
        else:
            retrieval_query = raw_query

        chunks = self._retrieve_chunks(
            query=retrieval_query,
            user_id=user_id,
            k=k,
            distance_threshold=distance_threshold,
            use_hybrid=use_hybrid,
            hybrid_retriever=hybrid_retriever,
            use_rerank=use_rerank,
            reranker=reranker,
        )

        # Phase 5: the cache key needs the retrieved chunk ids, so the
        # lookup happens here -- after retrieval, before the LLM call.
        cache = self._resolve_response_cache(response_cache, use_cache)
        cached = (
            self._lookup_cached(cache, user_id, raw_query, chunks)
            if cache is not None
            else None
        )
        if cached is not None:
            return {
                "answer": cached["answer"],
                "retrieval_query": cached["retrieval_query"],

                "retrieved_document_ids": cached["retrieved_document_ids"],

                "chunks": cached["chunks"],

                "cache_hit": True,
            }

        answer = None
        degraded = False
        usage_report: dict = {}
        try:
            answer = self.generate_answer(
                question=raw_query,
                chunks=chunks,
                chat_history=chat_history,
                usage=usage_report,
            )
        except Exception as error:
            # Phase 7: after Phase 6 retries are exhausted, an
            # unavailable LLM serves excerpts instead of an error.
            # Anything ineligible re-raises into the normal path.
            degraded_text = self._degraded_answer(
                error, chunks,
                self._resolve_degraded(degraded_tracker, use_degraded),
            )
            if degraded_text is None:
                raise
            answer = degraded_text
            degraded = True

        retrieved_document_ids = [
            chunk.metadata["document_id"]
            for chunk in chunks
        ]

        if not degraded:
            # Errors propagate before any store; refusals are skipped
            # inside. Degraded answers are never cached: the LLM may
            # have recovered by the next identical query.
            self._store_cached(
                cache, user_id, raw_query, chunks, retrieval_query, answer,
            )

        result = {
            "answer": answer,
            "retrieval_query": retrieval_query,

            "retrieved_document_ids": retrieved_document_ids,

            "chunks": [
                asdict(chunk)
                for chunk in chunks
            ]
        }
        if cache is not None:
            result["cache_hit"] = False
        if degraded:
            # Only present on the degraded path, like cache_hit.
            result["degraded"] = True
        if usage_report:
            # Only present when the LLM actually reported token counts,
            # so uncached/unmetered callers keep the historical shape.
            result["usage"] = dict(usage_report)
        return result
    def run_once_stream(
        self,
        raw_query: str,
        user_id: str,
        chat_history: list[dict] | None = None,
        k: int = 3,
        rewrite_query: bool = True,
        distance_threshold: float | None = None,
        use_hybrid: bool | None = None,
        hybrid_retriever=None,
        use_rerank: bool | None = None,
        reranker=None,
        use_cache: bool | None = None,
        response_cache=None,
        use_degraded: bool | None = None,
        degraded_tracker=None,
        usage: dict | None = None,
    ):

        if chat_history is None:
            chat_history = []

        if rewrite_query:

            retrieval_query = condense_question(
                chat_history,
                raw_query
            )

        else:

            retrieval_query = raw_query

        chunks = self._retrieve_chunks(
            query=retrieval_query,
            user_id=user_id,
            k=k,
            distance_threshold=distance_threshold,
            use_hybrid=use_hybrid,
            hybrid_retriever=hybrid_retriever,
            use_rerank=use_rerank,
            reranker=reranker,
        )

        # Phase 5: same lookup point as run_once -- a hit replays the
        # stored answer as stream tokens without touching the LLM.
        cache = self._resolve_response_cache(response_cache, use_cache)
        cached = (
            self._lookup_cached(cache, user_id, raw_query, chunks)
            if cache is not None
            else None
        )
        if cached is not None:
            yield cached["answer"]
            return

        # Phase 7: pull the first token eagerly. A failure before it
        # (retries already exhausted inside the generator) can still
        # become excerpts; once tokens flow, errors propagate -- the
        # client already has a partial answer that must not be
        # replaced.
        degraded = self._resolve_degraded(degraded_tracker, use_degraded)
        try:
            stream = self.generate_answer_stream(
                question=raw_query,
                chunks=chunks,
                chat_history=chat_history,
                usage=usage,
            )
            iterator = iter(stream)
            first = next(iterator)
        except StopIteration:
            return
        except Exception as error:
            degraded_text = self._degraded_answer(error, chunks, degraded)
            if degraded_text is None:
                raise
            yield degraded_text
            return

        parts = [first]
        yield first
        for text in iterator:
            parts.append(text)
            yield text

        self._store_cached(
            cache, user_id, raw_query, chunks,
            retrieval_query, "".join(parts),
        )
    def run_once_event_stream(
        self,
        raw_query: str,
        user_id: str,
        chat_history: list[dict] | None = None,
        k: int = 3,
        rewrite_query: bool = True,
        distance_threshold: float | None = None,
        use_hybrid: bool | None = None,
        hybrid_retriever=None,
        use_rerank: bool | None = None,
        reranker=None,
        use_cache: bool | None = None,
        response_cache=None,
        use_degraded: bool | None = None,
        degraded_tracker=None,
    ):
        """Yield semantic events, ending with either done or a generic error."""

        if chat_history is None:
            chat_history = []

        stage = "start"
        try:
            # --------------------------------------------
            # 1. Stream started
            # --------------------------------------------
            yield {
                "event": "start",
                "data": {
                    "raw_query": raw_query
                }
            }

            # --------------------------------------------
            # 2. Query rewriting
            # --------------------------------------------

            stage = "query_rewrite"

            if rewrite_query:

                retrieval_query = condense_question(
                    chat_history,
                    raw_query
                )

            else:

                retrieval_query = raw_query

            # --------------------------------------------
            # 3. Retrieval
            # --------------------------------------------

            stage = "retrieval"

            chunks = self._retrieve_chunks(
                query=retrieval_query,
                user_id=user_id,
                k=k,
                distance_threshold=distance_threshold,
                use_hybrid=use_hybrid,
                hybrid_retriever=hybrid_retriever,
                use_rerank=use_rerank,
                reranker=reranker,
            )

            retrieved_document_ids = [
                chunk.metadata["document_id"]
                for chunk in chunks
            ]

            yield {
                "event": "retrieval",
                "data": {
                    "retrieval_query": retrieval_query,
                    "retrieved_document_ids": retrieved_document_ids
                }
            }

            # --------------------------------------------
            # 3b. Response cache (Phase 5)
            # --------------------------------------------

            # The key needs the retrieved chunk ids, so the lookup sits
            # after retrieval and before generation. A hit replays the
            # stored answer through the same token/sources/done shape,
            # flagging itself in the done metadata.
            cache = self._resolve_response_cache(
                response_cache, use_cache)
            cached = (
                self._lookup_cached(cache, user_id, raw_query, chunks)
                if cache is not None
                else None
            )
            if cached is not None:
                yield {
                    "event": "token",
                    "data": {
                        "text": cached["answer"]
                    }
                }
                yield {
                    "event": "sources",
                    "data": {
                        "sources": cached["sources"]
                    }
                }
                yield {
                    "event": "done",
                    "data": {
                        "metadata": {
                            "cache": "hit"
                        }
                    }
                }
                return

            # --------------------------------------------
            # 4. Generation streaming
            # --------------------------------------------

            stage = "generation"

            # Phase 7: same first-token rule as run_once_stream -- a
            # pre-stream failure may become excerpts (counted in Redis);
            # a mid-stream failure keeps the existing error event.
            degraded = self._resolve_degraded(
                degraded_tracker, use_degraded)
            is_degraded = False
            parts = []
            usage_report: dict = {}
            try:
                stream = self.generate_answer_stream(
                    question=raw_query,
                    chunks=chunks,
                    chat_history=chat_history,
                    usage=usage_report,
                )
                iterator = iter(stream)
                first = next(iterator)
            except StopIteration:
                first = None
            except Exception as error:
                degraded_text = self._degraded_answer(
                    error, chunks, degraded)
                if degraded_text is None:
                    raise
                first = degraded_text
                is_degraded = True
            if first is not None:
                parts.append(first)
                yield {
                    "event": "token",
                    "data": {
                        "text": first
                    }
                }
            if not is_degraded:
                for text in iterator:
                    parts.append(text)

                    yield {
                        "event": "token",
                        "data": {
                            "text": text
                        }
                    }

            # --------------------------------------------
            # 5. Sources
            # --------------------------------------------

            stage = "sources"

            sources = [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.metadata["document_id"],
                    "source": chunk.metadata["source"],
                    "title": chunk.metadata["title"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "distance": chunk.distance
                }
                for chunk in chunks
            ]

            yield {
                "event": "sources",
                "data": {
                    "sources": sources
                }
            }

            # Cache only completed, successful, non-degraded answers:
            # refusals and empty outputs are re-evaluated next time, a
            # degraded answer must not outlive the outage that caused it,
            # and any failure below lands on the error event with
            # nothing stored.
            if not is_degraded:
                self._store_cached(
                    cache, user_id, raw_query, chunks,
                    retrieval_query, "".join(parts),
                )

            # --------------------------------------------
            # 6. Successful completion
            # --------------------------------------------

            stage = "done"

            metadata: dict = {}
            if cache is not None:
                metadata["cache"] = "miss"
            if is_degraded:
                metadata["degraded"] = True
            if usage_report and not is_degraded:
                # Metered completions report their token counts; degraded
                # answers never reach an LLM call, so there is nothing
                # truthful to attach on that path.
                metadata["usage"] = dict(usage_report)
            yield {
                "event": "done",
                # Uncached, non-degraded callers keep the exact
                # historical shape (``{}``).
                "data": {"metadata": metadata} if metadata else {}
            }

        except Exception:
            # Keep the traceback here. Cancellation/GeneratorExit is not an
            # application error and is deliberately not caught by Exception.
            logger.exception(
                "RAG stream failed during stage: %s",
                stage
            )

            yield {
                "event": "error",
                "data": {
                    "stage": stage,
                    "message": "The response could not be completed."
                }
            }



# ============================================================
# Evaluation dataset
# ============================================================

def load_eval_cases(path: str) -> list[EvalCase]:

    with open(path, "r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    return [
        EvalCase(**case)
        for case in raw_cases
    ]


# ============================================================
# Retrieval evaluation
# ============================================================

def calculate_retrieval_recall(
    expected_document_ids: list[str],
    retrieved_document_ids: list[str]
) -> float:

    if not expected_document_ids:
        return 1.0

    expected = set(expected_document_ids)
    retrieved = set(retrieved_document_ids)

    found = expected.intersection(retrieved)

    return len(found) / len(expected)


# ============================================================
# Simple deterministic answer evaluator
# ============================================================

def answer_contains_required_terms(
    answer: str,
    required_terms: list[str]
) -> bool:

    if not required_terms:
        return True

    answer_lower = answer.lower()

    return all(
        term.lower() in answer_lower
        for term in required_terms
    )


# ============================================================
# Full evaluation
# ============================================================

def evaluate_cases(
    rag: RAGSystem,
    cases: list[EvalCase],
    k: int,
    rewrite_query: bool,
    distance_threshold: float | None = None
) -> dict:

    rows = []

    retrieval_scores = []
    answer_scores = []
    refusal_scores = []

    for case in cases:

        history = case.history or []

        result = rag.run_once(
            raw_query=case.question,
            user_id=case.user_id,
            chat_history=history,
            k=k,
            rewrite_query=rewrite_query,
            distance_threshold=distance_threshold
        )

        answer = result["answer"]

        # ----------------------------------------------
        # Retrieval measurement
        # ----------------------------------------------

        if case.answerable:

            retrieval_recall = calculate_retrieval_recall(
                case.expected_document_ids,
                result["retrieved_document_ids"]
            )

            retrieval_scores.append(retrieval_recall)

            answer_correct = answer_contains_required_terms(
                answer,
                case.required_answer_terms
            )

            answer_scores.append(
                1 if answer_correct else 0
            )

        else:

            retrieval_recall = None

            refused_correctly = (
                answer.strip() == REFUSAL_MESSAGE
            )

            refusal_scores.append(
                1 if refused_correctly else 0
            )

            answer_correct = None

        rows.append({
            "id": case.id,
            "question": case.question,
            "answerable": case.answerable,
            "rewritten_query": result["retrieval_query"],
            "retrieved_documents": result[
                "retrieved_document_ids"
            ],
            "retrieval_recall": retrieval_recall,
            "answer_correct": answer_correct,
            "answer": answer
        })

    retrieval_average = (
        sum(retrieval_scores) / len(retrieval_scores)
        if retrieval_scores
        else 0
    )

    answer_accuracy = (
        sum(answer_scores) / len(answer_scores)
        if answer_scores
        else 0
    )

    refusal_accuracy = (
        sum(refusal_scores) / len(refusal_scores)
        if refusal_scores
        else 0
    )

    return {
        "retrieval_recall": retrieval_average,
        "answer_accuracy": answer_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "rows": rows
    }


# ============================================================
# Threshold calibration
# ============================================================

def calibrate_distance_threshold(
    rag: RAGSystem,
    cases: list[EvalCase],
    rewrite_query: bool = False
):

    """
    Find a distance threshold that best separates:

        answerable queries
        vs
        unanswerable queries

    IMPORTANT:
    In a serious project this should use a VALIDATION SET,
    not your final test set.
    """

    samples = []

    for case in cases:

        history = case.history or []

        if rewrite_query:
            query = condense_question(
                history,
                case.question
            )
        else:
            query = case.question

        chunks = rag.retrieve(
            query=query,
            user_id=case.user_id,
            k=1,
            distance_threshold=None
        )

        if not chunks:
            continue

        best_distance = chunks[0].distance

        samples.append(
            (
                best_distance,
                case.answerable
            )
        )

    if not samples:
        return None

    candidates = sorted(
        set(distance for distance, _ in samples)
    )

    best_threshold = None
    best_score = -1

    for threshold in candidates:

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for distance, answerable in samples:

            predicted_answerable = (
                distance <= threshold
            )

            if answerable and predicted_answerable:
                true_positive += 1

            elif answerable and not predicted_answerable:
                false_negative += 1

            elif not answerable and predicted_answerable:
                false_positive += 1

            else:
                true_negative += 1

        positives = true_positive + false_negative
        negatives = true_negative + false_positive

        tpr = (
            true_positive / positives
            if positives
            else 1
        )

        tnr = (
            true_negative / negatives
            if negatives
            else 1
        )

        balanced_accuracy = (tpr + tnr) / 2

        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_threshold = threshold

    return {
        "threshold": best_threshold,
        "balanced_accuracy": best_score,
        "samples": samples
    }


# ============================================================
# Chunk-size experiments
# ============================================================

def run_chunking_experiments(
    documents: list[Document],
    cases: list[EvalCase],
    user_id: str
):

    configurations = [
        (256, 32),
        (512, 64),
        (1024, 128)
    ]

    k_values = [
        1,
        3,
        5
    ]

    experiment_results = []

    for chunk_size, overlap in configurations:

        collection_name = (
            f"experiment_"
            f"{chunk_size}_"
            f"{overlap}"
        )

        rag = RAGSystem(
            collection_name=collection_name,
            reset=True
        )

        rag.ingest_documents(
            documents=documents,
            user_id=user_id,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        for k in k_values:

            metrics = evaluate_cases(
                rag=rag,
                cases=cases,
                k=k,
                rewrite_query=False
            )

            experiment_results.append({
                "chunk_size": chunk_size,
                "overlap": overlap,
                "k": k,
                "retrieval_recall":
                    metrics["retrieval_recall"],
                "answer_accuracy":
                    metrics["answer_accuracy"],
                "refusal_accuracy":
                    metrics["refusal_accuracy"]
            })

    return experiment_results


# ============================================================
# Query rewriting A/B test
# ============================================================

def query_rewrite_experiment(
    rag: RAGSystem,
    cases: list[EvalCase],
    k: int = 3
):

    without_rewrite = evaluate_cases(
        rag=rag,
        cases=cases,
        k=k,
        rewrite_query=False
    )

    with_rewrite = evaluate_cases(
        rag=rag,
        cases=cases,
        k=k,
        rewrite_query=True
    )

    return {
        "without_rewrite": {
            "retrieval_recall":
                without_rewrite["retrieval_recall"],
            "answer_accuracy":
                without_rewrite["answer_accuracy"]
        },

        "with_rewrite": {
            "retrieval_recall":
                with_rewrite["retrieval_recall"],
            "answer_accuracy":
                with_rewrite["answer_accuracy"]
        }
    }


# ============================================================
# MULTI-USER ISOLATION TEST
# ============================================================

def multi_user_isolation_test():

    rag = RAGSystem(
        collection_name="multi_user_test",
        reset=True
    )

    user_a_documents = [
        Document(
            document_id="falcon",
            source="falcon.txt",
            title="Falcon",
            text=(
                "Project Falcon uses PostgreSQL "
                "as its main relational database."
            )
        )
    ]

    user_b_documents = [
        Document(
            document_id="orion",
            source="orion.txt",
            title="Orion",
            text=(
                "Project Orion uses MongoDB "
                "as its main database."
            )
        )
    ]

    rag.ingest_documents(
        user_a_documents,
        user_id="user_A"
    )

    rag.ingest_documents(
        user_b_documents,
        user_id="user_B"
    )

    # User A deliberately asks about User B's content.
    results = rag.retrieve(
        query="Which database does Project Orion use?",
        user_id="user_A",
        k=5
    )

    # User A must NEVER receive a user_B chunk.
    for chunk in results:

        assert (
            chunk.metadata["user_id"]
            == "user_A"
        ), "SECURITY FAILURE: cross-user retrieval!"

    print("Multi-user isolation test passed.")


# ============================================================
# Demo
# ============================================================

def main():

    USER_ID = "eval_user"

    # Fix 6: use a project-anchored corpus path for the standalone demo.
    documents = load_txt_documents(PROJECT_ROOT / "data")

    rag = RAGSystem(
        collection_name="learning_rag",
        reset=True
    )

    rag.ingest_documents(
        documents=documents,
        user_id=USER_ID,
        chunk_size=512,
        chunk_overlap=64
    )

    evaluation_cases = load_eval_cases(
        PROJECT_ROOT / "eval_cases.json"
    )

    # --------------------------------------------------------
    # 1. Baseline
    # --------------------------------------------------------

    baseline = evaluate_cases(
        rag=rag,
        cases=evaluation_cases,
        k=3,
        rewrite_query=False
    )

    print("\nBASELINE")
    print(
        "Retrieval recall:",
        baseline["retrieval_recall"]
    )

    print(
        "Answer accuracy:",
        baseline["answer_accuracy"]
    )

    print(
        "Refusal accuracy:",
        baseline["refusal_accuracy"]
    )

    # --------------------------------------------------------
    # 2. Threshold calibration
    # --------------------------------------------------------

    threshold_result = calibrate_distance_threshold(
        rag,
        evaluation_cases
    )

    print("\nTHRESHOLD CALIBRATION")
    print(threshold_result)

    # --------------------------------------------------------
    # 3. Query rewrite experiment
    # --------------------------------------------------------

    rewrite_result = query_rewrite_experiment(
        rag,
        evaluation_cases,
        k=3
    )

    print("\nQUERY REWRITE A/B TEST")
    print(
        json.dumps(
            rewrite_result,
            indent=2
        )
    )

    # --------------------------------------------------------
    # 4. Multi-user security check
    # --------------------------------------------------------

    multi_user_isolation_test()


if __name__ == "__main__":
    main()
