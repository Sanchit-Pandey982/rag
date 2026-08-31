from __future__ import annotations

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


# ============================================================
# Configuration
# ============================================================

load_dotenv()

gemini_client = genai.Client()

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
def embed_documents(texts: list[str]) -> list[list[float]]:

    if not texts:
        return []

    result = gemini_client.models.embed_content(
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
def embed_query(text: str) -> list[float]:

    result = gemini_client.models.embed_content(
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

    root = Path(folder)

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

    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    if not response.text:
        return latest_query

    return response.text.strip()


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
    def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        chat_history: list[dict]
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

        response = gemini_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        return (
            response.text.strip()
            if response.text
            else REFUSAL_MESSAGE
        )

    # --------------------------------------------------------
    # COMPLETE PIPELINE
    # --------------------------------------------------------

    def run_once(
        self,
        raw_query: str,
        user_id: str,
        chat_history: list[dict] | None = None,
        k: int = 3,
        rewrite_query: bool = True,
        distance_threshold: float | None = None
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

        chunks = self.retrieve(
            query=retrieval_query,
            user_id=user_id,
            k=k,
            distance_threshold=distance_threshold
        )

        answer = self.generate_answer(
            question=raw_query,
            chunks=chunks,
            chat_history=chat_history
        )

        return {
            "answer": answer,
            "retrieval_query": retrieval_query,

            "retrieved_document_ids": [
                chunk.metadata["document_id"]
                for chunk in chunks
            ],

            "chunks": [
                asdict(chunk)
                for chunk in chunks
            ]
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

    documents = load_txt_documents("./data")

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
        "eval_cases.json"
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