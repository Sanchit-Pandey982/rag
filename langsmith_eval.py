"""RAG quality evaluation.

Two modes:

- default (LangSmith): upload the eval cases to a named dataset and run a
  hosted experiment. Requires ``LANGSMITH_API_KEY``.
- ``--local``: run the same cases against this process's RAG pipeline and
  fail when retrieval/answer/refusal metrics drop below env thresholds.
  This is the CI nightly gate; it needs only a Gemini key (and a reachable
  Chroma instance via ``CHROMA_URL`` when run inside Compose).

The RAG system and LangSmith client are constructed lazily so importing
this module requires no credentials or network.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from phase1 import (
    RAGSystem,
    evaluate_cases,
    load_eval_cases
)


DATASET_NAME = "rag-phase1-evaluation"
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MIN_RETRIEVAL_RECALL = 0.80
DEFAULT_MIN_ANSWER_ACCURACY = 0.80
DEFAULT_MIN_REFUSAL_ACCURACY = 0.80

_rag = None


def get_rag() -> RAGSystem:
    global _rag
    if _rag is None:
        _rag = RAGSystem(
            collection_name="learning_rag",
            chroma_path=str(PROJECT_ROOT / "chroma_data")
        )
    return _rag


def get_client():
    # Lazy import: --local and importing the module must not need LangSmith.
    from langsmith import Client
    return Client()


# ============================================================
# Target
# ============================================================

def target(inputs: dict) -> dict:

    result = get_rag().run_once(
        raw_query=inputs["question"],
        user_id=inputs["user_id"],
        chat_history=inputs.get("history", []),
        k=3,
        rewrite_query=inputs.get(
            "rewrite_query",
            False
        )
    )

    return {
        "answer": result["answer"],
        "retrieved_document_ids":
            result["retrieved_document_ids"]
    }


# ============================================================
# Retrieval evaluator
# ============================================================

def retrieval_recall_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
):

    expected = set(
        reference_outputs[
            "expected_document_ids"
        ]
    )

    retrieved = set(
        outputs[
            "retrieved_document_ids"
        ]
    )

    if not expected:
        score = 1.0

    else:
        score = (
            len(expected.intersection(retrieved))
            / len(expected)
        )

    return {
        "key": "retrieval_recall",
        "score": score
    }


# ============================================================
# Refusal evaluator
# ============================================================

def refusal_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
):

    answerable = reference_outputs["answerable"]

    answer = outputs["answer"]

    refusal = (
        "I do not have enough information "
        "to answer that."
    )

    # Fix 4: an answerable case must be checked for answer quality; it is not
    # automatically a correct refusal.
    if answerable:
        return {
            "key": "answer_correct",
            "score": int(
                all(
                    term.lower() in answer.lower()
                    for term in reference_outputs.get("required_answer_terms", [])
                )
                and answer.strip() != refusal
            )
        }

    return {
        "key": "correct_refusal",
        "score": int(
            answer.strip() == refusal
        )
    }


# ============================================================
# Create dataset once
# ============================================================

def create_dataset():

    client = get_client()
    cases = load_eval_cases(
        PROJECT_ROOT / "eval_cases.json"
    )

    # Fix 13: reuse the named dataset so setup is safe to rerun.
    dataset = next(iter(client.list_datasets(dataset_name=DATASET_NAME)), None)
    created = dataset is None
    if dataset is None:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Phase 1 RAG retrieval and answer evaluation"
        )

    # Fix 13: do not append duplicate examples when the dataset already exists.
    if not created:
        return dataset

    examples = []

    for case in cases:

        examples.append({

            "inputs": {
                "question": case.question,
                "user_id": case.user_id,
                "history": case.history or []
            },

            "outputs": {
                "expected_answer":
                    case.expected_answer,

                "expected_document_ids":
                    case.expected_document_ids,

                "answerable":
                    case.answerable,
                "required_answer_terms":
                    case.required_answer_terms
            }
        })

    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )
    return dataset


# ============================================================
# Run experiment
# ============================================================

def run_experiment():

    client = get_client()
    results = client.evaluate(
        target,

        data=DATASET_NAME,

        evaluators=[
            retrieval_recall_evaluator,
            refusal_evaluator
        ],

        experiment_prefix=(
            "chunk512-k3-no-rewrite"
        ),

        max_concurrency=2
    )

    print(results)
    return results


# ============================================================
# Local CI gate
# ============================================================

def _threshold(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def run_local_gate(k: int = 3, rewrite_query: bool = False) -> int:
    """Run eval cases locally and return a process exit code."""
    cases = load_eval_cases(PROJECT_ROOT / "eval_cases.json")
    metrics = evaluate_cases(
        get_rag(), cases, k=k, rewrite_query=rewrite_query
    )

    gates = {
        "retrieval_recall": _threshold(
            "EVAL_MIN_RETRIEVAL_RECALL", DEFAULT_MIN_RETRIEVAL_RECALL
        ),
        "answer_accuracy": _threshold(
            "EVAL_MIN_ANSWER_ACCURACY", DEFAULT_MIN_ANSWER_ACCURACY
        ),
        "refusal_accuracy": _threshold(
            "EVAL_MIN_REFUSAL_ACCURACY", DEFAULT_MIN_REFUSAL_ACCURACY
        ),
    }

    summary = {key: round(metrics[key], 4) for key in gates}
    summary["thresholds"] = gates
    summary["cases"] = len(cases)
    print(json.dumps(summary, indent=2))

    failed = [
        name for name, minimum in gates.items()
        if metrics[name] < minimum
    ]
    if failed:
        print(
            "Eval gate failed: " + ", ".join(
                f"{name}={metrics[name]:.4f} < {gates[name]:.4f}"
                for name in failed
            ),
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="run the local threshold gate instead of a LangSmith experiment",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--rewrite-query", action="store_true")
    args = parser.parse_args(argv)

    if args.local:
        return run_local_gate(k=args.k, rewrite_query=args.rewrite_query)

    run_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

