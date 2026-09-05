from pathlib import Path

from langsmith import Client

from phase1 import (
    RAGSystem,
    load_eval_cases
)


DATASET_NAME = "rag-phase1-evaluation"
PROJECT_ROOT = Path(__file__).resolve().parent

rag = RAGSystem(
    collection_name="learning_rag",
    chroma_path=str(PROJECT_ROOT / "chroma_data")
)

client = Client()


# ============================================================
# Target
# ============================================================

def target(inputs: dict) -> dict:

    result = rag.run_once(
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


if __name__ == "__main__":
    run_experiment()
