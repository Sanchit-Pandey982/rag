"""Phase 11 tests: agent routing + confidence loop, no live services.

Fake RAGService (in-memory chunks, scripted answers) exercises the
whole agent contract: env config parsing, direct vs retrieve routing,
confidence math, retry-until-pass, max-iteration cap with gave_up,
manual-loop fallback when langgraph is unavailable, fail-open
integration in RAGService.run_once, and legacy byte-identity when
the agent is disabled.
"""

import os
import unittest
from unittest.mock import patch

from phase1 import REFUSAL_MESSAGE, RetrievedChunk

from app.agents import config as agent_config
from app.agents import graph as agent_graph
from app.agents.nodes import (
    check_confidence_node,
    confidence_of,
    decide_route,
    generate_node,
    retrieve_node,
    should_retry,
)


def make_chunk(chunk_id="u:doc:0", distance=0.2, document_id="doc"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Grounded context about RAG.",
        distance=distance,
        metadata={"user_id": "alice", "document_id": document_id,
                  "source": "s.txt", "title": "T", "chunk_index": 0},
    )


class FakeRAG:
    """Minimal RAGSystem double behind RAGService-shaped access."""

    def __init__(self, chunks=None, answers=None, fail_generate=None):
        self._chunks = list(chunks) if chunks is not None else [make_chunk()]
        self._answers = list(answers or ["Grounded answer."])
        self.fail_generate = fail_generate
        self.retrieve_calls = []
        self.generate_calls = []

    def _retrieve_chunks(self, query, user_id, k=3,
                         distance_threshold=None, use_hybrid=None,
                         hybrid_retriever=None, use_rerank=None,
                         reranker=None):
        self.retrieve_calls.append({"query": query, "k": k})
        return list(self._chunks)

    def generate_answer(self, question, chunks, chat_history, usage=None):
        self.generate_calls.append({"question": question})
        if self.fail_generate is not None:
            raise self.fail_generate
        answer = self._answers[min(len(self.generate_calls) - 1,
                                   len(self._answers) - 1)]
        if usage is not None:
            usage.update({"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15, "model": "fake-model"})
        return answer

    @staticmethod
    def _resolve_degraded(tracker, use_degraded):
        return False, None

    def _degraded_answer(self, error, chunks, degraded):
        return None


class FakeRAGService:
    def __init__(self, rag):
        self.rag = rag
        self.hybrid_retriever = None
        self.reranker = None
        self.degradation_tracker = None


def agent_env(**overrides):
    base = {"AGENT_ENABLED": "false",
            "AGENT_MAX_ITERATIONS": "3",
            "AGENT_CONFIDENCE_THRESHOLD": "0.5"}
    base.update(overrides)
    return patch.dict(os.environ, base, clear=False)


def plain_agent_env():
    env = {k: v for k, v in os.environ.items()
           if k not in ("AGENT_ENABLED", "AGENT_MAX_ITERATIONS",
                        "AGENT_CONFIDENCE_THRESHOLD")}
    return patch.dict(os.environ, env, clear=True)


class ConfigTest(unittest.TestCase):
    def test_defaults_when_unset(self):
        with plain_agent_env():
            self.assertFalse(agent_config.agent_enabled())
            self.assertEqual(agent_config.agent_max_iterations(), 3)
            self.assertEqual(agent_config.agent_confidence_threshold(), 0.5)

    def test_enabled_truthy_values(self):
        for raw in ("1", "true", "TRUE", "yes", "on"):
            with agent_env(AGENT_ENABLED=raw):
                self.assertTrue(agent_config.agent_enabled())

    def test_invalid_falls_back_to_defaults(self):
        with agent_env(AGENT_MAX_ITERATIONS="many",
                       AGENT_CONFIDENCE_THRESHOLD="high"):
            self.assertEqual(agent_config.agent_max_iterations(), 3)
            self.assertEqual(agent_config.agent_confidence_threshold(), 0.5)

    def test_bounds_are_clamped(self):
        with agent_env(AGENT_MAX_ITERATIONS="0"):
            self.assertEqual(agent_config.agent_max_iterations(), 1)
        with agent_env(AGENT_MAX_ITERATIONS="99"):
            self.assertEqual(agent_config.agent_max_iterations(), 10)
        with agent_env(AGENT_CONFIDENCE_THRESHOLD="-1"):
            self.assertEqual(agent_config.agent_confidence_threshold(), 0.0)
        with agent_env(AGENT_CONFIDENCE_THRESHOLD="9"):
            self.assertEqual(agent_config.agent_confidence_threshold(), 1.0)


class RoutingTest(unittest.TestCase):
    def test_smalltalk_goes_direct(self):
        for query in ("hi", "Hello!", "hey", "thanks", "thank you",
                      "good morning", "bye", "ok"):
            self.assertEqual(decide_route(query), "direct",
                             f"query={query!r}")

    def test_content_queries_retrieve(self):
        for query in ("What is RAG?", "hi, explain embeddings",
                      "hello world deployment steps",
                      "thanks but how do refunds work?"):
            self.assertEqual(decide_route(query), "retrieve",
                             f"query={query!r}")

    def test_empty_query_retrieves(self):
        self.assertEqual(decide_route(""), "retrieve")
        self.assertEqual(decide_route("   "), "retrieve")


class ConfidenceTest(unittest.TestCase):
    def test_refusal_and_empty_score_zero(self):
        self.assertEqual(
            confidence_of(REFUSAL_MESSAGE, [make_chunk()], "retrieve"), 0.0)
        self.assertEqual(confidence_of("", [make_chunk()], "retrieve"), 0.0)
        self.assertEqual(confidence_of(None, [make_chunk()], "retrieve"), 0.0)

    def test_no_chunks_scores_zero(self):
        self.assertEqual(confidence_of("An answer", [], "retrieve"), 0.0)

    def test_direct_answer_scores_one(self):
        self.assertEqual(confidence_of("Hello there!", [], "direct"), 1.0)

    def test_distance_mapping(self):
        good = confidence_of("An answer",
                             [make_chunk(distance=0.2)], "retrieve")
        self.assertAlmostEqual(good, 0.9)
        mid = confidence_of("An answer",
                            [make_chunk(distance=1.0)], "retrieve")
        self.assertAlmostEqual(mid, 0.5)
        bad = confidence_of("An answer",
                            [make_chunk(distance=2.0)], "retrieve")
        self.assertAlmostEqual(bad, 0.0)

    def test_should_retry_gate(self):
        self.assertTrue(should_retry(
            {"route": "retrieve", "confidence": 0.1, "attempts": 1}, 0.5, 3))
        self.assertFalse(should_retry(
            {"route": "retrieve", "confidence": 0.9, "attempts": 1}, 0.5, 3))
        self.assertFalse(should_retry(
            {"route": "retrieve", "confidence": 0.1, "attempts": 3}, 0.5, 3))
        self.assertFalse(should_retry(
            {"route": "direct", "confidence": 0.0, "attempts": 0}, 0.5, 3))

    def test_check_confidence_node(self):
        update = check_confidence_node(
            {"answer": "Hi!", "chunks": [], "route": "direct"}, 0.5)
        self.assertEqual(update, {"confidence": 1.0})


class RetrieveNodeTest(unittest.TestCase):
    def test_reuses_pipeline_and_records_query(self):
        rag = FakeRAG(chunks=[make_chunk(distance=0.3)])
        service = FakeRAGService(rag)
        state = {"raw_query": "What is RAG?", "user_id": "alice",
                 "chat_history": [], "k": 3, "rewrite_query": False,
                 "distance_threshold": None, "attempts": 0}
        update = retrieve_node(state, service)
        self.assertEqual(update["retrieval_query"], "What is RAG?")
        self.assertEqual(len(update["chunks"]), 1)
        self.assertEqual(len(rag.retrieve_calls), 1)

    def test_retrieval_failure_is_recorded_not_raised(self):
        class BrokenRAG(FakeRAG):
            def _retrieve_chunks(self, *a, **k):
                raise ConnectionError("chroma down")
        update = retrieve_node(
            {"raw_query": "q", "user_id": "u", "chat_history": [],
             "k": 3, "rewrite_query": False, "distance_threshold": None,
             "attempts": 0},
            FakeRAGService(BrokenRAG()))
        self.assertEqual(update["chunks"], [])
        self.assertIn("retrieval", update["error"])

    def test_generate_node_captures_usage(self):
        rag = FakeRAG()
        update = generate_node(
            {"raw_query": "q", "chunks": [make_chunk()],
             "chat_history": []},
            FakeRAGService(rag))
        self.assertEqual(update["answer"], "Grounded answer.")
        self.assertEqual(update["usage"]["prompt_tokens"], 10)

    def test_generate_failure_is_recorded_not_raised(self):
        rag = FakeRAG(fail_generate=TimeoutError("llm down"))
        update = generate_node(
            {"raw_query": "q", "chunks": [make_chunk()],
             "chat_history": []},
            FakeRAGService(rag))
        self.assertIsNone(update["answer"])
        self.assertIn("_error_obj", update)


class RunAgentTest(unittest.TestCase):
    def test_disabled_returns_none(self):
        service = FakeRAGService(FakeRAG())
        with agent_env(AGENT_ENABLED="false"):
            self.assertIsNone(agent_graph.run_agent(
                service, raw_query="What is RAG?", user_id="alice"))
        with agent_env():
            self.assertIsNone(agent_graph.run_agent(
                service, raw_query="What is RAG?", user_id="alice",
                use_agent=False))

    def test_direct_route_skips_retrieval(self):
        rag = FakeRAG()
        service = FakeRAGService(rag)
        with agent_env():
            result = agent_graph.run_agent(
                service, raw_query="hello", user_id="alice",
                use_agent=True)
        self.assertEqual(rag.retrieve_calls, [])
        self.assertEqual(rag.generate_calls, [])
        self.assertTrue(result["answer"])
        self.assertEqual(result["retrieved_document_ids"], [])
        self.assertEqual(result["agent"]["route"], "direct")
        self.assertEqual(result["agent"]["confidence"], 1.0)
        self.assertFalse(result["agent"]["gave_up"])

    def test_happy_path_single_attempt(self):
        rag = FakeRAG(chunks=[make_chunk(distance=0.2)])
        service = FakeRAGService(rag)
        with agent_env():
            result = agent_graph.run_agent(
                service, raw_query="What is RAG?", user_id="alice",
                use_agent=True)
        self.assertEqual(result["answer"], "Grounded answer.")
        self.assertEqual(result["retrieved_document_ids"], ["doc"])
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(result["usage"]["prompt_tokens"], 10)
        self.assertEqual(result["agent"]["route"], "retrieve")
        self.assertEqual(result["agent"]["attempts"], 1)
        self.assertGreaterEqual(result["agent"]["confidence"], 0.5)
        self.assertFalse(result["agent"]["gave_up"])
        self.assertNotIn("degraded", result)

    def test_weak_answer_retries_with_wider_k(self):
        rag = FakeRAG(chunks=[make_chunk(distance=1.8)],
                      answers=[REFUSAL_MESSAGE, "Grounded answer."])
        service = FakeRAGService(rag)
        with agent_env():
            result = agent_graph.run_agent(
                service, raw_query="Obscure detail?", user_id="alice",
                use_agent=True)
        # First attempt refuses (confidence 0) -> retry widens k.
        self.assertGreaterEqual(result["agent"]["attempts"], 1)
        self.assertEqual(len(rag.generate_calls),
                         result["agent"]["attempts"])
        ks = [call["k"] for call in rag.retrieve_calls]
        self.assertTrue(all(k >= 3 for k in ks))
        if len(ks) > 1:
            self.assertGreater(ks[-1], ks[0])

    def test_max_iterations_cap_and_gave_up(self):
        rag = FakeRAG(chunks=[make_chunk(distance=1.9)],
                      answers=[REFUSAL_MESSAGE])
        service = FakeRAGService(rag)
        with agent_env():
            result = agent_graph.run_agent(
                service, raw_query="Never answerable?", user_id="alice",
                use_agent=True, max_iterations=2)
        self.assertEqual(result["agent"]["attempts"], 2)
        self.assertEqual(len(rag.generate_calls), 2)
        self.assertTrue(result["agent"]["gave_up"])
        self.assertEqual(result["answer"], REFUSAL_MESSAGE)

    def test_langgraph_engine_compiles_and_runs(self):
        rag = FakeRAG(chunks=[make_chunk(distance=0.2)])
        service = FakeRAGService(rag)
        graph = agent_graph.build_graph(service)
        self.assertTrue(callable(getattr(graph, "invoke", None)))
        with agent_env(), \
                patch.object(agent_graph, "_run_manual",
                             side_effect=AssertionError("must not fallback")):
            result = agent_graph.run_agent(
                service, raw_query="What is RAG?", user_id="alice",
                use_agent=True)
        self.assertEqual(result["answer"], "Grounded answer.")
        self.assertEqual(result["agent"]["attempts"], 1)

    def test_manual_loop_matches_when_langgraph_missing(self):
        rag = FakeRAG(chunks=[make_chunk(distance=0.2)])
        service = FakeRAGService(rag)
        with agent_env(), \
                patch.object(agent_graph, "build_graph",
                             side_effect=ImportError("no langgraph")):
            result = agent_graph.run_agent(
                service, raw_query="What is RAG?", user_id="alice",
                use_agent=True)
        self.assertEqual(result["answer"], "Grounded answer.")
        self.assertEqual(result["agent"]["attempts"], 1)
        self.assertFalse(result["agent"]["gave_up"])

    def test_terminal_generate_failure_returns_refusal(self):
        rag = FakeRAG(fail_generate=ValueError("bad request"))
        service = FakeRAGService(rag)
        with agent_env():
            result = agent_graph.run_agent(
                service, raw_query="What is RAG?", user_id="alice",
                use_agent=True, max_iterations=1)
        # Bugs are not LLM-outages: refusal, not degraded excerpts.
        self.assertEqual(result["answer"], REFUSAL_MESSAGE)
        self.assertNotIn("degraded", result)

    def test_input_validation_mirrors_retrieve(self):
        service = FakeRAGService(FakeRAG())
        with agent_env(), self.assertRaises(ValueError):
            agent_graph.run_agent(service, raw_query="q", user_id="u",
                                  k=0, use_agent=True)
        with agent_env(), self.assertRaises(ValueError):
            agent_graph.run_agent(service, raw_query="", user_id="u",
                                  use_agent=True)


class IntegrationTest(unittest.TestCase):
    def _request(self, query="What is RAG?"):
        from app.schemas.chat import ChatRequest
        return ChatRequest(raw_query=query, user_id="alice")

    def test_rag_service_routes_to_agent_when_enabled(self):
        from app.services.rag_services import RAGService
        rag = FakeRAG(chunks=[make_chunk(distance=0.2)])
        service = RAGService(rag=rag)
        with agent_env():
            result = service.run_once(self._request(), use_agent=True)
        self.assertEqual(result["answer"], "Grounded answer.")
        self.assertIn("agent", result)
        self.assertEqual(result["agent"]["route"], "retrieve")

    def test_rag_service_legacy_when_disabled(self):
        from app.services.rag_services import RAGService

        class LegacyRAG(FakeRAG):
            def run_once(self, **kwargs):
                return {"answer": "Legacy answer.",
                        "retrieval_query": kwargs.get("raw_query", ""),
                        "retrieved_document_ids": ["doc"],
                        "chunks": []}

        service = RAGService(rag=LegacyRAG())
        with agent_env(AGENT_ENABLED="false"):
            result = service.run_once(self._request())
        self.assertEqual(result, {"answer": "Legacy answer.",
                                 "retrieval_query": "What is RAG?",
                                 "retrieved_document_ids": ["doc"],
                                 "chunks": []})

    def test_rag_service_fails_open_to_legacy(self):
        from app.services.rag_services import RAGService

        class LegacyRAG(FakeRAG):
            def run_once(self, **kwargs):
                return {"answer": "Legacy answer.",
                        "retrieval_query": "q",
                        "retrieved_document_ids": [],
                        "chunks": []}

        service = RAGService(rag=LegacyRAG())
        with agent_env(), \
                patch("app.agents.graph.run_agent",
                      side_effect=RuntimeError("agent bug")):
            result = service.run_once(self._request(), use_agent=True)
        self.assertEqual(result["answer"], "Legacy answer.")

    def test_orchestration_threads_use_agent(self):
        from app.services import chat_orchestration as orchestration
        seen = {}

        class SpyService:
            def run_once(self, payload, use_cache=None,
                         use_degraded=None, use_agent=None):
                seen["use_agent"] = use_agent
                return {"answer": "ok", "retrieval_query": "q",
                        "retrieved_document_ids": [], "chunks": []}

        payload = self._request()
        with agent_env():
            result = orchestration.run_chat_once(
                SpyService(), None, payload, use_agent=True)
        self.assertEqual(result["answer"], "ok")
        self.assertTrue(seen["use_agent"])


if __name__ == "__main__":
    unittest.main()
