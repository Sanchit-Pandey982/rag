"""Phase 12 tests: web search + calculator tools, no live services.

Covers tool config parsing, safe calculator evaluation (including
injection rejection), web backend selection/parsing/failure modes via
patched transports, registry selection/extensibility, and agent
integration: calculator answers math directly, web results ground
generation for current events, tool failures fall back to retrieval,
and the best attempt wins across retries.
"""

import os
import unittest
from unittest.mock import patch

from phase1 import REFUSAL_MESSAGE, RetrievedChunk

from app.agents import graph as agent_graph
from app.agents.nodes import tools_node, track_best, web_results_to_chunks
from app.tools import base as tool_base
from app.tools import config as tool_config
from app.tools.calculator import (
    CalculatorTool,
    format_result,
    safe_evaluate,
)
from app.tools.registry import (
    ToolRegistry,
    default_registry,
    resolve_registry,
)
from app.tools.web_search import WebSearchTool


def tool_env(**overrides):
    base = {"SEARCH_BACKEND": "none",
            "SEARCH_MAX_RESULTS": "5",
            "SEARCH_TIMEOUT_SECONDS": "10",
            "CALCULATOR_MAX_EXPR_LEN": "200"}
    env = {k: v for k, v in os.environ.items()
           if k not in base and k not in ("TAVILY_API_KEY",
                                          "SERPAPI_API_KEY")}
    base.update(overrides)
    return patch.dict(os.environ, {**env, **base}, clear=True)


def agent_env(**overrides):
    base = {"AGENT_ENABLED": "false",
            "AGENT_MAX_ITERATIONS": "3",
            "AGENT_CONFIDENCE_THRESHOLD": "0.5"}
    base.update(overrides)
    return patch.dict(os.environ, base, clear=False)


def make_chunk(chunk_id="u:doc:0", distance=0.2, document_id="doc"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Grounded context about RAG.",
        distance=distance,
        metadata={"user_id": "alice", "document_id": document_id,
                  "source": "s.txt", "title": "T", "chunk_index": 0},
    )


class ToolConfigTest(unittest.TestCase):
    def test_defaults(self):
        with tool_env():
            self.assertEqual(tool_config.search_backend(), "none")
            self.assertEqual(tool_config.search_max_results(), 5)
            self.assertEqual(tool_config.search_timeout_seconds(), 10)
            self.assertEqual(tool_config.calculator_max_expr_len(), 200)
            self.assertEqual(tool_config.tavily_api_key(), "")
            self.assertEqual(tool_config.serpapi_api_key(), "")

    def test_backend_selection_and_unknown_falls_back_to_none(self):
        with tool_env(SEARCH_BACKEND="tavily"):
            self.assertEqual(tool_config.search_backend(), "tavily")
        with tool_env(SEARCH_BACKEND="SerpAPI"):
            self.assertEqual(tool_config.search_backend(), "serpapi")
        with tool_env(SEARCH_BACKEND="google"):
            self.assertEqual(tool_config.search_backend(), "none")

    def test_bounds_clamped_and_invalid_defaulted(self):
        with tool_env(SEARCH_MAX_RESULTS="0"):
            self.assertEqual(tool_config.search_max_results(), 1)
        with tool_env(SEARCH_MAX_RESULTS="99"):
            self.assertEqual(tool_config.search_max_results(), 10)
        with tool_env(SEARCH_MAX_RESULTS="many"):
            self.assertEqual(tool_config.search_max_results(), 5)
        with tool_env(SEARCH_TIMEOUT_SECONDS="0"):
            self.assertEqual(tool_config.search_timeout_seconds(), 1)
        with tool_env(CALCULATOR_MAX_EXPR_LEN="0"):
            self.assertEqual(tool_config.calculator_max_expr_len(), 1)


class CalculatorTest(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        self.assertEqual(safe_evaluate("2+2*3"), 8)
        self.assertEqual(safe_evaluate("(2+2)*3"), 12)
        self.assertEqual(safe_evaluate("2**10"), 1024)
        self.assertEqual(safe_evaluate("7 % 3"), 1)
        self.assertEqual(safe_evaluate("7 // 3"), 2)
        self.assertEqual(safe_evaluate("-4 + +2"), -2)
        self.assertAlmostEqual(safe_evaluate("1/3"), 1 / 3)

    def test_functions_and_constants(self):
        self.assertEqual(safe_evaluate("sqrt(144)"), 12.0)
        self.assertEqual(safe_evaluate("pow(2, 8)"), 256.0)
        self.assertEqual(safe_evaluate("factorial(5)"), 120)
        self.assertEqual(safe_evaluate("abs(-3)"), 3)
        self.assertAlmostEqual(safe_evaluate("sin(pi/2)"), 1.0)
        self.assertAlmostEqual(safe_evaluate("pi"), 3.141592653589793)

    def test_format_result_compact(self):
        self.assertEqual(format_result(96.0), "96")
        self.assertEqual(format_result(3.5), "3.5")

    def test_division_by_zero_is_an_error_dict(self):
        out = CalculatorTool().execute("1/0")
        self.assertFalse(out["ok"])
        self.assertIn("zero", out["error"])

    def test_injection_is_rejected(self):
        for evil in ("__import__('os').system('x')",
                     "eval('1+1')",
                     "open('/etc/passwd').read()",
                     "(lambda: 1)()",
                     "x = 1",
                     "os.system",
                     "[1,2][0]",
                     "True + 1"):
            with self.assertRaises(ValueError, msg=evil):
                safe_evaluate(evil)
            out = CalculatorTool().execute(evil)
            self.assertFalse(out["ok"], msg=evil)

    def test_empty_and_bad_input(self):
        self.assertFalse(CalculatorTool().execute("").get("ok"))
        self.assertFalse(CalculatorTool().execute("2+").get("ok"))
        with self.assertRaises(TypeError):
            CalculatorTool().execute(42)

    def test_match_claims_math_only(self):
        tool = CalculatorTool()
        self.assertEqual(tool.match("calculate 12*8"), "12*8")
        self.assertEqual(tool.match("What is 12 * 8?"), "12 * 8")
        self.assertEqual(tool.match("sqrt(144)"), "sqrt(144)")
        self.assertIsNone(tool.match("What is RAG?"))
        self.assertIsNone(tool.match("hi"))
        self.assertIsNone(tool.match("42"))
        self.assertIsNone(tool.match(""))

    def test_execute_success_shape(self):
        out = CalculatorTool().execute("12 * 8")
        self.assertEqual(out, {"ok": True, "expression": "12 * 8",
                               "result": 96, "display": "96"})


class WebSearchTest(unittest.TestCase):
    def test_match_claims_recency_only(self):
        tool = WebSearchTool()
        self.assertEqual(tool.match("latest AI news"),
                         "latest AI news")
        self.assertIsNotNone(tool.match("Who won the election?"))
        self.assertIsNotNone(tool.match("AI breakthroughs 2026"))
        self.assertIsNone(tool.match("What is RAG?"))
        self.assertIsNone(tool.match("Explain embeddings."))
        self.assertIsNone(tool.match(""))

    def test_unconfigured_backend_is_graceful(self):
        with tool_env():
            out = WebSearchTool().execute("latest AI news")
        self.assertFalse(out["ok"])
        self.assertIn("not configured", out["error"])

    def test_missing_key_names_the_env_var(self):
        with tool_env(SEARCH_BACKEND="tavily"):
            out = WebSearchTool().execute("latest AI news")
        self.assertFalse(out["ok"])
        self.assertIn("TAVILY_API_KEY", out["error"])
        with tool_env(SEARCH_BACKEND="serpapi"):
            out = WebSearchTool().execute("latest AI news")
        self.assertFalse(out["ok"])
        self.assertIn("SERPAPI_API_KEY", out["error"])

    def test_tavily_response_parsing(self):
        body = {"results": [
            {"title": "T1", "content": "S1", "url": "http://a"},
            {"title": "T2", "content": "S2", "url": "http://b"},
            {"title": "", "content": "", "url": "http://empty"},
        ]}
        with tool_env(SEARCH_BACKEND="tavily",
                       TAVILY_API_KEY="k"), \
                patch("app.tools.web_search._post_json",
                      return_value=body) as post:
            out = WebSearchTool().execute("latest AI news")
        self.assertTrue(out["ok"])
        self.assertEqual(out["backend"], "tavily")
        self.assertEqual(len(out["results"]), 2)
        self.assertEqual(out["results"][0],
                         {"title": "T1", "snippet": "S1",
                          "url": "http://a"})
        post.assert_called_once()

    def test_serpapi_response_parsing(self):
        body = {"organic_results": [
            {"title": "T", "snippet": "S", "link": "http://x"},
        ]}
        with tool_env(SEARCH_BACKEND="serpapi",
                       SERPAPI_API_KEY="k"), \
                patch("app.tools.web_search._get_json",
                      return_value=body):
            out = WebSearchTool().execute("election news")
        self.assertTrue(out["ok"])
        self.assertEqual(out["results"],
                         [{"title": "T", "snippet": "S",
                           "url": "http://x"}])

    def test_transport_failure_is_graceful(self):
        with tool_env(SEARCH_BACKEND="tavily",
                       TAVILY_API_KEY="k"), \
                patch("app.tools.web_search._post_json",
                      side_effect=TimeoutError("slow")):
            out = WebSearchTool().execute("latest news")
        self.assertFalse(out["ok"])
        self.assertIn("failed", out["error"])

    def test_empty_query_and_bad_type(self):
        with tool_env():
            self.assertFalse(
                WebSearchTool().execute("  ")["ok"])
        with self.assertRaises(TypeError):
            WebSearchTool().execute(None)


class RegistryTest(unittest.TestCase):
    def test_default_registry_order_and_cards(self):
        registry = default_registry()
        self.assertEqual([t.name for t in registry.all()],
                         ["calculator", "web_search"])
        cards = registry.describe()
        self.assertEqual(len(cards), 2)
        self.assertTrue(all("description" in c for c in cards))

    def test_get_and_unknown(self):
        registry = default_registry()
        self.assertIsInstance(registry.get("calculator"),
                              CalculatorTool)
        self.assertIsNone(registry.get("nope"))

    def test_select_priority_calculator_first(self):
        registry = default_registry()
        tool, tool_input = registry.select("calculate 2+2")
        self.assertEqual(tool.name, "calculator")
        self.assertEqual(tool_input, "2+2")

    def test_select_none_for_plain_factual(self):
        self.assertIsNone(default_registry().select("What is RAG?"))

    def test_register_custom_tool(self):
        class EchoTool(tool_base.BaseTool):
            name = "echo"
            description = "Echoes."
            def match(self, query):
                return query if query == "echo" else None
            def execute(self, tool_input):
                return {"ok": True, "echo": tool_input}

        registry = ToolRegistry([EchoTool()])
        self.assertEqual(registry.select("echo")[0].name, "echo")
        with self.assertRaises(TypeError):
            registry.register(object())

        class NamelessTool(tool_base.BaseTool):
            name = ""
            description = "No name."
            def match(self, query):
                return None
            def execute(self, tool_input):
                return {"ok": False, "error": "x"}

        with self.assertRaises(ValueError):
            registry.register(NamelessTool())

    def test_resolve_registry_variants(self):
        registry = default_registry()
        self.assertIs(resolve_registry(registry), registry)
        self.assertEqual(
            [t.name for t in resolve_registry([]).all()], [])
        custom = resolve_registry([CalculatorTool()])
        self.assertEqual([t.name for t in custom.all()],
                         ["calculator"])
        self.assertEqual(
            [t.name for t in resolve_registry(None).all()],
            ["calculator", "web_search"])


class FakeRAG:
    def __init__(self, chunks=None, answers=None, chunks_seq=None):
        self._chunks = list(chunks) if chunks is not None else [make_chunk()]
        self._chunks_seq = list(chunks_seq) if chunks_seq else None
        self._answers = list(answers or ["Grounded answer."])
        self.retrieve_calls = []
        self.generate_calls = []
        self.generate_chunks = []

    def _retrieve_chunks(self, query, user_id, k=3,
                         distance_threshold=None, use_hybrid=None,
                         hybrid_retriever=None, use_rerank=None,
                         reranker=None):
        self.retrieve_calls.append({"query": query, "k": k})
        if self._chunks_seq is not None:
            idx = min(len(self.retrieve_calls) - 1,
                      len(self._chunks_seq) - 1)
            return list(self._chunks_seq[idx])
        return list(self._chunks)

    def generate_answer(self, question, chunks, chat_history, usage=None):
        self.generate_calls.append({"question": question})
        self.generate_chunks.append(list(chunks))
        return self._answers[min(len(self.generate_calls) - 1,
                                 len(self._answers) - 1)]

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


class AgentToolsTest(unittest.TestCase):
    def test_calculator_answers_math_directly(self):
        rag = FakeRAG()
        service = FakeRAGService(rag)
        with agent_env(), tool_env():
            result = agent_graph.run_agent(
                service, raw_query="What is 12 * 8?", user_id="alice",
                use_agent=True)
        self.assertEqual(result["answer"], "The result of 12 * 8 is 96.")
        self.assertEqual(rag.retrieve_calls, [])
        self.assertEqual(rag.generate_calls, [])
        self.assertEqual(result["agent"]["tools_used"], ["calculator"])
        self.assertEqual(result["agent"]["attempts"], 1)
        self.assertEqual(result["agent"]["confidence"], 1.0)
        self.assertFalse(result["agent"]["gave_up"])

    def test_plain_factual_uses_no_tools(self):
        rag = FakeRAG()
        service = FakeRAGService(rag)
        with agent_env(), tool_env():
            result = agent_graph.run_agent(
                service, raw_query="What is RAG?", user_id="alice",
                use_agent=True)
        self.assertEqual(result["answer"], "Grounded answer.")
        self.assertEqual(result["agent"]["tools_used"], [])
        self.assertEqual(len(rag.retrieve_calls), 1)

    def test_web_search_grounds_current_events(self):
        body = {"results": [
            {"title": "Big launch", "content": "Announced today.",
             "url": "http://news/a"},
        ]}
        rag = FakeRAG()
        service = FakeRAGService(rag)
        with agent_env(), tool_env(SEARCH_BACKEND="tavily",
                                   TAVILY_API_KEY="k"), \
                patch("app.tools.web_search._post_json",
                      return_value=body):
            result = agent_graph.run_agent(
                service, raw_query="latest AI news today",
                user_id="alice", use_agent=True)
        self.assertEqual(result["answer"], "Grounded answer.")
        # Web chunks (not Chroma) grounded generation.
        self.assertEqual(rag.retrieve_calls, [])
        web_ids = [c.metadata["document_id"]
                   for c in rag.generate_chunks[0]]
        self.assertEqual(web_ids, ["web_search"])
        self.assertEqual(result["retrieved_document_ids"],
                         ["web_search"])
        self.assertEqual(result["agent"]["tools_used"], ["web_search"])
        self.assertFalse(result["agent"]["gave_up"])

    def test_unconfigured_search_falls_back_to_retrieval(self):
        rag = FakeRAG()
        service = FakeRAGService(rag)
        with agent_env(), tool_env():
            result = agent_graph.run_agent(
                service, raw_query="latest AI news today",
                user_id="alice", use_agent=True)
        self.assertEqual(len(rag.retrieve_calls), 1)
        self.assertEqual(result["answer"], "Grounded answer.")
        self.assertEqual(result["agent"]["tools_used"], [])

    def test_exploding_tool_never_breaks_the_turn(self):
        class BoomTool(tool_base.BaseTool):
            name = "boom"
            description = "Always explodes."
            def match(self, query):
                return query
            def execute(self, tool_input):
                raise RuntimeError("boom")

        rag = FakeRAG()
        service = FakeRAGService(rag)
        with agent_env(), tool_env():
            result = agent_graph.run_agent(
                service, raw_query="What is RAG?", user_id="alice",
                use_agent=True, tools=[BoomTool()])
        self.assertEqual(result["answer"], "Grounded answer.")
        self.assertEqual(len(rag.retrieve_calls), 1)

    def test_best_attempt_wins_across_retries(self):
        rag = FakeRAG(
            chunks_seq=[[make_chunk(distance=1.9)],
                        [make_chunk(distance=0.1)]],
            answers=["A mediocre statement.", "The great answer."])
        service = FakeRAGService(rag)
        with agent_env(), tool_env():
            result = agent_graph.run_agent(
                service, raw_query="Explain embeddings?",
                user_id="alice", use_agent=True)
        # Attempt 1 scores ~0.05 -> retry; attempt 2 scores ~0.95.
        self.assertEqual(result["answer"], "The great answer.")
        self.assertGreaterEqual(result["agent"]["confidence"], 0.5)
        self.assertFalse(result["agent"]["gave_up"])

    def test_tools_node_records_failed_tool(self):
        update = tools_node(
            {"raw_query": "latest news", "attempts": 0,
             "tool_calls": [], "tools_used": []},
            default_registry())
        # No backend configured in test env -> graceful miss.
        self.assertEqual(update["tool_step"], "retrieve")
        self.assertEqual(update["tools_used"], [])
        self.assertEqual(len(update["tool_calls"]), 1)
        self.assertFalse(update["tool_calls"][0]["output"]["ok"])

    def test_track_best_keeps_maximum(self):
        state = {"answer": "a", "confidence": 0.2, "chunks": [],
                 "usage": {}, "best_confidence": -1.0}
        update = track_best(state)
        self.assertEqual(update["best_confidence"], 0.2)
        state.update(update)
        state.update({"answer": "b", "confidence": 0.1})
        self.assertEqual(track_best(state), {})

    def test_web_results_to_chunks_shape(self):
        chunks = web_results_to_chunks(
            [{"title": "T", "snippet": "S", "url": "http://x"}])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["document_id"],
                         "web_search")
        self.assertEqual(chunks[0].metadata["source"], "http://x")
        self.assertEqual(web_results_to_chunks([]), [])

    def test_manual_loop_matches_with_tools(self):
        rag = FakeRAG()
        service = FakeRAGService(rag)
        with agent_env(), tool_env(), \
                patch.object(agent_graph, "build_graph",
                             side_effect=ImportError("no langgraph")):
            result = agent_graph.run_agent(
                service, raw_query="calculate 7*6", user_id="alice",
                use_agent=True)
        self.assertEqual(result["answer"], "The result of 7*6 is 42.")
        self.assertEqual(result["agent"]["tools_used"], ["calculator"])


if __name__ == "__main__":
    unittest.main()
