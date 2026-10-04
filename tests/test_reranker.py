"""Phase 3a tests: cross-encoder reranking of retrieved chunks.

No live model, Redis, Mongo, or AI: the cross-encoder is a fake injected
into CrossEncoderReranker, and RAGSystem retrieval is stubbed. Covers
rescoring wins, top-k truncation, disabled-flag passthrough, graceful
fallback on model failure, lazy loading, and the _retrieve_chunks
integration (rerank runs after hybrid fetch, before generation).
"""

import os
import unittest
from unittest.mock import Mock, patch

from phase1 import RetrievedChunk

from app.services.reranker import (
    CrossEncoderReranker,
    reranker_enabled,
    reranker_model_name,
    reranker_top_k,
)


def make_chunk(chunk_id, text="text", distance=0.5):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        distance=distance,
        metadata={"user_id": "alice", "document_id": "doc",
                  "source": "s.txt", "title": "T", "chunk_index": 0},
    )


class FakeModel:
    """Scores pairs by keyword overlap: chunks mentioning the query's
    rare term score highest, simulating a cross-encoder preference."""

    def __init__(self, scores=None, fail=False):
        self._scores = scores
        self.fail = fail
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        if self.fail:
            raise RuntimeError("model boom")
        if self._scores is not None:
            return list(self._scores)
        return [1.0 if "zebracorn" in doc else 0.0 for _, doc in pairs]


class ConfigTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RERANKER_ENABLED", None)
            self.assertFalse(reranker_enabled())

    def test_model_default_and_override(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RERANKER_MODEL", None)
            self.assertEqual(reranker_model_name(),
                             "cross-encoder/ms-marco-MiniLM-L-6-v2")
        with patch.dict(os.environ, {"RERANKER_MODEL": "custom/model"}):
            self.assertEqual(reranker_model_name(), "custom/model")

    def test_top_k_default_and_invalid(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RERANKER_TOP_K", None)
            self.assertEqual(reranker_top_k(), 5)
        with patch.dict(os.environ, {"RERANKER_TOP_K": "not-a-number"}):
            self.assertEqual(reranker_top_k(), 5)
        with patch.dict(os.environ, {"RERANKER_TOP_K": "-3"}):
            self.assertEqual(reranker_top_k(), 5)


class RerankTests(unittest.TestCase):
    def test_rescoring_promotes_best_match(self):
        chunks = [make_chunk("v1", "generic database overview"),
                  make_chunk("v2", "generic caching overview"),
                  make_chunk("kw", "zebracorn error code XZ-9919 fix")]
        reranker = CrossEncoderReranker(model=FakeModel())
        out = reranker.rerank("zebracorn XZ-9919", chunks, top_k=3)
        self.assertEqual([c.chunk_id for c in out], ["kw", "v1", "v2"])

    def test_top_k_truncates(self):
        chunks = [make_chunk(f"c{i}") for i in range(5)]
        reranker = CrossEncoderReranker(
            model=FakeModel(scores=[1, 2, 3, 4, 5]))
        out = reranker.rerank("q", chunks, top_k=2)
        self.assertEqual([c.chunk_id for c in out], ["c4", "c3"])

    def test_top_k_defaults_to_env(self):
        chunks = [make_chunk(f"c{i}") for i in range(4)]
        reranker = CrossEncoderReranker(
            model=FakeModel(scores=[4, 3, 2, 1]))
        with patch.dict(os.environ, {"RERANKER_TOP_K": "2"}):
            out = reranker.rerank("q", chunks)
        self.assertEqual(len(out), 2)

    def test_empty_input_never_touches_model(self):
        model = FakeModel()
        reranker = CrossEncoderReranker(model=model)
        self.assertEqual(reranker.rerank("q", [], top_k=5), [])
        self.assertEqual(model.calls, [])

    def test_model_failure_falls_back_to_input_order(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        reranker = CrossEncoderReranker(model=FakeModel(fail=True))
        with self.assertLogs("app.services.reranker", level="ERROR"):
            out = reranker.rerank("q", chunks, top_k=2)
        self.assertEqual([c.chunk_id for c in out], ["a", "b"])

    def test_lazy_model_not_built_until_first_rerank(self):
        reranker = CrossEncoderReranker(model_name="some/model")
        self.assertIsNone(reranker._model)
        # Injecting a model directly never imports the package.
        reranker2 = CrossEncoderReranker(model=FakeModel(scores=[1.0]))
        out = reranker2.rerank("q", [make_chunk("a")], top_k=1)
        self.assertEqual(len(out), 1)

    def test_rerank_fused_disabled_truncates(self):
        chunks = [make_chunk(f"c{i}") for i in range(6)]
        reranker = CrossEncoderReranker(model=FakeModel())
        out = reranker.rerank_fused("q", chunks, k=3, use_rerank=False)
        self.assertEqual([c.chunk_id for c in out], ["c0", "c1", "c2"])
        # Model untouched when disabled.
        self.assertEqual(reranker._model.calls, [])

    def test_rerank_fused_enabled_rescores_to_k(self):
        chunks = [make_chunk("v1", "generic"),
                  make_chunk("kw", "zebracorn fix"),
                  make_chunk("v2", "generic")]
        reranker = CrossEncoderReranker(model=FakeModel())
        out = reranker.rerank_fused("zebracorn", chunks, k=2,
                                    use_rerank=True)
        self.assertEqual([c.chunk_id for c in out], ["kw", "v1"])


class RetrieveChunksIntegrationTests(unittest.TestCase):
    """_retrieve_chunks applies rerank after the retrieval legs."""

    def _rag_with_candidates(self, candidates):
        rag = Mock()
        rag.retrieve.return_value = list(candidates)
        # Avoid the cached-instance path interfering across tests.
        for attr in ("_hybrid_retriever", "_reranker"):
            if attr in rag.__dict__:
                del rag.__dict__[attr]
        return rag

    def test_rerank_runs_after_vector_fetch(self):
        from phase1 import RAGSystem
        candidates = [make_chunk("v1", "generic"),
                      make_chunk("kw", "zebracorn fix")]
        rag = self._rag_with_candidates(candidates)
        fake_reranker = CrossEncoderReranker(model=FakeModel())
        out = RAGSystem._retrieve_chunks(
            rag, "zebracorn", "alice", k=1,
            use_hybrid=False, use_rerank=True, reranker=fake_reranker)
        self.assertEqual([c.chunk_id for c in out], ["kw"])
        # Vector leg over-fetches the full window for reranking.
        self.assertEqual(rag.retrieve.call_args.kwargs["k"], 10)

    def test_disabled_rerank_preserves_retrieval_order(self):
        from phase1 import RAGSystem
        candidates = [make_chunk("v1"), make_chunk("v2")]
        rag = self._rag_with_candidates(candidates)
        out = RAGSystem._retrieve_chunks(
            rag, "q", "alice", k=2, use_hybrid=False, use_rerank=False)
        self.assertEqual([c.chunk_id for c in out], ["v1", "v2"])

    def test_reranker_failure_keeps_retrieval_order(self):
        from phase1 import RAGSystem
        candidates = [make_chunk("v1"), make_chunk("v2")]
        rag = self._rag_with_candidates(candidates)
        bad = CrossEncoderReranker(model=FakeModel(fail=True))
        with self.assertLogs("app.services.reranker", level="ERROR"):
            out = RAGSystem._retrieve_chunks(
                rag, "q", "alice", k=2, use_hybrid=False,
                use_rerank=True, reranker=bad)
        self.assertEqual([c.chunk_id for c in out], ["v1", "v2"])

    def test_hybrid_and_rerank_combine(self):
        from phase1 import RAGSystem
        fused = [make_chunk("h1"), make_chunk("h2"), make_chunk("h3")]
        fake_hybrid = Mock()
        fake_hybrid.hybrid_retrieve.return_value = list(fused)
        fake_reranker = CrossEncoderReranker(
            model=FakeModel(scores=[1.0, 3.0, 2.0]))
        rag = self._rag_with_candidates([])
        out = RAGSystem._retrieve_chunks(
            rag, "q", "alice", k=2,
            use_hybrid=True, hybrid_retriever=fake_hybrid,
            use_rerank=True, reranker=fake_reranker)
        fake_hybrid.hybrid_retrieve.assert_called_once()
        self.assertEqual(fake_hybrid.hybrid_retrieve.call_args.kwargs["k"],
                         10)
        self.assertEqual([c.chunk_id for c in out], ["h2", "h3"])


if __name__ == "__main__":
    unittest.main()
