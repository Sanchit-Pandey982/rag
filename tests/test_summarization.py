"""Phase 1 tests: long-conversation summarization, no live services.

Real ConversationService + real orchestration against in-memory async
Mongo doubles; summarization is a fake callable (or a patched Gemini
client for the phase1 unit tests). Covers threshold gating, summary
storage, incremental reuse, re-summarization after max age, failure
fallback, and ownership scoping.
"""

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.schemas.chat import ChatRequest
from app.services import chat_orchestration as orchestration
from app.services.conversation_service import (
    SUMMARY_KEEP_RECENT,
    ConversationNotFound,
    ConversationService,
    summary_max_age,
    summary_threshold,
)


class FakeCursor:
    def __init__(self, documents):
        self._documents = list(documents)

    def sort(self, key, direction=1):
        self._documents.sort(
            key=lambda document: document.get(key),
            reverse=(direction < 0),
        )
        return self

    async def to_list(self, length=None):
        documents = self._documents[:length] if length else list(self._documents)
        return [dict(document) for document in documents]


class FakeCollection:
    """Minimal async Motor double: exact-match filters only (all we use)."""

    def __init__(self):
        self.documents = []

    async def create_index(self, keys, **kwargs):
        return "index"

    async def insert_one(self, document):
        self.documents.append(dict(document))
        return SimpleNamespace(inserted_id=document.get("message_id"))

    async def find_one(self, filt):
        for document in self.documents:
            if all(document.get(k) == v for k, v in filt.items()):
                return dict(document)
        return None

    def find(self, filt):
        return FakeCursor([
            document for document in self.documents
            if all(document.get(k) == v for k, v in filt.items())
        ])

    async def update_one(self, filt, update):
        matched = 0
        for document in self.documents:
            if all(document.get(k) == v for k, v in filt.items()):
                document.update(update.get("$set", {}))
                matched = 1
                break
        return SimpleNamespace(matched_count=matched)

    async def update_many(self, filt, update):
        matched = 0
        for document in self.documents:
            if all(document.get(k) == v for k, v in filt.items()):
                document.update(update.get("$set", {}))
                matched += 1
        return SimpleNamespace(matched_count=matched)


def run(coro):
    return asyncio.run(coro)


def fake_summarize_fn(messages, previous_summary=None):
    return f"Summary of {len(messages)} messages." + (
        " Extends previous." if previous_summary else ""
    )


class SummarizationTests(unittest.TestCase):
    def setUp(self):
        self.conversations = FakeCollection()
        self.messages = FakeCollection()
        self.service = ConversationService(self.conversations, self.messages)
        self.conversation_id = run(
            self.service.create_conversation("alice")
        )["conversation_id"]

    def complete_turns(self, count, start=0):
        for index in range(start, start + count):
            turn_id = run(self.service.start_turn(
                self.conversation_id, "alice", f"question {index}"
            ))
            run(self.service.complete_turn(
                self.conversation_id, "alice", turn_id, f"answer {index}"
            ))
        # Distinct timestamps in insertion order: Windows clock granularity
        # can stamp rapid inserts identically, and ties have no defined
        # user-before-assistant order.
        base = datetime.now(timezone.utc)
        for seq, document in enumerate(self.messages.documents):
            document["created_at"] = base + timedelta(milliseconds=seq)

    def context_history(self, summarize_fn=fake_summarize_fn):
        return run(orchestration.build_context_history(
            self.service, self.conversation_id, "alice",
            summarize_fn=summarize_fn,
        ))

    # -- threshold gating ------------------------------------------

    def test_short_conversation_returns_plain_history(self):
        calls = []

        def counting_fn(messages, previous_summary=None):
            calls.append((messages, previous_summary))
            return "unused"

        self.complete_turns(3)
        history = self.context_history(summarize_fn=counting_fn)
        self.assertEqual(calls, [])
        self.assertEqual(len(history), 6)
        self.assertTrue(all(item["role"] != "system" for item in history))

    def test_long_conversation_returns_summary_plus_recent(self):
        # 15 turns = 30 completed messages, over the default threshold of 20.
        self.complete_turns(15)
        history = self.context_history()
        self.assertGreater(
            run(self.service.count_completed_messages(
                self.conversation_id, "alice")),
            summary_threshold(),
        )
        self.assertEqual(history[0]["role"], "system")
        self.assertIn("Summary of 22 messages.", history[0]["content"])
        recent = history[1:]
        self.assertEqual(len(recent), SUMMARY_KEEP_RECENT)
        self.assertEqual(recent[-2:], [
            {"role": "user", "content": "question 14"},
            {"role": "assistant", "content": "answer 14"},
        ])

    def test_summary_is_stored_on_the_conversation(self):
        self.complete_turns(15)
        self.context_history()
        state = run(self.service.get_summary_state(
            self.conversation_id, "alice"))
        self.assertIn("Summary of 22 messages.", state["summary"])
        self.assertEqual(state["summary_message_count"], 22)

    # -- incremental behavior ---------------------------------------

    def test_summary_reused_before_max_age(self):
        self.complete_turns(15)
        self.context_history()
        calls = []

        def counting_fn(messages, previous_summary=None):
            calls.append((messages, previous_summary))
            return "fresh"

        history = self.context_history(summarize_fn=counting_fn)
        self.assertEqual(calls, [])
        self.assertIn("Summary of 22 messages.", history[0]["content"])

    def test_resummarize_after_max_age_new_messages(self):
        self.complete_turns(15)
        self.context_history()
        # Each turn adds 2 messages; max age defaults to 10.
        self.complete_turns(summary_max_age() // 2 + 1, start=15)
        seen = []

        def recording_fn(messages, previous_summary=None):
            seen.append(previous_summary)
            return "Updated summary."

        history = self.context_history(summarize_fn=recording_fn)
        self.assertEqual(len(seen), 1)
        self.assertIn("Summary of 22 messages.", seen[0])
        self.assertEqual(history[0]["content"], "Updated summary.")

    def test_failed_summarization_falls_back_to_recent(self):
        self.complete_turns(15)
        history = self.context_history(summarize_fn=lambda m, p=None: "")
        self.assertEqual(len(history), SUMMARY_KEEP_RECENT)
        self.assertTrue(all(item["role"] != "system" for item in history))
        state = run(self.service.get_summary_state(
            self.conversation_id, "alice"))
        self.assertIsNone(state["summary"])

    def test_failed_resummarize_keeps_old_summary(self):
        self.complete_turns(15)
        self.context_history()
        history = self.context_history(
            summarize_fn=Mock(side_effect=RuntimeError("LLM down")))
        self.assertIn("Summary of 22 messages.", history[0]["content"])

    # -- ownership ---------------------------------------------------

    def test_summary_state_scoped_to_owner(self):
        self.complete_turns(15)
        self.context_history()
        with self.assertRaises(ConversationNotFound):
            run(self.service.get_summary_state(self.conversation_id, "bob"))

    # -- prepare_chat integration ------------------------------------

    def test_prepare_chat_replaces_history_with_summary(self):
        self.complete_turns(15)
        payload = ChatRequest(
            raw_query="What did we decide?",
            user_id="alice",
            chat_history=[{"role": "user", "content": "forged"}],
            conversation_id=self.conversation_id,
        )
        rag_payload, turn_id = orchestration.prepare_chat(
            payload, self.service, summarize_fn=fake_summarize_fn
        )
        self.assertIsNotNone(turn_id)
        roles = [message.role for message in rag_payload.chat_history]
        self.assertEqual(roles[0], "system")
        self.assertEqual(len(roles), SUMMARY_KEEP_RECENT + 1)
        self.assertNotIn("forged", [
            message.content for message in rag_payload.chat_history])


class SummarizeFunctionTests(unittest.TestCase):
    def test_empty_input_returns_empty_without_llm(self):
        from phase1 import summarize_conversation_history
        with patch("phase1.get_gemini_client") as client_factory:
            result = summarize_conversation_history([], None)
        self.assertEqual(result, "")
        client_factory.assert_not_called()

    def test_summarize_returns_stripped_text(self):
        from phase1 import summarize_conversation_history
        fake_client = SimpleNamespace(models=SimpleNamespace(
            generate_content=Mock(return_value=SimpleNamespace(
                text="  A short summary.  "))
        ))
        with patch("phase1.get_gemini_client", return_value=fake_client):
            result = summarize_conversation_history(
                [{"role": "user", "content": "hello"}])
        self.assertEqual(result, "A short summary.")

    def test_summarize_failure_returns_empty(self):
        from phase1 import summarize_conversation_history
        fake_client = SimpleNamespace(models=SimpleNamespace(
            generate_content=Mock(side_effect=RuntimeError("boom"))
        ))
        with patch("phase1.get_gemini_client", return_value=fake_client):
            result = summarize_conversation_history(
                [{"role": "user", "content": "hello"}])
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
