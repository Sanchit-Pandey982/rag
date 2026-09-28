"""Phase 3.5: persistent MongoDB conversations and message history.

This service owns conversation persistence only. It never embeds, retrieves,
prompts, or generates -- those remain RAG responsibilities in ``phase1.py``.

Data model (two collections, so a long conversation can never outgrow a
single MongoDB document):

    conversations: conversation_id, user_id, title, created_at, updated_at
    messages:      message_id, conversation_id, user_id, role, content,
                   turn_id, status, created_at

Every field earns its place: ``user_id`` on both collections lets every
query enforce ownership without a join; ``turn_id`` pairs one user message
with its assistant reply; ``status`` separates completed turns (valid RAG
context) from pending/failed/cancelled ones (diagnostics only);
``created_at`` gives chronological order; ``*_id`` values are backend
uuids, never browser-chosen.
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

# Bounded context sent to the model: MongoDB keeps the full conversation,
# RAG receives only the most recent completed messages (matches the
# ChatRequest.chat_history cap of 20).
RAG_HISTORY_LIMIT = 20

COMPLETED = "completed"
PENDING = "pending"
FAILED = "failed"
CANCELLED = "cancelled"


class ConversationNotFound(Exception):
    """No conversation exists for this id *and* authenticated user.

    Missing and not-owned map to the same error so callers return 404 for
    both without leaking whether another user's conversation exists.
    """


class ConversationStoreUnavailable(Exception):
    """MongoDB could not be read or changed; never surfaces internals."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationService:
    def __init__(self, conversations, messages):
        self.conversations = conversations
        self.messages = messages

    async def ensure_indexes(self) -> None:
        # Uniqueness reinforces correctness: backend uuids must never collide.
        await self.conversations.create_index("conversation_id", unique=True)
        # Supports "list my conversations, most recently active first".
        await self.conversations.create_index([("user_id", 1), ("updated_at", -1)])
        await self.messages.create_index("message_id", unique=True)
        # Supports the ownership-scoped chronological history load
        # (find user+conversation, sort by time) from a single index.
        await self.messages.create_index(
            [("user_id", 1), ("conversation_id", 1), ("created_at", 1)]
        )

    async def create_conversation(
        self, user_id: str, title: str | None = None
    ) -> dict[str, Any]:
        now = _utcnow()
        document = {
            "conversation_id": str(uuid4()),
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self.conversations.insert_one(document)
        except PyMongoError as error:
            logger.exception("Could not create conversation")
            raise ConversationStoreUnavailable("Conversation store unavailable") from error
        return document

    async def get_conversation(
        self, conversation_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Ownership-scoped lookup: id alone never grants access."""
        try:
            return await self.conversations.find_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
            })
        except PyMongoError as error:
            logger.exception("Could not read conversation")
            raise ConversationStoreUnavailable("Conversation store unavailable") from error

    async def list_conversations(
        self, user_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        try:
            cursor = (
                self.conversations.find({"user_id": user_id})
                .sort("updated_at", -1)
            )
            return await cursor.to_list(length=limit)
        except PyMongoError as error:
            logger.exception("Could not list conversations")
            raise ConversationStoreUnavailable("Conversation store unavailable") from error

    async def load_chat_history(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = RAG_HISTORY_LIMIT,
    ) -> list[dict[str, str]]:
        """Recent *completed* turns as RAG ``chat_history`` dicts.

        Only ``status == completed`` messages are returned, oldest first, so
        failed/cancelled partials never enter query rewriting or generation.
        """
        try:
            cursor = (
                self.messages.find({
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "status": COMPLETED,
                })
                .sort("created_at", -1)
            )
            documents = await cursor.to_list(length=limit)
        except PyMongoError as error:
            logger.exception("Could not load conversation history")
            raise ConversationStoreUnavailable("Conversation store unavailable") from error
        documents.reverse()
        return [
            {"role": document["role"], "content": document["content"]}
            for document in documents
        ]

    async def list_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Full turn record (all statuses) for UI/diagnostics, oldest first."""
        try:
            cursor = (
                self.messages.find({
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                })
                .sort("created_at", 1)
            )
            return await cursor.to_list(length=limit)
        except PyMongoError as error:
            logger.exception("Could not list conversation messages")
            raise ConversationStoreUnavailable("Conversation store unavailable") from error

    async def start_turn(
        self, conversation_id: str, user_id: str, content: str
    ) -> str:
        """Persist the user message as pending and open a turn.

        The user message starts as ``pending`` (not ``completed``) so a turn
        that never finishes leaves no orphan user message in future RAG
        history. Raises ConversationNotFound when the id is unknown or owned
        by someone else.
        """
        conversation = await self.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)
        turn_id = str(uuid4())
        try:
            await self.messages.insert_one({
                "message_id": str(uuid4()),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": content,
                "turn_id": turn_id,
                "status": PENDING,
                "created_at": _utcnow(),
            })
        except PyMongoError as error:
            logger.exception("Could not save user message")
            raise ConversationStoreUnavailable("Conversation store unavailable") from error
        return turn_id

    async def complete_turn(
        self,
        conversation_id: str,
        user_id: str,
        turn_id: str,
        assistant_content: str,
    ) -> None:
        """Mark the turn completed and persist one logical assistant message."""
        now = _utcnow()
        try:
            await self.messages.update_many(
                {"turn_id": turn_id, "user_id": user_id, "role": "user"},
                {"$set": {"status": COMPLETED}},
            )
            await self.messages.insert_one({
                "message_id": str(uuid4()),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": assistant_content,
                "turn_id": turn_id,
                "status": COMPLETED,
                "created_at": now,
            })
            await self.conversations.update_one(
                {"conversation_id": conversation_id, "user_id": user_id},
                {"$set": {"updated_at": now}},
            )
        except PyMongoError as error:
            logger.exception("Could not save completed turn")
            raise ConversationStoreUnavailable("Conversation store unavailable") from error

    async def fail_turn(
        self,
        conversation_id: str,
        user_id: str,
        turn_id: str,
        status: str = FAILED,
        partial_content: str = "",
    ) -> None:
        """Keep a failed/cancelled turn out of future RAG history.

        The partial assistant text is stored with a non-completed status for
        diagnostics only; ``load_chat_history`` excludes it, and the pending
        user message is flipped to the same terminal status so it cannot
        become an orphan completed message either.
        """
        if status not in (FAILED, CANCELLED):
            raise ValueError("fail_turn status must be failed or cancelled")
        try:
            await self.messages.update_many(
                {"turn_id": turn_id, "user_id": user_id, "role": "user"},
                {"$set": {"status": status}},
            )
            await self.messages.insert_one({
                "message_id": str(uuid4()),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": partial_content,
                "turn_id": turn_id,
                "status": status,
                "created_at": _utcnow(),
            })
        except PyMongoError as error:
            logger.exception("Could not record failed turn")
            raise ConversationStoreUnavailable("Conversation store unavailable") from error
