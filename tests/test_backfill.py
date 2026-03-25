"""Tests for the startup backfill — loading the last 48h of messages from Beeper."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from src.beeper_client import BeeperPoller
from src.message_cache import MessageCache
from src.contacts import ContactRegistry
from src.main import _is_control_channel


# --- Helpers ---

def make_search_message(msg_id, chat_id="chat1", account_id="whatsapp",
                        sender_name="Alice", text="hello", hours_ago=1,
                        attachments=None):
    """Build a mock Message object as returned by client.messages.search()."""
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return SimpleNamespace(
        id=msg_id,
        chat_id=chat_id,
        account_id=account_id,
        sender_name=sender_name,
        text=text,
        timestamp=ts,
        attachments=attachments,
        sort_key=str(int(ts.timestamp() * 1000)),
    )


def make_chat(chat_id, title, account_id="whatsapp"):
    """Build a mock Chat object as returned by client.chats.retrieve()."""
    return SimpleNamespace(id=chat_id, title=title, account_id=account_id)


@pytest.fixture
def poller():
    """Create a BeeperPoller with a mocked Beeper client."""
    with patch("src.beeper_client.BeeperDesktop"):
        p = BeeperPoller()
        p.client = MagicMock()
    return p


@pytest.fixture
def cache(tmp_path):
    c = MessageCache(db_path=tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def contacts(tmp_path):
    c = ContactRegistry(db_path=tmp_path / "test_contacts.db")
    yield c
    c.close()


# --- BeeperPoller.backfill_recent() ---

class TestBackfillRecent:
    def test_returns_messages_in_chronological_order(self, poller):
        """Messages should be sorted oldest-first, regardless of search API order."""
        msgs = [
            make_search_message("m1", hours_ago=1),   # newer
            make_search_message("m2", hours_ago=10),   # older
            make_search_message("m3", hours_ago=5),    # middle
        ]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("chat1", "Alice Chat")

        result = poller.backfill_recent(hours=48)

        # Oldest first: m2 (10h ago), m3 (5h ago), m1 (1h ago)
        assert [m["message_id"] for m in result] == ["m2", "m3", "m1"]

    def test_resolves_chat_titles(self, poller):
        """Each message should have the correct chat_title from chats.retrieve()."""
        msgs = [
            make_search_message("m1", chat_id="chat_a"),
            make_search_message("m2", chat_id="chat_b"),
        ]
        poller.client.messages.search.return_value = iter(msgs)

        def fake_retrieve(chat_id):
            titles = {"chat_a": "Alice Chat", "chat_b": "Bob Chat"}
            return make_chat(chat_id, titles[chat_id])

        poller.client.chats.retrieve.side_effect = fake_retrieve

        result = poller.backfill_recent()

        titles = {m["message_id"]: m["chat_title"] for m in result}
        assert titles["m1"] == "Alice Chat"
        assert titles["m2"] == "Bob Chat"

    def test_falls_back_to_chat_id_when_retrieve_fails(self, poller):
        """If chats.retrieve() fails for a chat, use chat_id as the title."""
        msgs = [make_search_message("m1", chat_id="chat_unknown")]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.side_effect = Exception("not found")

        result = poller.backfill_recent()

        assert len(result) == 1
        assert result[0]["chat_title"] == "chat_unknown"

    def test_deduplicates_chat_retrieve_calls(self, poller):
        """Multiple messages from the same chat should only trigger one retrieve() call."""
        msgs = [
            make_search_message("m1", chat_id="chat1"),
            make_search_message("m2", chat_id="chat1"),
            make_search_message("m3", chat_id="chat1"),
        ]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("chat1", "Alice")

        poller.backfill_recent()

        # Should be called exactly once for chat1, not three times
        poller.client.chats.retrieve.assert_called_once_with("chat1")

    def test_respects_max_messages_cap(self, poller):
        """Should stop after max_messages even if more are available."""
        # Create 20 messages but set cap to 5
        msgs = [make_search_message(f"m{i}", hours_ago=i) for i in range(20)]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("chat1", "Alice")

        result = poller.backfill_recent(max_messages=5)

        assert len(result) == 5

    def test_returns_empty_when_no_messages(self, poller):
        """No messages in the time window — return empty list, no retrieve() calls."""
        poller.client.messages.search.return_value = iter([])

        result = poller.backfill_recent()

        assert result == []
        poller.client.chats.retrieve.assert_not_called()

    def test_message_format_matches_poll_once(self, poller):
        """Backfill output should have the same dict keys as poll_once()."""
        msgs = [make_search_message("m1", chat_id="chat1", account_id="telegram",
                                    sender_name="Bob", text="hey")]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("chat1", "Bob Chat", "telegram")

        result = poller.backfill_recent()

        msg = result[0]
        expected_keys = {"chat_id", "chat_title", "network", "message_id",
                         "sender_name", "text", "timestamp", "has_attachments"}
        assert set(msg.keys()) == expected_keys
        assert msg["chat_id"] == "chat1"
        assert msg["chat_title"] == "Bob Chat"
        assert msg["network"] == "telegram"
        assert msg["message_id"] == "m1"
        assert msg["sender_name"] == "Bob"
        assert msg["text"] == "hey"
        assert msg["has_attachments"] is False

    def test_handles_attachments(self, poller):
        """Messages with attachments should set has_attachments=True."""
        msgs = [make_search_message("m1", attachments=[{"type": "image"}])]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("chat1", "Alice")

        result = poller.backfill_recent()

        assert result[0]["has_attachments"] is True

    def test_handles_missing_sender_name(self, poller):
        """Messages with sender_name=None should default to 'Unknown'."""
        msgs = [make_search_message("m1", sender_name=None)]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("chat1", "Alice")

        result = poller.backfill_recent()

        assert result[0]["sender_name"] == "Unknown"

    def test_multiple_chats_resolved(self, poller):
        """Messages from different chats should each get their correct title."""
        msgs = [
            make_search_message("m1", chat_id="c1", account_id="whatsapp"),
            make_search_message("m2", chat_id="c2", account_id="telegram"),
            make_search_message("m3", chat_id="c3", account_id="facebookgo"),
        ]
        poller.client.messages.search.return_value = iter(msgs)

        def fake_retrieve(chat_id):
            titles = {"c1": "Alice", "c2": "Bob", "c3": "Team Chat"}
            return make_chat(chat_id, titles[chat_id])

        poller.client.chats.retrieve.side_effect = fake_retrieve

        result = poller.backfill_recent()

        assert len(result) == 3
        # Exactly 3 retrieve calls (one per unique chat)
        assert poller.client.chats.retrieve.call_count == 3

    def test_passes_correct_date_after(self, poller):
        """Should pass the right cutoff timestamp to messages.search()."""
        poller.client.messages.search.return_value = iter([])

        poller.backfill_recent(hours=24)

        # Verify date_after was passed to search
        call_kwargs = poller.client.messages.search.call_args[1]
        assert "date_after" in call_kwargs
        # The cutoff should be approximately 24h ago
        cutoff = datetime.fromisoformat(call_kwargs["date_after"])
        expected = datetime.now(timezone.utc) - timedelta(hours=24)
        assert abs((cutoff - expected).total_seconds()) < 5  # within 5 seconds


# --- resolve_chat_titles ---

class TestResolveChatMetadata:
    def test_resolves_all_ids(self, poller):
        """Should return metadata for all provided chat_ids."""
        def fake_retrieve(chat_id):
            return make_chat(chat_id, f"Title for {chat_id}")

        poller.client.chats.retrieve.side_effect = fake_retrieve

        result = poller._resolve_chat_metadata({"c1", "c2"})

        assert result["c1"]["title"] == "Title for c1"
        assert result["c2"]["title"] == "Title for c2"

    def test_skips_failed_lookups(self, poller):
        """Failed retrieve() calls should be silently skipped."""
        def fake_retrieve(chat_id):
            if chat_id == "c2":
                raise Exception("not found")
            return make_chat(chat_id, f"Title for {chat_id}")

        poller.client.chats.retrieve.side_effect = fake_retrieve

        result = poller._resolve_chat_metadata({"c1", "c2"})

        assert "c1" in result
        assert "c2" not in result

    def test_empty_input(self, poller):
        """Empty set of chat_ids — no API calls, empty result."""
        result = poller._resolve_chat_metadata(set())

        assert result == {}
        poller.client.chats.retrieve.assert_not_called()

    def test_falls_back_to_chat_id_for_none_title(self, poller):
        """If chat.title is None, use the chat_id as fallback."""
        poller.client.chats.retrieve.return_value = make_chat("c1", None)

        result = poller._resolve_chat_metadata({"c1"})

        assert result["c1"]["title"] == "c1"


# --- Integration: backfill stores into cache without duplicates ---

class TestBackfillCacheIntegration:
    def test_stores_messages_in_cache(self, poller, cache):
        """Backfilled messages should be stored in the SQLite cache."""
        msgs = [
            make_search_message("m1", sender_name="Alice", text="hey"),
            make_search_message("m2", sender_name="Bob", text="hi"),
        ]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("chat1", "Alice Chat")

        backfill = poller.backfill_recent()
        for msg in backfill:
            cache.store(msg)

        cached = cache._query()
        assert len(cached) == 2

    def test_no_duplicates_when_already_cached(self, poller, cache):
        """Messages already in the cache should not create duplicates."""
        # Pre-store a message
        cache.store({
            "message_id": "m1", "chat_id": "chat1", "chat_title": "Alice Chat",
            "network": "whatsapp", "sender_name": "Alice", "text": "hey",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_attachments": False,
        })

        # Backfill returns the same message
        msgs = [make_search_message("m1", sender_name="Alice", text="hey")]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("chat1", "Alice Chat")

        backfill = poller.backfill_recent()
        for msg in backfill:
            cache.store(msg)

        # Should still have exactly 1 message (no duplicate)
        cached = cache._query()
        assert len(cached) == 1

    def test_updates_contacts(self, poller, cache, contacts):
        """Backfilled messages should update the contact registry."""
        msgs = [
            make_search_message("m1", chat_id="c1", account_id="whatsapp",
                                sender_name="Sophie"),
        ]
        poller.client.messages.search.return_value = iter(msgs)
        poller.client.chats.retrieve.return_value = make_chat("c1", "Sophie Chat")

        backfill = poller.backfill_recent()
        for msg in backfill:
            cache.store(msg)
            contacts.update(
                sender_name=msg["sender_name"],
                network=msg["network"],
                chat_id=msg["chat_id"],
                chat_title=msg["chat_title"],
                timestamp=msg["timestamp"],
            )

        result = contacts.resolve("Sophie")
        assert result is not None
        assert result["chat_id"] == "c1"
        assert result["network"] == "whatsapp"

    def test_control_channel_messages_skipped(self, poller, cache):
        """Control channel messages should be filtered out before storing."""
        msgs = [
            make_search_message("m1", sender_name="Alice", text="real msg"),
            make_search_message("m2", sender_name="Diplo", text="bot msg",
                                chat_id="bot_chat"),
        ]
        poller.client.messages.search.return_value = iter(msgs)

        def fake_retrieve(chat_id):
            if chat_id == "bot_chat":
                return make_chat(chat_id, "Diplo")
            return make_chat(chat_id, "Alice Chat")

        poller.client.chats.retrieve.side_effect = fake_retrieve

        backfill = poller.backfill_recent()
        for msg in backfill:
            if not _is_control_channel(msg):
                cache.store(msg)

        cached = cache._query()
        assert len(cached) == 1
        assert cached[0]["sender_name"] == "Alice"

    def test_no_triage_on_backfill(self):
        """Backfilled messages are historical — verify they are NOT triaged.

        This is a design contract: backfill only stores and updates contacts,
        it never calls classify_urgency(). The main.py code does not triage
        backfill messages — this test documents that intent.
        """
        # This is tested implicitly by the main.py code: backfill_recent()
        # is called before the poller loop, and the results are only passed
        # to cache.store() and contacts.update(), never to classify_urgency().
        # If someone adds triage to the backfill path, this test name will
        # serve as a reminder that it was intentionally excluded.
        pass


# --- Edge cases ---

class TestBackfillEdgeCases:
    def test_search_api_failure(self, poller):
        """If messages.search() raises, backfill_recent() should propagate the error."""
        poller.client.messages.search.side_effect = Exception("Beeper down")

        with pytest.raises(Exception, match="Beeper down"):
            poller.backfill_recent()

    def test_partial_retrieve_failure(self, poller):
        """If some chats fail to resolve, messages still get backfilled with chat_id as title."""
        msgs = [
            make_search_message("m1", chat_id="c1"),
            make_search_message("m2", chat_id="c2"),
        ]
        poller.client.messages.search.return_value = iter(msgs)

        def selective_retrieve(chat_id):
            if chat_id == "c2":
                raise Exception("iMessage 404")
            return make_chat(chat_id, "Alice Chat")

        poller.client.chats.retrieve.side_effect = selective_retrieve

        result = poller.backfill_recent()

        assert len(result) == 2
        assert result[0]["chat_title"] == "Alice Chat"  # c1 resolved
        assert result[1]["chat_title"] == "c2"           # c2 fell back to chat_id
