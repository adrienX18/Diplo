"""Tests for network-based message filtering (by_network, resolve_network, search plan)."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from src.message_cache import MessageCache, resolve_network, NETWORK_ALIASES
from src.assistant import _execute_search, handle_user_message
from src.conversation import ConversationHistory


def make_msg(message_id, sender_name="Alice", text="hello", chat_title="Alice Chat",
             network="whatsapp", hours_ago=1, chat_id=None, timestamp=None):
    ts = timestamp or (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "message_id": message_id,
        "chat_id": chat_id or f"chat_{message_id}",
        "chat_title": chat_title,
        "network": network,
        "sender_name": sender_name,
        "text": text,
        "timestamp": ts,
        "has_attachments": False,
    }


@pytest.fixture
def cache(tmp_path):
    c = MessageCache(db_path=tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def convo(tmp_path):
    c = ConversationHistory(db_path=tmp_path / "test.db")
    yield c
    c.close()


# ---------- resolve_network ----------

class TestResolveNetwork:
    def test_messenger_aliases(self):
        assert resolve_network("messenger") == "facebookgo"
        assert resolve_network("facebook") == "facebookgo"
        assert resolve_network("facebook messenger") == "facebookgo"
        assert resolve_network("fb") == "facebookgo"
        assert resolve_network("fb messenger") == "facebookgo"

    def test_instagram_aliases(self):
        assert resolve_network("instagram") == "instagramgo"
        assert resolve_network("insta") == "instagramgo"
        assert resolve_network("ig") == "instagramgo"

    def test_imessage_aliases(self):
        assert resolve_network("imessage") == "imessagego"
        assert resolve_network("imsg") == "imessagego"

    def test_whatsapp_aliases(self):
        assert resolve_network("whatsapp") == "whatsapp"
        assert resolve_network("wa") == "whatsapp"

    def test_telegram_aliases(self):
        assert resolve_network("telegram") == "telegram"
        assert resolve_network("tg") == "telegram"

    def test_twitter_aliases(self):
        assert resolve_network("twitter") == "twitter"
        assert resolve_network("x") == "twitter"

    def test_other_networks(self):
        assert resolve_network("slack") == "slack"
        assert resolve_network("discord") == "discord"
        assert resolve_network("linkedin") == "linkedin"
        assert resolve_network("signal") == "signal"
        assert resolve_network("sms") == "sms"

    def test_beeper_alias(self):
        assert resolve_network("beeper") == "hungryserv"
        assert resolve_network("hungryserv") == "hungryserv"

    def test_case_insensitive(self):
        assert resolve_network("Messenger") == "facebookgo"
        assert resolve_network("INSTAGRAM") == "instagramgo"
        assert resolve_network("WhatsApp") == "whatsapp"
        assert resolve_network("Facebook Messenger") == "facebookgo"

    def test_strips_whitespace(self):
        assert resolve_network("  messenger  ") == "facebookgo"
        assert resolve_network(" instagram ") == "instagramgo"

    def test_unknown_passthrough(self):
        """Unknown network names are returned lowercased as-is."""
        assert resolve_network("somethingelse") == "somethingelse"
        assert resolve_network("NewPlatform") == "newplatform"

    def test_already_beeper_id(self):
        """Beeper internal IDs should pass through correctly."""
        assert resolve_network("facebookgo") == "facebookgo"
        assert resolve_network("instagramgo") == "instagramgo"


# ---------- MessageCache.by_network ----------

class TestByNetwork:
    def test_returns_messages_for_network(self, cache):
        cache.store(make_msg("m1", network="facebookgo", sender_name="Louis"))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Sophie"))
        cache.store(make_msg("m3", network="facebookgo", sender_name="Tom"))

        results = cache.by_network("facebookgo")
        assert len(results) == 2
        assert all(r["network"] == "facebookgo" for r in results)

    def test_accepts_natural_name(self, cache):
        """'messenger' should match 'facebookgo' messages."""
        cache.store(make_msg("m1", network="facebookgo", sender_name="Louis"))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Sophie"))

        results = cache.by_network("messenger")
        assert len(results) == 1
        assert results[0]["sender_name"] == "Louis"

    def test_accepts_instagram_alias(self, cache):
        cache.store(make_msg("m1", network="instagramgo", sender_name="Natasha"))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Sophie"))

        results = cache.by_network("insta")
        assert len(results) == 1
        assert results[0]["sender_name"] == "Natasha"

    def test_case_insensitive_query(self, cache):
        cache.store(make_msg("m1", network="facebookgo"))

        assert len(cache.by_network("MESSENGER")) == 1
        assert len(cache.by_network("Facebook")) == 1

    def test_empty_for_no_matches(self, cache):
        cache.store(make_msg("m1", network="whatsapp"))

        assert cache.by_network("messenger") == []

    def test_respects_limit(self, cache):
        for i in range(20):
            cache.store(make_msg(f"m{i}", network="facebookgo"))

        results = cache.by_network("messenger", limit=5)
        assert len(results) == 5

    def test_returns_newest_first(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg("m1", network="facebookgo",
                             timestamp=(now - timedelta(hours=3)).isoformat()))
        cache.store(make_msg("m2", network="facebookgo",
                             timestamp=(now - timedelta(hours=1)).isoformat()))

        results = cache.by_network("messenger")
        assert results[0]["message_id"] == "m2"
        assert results[1]["message_id"] == "m1"

    def test_whatsapp_passthrough(self, cache):
        cache.store(make_msg("m1", network="whatsapp"))
        assert len(cache.by_network("whatsapp")) == 1
        assert len(cache.by_network("wa")) == 1


def make_msg_with_ts(message_id, network="whatsapp", hours_ago=1, **kwargs):
    """Helper that lets you set timestamp via hours_ago."""
    defaults = {
        "sender_name": "Alice",
        "text": "hello",
        "chat_title": "Alice Chat",
        "chat_id": f"chat_{message_id}",
    }
    defaults.update(kwargs)
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "message_id": message_id,
        "network": network,
        "timestamp": ts,
        "has_attachments": False,
        **defaults,
    }


# ---------- _execute_search with network filter ----------

class TestExecuteSearchNetwork:
    def test_network_only(self, cache):
        cache.store(make_msg("m1", network="facebookgo", sender_name="Louis"))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Sophie"))
        cache.store(make_msg("m3", network="facebookgo", sender_name="Tom"))

        results = _execute_search({"network": "messenger"}, cache)
        assert len(results) == 2
        assert all(r["network"] == "facebookgo" for r in results)

    def test_network_with_sender(self, cache):
        """network + sender should intersect: only messages from that sender on that network."""
        cache.store(make_msg("m1", network="facebookgo", sender_name="Louis", chat_id="chat_louis"))
        cache.store(make_msg("m2", network="facebookgo", sender_name="Tom", chat_id="chat_tom"))
        cache.store(make_msg("m3", network="whatsapp", sender_name="Louis", chat_id="chat_louis_wa"))

        results = _execute_search({"network": "messenger", "sender": "Louis"}, cache)
        # Should only have Louis's messages on Messenger (+ full chat context for Louis's chats on messenger)
        assert all(r["network"] == "facebookgo" for r in results)
        networks = {r["network"] for r in results}
        assert networks == {"facebookgo"}

    def test_network_with_since_last_seen(self, cache):
        """network + since_last_seen should only return recent messages from that network."""
        now = datetime.now(timezone.utc)
        # Old message
        cache.store(make_msg("m1", network="facebookgo",
                             timestamp=(now - timedelta(hours=5)).isoformat()))
        # New message
        cache.store(make_msg("m2", network="facebookgo",
                             timestamp=(now - timedelta(minutes=10)).isoformat()))
        cache.store(make_msg("m3", network="whatsapp",
                             timestamp=(now - timedelta(minutes=5)).isoformat()))

        # Set last_seen to 1h ago
        cache._conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES ('last_seen_at', ?)",
            ((now - timedelta(hours=1)).isoformat(),),
        )
        cache._conn.commit()

        results = _execute_search({"network": "messenger", "since_last_seen": True}, cache)
        assert len(results) == 1
        assert results[0]["message_id"] == "m2"

    def test_network_with_hours(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg("m1", network="facebookgo",
                             timestamp=(now - timedelta(hours=5)).isoformat()))
        cache.store(make_msg("m2", network="facebookgo",
                             timestamp=(now - timedelta(hours=1)).isoformat()))

        results = _execute_search({"network": "messenger", "hours": 2}, cache)
        assert len(results) == 1
        assert results[0]["message_id"] == "m2"

    def test_network_with_search(self, cache):
        cache.store(make_msg("m1", network="facebookgo", text="let's grab coffee"))
        cache.store(make_msg("m2", network="facebookgo", text="see you later"))
        cache.store(make_msg("m3", network="whatsapp", text="coffee time"))

        results = _execute_search({"network": "messenger", "search": "coffee"}, cache)
        assert len(results) == 1
        assert results[0]["message_id"] == "m1"

    def test_network_is_specific_filter(self, cache):
        """Network alone should count as a specific filter — don't fall back to since_last_seen."""
        cache.store(make_msg("m1", network="facebookgo", sender_name="Louis"))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Sophie"))

        # Set last_seen so since_last_seen would return nothing
        cache.touch_last_seen()

        results = _execute_search({"network": "messenger"}, cache)
        assert len(results) == 1
        assert results[0]["network"] == "facebookgo"

    def test_network_no_results_does_not_fallback(self, cache):
        """If network filter returns nothing, that's the answer — no fallback."""
        cache.store(make_msg("m1", network="whatsapp"))

        results = _execute_search({"network": "signal"}, cache)
        assert results == []

    def test_network_only_with_hours(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg("m1", network="facebookgo",
                             timestamp=(now - timedelta(minutes=30)).isoformat()))
        cache.store(make_msg("m2", network="whatsapp",
                             timestamp=(now - timedelta(minutes=30)).isoformat()))

        results = _execute_search({"network": "messenger", "hours": 1}, cache)
        assert len(results) == 1
        assert results[0]["network"] == "facebookgo"

    def test_deduplicates_results(self, cache):
        """Network + sender on same chat shouldn't produce duplicates."""
        cache.store(make_msg("m1", network="facebookgo", sender_name="Louis", chat_id="chat1"))

        results = _execute_search({"network": "messenger", "sender": "Louis"}, cache)
        ids = [r["message_id"] for r in results]
        assert len(ids) == len(set(ids))


# ---------- Full integration: handle_user_message with network ----------

class TestHandleUserMessageNetwork:
    @pytest.fixture(autouse=True)
    def _clear_pending(self):
        import src.assistant
        src.assistant._pending_action = None

    @pytest.mark.asyncio
    async def test_messenger_query_returns_results(self, cache, convo):
        """'show me messenger messages' should find facebookgo messages."""
        cache.store(make_msg("m1", network="facebookgo", sender_name="Louis",
                             text="salut!", chat_title="Louis Gallais"))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Sophie",
                             text="hey"))

        mock_search_plan = json.dumps({"network": "messenger"})
        mock_response = "Louis sent you a message on Messenger: salut!"

        with patch("src.assistant.complete", new_callable=AsyncMock,
                   side_effect=["query", mock_search_plan, mock_response]):
            response, queried = await handle_user_message(
                "show me my messenger convos", cache, convo)

        assert queried is True
        assert "Louis" in response or "salut" in response or "Messenger" in response

    @pytest.mark.asyncio
    async def test_instagram_query_returns_results(self, cache, convo):
        cache.store(make_msg("m1", network="instagramgo", sender_name="Natasha",
                             text="hey cutie"))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Sophie",
                             text="hey"))

        mock_search_plan = json.dumps({"network": "instagram"})
        mock_response = "Natasha messaged you on Instagram."

        with patch("src.assistant.complete", new_callable=AsyncMock,
                   side_effect=["query", mock_search_plan, mock_response]):
            response, queried = await handle_user_message(
                "any instagram messages?", cache, convo)

        assert queried is True

    @pytest.mark.asyncio
    async def test_network_with_sender_in_plan(self, cache, convo):
        """'messenger messages from Louis' should filter by both."""
        cache.store(make_msg("m1", network="facebookgo", sender_name="Louis",
                             text="salut", chat_id="chat_louis"))
        cache.store(make_msg("m2", network="facebookgo", sender_name="Tom",
                             text="hey", chat_id="chat_tom"))

        mock_search_plan = json.dumps({"network": "messenger", "sender": "Louis"})
        mock_response = "Louis on Messenger: salut"

        with patch("src.assistant.complete", new_callable=AsyncMock,
                   side_effect=["query", mock_search_plan, mock_response]):
            response, queried = await handle_user_message(
                "facebook messages from Louis", cache, convo)

        assert queried is True
