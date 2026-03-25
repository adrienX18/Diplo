"""Tests for the Sonnet-powered assistant."""

import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta, timezone

from src.assistant import (
    _execute_search, handle_user_message, _to_local, _pending_action, _parse_json,
    _find_dm_chat, _cache_sent_message, _USER_SENDER_LABEL, _feedback_ack, _FEEDBACK_ACKS,
    _set_pending, _get_pending, _pending_action_lock, PENDING_ACTION_TTL_SECONDS,
    _last_diplo_turn_was_empty, _display_sender, _is_owner_recipient,
    _route_intent, _extract_query_plan, _extract_reply_plan,
    _extract_automation_plan, _parse_intent,
    _extract_debug_plan, _handle_debug, _format_debug_entries,
)
from src.conversation import ConversationHistory
from src.message_cache import MessageCache
from src.contacts import ContactRegistry
from src.config import USER_NAME, USER_SENDER_IDS


def make_msg(message_id, sender_name="Alice", text="hello", chat_title="Alice Chat",
             network="whatsapp", hours_ago=1, chat_id=None):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
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
def contacts(tmp_path):
    c = ContactRegistry(db_path=tmp_path / "test_contacts.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _clear_pending_action():
    """Reset pending action state between tests."""
    import src.assistant
    src.assistant._pending_action = None
    yield
    src.assistant._pending_action = None


class TestPendingActionTTL:
    def test_fresh_action_is_returned(self):
        _set_pending({"chat_id": "chat1", "text": "hi"})
        assert _get_pending() is not None
        assert _get_pending()["chat_id"] == "chat1"

    def test_expired_action_returns_none(self):
        _set_pending({"chat_id": "chat1", "text": "hi"})
        # Backdate the creation time past the TTL
        import src.assistant
        src.assistant._pending_action["_created_at"] = time.monotonic() - PENDING_ACTION_TTL_SECONDS - 1
        assert _get_pending() is None
        # Should also clear the global
        assert src.assistant._pending_action is None

    def test_set_pending_none_clears(self):
        _set_pending({"chat_id": "chat1", "text": "hi"})
        _set_pending(None)
        assert _get_pending() is None

    def test_set_pending_stamps_created_at(self):
        before = time.monotonic()
        _set_pending({"chat_id": "chat1"})
        after = time.monotonic()
        pending = _get_pending()
        assert before <= pending["_created_at"] <= after

    @pytest.mark.asyncio
    async def test_expired_pending_ignored_in_handle_user_message(self, cache):
        """A stale 'yes' should NOT trigger the expired pending action."""
        import src.assistant
        _set_pending({
            "chat_id": "chat1",
            "chat_title": "Sophie",
            "text": "stale message",
            "recipient_name": "Sophie",
            "network": "whatsapp",
        })
        # Expire it
        src.assistant._pending_action["_created_at"] = time.monotonic() - PENDING_ACTION_TTL_SECONDS - 1

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "casual",  # router
                "Hey!",  # Opus response
            ]
            result, queried = await handle_user_message("yes", cache)

        # Should NOT have sent anything — the pending action was expired
        assert "Sent" not in result
        assert src.assistant._pending_action is None


    @pytest.mark.asyncio
    async def test_modify_preserves_pending_with_valid_ttl(self, cache):
        """After modify, the pending action is still valid (not expired)."""
        import src.assistant

        _set_pending({
            "chat_id": "chat_sophie",
            "chat_title": "Sophie",
            "text": "I'll be there tonight",
            "recipient_name": "Sophie",
            "network": "whatsapp",
        })

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = json.dumps({
                "action": "modify",
                "text": "I'll be there tomorrow",
                "response": "Updated. Send it?",
            })
            result, queried = await handle_user_message("change tonight to tomorrow", cache)

        # Pending action should still exist with updated text
        pending = _get_pending()
        assert pending is not None
        assert pending["text"] == "I'll be there tomorrow"
        # TTL should still be valid (not expired by the modify)
        assert "_created_at" in pending


class TestPendingActionLock:
    @pytest.mark.asyncio
    async def test_concurrent_messages_do_not_double_send(self, cache):
        """Two rapid messages should not both trigger the same pending action."""
        import src.assistant

        client = MagicMock()
        send_count = 0

        async def mock_send(c, chat_id, text):
            nonlocal send_count
            send_count += 1
            return True

        _set_pending({
            "chat_id": "chat_sophie",
            "chat_title": "Sophie",
            "text": "draft message",
            "recipient_name": "Sophie",
            "network": "whatsapp",
        })

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.send_message", side_effect=mock_send):
            mock_complete.return_value = '{"action": "confirm", "text": "draft message"}'

            # Fire two "yes" messages concurrently
            results = await asyncio.gather(
                handle_user_message("yes", cache, beeper_client=client),
                handle_user_message("yes", cache, beeper_client=client),
            )

        # Exactly one should have sent, the other should have gone through normal flow
        sent_results = [r for r, _ in results if "Sent" in r]
        assert len(sent_results) == 1
        assert send_count == 1


class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"action": "send"}') == {"action": "send"}

    def test_json_in_code_block(self):
        raw = '```json\n{"action": "send", "text": "hey"}\n```'
        assert _parse_json(raw) == {"action": "send", "text": "hey"}

    def test_json_in_bare_code_block(self):
        raw = '```\n{"no_query": true}\n```'
        assert _parse_json(raw) == {"no_query": True}

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json("not json at all")

    def test_whitespace_around_json(self):
        assert _parse_json('  {"action": "send"}  ') == {"action": "send"}

    def test_json_after_prose(self):
        raw = 'Here is my analysis of the message.\n\n{"action": "send", "text": "hey"}'
        assert _parse_json(raw) == {"action": "send", "text": "hey"}


class TestExecuteSearch:
    def test_recent_by_hours(self, cache):
        cache.store(make_msg("m1", hours_ago=2))
        cache.store(make_msg("m2", hours_ago=30))

        results = _execute_search({"hours": 24}, cache)
        assert len(results) == 1

    def test_filter_by_sender(self, cache):
        cache.store(make_msg("m1", sender_name="Sophie"))
        cache.store(make_msg("m2", sender_name="Bob"))

        results = _execute_search({"sender": "Sophie"}, cache)
        assert len(results) == 1
        assert results[0]["sender_name"] == "Sophie"

    def test_search_text(self, cache):
        cache.store(make_msg("m1", text="fundraising deck is ready"))
        cache.store(make_msg("m2", text="want coffee?"))

        results = _execute_search({"search": "fundraising"}, cache)
        assert len(results) == 1

    def test_filter_by_chat(self, cache):
        cache.store(make_msg("m1", chat_title="Team Chat"))
        cache.store(make_msg("m2", chat_title="Sophie"))

        results = _execute_search({"chat": "Team"}, cache)
        assert len(results) == 1

    def test_deduplicates_results(self, cache):
        cache.store(make_msg("m1", sender_name="Sophie", text="Sophie says fundraising"))

        # This message matches both sender and search
        results = _execute_search({"sender": "Sophie", "search": "fundraising"}, cache)
        assert len(results) == 1

    def test_time_filter_with_sender(self, cache):
        cache.store(make_msg("m1", sender_name="Sophie", hours_ago=2))
        cache.store(make_msg("m2", sender_name="Sophie", hours_ago=50))

        results = _execute_search({"sender": "Sophie", "hours": 24}, cache)
        assert len(results) == 1

    def test_results_sorted_chronologically(self, cache):
        cache.store(make_msg("m1", hours_ago=5))
        cache.store(make_msg("m2", hours_ago=1))
        cache.store(make_msg("m3", hours_ago=3))

        results = _execute_search({"hours": 24}, cache)
        assert len(results) == 3
        # Oldest first
        assert results[0]["message_id"] == "m1"
        assert results[2]["message_id"] == "m2"

    def test_empty_cache_returns_empty(self, cache):
        results = _execute_search({"hours": 24}, cache)
        assert results == []

    def test_specific_filter_with_no_matches_returns_empty(self, cache):
        """If sender filter matches nothing, should NOT fall back to all messages."""
        cache.store(make_msg("m1", sender_name="Bob", text="hey"))
        cache.store(make_msg("m2", sender_name="Charlie", text="yo"))

        results = _execute_search({"sender": "Sophie", "since_last_seen": True}, cache)
        assert results == []

    def test_specific_search_with_no_matches_returns_empty(self, cache):
        """If text search matches nothing, should NOT fall back to all messages."""
        cache.store(make_msg("m1", text="let's get coffee"))

        results = _execute_search({"search": "fundraising"}, cache)
        assert results == []

    def test_specific_chat_with_no_matches_returns_empty(self, cache):
        """If chat filter matches nothing, should NOT fall back to all messages."""
        cache.store(make_msg("m1", chat_title="Team Chat"))

        results = _execute_search({"chat": "Nonexistent"}, cache)
        assert results == []


class TestToLocal:
    def test_converts_utc_to_pacific(self):
        # 2026-03-05T06:00:00Z = 2026-03-04 22:00 PST (UTC-8, before DST)
        result = _to_local("2026-03-05T06:00:00+00:00", "America/Los_Angeles")
        assert result == "2026-03-04 22:00"

    def test_converts_utc_to_paris(self):
        # 2026-03-05T06:00:00Z = 2026-03-05 07:00 CET (UTC+1)
        result = _to_local("2026-03-05T06:00:00+00:00", "Europe/Paris")
        assert result == "2026-03-05 07:00"

    def test_handles_naive_timestamp(self):
        # Naive timestamps are treated as UTC
        result = _to_local("2026-03-05T06:00:00", "America/Los_Angeles")
        assert result == "2026-03-04 22:00"

    def test_fallback_on_invalid_timezone(self):
        result = _to_local("2026-03-05T06:00:00+00:00", "Invalid/Zone")
        assert result == "2026-03-05 06:00"


class TestHandleUserMessage:
    @pytest.mark.asyncio
    async def test_end_to_end(self, cache):
        cache.store(make_msg("m1", sender_name="Sophie", text="meeting at 3pm"))
        cache.store(make_msg("m2", sender_name="Bob", text="lunch?"))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"sender": "Sophie"}',  # query extractor
                "Sophie said she has a meeting at 3pm.",  # Opus response
            ]
            result, queried = await handle_user_message("what did Sophie say?", cache)

        assert "Sophie" in result
        assert "3pm" in result
        assert queried is True


class TestReplyAction:
    @pytest.mark.asyncio
    async def test_direct_send(self, cache, contacts):
        """Simple reply sends directly without confirmation."""
        contacts.update("Sophie", "whatsapp", "chat_sophie", "Sophie", "2026-03-05T10:00:00+00:00")
        cache.store(make_msg("m1", sender_name="Sophie", text="are you coming?", chat_title="Sophie"))

        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "I\'ll be 5 min late"}',  # reply extractor
                '{"action": "send", "text": "I\'ll be 5 min late!"}',  # compose decision
            ]
            result, queried = await handle_user_message(
                "tell Sophie I'll be 5 min late", cache, contacts=contacts, beeper_client=client,
            )

        assert "Sent" in result
        assert "Sophie" in result
        assert "I'll be 5 min late!" in result  # sent message content in convo history
        client.messages.send.assert_called_once_with("chat_sophie", text="I'll be 5 min late!")
        assert queried is False

    @pytest.mark.asyncio
    async def test_confirm_flow(self, cache, contacts):
        """Complex reply asks for confirmation, then sends on 'yes'."""
        import src.assistant

        contacts.update("Marc", "linkedin", "chat_marc", "Marc", "2026-03-05T10:00:00+00:00")

        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        # Step 1: Adrien asks to reply, Opus decides to confirm
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            confirm_response = json.dumps({
                "action": "confirm",
                "text": "Hi Marc, we are interested but would like to discuss terms.",
                "response": "I'd send this to Marc on LinkedIn: \"Hi Marc, we are interested but would like to discuss terms.\" Go ahead?",
            })
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Marc", "message": "interested but need to discuss terms"}',  # reply extractor
                confirm_response,
            ]
            result, queried = await handle_user_message(
                "reply to Marc saying we're interested but need to discuss terms",
                cache, contacts=contacts, beeper_client=client,
            )

        assert "Go ahead" in result or "Marc" in result
        assert src.assistant._pending_action is not None
        client.messages.send.assert_not_called()

        # Step 2: Adrien confirms
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"action": "confirm", "text": "Hi Marc, we are interested but would like to discuss terms."}'
            result, queried = await handle_user_message(
                "yes", cache, contacts=contacts, beeper_client=client,
            )

        assert "Sent" in result
        client.messages.send.assert_called_once()
        assert src.assistant._pending_action is None

    @pytest.mark.asyncio
    async def test_cancel_pending(self, cache, contacts):
        """Adrien cancels a pending reply."""
        import src.assistant

        _set_pending({
            "chat_id": "chat1",
            "chat_title": "Sophie",
            "text": "draft message",
            "recipient_name": "Sophie",
            "network": "whatsapp",
        })

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"action": "cancel", "response": "Got it, cancelled."}'
            result, queried = await handle_user_message("nevermind", cache)

        assert "cancel" in result.lower()
        assert src.assistant._pending_action is None

    @pytest.mark.asyncio
    async def test_unrelated_clears_pending(self, cache, contacts):
        """An unrelated message clears the pending action and processes normally."""
        import src.assistant

        _set_pending({
            "chat_id": "chat1",
            "chat_title": "Sophie",
            "text": "draft message",
            "recipient_name": "Sophie",
            "network": "whatsapp",
        })

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                '{"action": "unrelated"}',  # confirmation check
                "query",  # router
                '{"since_last_seen": true}',  # query extractor
                "Nothing new.",  # Opus response
            ]
            result, queried = await handle_user_message("what's new?", cache)

        assert src.assistant._pending_action is None
        assert queried is True

    @pytest.mark.asyncio
    async def test_recipient_not_found(self, cache, contacts):
        """Reply to unknown person returns helpful message."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Nobody", "message": "hello"}',  # reply extractor
            ]
            result, queried = await handle_user_message(
                "tell Nobody hello", cache, contacts=contacts,
            )

        assert "couldn't find" in result.lower()

    @pytest.mark.asyncio
    async def test_network_override(self, cache, contacts):
        """Specifying a network uses that network even if not most recent."""
        contacts.update("Sophie", "whatsapp", "chat_wa", "Sophie WA", "2026-03-01T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat_tg", "Sophie TG", "2026-03-05T14:00:00+00:00")

        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "hey", "network": "whatsapp"}',  # reply extractor
                '{"action": "send", "text": "Hey!"}',  # compose decision
            ]
            result, queried = await handle_user_message(
                "tell Sophie on whatsapp hey", cache, contacts=contacts, beeper_client=client,
            )

        assert "Sent" in result
        assert "whatsapp" in result
        client.messages.send.assert_called_once_with("chat_wa", text="Hey!")

    @pytest.mark.asyncio
    async def test_send_failure(self, cache, contacts):
        """Beeper send failure is reported to Adrien."""
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")

        client = MagicMock()
        client.messages.send = MagicMock(side_effect=Exception("network error"))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "hey"}',  # reply extractor
                '{"action": "send", "text": "Hey!"}',  # compose decision
            ]
            result, queried = await handle_user_message(
                "tell Sophie hey", cache, contacts=contacts, beeper_client=client,
            )

        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_modify_flow(self, cache, contacts):
        """Adrien modifies a pending reply before sending."""
        import src.assistant

        _set_pending({
            "chat_id": "chat_sophie",
            "chat_title": "Sophie",
            "text": "I'll be there tonight",
            "recipient_name": "Sophie",
            "network": "whatsapp",
        })

        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        # Step 1: Adrien says to change it
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = json.dumps({
                "action": "modify",
                "text": "I'll be there tomorrow",
                "response": "Updated to tomorrow. Send it?",
            })
            result, queried = await handle_user_message(
                "change tonight to tomorrow", cache, contacts=contacts, beeper_client=client,
            )

        assert "tomorrow" in result.lower()
        assert src.assistant._pending_action is not None
        assert src.assistant._pending_action["text"] == "I'll be there tomorrow"
        client.messages.send.assert_not_called()

        # Step 2: Adrien confirms modified version
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"action": "confirm", "text": "I\'ll be there tomorrow"}'
            result, queried = await handle_user_message(
                "yes", cache, contacts=contacts, beeper_client=client,
            )

        assert "Sent" in result
        assert "I'll be there tomorrow" in result  # sent message content in response
        client.messages.send.assert_called_once_with("chat_sophie", text="I'll be there tomorrow")

        # Verify the sent message was cached
        msgs = cache.by_chat_id("chat_sophie", limit=10)
        sent = [m for m in msgs if m["message_id"].startswith("sent_")]
        assert len(sent) == 1
        assert sent[0]["text"] == "I'll be there tomorrow"

    @pytest.mark.asyncio
    async def test_compose_json_parse_failure(self, cache, contacts):
        """If Opus returns garbage, fall back to confirmation with raw intent."""
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "thanks for the help"}',  # reply extractor
                "Sure, I'll compose that for you!",  # invalid JSON from Opus
            ]
            result, queried = await handle_user_message(
                "tell Sophie thanks for the help", cache, contacts=contacts,
            )

        # Should fall back to a confirmation with the raw intent
        assert "Sophie" in result
        assert "thanks for the help" in result

    @pytest.mark.asyncio
    async def test_reply_without_beeper_client(self, cache, contacts):
        """Reply when beeper_client is None gives a clear error."""
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "hey"}',  # reply extractor
                '{"action": "send", "text": "Hey!"}',  # compose decision
            ]
            result, queried = await handle_user_message(
                "tell Sophie hey", cache, contacts=contacts, beeper_client=None,
            )

        assert "can't send" in result.lower()


class TestFindDmChat:
    @pytest.mark.asyncio
    async def test_registers_chat_title_not_user_typed_name(self, contacts):
        """_find_dm_chat should register the Beeper chat title, not what Adrien typed."""
        mock_chat = MagicMock()
        mock_chat.title = "Sophie Martin"
        mock_chat.id = "chat_dm_sophie"
        mock_chat.account_id = "whatsapp"
        mock_chat.type = "single"

        client = MagicMock()
        client.chats.search = MagicMock(return_value=[mock_chat])

        result = await _find_dm_chat(client, "Sophie", contacts)

        assert result is not None
        assert result["sender_name"] == "Sophie Martin"  # Beeper's name, not "Sophie"
        assert result["chat_id"] == "chat_dm_sophie"

        # Verify the contact was registered under the canonical name
        stored = contacts.lookup("Sophie Martin")
        assert len(stored) >= 1
        assert stored[0]["chat_id"] == "chat_dm_sophie"

    @pytest.mark.asyncio
    async def test_does_not_create_duplicate_with_user_typed_name(self, contacts):
        """If Beeper knows 'Sophie Martin', we shouldn't also create a 'Sophie' entry."""
        contacts.update("Sophie Martin", "whatsapp", "chat_old", "Sophie Martin", "2026-03-04T10:00:00+00:00")

        mock_chat = MagicMock()
        mock_chat.title = "Sophie Martin"
        mock_chat.id = "chat_dm_sophie"
        mock_chat.account_id = "whatsapp"
        mock_chat.type = "single"

        client = MagicMock()
        client.chats.search = MagicMock(return_value=[mock_chat])

        await _find_dm_chat(client, "Sophie", contacts)

        # Should update the existing "Sophie Martin" entry, not create a new "Sophie" one
        all_sophies = contacts.lookup("Sophie")
        chat_ids = {c["chat_id"] for c in all_sophies}
        sender_names = {c["sender_name"] for c in all_sophies}
        # Only "Sophie Martin" should exist, not a separate "Sophie"
        assert "Sophie" not in sender_names or sender_names == {"Sophie Martin"}

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, contacts):
        """If Beeper search returns no matching DM, return None."""
        mock_chat = MagicMock()
        mock_chat.title = "Unrelated Group Chat"
        mock_chat.id = "chat_group"
        mock_chat.account_id = "whatsapp"
        mock_chat.type = "group"

        client = MagicMock()
        client.chats.search = MagicMock(return_value=[mock_chat])

        result = await _find_dm_chat(client, "Sophie", contacts)
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_group_chats_with_matching_title(self, contacts):
        """A group chat titled 'Sophie's Birthday' should not match when looking for Sophie's DM."""
        group_chat = MagicMock()
        group_chat.title = "Sophie's Birthday Party"
        group_chat.id = "chat_group_bday"
        group_chat.account_id = "whatsapp"
        group_chat.type = "group"

        dm_chat = MagicMock()
        dm_chat.title = "Sophie Martin"
        dm_chat.id = "chat_dm_sophie"
        dm_chat.account_id = "whatsapp"
        dm_chat.type = "single"

        client = MagicMock()
        client.chats.search = MagicMock(return_value=[group_chat, dm_chat])

        result = await _find_dm_chat(client, "Sophie", contacts)
        assert result is not None
        assert result["chat_id"] == "chat_dm_sophie"

    @pytest.mark.asyncio
    async def test_all_results_are_groups_returns_none(self, contacts):
        """If Beeper only returns group chats, return None even if titles match."""
        group1 = MagicMock()
        group1.title = "Sophie and Friends"
        group1.id = "chat_group1"
        group1.account_id = "whatsapp"
        group1.type = "group"

        group2 = MagicMock()
        group2.title = "Sophie's Work Chat"
        group2.id = "chat_group2"
        group2.account_id = "slack"
        group2.type = "group"

        client = MagicMock()
        client.chats.search = MagicMock(return_value=[group1, group2])

        result = await _find_dm_chat(client, "Sophie", contacts)
        assert result is None

    @pytest.mark.asyncio
    async def test_chat_without_type_field_still_works(self, contacts):
        """Chats missing the type attribute (e.g. older SDK) are not skipped."""
        mock_chat = MagicMock(spec=["title", "id", "account_id"])
        mock_chat.title = "Sophie"
        mock_chat.id = "chat_sophie"
        mock_chat.account_id = "whatsapp"

        client = MagicMock()
        client.chats.search = MagicMock(return_value=[mock_chat])

        result = await _find_dm_chat(client, "Sophie", contacts)
        assert result is not None
        assert result["chat_id"] == "chat_sophie"

    @pytest.mark.asyncio
    async def test_beeper_search_failure_returns_none(self, contacts):
        """If Beeper search throws, return None gracefully."""
        client = MagicMock()
        client.chats.search = MagicMock(side_effect=Exception("Beeper down"))

        result = await _find_dm_chat(client, "Sophie", contacts)
        assert result is None


class TestSentMessageCaching:
    """Verify that messages sent on Adrien's behalf are stored in the cache."""

    def test_cache_sent_message_stores_correctly(self, cache):
        """_cache_sent_message inserts a message with Adrien as sender."""
        _cache_sent_message(cache, "chat_sophie", "whatsapp", "Sophie", "I'll be late!")

        msgs = cache.by_chat_id("chat_sophie", limit=10)
        assert len(msgs) == 1
        assert msgs[0]["sender_name"] == _USER_SENDER_LABEL
        assert msgs[0]["text"] == "I'll be late!"
        assert msgs[0]["network"] == "whatsapp"
        assert msgs[0]["chat_title"] == "Sophie"
        assert msgs[0]["message_id"].startswith("sent_")

    @pytest.mark.asyncio
    async def test_direct_send_caches_message(self, cache, contacts):
        """After a direct send, the sent message appears in the cache."""
        contacts.update("Sophie", "whatsapp", "chat_sophie", "Sophie", "2026-03-05T10:00:00+00:00")
        cache.store(make_msg("m1", sender_name="Sophie", text="are you coming?",
                             chat_title="Sophie", hours_ago=0))

        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "yes"}',  # reply extractor
                '{"action": "send", "text": "Yep, on my way!"}',  # compose decision
            ]
            await handle_user_message(
                "tell Sophie yes", cache, contacts=contacts, beeper_client=client,
            )

        # Cache should now have the original message + the sent one
        msgs = cache.by_chat_id("chat_sophie", limit=10)
        sent = [m for m in msgs if m["message_id"].startswith("sent_")]
        assert len(sent) == 1
        assert sent[0]["text"] == "Yep, on my way!"
        assert sent[0]["sender_name"] == _USER_SENDER_LABEL

    @pytest.mark.asyncio
    async def test_confirmed_send_caches_message(self, cache, contacts):
        """After a confirmed send, the sent message appears in the cache."""
        import src.assistant

        contacts.update("Marc", "linkedin", "chat_marc", "Marc", "2026-03-05T10:00:00+00:00")

        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        # Step 1: Opus asks for confirmation
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Marc", "message": "interested"}',  # reply extractor
                json.dumps({
                    "action": "confirm",
                    "text": "Hi Marc, we're interested.",
                    "response": "Send this to Marc? \"Hi Marc, we're interested.\"",
                }),
            ]
            await handle_user_message(
                "reply to Marc saying interested", cache, contacts=contacts, beeper_client=client,
            )

        assert src.assistant._pending_action is not None
        assert cache.by_chat_id("chat_marc", limit=10) == []  # not cached yet

        # Step 2: Adrien confirms
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"action": "confirm", "text": "Hi Marc, we\'re interested."}'
            await handle_user_message("yes", cache, contacts=contacts, beeper_client=client)

        msgs = cache.by_chat_id("chat_marc", limit=10)
        sent = [m for m in msgs if m["message_id"].startswith("sent_")]
        assert len(sent) == 1
        assert sent[0]["text"] == "Hi Marc, we're interested."

    @pytest.mark.asyncio
    async def test_failed_send_does_not_cache(self, cache, contacts):
        """If sending fails, nothing is cached."""
        contacts.update("Sophie", "whatsapp", "chat_sophie", "Sophie", "2026-03-05T10:00:00+00:00")

        client = MagicMock()
        client.messages.send = MagicMock(side_effect=Exception("network error"))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "hey"}',  # reply extractor
                '{"action": "send", "text": "Hey!"}',  # compose decision
            ]
            await handle_user_message(
                "tell Sophie hey", cache, contacts=contacts, beeper_client=client,
            )

        msgs = cache.by_chat_id("chat_sophie", limit=10)
        sent = [m for m in msgs if m["message_id"].startswith("sent_")]
        assert len(sent) == 0


class TestFeedbackStoresRawText:
    @pytest.mark.asyncio
    async def test_stores_user_raw_words_not_sonnet_summary(self, cache):
        """Feedback should store the user's exact message, not Sonnet's summary."""
        user_text = "when Sophie messages about the Acme deal, always flag it as urgent even if it doesn't sound urgent"

        stored_feedback = None

        def capture_feedback(text):
            nonlocal stored_feedback
            stored_feedback = text

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.append_feedback", side_effect=capture_feedback):
            mock_complete.return_value = "feedback"  # router classifies as feedback
            result, queried = await handle_user_message(user_text, cache)

        # Should store the user's raw text
        assert stored_feedback == user_text
        assert queried is False

    @pytest.mark.asyncio
    async def test_feedback_no_opus_call(self, cache):
        """Feedback acknowledgment should NOT make an Opus call — only router for intent."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.append_feedback"):
            mock_complete.return_value = "feedback"  # router
            result, queried = await handle_user_message("that wasn't urgent", cache)

        # Only 1 call (router). No second call for response.
        assert mock_complete.call_count == 1
        assert result in _FEEDBACK_ACKS


class TestFeedbackAck:
    def test_returns_string(self):
        assert isinstance(_feedback_ack(), str)

    def test_returns_from_list(self):
        for _ in range(20):
            assert _feedback_ack() in _FEEDBACK_ACKS


@pytest.fixture
def convo(tmp_path):
    c = ConversationHistory(db_path=tmp_path / "test_convo.db")
    yield c
    c.close()


class TestDisplaySender:
    """Verify _display_sender labels the user's messages and leaves others alone."""

    @pytest.mark.skipif(not USER_SENDER_IDS, reason="USER_SENDER_IDS not configured")
    def test_sender_id_match(self):
        """First USER_SENDER_IDS entry should be labeled as '(you)'."""
        assert _display_sender(USER_SENDER_IDS[0]) == f"{USER_NAME} (you)"

    @pytest.mark.skipif(not USER_SENDER_IDS, reason="USER_SENDER_IDS not configured")
    def test_case_insensitive(self):
        assert _display_sender(USER_SENDER_IDS[0].upper()) == f"{USER_NAME} (you)"

    def test_other_sender_unchanged(self):
        assert _display_sender("Pierre-Louis Biojout") == "Pierre-Louis Biojout"

    def test_empty_string(self):
        assert _display_sender("") == ""


class TestLastDiploTurnWasEmpty:
    """Verify insistence detection by checking Diplo's last response."""

    def test_no_turns(self, convo):
        assert _last_diplo_turn_was_empty(convo) is False

    def test_none_convo(self):
        assert _last_diplo_turn_was_empty(None) is False

    def test_normal_response(self, convo):
        convo.add_turn("assistant", "Here are your messages from Sophie...")
        assert _last_diplo_turn_was_empty(convo) is False

    def test_no_new_messages(self, convo):
        convo.add_turn("user", "what's new?")
        convo.add_turn("assistant", "No new messages since you last checked.")
        assert _last_diplo_turn_was_empty(convo) is True

    def test_nothing_new(self, convo):
        convo.add_turn("assistant", "Nothing new on WhatsApp either.")
        assert _last_diplo_turn_was_empty(convo) is True

    def test_all_quiet(self, convo):
        convo.add_turn("assistant", "All quiet across the board.")
        assert _last_diplo_turn_was_empty(convo) is True

    def test_still_nothing(self, convo):
        convo.add_turn("assistant", "Still nothing from her in my cache.")
        assert _last_diplo_turn_was_empty(convo) is True

    def test_hasnt_synced(self, convo):
        convo.add_turn("assistant", "Whatever she sent hasn't synced to me yet.")
        assert _last_diplo_turn_was_empty(convo) is True

    def test_only_checks_most_recent_assistant_turn(self, convo):
        """An old 'nothing new' followed by a normal response should NOT trigger."""
        convo.add_turn("assistant", "No new messages.")
        convo.add_turn("user", "thanks")
        convo.add_turn("assistant", "You're welcome!")
        convo.add_turn("user", "check again")
        assert _last_diplo_turn_was_empty(convo) is False

    def test_after_successful_hours_search_does_not_trigger(self, convo):
        """After Diplo finds messages via hours search, insistence should NOT trigger."""
        convo.add_turn("assistant", "No new messages since you last checked.")
        convo.add_turn("user", "check again")
        convo.add_turn("assistant", "Here's what I found in the last hour: Maman sent 3 messages...")
        convo.add_turn("user", "go back further")
        assert _last_diplo_turn_was_empty(convo) is False


class TestExplicitHoursSearch:
    """Verify _execute_search handles explicit hours correctly."""

    def test_hours_without_filters(self, cache):
        """hours queries recent(hours=N), bypassing last_seen_at."""
        cache.store(make_msg("m1", hours_ago=0.5))
        cache.store(make_msg("m2", hours_ago=3))

        results = _execute_search({"hours": 1}, cache)
        assert len(results) == 1
        assert results[0]["message_id"] == "m1"

    def test_hours_wider_window(self, cache):
        """Increasing hours includes older messages."""
        cache.store(make_msg("m1", hours_ago=0.5))
        cache.store(make_msg("m2", hours_ago=3))

        results = _execute_search({"hours": 4}, cache)
        assert len(results) == 2

    def test_hours_with_sender_filter(self, cache):
        """hours + sender returns both sides of the conversation within the time window."""
        cache.store(make_msg("m1", sender_name="Maman", hours_ago=0.5,
                             chat_title="Maman", chat_id="chat_maman"))
        cache.store(make_msg("m2", sender_name="@alemercier:beeper.com", hours_ago=0.4,
                             chat_title="Maman", chat_id="chat_maman"))
        cache.store(make_msg("m3", sender_name="Maman", hours_ago=3,
                             chat_title="Maman", chat_id="chat_maman"))

        results = _execute_search({"sender": "Maman", "hours": 1}, cache)
        ids = {m["message_id"] for m in results}
        # m1 and m2 are within 1h, m3 is too old
        assert "m1" in ids
        assert "m2" in ids  # Adrien's reply included via chat expansion
        assert "m3" not in ids

    def test_hours_bypasses_last_seen(self, cache):
        """hours should find messages even if they're before last_seen_at."""
        # Set last_seen_at to now
        cache.touch_last_seen()
        # Store a message from 30 min ago (before last_seen_at)
        cache.store(make_msg("m1", hours_ago=0.5))

        # since_last_seen would miss this message
        assert len(_execute_search({"since_last_seen": True}, cache)) == 0
        # hours finds it
        assert len(_execute_search({"hours": 1}, cache)) == 1

    def test_hours_empty_cache(self, cache):
        results = _execute_search({"hours": 1}, cache)
        assert results == []

    def test_hours_results_sorted_chronologically(self, cache):
        cache.store(make_msg("m1", hours_ago=0.8))
        cache.store(make_msg("m2", hours_ago=0.2))
        cache.store(make_msg("m3", hours_ago=0.5))

        results = _execute_search({"hours": 1}, cache)
        assert len(results) == 3
        assert results[0]["message_id"] == "m1"  # oldest first
        assert results[2]["message_id"] == "m2"  # newest last

    def test_large_hours_window(self, cache):
        """Large hour values (e.g. 336 = 2 weeks) should work for broad searches."""
        cache.store(make_msg("m1", hours_ago=1))
        cache.store(make_msg("m2", hours_ago=48))
        cache.store(make_msg("m3", hours_ago=168))  # 1 week
        cache.store(make_msg("m4", hours_ago=330))  # ~2 weeks

        results = _execute_search({"hours": 336}, cache)
        assert len(results) == 4

    def test_hours_with_search_filter(self, cache):
        """hours + search text narrows by both time and content."""
        cache.store(make_msg("m1", text="fundraising deck", hours_ago=1))
        cache.store(make_msg("m2", text="fundraising update", hours_ago=50))
        cache.store(make_msg("m3", text="coffee tomorrow?", hours_ago=1))

        results = _execute_search({"search": "fundraising", "hours": 24}, cache)
        assert len(results) == 1
        assert results[0]["message_id"] == "m1"

    def test_hours_with_network_filter(self, cache):
        """hours + network narrows by both time and platform."""
        cache.store(make_msg("m1", network="whatsapp", hours_ago=1))
        cache.store(make_msg("m2", network="whatsapp", hours_ago=50))
        cache.store(make_msg("m3", network="telegram", hours_ago=1))

        results = _execute_search({"network": "whatsapp", "hours": 24}, cache)
        assert len(results) == 1
        assert results[0]["message_id"] == "m1"

    def test_legacy_lookback_hours_still_works(self, cache):
        """Legacy lookback_hours key should be treated as hours in _execute_search.

        This ensures backwards compatibility if Sonnet still emits lookback_hours.
        The normalization happens in handle_user_message, but _execute_search
        should handle hours directly."""
        cache.store(make_msg("m1", hours_ago=0.5))
        cache.store(make_msg("m2", hours_ago=3))

        # _execute_search only knows about "hours" now, not "lookback_hours".
        # The normalization happens upstream in handle_user_message.
        results = _execute_search({"hours": 1}, cache)
        assert len(results) == 1

    def test_hours_without_filters_ignores_last_seen(self, cache):
        """hours alone should return messages regardless of last_seen_at watermark."""
        cache.touch_last_seen()
        # Messages before last_seen
        cache.store(make_msg("m1", hours_ago=2))
        cache.store(make_msg("m2", hours_ago=0.5))

        results = _execute_search({"hours": 3}, cache)
        assert len(results) == 2


class TestInsistenceSafetyNet:
    """Verify the code-level safety net overrides since_last_seen to hours when Diplo just said 'nothing new'."""

    @pytest.fixture
    def convo(self, tmp_path):
        c = ConversationHistory(db_path=tmp_path / "test_convo_safety.db")
        yield c
        c.close()

    @pytest.mark.asyncio
    async def test_insistence_overrides_since_last_seen(self, cache, convo):
        """When Diplo said 'nothing new' and Sonnet returns since_last_seen again, override to hours=1."""
        cache.touch_last_seen()
        # Store a message from 30 min ago (before last_seen_at)
        cache.store(make_msg("m1", sender_name="Maman", text="coucou", hours_ago=0.5))

        # Diplo said nothing new on the last turn
        convo.add_turn("user", "what's new?")
        convo.add_turn("assistant", "No new messages since you last checked.")

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"since_last_seen": true}',  # query extractor repeats the same plan (wrong)
                "Found something: Maman said coucou.",  # Opus response
            ]
            result, queried = await handle_user_message(
                "are you sure? check again", cache, convo=convo,
            )

        # Should have found the message via hours override
        assert "Maman" in result or "coucou" in result
        # Explicit hours queries should NOT advance the watermark
        assert queried is False

    @pytest.mark.asyncio
    async def test_no_override_when_diplo_found_messages(self, cache, convo):
        """When Diplo's last turn had results, since_last_seen should NOT be overridden."""
        convo.add_turn("user", "what's new?")
        convo.add_turn("assistant", "Sophie sent you 3 messages about the project.")

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"since_last_seen": true}',  # query extractor
                "Nothing new since then.",  # Opus response
            ]
            result, queried = await handle_user_message(
                "anything else?", cache, convo=convo,
            )

        # Normal flow — queried_cache should be True (no hours override)
        assert queried is True

    @pytest.mark.asyncio
    async def test_explicit_hours_does_not_advance_watermark(self, cache, convo):
        """Explicit hours queries should return queried_cache=False so last_seen_at doesn't advance."""
        convo.add_turn("assistant", "All quiet across the board.")

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 1}',  # query extractor correctly detects insistence
                "Here's what I found...",  # Opus response
            ]
            result, queried = await handle_user_message(
                "check again", cache, convo=convo,
            )

        assert queried is False

    @pytest.mark.asyncio
    async def test_explicit_hours_from_sonnet(self, cache, convo):
        """When the query extractor returns hours, it should work without the safety net."""
        cache.store(make_msg("m1", sender_name="Maman", text="hello", hours_ago=0.5))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 2}',  # query extractor
                "Maman said hello 30 min ago.",  # Opus response
            ]
            result, queried = await handle_user_message(
                "go back 2 hours", cache, convo=convo,
            )

        assert queried is False

    @pytest.mark.asyncio
    async def test_wider_hours_on_follow_up(self, cache, convo):
        """Adrien asks to go further back after an initial search."""
        cache.store(make_msg("m1", hours_ago=0.5))
        cache.store(make_msg("m2", hours_ago=4))

        convo.add_turn("assistant", "Here's the last hour: 1 message from Alice.")
        convo.add_turn("user", "go back further, 6 hours")

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 6}',  # query extractor
                "Found 2 messages in the last 6 hours...",  # Opus response
            ]
            result, queried = await handle_user_message(
                "go back further, 6 hours", cache, convo=convo,
            )

        assert queried is False

    @pytest.mark.asyncio
    async def test_legacy_lookback_hours_normalized_to_hours(self, cache, convo):
        """If the query extractor still emits lookback_hours, it gets normalized to hours."""
        cache.store(make_msg("m1", sender_name="Maman", text="hello", hours_ago=0.5))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"lookback_hours": 2}',  # legacy key
                "Maman said hello 30 min ago.",  # Opus response
            ]
            result, queried = await handle_user_message(
                "go back 2 hours", cache, convo=convo,
            )

        # Should still work — normalized to hours
        assert queried is False

    @pytest.mark.asyncio
    async def test_broad_search_with_large_hours(self, cache, convo):
        """'Search the last 2 weeks' should use hours=336 and find old messages."""
        cache.store(make_msg("m1", sender_name="Sophie", text="hey", hours_ago=1))
        cache.store(make_msg("m2", sender_name="Marc", text="dinner?", hours_ago=100))
        cache.store(make_msg("m3", sender_name="Natasha", text="call me", hours_ago=200))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 336}',  # query extractor — 2 weeks
                "Found 3 messages across 2 weeks...",  # Opus response
            ]
            result, queried = await handle_user_message(
                "search the last 2 weeks", cache, convo=convo,
            )

        assert queried is False
        # Verify the Opus call received all 3 messages (3rd call = Opus)
        opus_call = mock_complete.call_args_list[2]
        opus_user_msg = opus_call[1]["messages"][0]["content"]
        assert "3 messages found" in opus_user_msg

    @pytest.mark.asyncio
    async def test_hours_with_sender_finds_messages(self, cache, convo):
        """'Check Sophie messages from last week' should combine sender + hours."""
        cache.store(make_msg("m1", sender_name="Sophie", text="hey", hours_ago=1,
                             chat_title="Sophie", chat_id="chat_sophie"))
        cache.store(make_msg("m2", sender_name="Sophie", text="old msg", hours_ago=200,
                             chat_title="Sophie", chat_id="chat_sophie"))
        cache.store(make_msg("m3", sender_name="Marc", text="unrelated", hours_ago=1))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"sender": "Sophie", "hours": 168}',  # query extractor — 1 week
                "Sophie said hey recently.",  # Opus response
            ]
            result, queried = await handle_user_message(
                "check my messages from Sophie in the last week", cache, convo=convo,
            )

        assert queried is False
        # Verify only Sophie's messages within 168h were passed to Opus (3rd call = Opus)
        opus_call = mock_complete.call_args_list[2]
        opus_user_msg = opus_call[1]["messages"][0]["content"]
        assert "Sophie" in opus_user_msg
        assert "Marc" not in opus_user_msg


class TestLookbackNoteThreshold:
    """Verify that the 'want to go further back?' note only shows for small lookbacks."""

    @pytest.fixture
    def convo(self, tmp_path):
        c = ConversationHistory(db_path=tmp_path / "test_convo_note.db")
        yield c
        c.close()

    @pytest.mark.asyncio
    async def test_small_hours_shows_lookback_note(self, cache, convo):
        """For small lookbacks (insistence), Opus should be told to offer going further back."""
        cache.store(make_msg("m1", hours_ago=0.5))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 1}',  # query extractor
                "Here are messages from the last hour.",  # Opus response
            ]
            await handle_user_message("check again", cache, convo=convo)

        # Opus call should include the lookback note (3rd call)
        opus_call = mock_complete.call_args_list[2]
        opus_user_msg = opus_call[1]["messages"][0]["content"]
        assert "lookback search" in opus_user_msg
        assert "go further back" in opus_user_msg

    @pytest.mark.asyncio
    async def test_large_hours_no_lookback_note(self, cache, convo):
        """For large explicit ranges (2 weeks), Opus should NOT be told to offer going further back."""
        cache.store(make_msg("m1", hours_ago=1))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 336}',  # query extractor
                "Here are messages from the last 2 weeks.",  # Opus response
            ]
            await handle_user_message("search the last 2 weeks", cache, convo=convo)

        # Opus call should NOT include the lookback note (3rd call)
        opus_call = mock_complete.call_args_list[2]
        opus_user_msg = opus_call[1]["messages"][0]["content"]
        assert "lookback search" not in opus_user_msg
        assert "go further back" not in opus_user_msg

    @pytest.mark.asyncio
    async def test_24h_boundary_shows_note(self, cache, convo):
        """24h is the threshold — should still show the note."""
        cache.store(make_msg("m1", hours_ago=0.5))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 24}',  # query extractor
                "Here are messages from the last day.",  # Opus response
            ]
            await handle_user_message("check the last 24 hours", cache, convo=convo)

        opus_call = mock_complete.call_args_list[2]
        opus_user_msg = opus_call[1]["messages"][0]["content"]
        assert "lookback search" in opus_user_msg

    @pytest.mark.asyncio
    async def test_25h_no_note(self, cache, convo):
        """25h is past the threshold — no note."""
        cache.store(make_msg("m1", hours_ago=0.5))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 25}',  # query extractor
                "Here are messages.",  # Opus response
            ]
            await handle_user_message("check the last 25 hours", cache, convo=convo)

        opus_call = mock_complete.call_args_list[2]
        opus_user_msg = opus_call[1]["messages"][0]["content"]
        assert "lookback search" not in opus_user_msg


class TestDeepSearchScenario:
    """Regression test for the March 12 'deep search' failure.

    Reproduces the exact scenario where Adrien asked for a 2-week deep search
    and Diplo repeatedly returned 'no new messages since last check' because
    since_last_seen was used instead of a time-based query.
    """

    @pytest.fixture
    def convo(self, tmp_path):
        c = ConversationHistory(db_path=tmp_path / "test_convo_deep.db")
        yield c
        c.close()

    @pytest.mark.asyncio
    async def test_deep_search_finds_old_messages(self, cache, convo):
        """A 'deep search' request should find messages across the full cache."""
        # Simulate last_seen_at being very recent (Adrien just chatted)
        cache.touch_last_seen()

        # Store messages spanning 2 weeks
        cache.store(make_msg("m1", sender_name="Sophie", text="dinner Saturday?",
                             hours_ago=2, network="whatsapp"))
        cache.store(make_msg("m2", sender_name="Marc", text="fundraising deck ready",
                             hours_ago=48, network="whatsapp"))
        cache.store(make_msg("m3", sender_name="Natasha", text="hey are you free?",
                             hours_ago=120, network="instagramgo"))
        cache.store(make_msg("m4", sender_name="Julian", text="intro sent to Nader",
                             hours_ago=240, network="whatsapp"))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"hours": 336}',  # query extractor — correctly interprets "last 2 weeks"
                "Found 4 messages across 2 weeks. Sophie, Marc, Natasha, Julian.",
            ]
            result, queried = await handle_user_message(
                "do a deep search of all my messages from the last 2 weeks",
                cache, convo=convo,
            )

        # Verify all 4 messages were found despite last_seen_at being recent
        opus_call = mock_complete.call_args_list[2]
        opus_user_msg = opus_call[1]["messages"][0]["content"]
        assert "4 messages found" in opus_user_msg
        assert "Sophie" in opus_user_msg
        assert "Julian" in opus_user_msg
        # Watermark should NOT advance (explicit hours query)
        assert queried is False

    @pytest.mark.asyncio
    async def test_since_last_seen_would_miss_old_messages(self, cache, convo):
        """Confirms that since_last_seen alone misses messages before the watermark."""
        cache.touch_last_seen()
        cache.store(make_msg("m1", sender_name="Sophie", text="hey", hours_ago=2))

        # since_last_seen returns nothing (message is before watermark)
        results = _execute_search({"since_last_seen": True}, cache)
        assert len(results) == 0

        # hours=48 finds it
        results = _execute_search({"hours": 48}, cache)
        assert len(results) == 1


class TestSenderSearchExpansion:
    """Verify that sender filter fetches full conversation (both sides)."""

    def test_sender_includes_both_sides(self, cache):
        """Searching by sender should also return Adrien's replies in the same chat."""
        cache.store(make_msg("m1", sender_name="Natasha", chat_title="Natasha",
                             chat_id="chat_natasha", text="hey", hours_ago=1))
        cache.store(make_msg("m2", sender_name="@alemercier:beeper.com", chat_title="Natasha",
                             chat_id="chat_natasha", text="hey back", hours_ago=0.9))

        results = _execute_search({"sender": "Natasha"}, cache)
        ids = {m["message_id"] for m in results}
        assert "m1" in ids  # Natasha's message
        assert "m2" in ids  # Adrien's reply

    def test_sender_multiple_chats_expanded(self, cache):
        """If sender appears in multiple chats, all are expanded."""
        cache.store(make_msg("m1", sender_name="Sophie", chat_title="Sophie DM",
                             chat_id="dm", hours_ago=1))
        cache.store(make_msg("m2", sender_name="@alemercier:beeper.com", chat_title="Sophie DM",
                             chat_id="dm", text="reply in dm", hours_ago=0.9))
        cache.store(make_msg("m3", sender_name="Sophie", chat_title="Group Chat",
                             chat_id="group", hours_ago=0.8))
        cache.store(make_msg("m4", sender_name="Bob", chat_title="Group Chat",
                             chat_id="group", text="bob in group", hours_ago=0.7))

        results = _execute_search({"sender": "Sophie"}, cache)
        ids = {m["message_id"] for m in results}
        assert ids == {"m1", "m2", "m3", "m4"}

    def test_sender_none_skipped(self, cache):
        """sender=None should not crash or call by_sender."""
        cache.store(make_msg("m1", hours_ago=1))
        results = _execute_search({"sender": None, "hours": 24}, cache)
        assert len(results) == 1  # falls through to time-based query


class TestIsOwnerRecipient:
    """Tests for _is_owner_recipient — blocks Beeper sends to the user himself."""

    def test_direct_self_references(self):
        assert _is_owner_recipient("me") is True
        assert _is_owner_recipient("myself") is True
        assert _is_owner_recipient("moi") is True

    def test_user_name_match(self):
        assert _is_owner_recipient(USER_NAME) is True
        assert _is_owner_recipient(USER_NAME.upper()) is True

    def test_case_insensitive(self):
        assert _is_owner_recipient("ME") is True
        assert _is_owner_recipient("Moi") is True
        assert _is_owner_recipient("  Me  ") is True

    @pytest.mark.skipif(not USER_SENDER_IDS, reason="USER_SENDER_IDS not configured")
    def test_user_sender_ids(self):
        assert _is_owner_recipient(USER_SENDER_IDS[0]) is True

    def test_other_people_not_blocked(self):
        assert _is_owner_recipient("Sophie") is False
        assert _is_owner_recipient("Marc") is False
        assert _is_owner_recipient("PLB") is False
        assert _is_owner_recipient("team") is False
        assert _is_owner_recipient("meeting") is False


class TestBlockSelfSendViaBeper:
    """Integration tests: reply intents targeting the user are blocked from Beeper."""

    @pytest.mark.asyncio
    async def test_reply_to_me_blocked(self, cache, contacts):
        """'tell me ...' misinterpreted as reply intent should not send via Beeper."""
        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router (wrong — should be query)
                '{"recipient": "me", "message": "here is your summary"}',  # reply extractor
            ]
            result, queried = await handle_user_message(
                "tell me a summary", cache, contacts=contacts, beeper_client=client,
            )

        # Should NOT have called Beeper send
        client.messages.send.assert_not_called()
        # Should return the intent text instead
        assert "summary" in result.lower()

    @pytest.mark.asyncio
    async def test_reply_to_user_name_blocked(self, cache, contacts):
        """Reply to the user's name should be blocked."""
        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router (wrong — should be query)
                f'{{"recipient": "{USER_NAME}", "message": "notification text"}}',  # reply extractor
            ]
            result, queried = await handle_user_message(
                "notify me about new messages", cache, contacts=contacts, beeper_client=client,
            )

        client.messages.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_reply_to_moi_blocked(self, cache, contacts):
        """Reply to 'moi' (French) should be blocked."""
        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router (wrong — should be query)
                '{"recipient": "moi", "message": "voici le résumé"}',  # reply extractor
            ]
            result, queried = await handle_user_message(
                "dis moi ce qui se passe", cache, contacts=contacts, beeper_client=client,
            )

        client.messages.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_reply_to_other_person_still_works(self, cache, contacts):
        """Replies to actual other people should still go through normally."""
        contacts.update("Sophie", "whatsapp", "chat_sophie", "Sophie", "2026-03-05T10:00:00+00:00")
        cache.store(make_msg("m1", sender_name="Sophie", text="hey", chat_title="Sophie"))

        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "on my way"}',  # reply extractor
                '{"action": "send", "text": "On my way!"}',  # compose decision
            ]
            result, queried = await handle_user_message(
                "tell Sophie I'm on my way", cache, contacts=contacts, beeper_client=client,
            )

        assert "Sent" in result
        client.messages.send.assert_called_once()


class TestQuestionNotClassifiedAsFeedback:
    """Messages ending with '?' should never be treated as feedback,
    even when Sonnet misclassifies them. This prevents canned 'Got it.' / 'Noted.'
    responses to real questions like 'how would you do it?' or 'why are you so dumb??'."""

    @pytest.mark.asyncio
    async def test_question_mark_bypasses_feedback(self, cache):
        """A message ending with '?' that the router classifies as feedback should
        fall through to the casual pipeline instead."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.append_feedback") as mock_append:
            mock_complete.side_effect = [
                "feedback",                             # router misclassifies
                "Here's my honest answer...",           # Opus responds (casual)
            ]
            result, queried = await handle_user_message(
                "Why are you sometimes so smart and sometimes so dumb??", cache,
            )

        # Should NOT have stored feedback
        mock_append.assert_not_called()
        # Should have made 2 LLM calls (router + Opus), not just 1
        assert mock_complete.call_count == 2
        # Response should be from Opus, not a canned ack
        assert result not in _FEEDBACK_ACKS
        # queried_cache should be False (treated as casual, no cache dump)
        assert queried is False

    @pytest.mark.asyncio
    async def test_question_with_single_question_mark(self, cache):
        """Single '?' also triggers the guard."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.append_feedback") as mock_append:
            mock_complete.side_effect = [
                "feedback",  # router misclassifies
                "Let me explain...",  # Opus responds (casual)
            ]
            result, queried = await handle_user_message("How would you do it?", cache)

        mock_append.assert_not_called()
        assert result not in _FEEDBACK_ACKS
        assert queried is False

    @pytest.mark.asyncio
    async def test_real_feedback_still_works(self, cache):
        """Statements (no '?') classified as feedback should still be stored."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.append_feedback") as mock_append:
            mock_complete.return_value = "feedback"  # router
            result, _ = await handle_user_message("your summaries are too long", cache)

        mock_append.assert_called_once_with("your summaries are too long")
        assert result in _FEEDBACK_ACKS

    @pytest.mark.asyncio
    async def test_feedback_with_trailing_whitespace_and_question(self, cache):
        """Trailing whitespace after '?' should still be caught."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.append_feedback") as mock_append:
            mock_complete.side_effect = [
                "feedback",  # router misclassifies
                "Good question...",  # Opus responds (casual)
            ]
            result, _ = await handle_user_message("Have you done it?  ", cache)

        mock_append.assert_not_called()

    @pytest.mark.asyncio
    async def test_question_does_not_dump_cache_to_opus(self, cache):
        """When a question bypasses feedback, Opus should NOT get 24h of messages.
        The intent is overridden to casual so Opus gets just the question."""
        # Populate cache with messages that should NOT appear in the Opus call
        cache.store(make_msg("m1", text="unrelated cached message", hours_ago=2))
        cache.store(make_msg("m2", text="another cached message", hours_ago=3))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.append_feedback") as mock_append:
            mock_complete.side_effect = [
                "feedback",  # router misclassifies
                "Good question, let me think...",  # Opus responds (casual)
            ]
            result, _ = await handle_user_message("Why do you keep failing?", cache)

        # The Opus call (second call) should not contain cached messages
        opus_call = mock_complete.call_args_list[1]
        opus_user_msg = opus_call.kwargs.get("messages", opus_call[1].get("messages", [{}]))[0]["content"]
        assert "unrelated cached message" not in opus_user_msg
        assert "another cached message" not in opus_user_msg


# ---------------------------------------------------------------------------
# Two-stage intent classification tests
# ---------------------------------------------------------------------------


class TestParseIntent:
    """Tests for the _parse_intent helper that normalizes router output."""

    def test_exact_match(self):
        assert _parse_intent("query") == "query"
        assert _parse_intent("reply") == "reply"
        assert _parse_intent("automation") == "automation"
        assert _parse_intent("feedback") == "feedback"
        assert _parse_intent("casual") == "casual"
        assert _parse_intent("timezone") == "timezone"

    def test_whitespace_and_case(self):
        assert _parse_intent("  Query  ") == "query"
        assert _parse_intent("REPLY") == "reply"
        assert _parse_intent("  Casual\n") == "casual"

    def test_trailing_period(self):
        assert _parse_intent("query.") == "query"
        assert _parse_intent("feedback.") == "feedback"

    def test_intent_in_sentence(self):
        """Router might return prose — extract the intent word."""
        assert _parse_intent("The intent is query") == "query"
        assert _parse_intent("I think this is a reply intent") == "reply"

    def test_unknown_defaults_to_query(self):
        assert _parse_intent("unknown") == "query"
        assert _parse_intent("banana") == "query"
        assert _parse_intent("") == "query"

    def test_garbled_with_valid_word(self):
        assert _parse_intent("```json\n\"automation\"\n```") == "automation"


class TestRouteIntent:
    """Tests for the _route_intent router function."""

    @pytest.mark.asyncio
    async def test_returns_parsed_intent(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "query"
            result = await _route_intent("what's new?")
        assert result == "query"

    @pytest.mark.asyncio
    async def test_includes_convo_context_in_prompt(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "query"
            await _route_intent("tell me more", convo_context="Diplo: Sophie said hello.")

        # Verify convo context was passed to the LLM
        call_kwargs = mock_complete.call_args[1]
        assert "Sophie said hello" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_garbled_response_defaults_to_query(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "I'm not sure what to do here"
            result = await _route_intent("something ambiguous")
        assert result == "query"

    @pytest.mark.asyncio
    async def test_uses_search_plan_model(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "casual"
            await _route_intent("hey!")

        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["max_tokens"] == 10

    @pytest.mark.asyncio
    async def test_query_description_mentions_calendar_and_email(self):
        """The router prompt must mention calendar/schedule/email so Sonnet
        routes those queries as 'query' rather than 'casual'."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "query"
            await _route_intent("what's on my calendar?")

        system_prompt = mock_complete.call_args[1]["system"]
        assert "calendar" in system_prompt
        assert "schedule" in system_prompt or "email" in system_prompt


class TestExtractQueryPlan:
    """Tests for the focused query extractor."""

    @pytest.mark.asyncio
    async def test_returns_parsed_plan(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"since_last_seen": true}'
            result = await _extract_query_plan("what's new?", last_seen=None)
        assert result == {"since_last_seen": True}

    @pytest.mark.asyncio
    async def test_includes_last_seen_in_prompt(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"since_last_seen": true}'
            await _extract_query_plan("what's new?", last_seen="2026-03-12T10:00:00+00:00")

        call_kwargs = mock_complete.call_args[1]
        assert "2026-03-12T10:00:00+00:00" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_parse_failure_defaults_to_24h(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "I can't parse this"
            result = await _extract_query_plan("something", last_seen=None)
        assert result == {"hours": 24}


class TestExtractReplyPlan:
    """Tests for the focused reply extractor."""

    @pytest.mark.asyncio
    async def test_returns_parsed_plan(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"recipient": "Sophie", "message": "I\'ll be late"}'
            result = await _extract_reply_plan("tell Sophie I'll be late")
        assert result["recipient"] == "Sophie"
        assert "late" in result["message"]

    @pytest.mark.asyncio
    async def test_includes_convo_context(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"recipient": "Sophie", "message": "thanks"}'
            await _extract_reply_plan("tell her thanks", convo_context="Diplo: Sophie said hello.")

        call_kwargs = mock_complete.call_args[1]
        assert "Sophie said hello" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_parse_failure_returns_empty(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "I can't parse this"
            result = await _extract_reply_plan("something")
        assert result == {}


class TestExtractAutomationPlan:
    """Tests for the focused automation extractor."""

    @pytest.mark.asyncio
    async def test_create_scheduled(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = json.dumps({
                "create_automation": {
                    "description": "Morning summary",
                    "schedule": "0 9 * * *",
                    "action": "summarize my messages",
                }
            })
            result = await _extract_automation_plan("every morning at 9am summarize my messages")
        assert "create_automation" in result
        assert result["create_automation"]["schedule"] == "0 9 * * *"

    @pytest.mark.asyncio
    async def test_list_automations(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"list_automations": true}'
            result = await _extract_automation_plan("show my automations")
        assert result.get("list_automations") is True

    @pytest.mark.asyncio
    async def test_parse_failure_returns_empty(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "I don't understand"
            result = await _extract_automation_plan("something")
        assert result == {}


class TestTwoStageIntegration:
    """Integration tests verifying the full two-stage flow."""

    @pytest.mark.asyncio
    async def test_casual_greeting_skips_cache(self, cache):
        """Casual messages should only call router + Opus (no query extractor)."""
        cache.store(make_msg("m1", text="should not appear"))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "casual",  # router
                "Hey boss!",  # Opus response
            ]
            result, queried = await handle_user_message("hey!", cache)

        assert mock_complete.call_count == 2  # router + Opus only
        assert queried is False
        assert "boss" in result.lower() or "Hey" in result

    @pytest.mark.asyncio
    async def test_feedback_only_calls_router(self, cache):
        """Feedback messages should only call the router (no extractor or Opus)."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.append_feedback"):
            mock_complete.return_value = "feedback"
            result, queried = await handle_user_message("that wasn't urgent", cache)

        assert mock_complete.call_count == 1  # router only
        assert result in _FEEDBACK_ACKS

    @pytest.mark.asyncio
    async def test_query_calls_router_then_extractor_then_opus(self, cache):
        """Query messages should call router + query extractor + Opus."""
        cache.store(make_msg("m1", sender_name="Sophie", text="hey"))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",  # router
                '{"sender": "Sophie"}',  # query extractor
                "Sophie said hey.",  # Opus response
            ]
            result, queried = await handle_user_message("what did Sophie say?", cache)

        assert mock_complete.call_count == 3
        assert queried is True

    @pytest.mark.asyncio
    async def test_reply_calls_router_then_extractor_then_compose(self, cache, contacts):
        """Reply messages should call router + reply extractor + compose."""
        contacts.update("Sophie", "whatsapp", "chat_sophie", "Sophie", "2026-03-05T10:00:00+00:00")
        cache.store(make_msg("m1", sender_name="Sophie", text="hey", chat_title="Sophie"))

        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "on my way"}',  # reply extractor
                '{"action": "send", "text": "On my way!"}',  # compose decision
            ]
            result, queried = await handle_user_message(
                "tell Sophie I'm on my way", cache, contacts=contacts, beeper_client=client,
            )

        assert mock_complete.call_count == 3
        assert "Sent" in result

    @pytest.mark.asyncio
    async def test_reply_extractor_failure_falls_through_to_query(self, cache):
        """If the reply extractor returns no recipient, fall through to query."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "reply",  # router (possibly wrong)
                '{}',  # reply extractor returns empty (failure)
                '{"hours": 24}',  # query extractor (fallback)
                "Nothing found.",  # Opus response
            ]
            result, queried = await handle_user_message("something ambiguous", cache)

        # Should have fallen through to query path
        assert mock_complete.call_count == 4  # router + reply extractor + query extractor + Opus

    @pytest.mark.asyncio
    async def test_timezone_calls_router_then_extractor(self, cache):
        """Timezone messages call router + timezone extractor."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "timezone",  # router
                "Europe/Paris",  # timezone extractor
            ]
            result, queried = await handle_user_message("I'm in Paris", cache)

        assert mock_complete.call_count == 2
        assert "Europe/Paris" in result
        assert queried is False


class TestDebugIntent:
    """Tests for the debug intent routing and handling."""

    @pytest.mark.asyncio
    async def test_debug_intent_routes_correctly(self, cache):
        """'why wasn't that urgent?' should route to debug and return a debug response."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.get_llm_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.search.return_value = [{
                "id": "abc123",
                "timestamp": "2026-03-13T10:00:00+00:00",
                "context_id": "ctx1",
                "call_type": "triage",
                "model": "claude-sonnet-4-6",
                "model_used": "claude-sonnet-4-6",
                "system_prompt": "You classify urgency...",
                "user_prompt": "Sophie: hey can you help me?",
                "response": "not_urgent",
                "input_tokens": 500,
                "output_tokens": 10,
                "latency_ms": 200,
                "status": "success",
                "error": None,
            }]
            mock_get_logger.return_value = mock_logger

            mock_complete.side_effect = [
                "debug",  # router
                '{"hours": 2, "call_type": "triage", "text": "Sophie"}',  # debug plan
                "I found the triage call for Sophie's message. It was classified as not urgent because...",  # debug response
            ]
            result, queried = await handle_user_message(
                "why wasn't Sophie's message urgent?", cache,
            )

        assert queried is False
        assert "Sophie" in result or "triage" in result or "not urgent" in result.lower()
        # router + debug_plan + debug_response = 3 LLM calls
        assert mock_complete.call_count == 3

    @pytest.mark.asyncio
    async def test_debug_no_logger_returns_message(self, cache):
        """If LLM logger isn't initialized, debug returns a clear message."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.get_llm_logger", return_value=None):
            mock_complete.side_effect = [
                "debug",  # router
                '{"hours": 2}',  # debug plan
            ]
            result, queried = await handle_user_message(
                "show me recent errors", cache,
            )

        assert "logging isn't enabled" in result.lower() or "can't look into" in result.lower()
        assert queried is False

    @pytest.mark.asyncio
    async def test_debug_no_matching_calls(self, cache):
        """When no LLM calls match the debug query, return a helpful message."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.get_llm_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.search.return_value = []
            mock_get_logger.return_value = mock_logger

            mock_complete.side_effect = [
                "debug",  # router
                '{"hours": 1, "call_type": "triage", "text": "NonexistentPerson"}',  # debug plan
            ]
            result, queried = await handle_user_message(
                "why wasn't NonexistentPerson's message urgent?", cache,
            )

        assert "No LLM calls found" in result or "no" in result.lower()
        assert queried is False

    @pytest.mark.asyncio
    async def test_debug_errors_only_filter(self, cache):
        """'show me recent errors' should search with errors_only=True."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.get_llm_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.search.return_value = []
            mock_get_logger.return_value = mock_logger

            mock_complete.side_effect = [
                "debug",  # router
                '{"hours": 24, "errors_only": true}',  # debug plan
            ]
            result, queried = await handle_user_message(
                "show me recent errors", cache,
            )

        # Verify search was called with status="error"
        mock_logger.search.assert_called_once_with(
            text=None,
            call_type=None,
            hours=24,
            status="error",
            limit=10,
        )


class TestExtractDebugPlan:
    """Tests for _extract_debug_plan parsing."""

    @pytest.mark.asyncio
    async def test_valid_debug_plan(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"hours": 2, "call_type": "triage", "text": "Sophie"}'
            plan = await _extract_debug_plan("why wasn't Sophie's message urgent?")

        assert plan["hours"] == 2
        assert plan["call_type"] == "triage"
        assert plan["text"] == "Sophie"

    @pytest.mark.asyncio
    async def test_malformed_debug_plan_defaults(self):
        """When Sonnet returns garbage, fall back to a sensible default."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "I don't understand"
            plan = await _extract_debug_plan("what happened?")

        assert plan == {"hours": 2}

    @pytest.mark.asyncio
    async def test_debug_plan_with_errors_only(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"hours": 24, "errors_only": true}'
            plan = await _extract_debug_plan("show me recent errors")

        assert plan["hours"] == 24
        assert plan["errors_only"] is True

    @pytest.mark.asyncio
    async def test_debug_plan_with_convo_context(self):
        """Conversation context is passed through to the LLM call."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"hours": 1, "call_type": "response"}'
            await _extract_debug_plan("why did you say that?", convo_context="[10:00 User]: check Sophie\n[10:01 You (Diplo)]: All quiet!")

        # Verify the system prompt included the conversation context
        call_args = mock_complete.call_args
        assert "check Sophie" in call_args.kwargs["system"] or "check Sophie" in str(call_args)


class TestFormatDebugEntries:
    """Tests for _format_debug_entries formatting."""

    def test_single_entry(self):
        calls = [{
            "timestamp": "2026-03-13T10:00:00+00:00",
            "call_type": "triage",
            "model": "claude-sonnet-4-6",
            "model_used": "claude-sonnet-4-6",
            "system_prompt": "You classify urgency",
            "user_prompt": "Sophie: hey",
            "response": "not_urgent",
            "input_tokens": 500,
            "output_tokens": 10,
            "latency_ms": 200,
            "status": "success",
            "error": None,
            "context_id": "ctx1",
        }]
        result = _format_debug_entries(calls, "America/Los_Angeles")
        assert "triage" in result
        assert "claude-sonnet-4-6" in result
        assert "not_urgent" in result
        assert "ctx1" in result

    def test_multiple_entries_separated_by_divider(self):
        calls = [
            {
                "timestamp": "2026-03-13T10:00:00+00:00",
                "call_type": "triage",
                "model": "claude-sonnet-4-6",
                "model_used": "claude-sonnet-4-6",
                "system_prompt": "sys1",
                "user_prompt": "usr1",
                "response": "resp1",
                "input_tokens": 100,
                "output_tokens": 10,
                "latency_ms": 100,
                "status": "success",
                "error": None,
                "context_id": None,
            },
            {
                "timestamp": "2026-03-13T10:05:00+00:00",
                "call_type": "response",
                "model": "claude-opus-4-6",
                "model_used": "claude-opus-4-6",
                "system_prompt": "sys2",
                "user_prompt": "usr2",
                "response": "resp2",
                "input_tokens": 200,
                "output_tokens": 50,
                "latency_ms": 500,
                "status": "success",
                "error": None,
                "context_id": None,
            },
        ]
        result = _format_debug_entries(calls, "America/Los_Angeles")
        assert "---" in result  # separator between entries
        assert "triage" in result
        assert "response" in result

    def test_error_entry_shows_error(self):
        calls = [{
            "timestamp": "2026-03-13T10:00:00+00:00",
            "call_type": "response",
            "model": "claude-opus-4-6",
            "model_used": "claude-opus-4-6",
            "system_prompt": "sys",
            "user_prompt": "usr",
            "response": None,
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": None,
            "status": "error",
            "error": "Connection timeout",
            "context_id": None,
        }]
        result = _format_debug_entries(calls, "UTC")
        assert "Connection timeout" in result
        assert "error" in result.lower()

    def test_truncation_of_long_prompts(self):
        calls = [{
            "timestamp": "2026-03-13T10:00:00+00:00",
            "call_type": "triage",
            "model": "claude-sonnet-4-6",
            "model_used": "claude-sonnet-4-6",
            "system_prompt": "A" * 1000,  # > 500 char limit
            "user_prompt": "B" * 1500,  # > 800 char limit
            "response": "C" * 1000,  # > 500 char limit
            "input_tokens": 100,
            "output_tokens": 10,
            "latency_ms": 100,
            "status": "success",
            "error": None,
            "context_id": None,
        }]
        result = _format_debug_entries(calls, "UTC")
        # System prompt truncated to 500
        assert "A" * 500 in result
        assert "A" * 501 not in result
        # User prompt truncated to 800
        assert "B" * 800 in result
        assert "B" * 801 not in result
        # Response truncated to 500
        assert "C" * 500 in result
        assert "C" * 501 not in result


class TestHandleDebug:
    """Tests for _handle_debug directly."""

    @pytest.mark.asyncio
    async def test_handle_debug_with_results(self):
        """Verifies Opus gets the formatted logs and produces a response."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.get_llm_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.search.return_value = [{
                "id": "abc",
                "timestamp": "2026-03-13T10:00:00+00:00",
                "context_id": "ctx1",
                "call_type": "triage",
                "model": "claude-sonnet-4-6",
                "model_used": "claude-sonnet-4-6",
                "system_prompt": "Classify urgency",
                "user_prompt": "Sophie: urgent meeting",
                "response": "not_urgent",
                "input_tokens": 500,
                "output_tokens": 10,
                "latency_ms": 200,
                "status": "success",
                "error": None,
            }]
            mock_get_logger.return_value = mock_logger

            mock_complete.return_value = "The triage model saw Sophie's message but classified it as not urgent because..."

            result = await _handle_debug(
                plan={"hours": 2, "call_type": "triage", "text": "Sophie"},
                user_text="why wasn't Sophie's message urgent?",
                session_context="",
                tz_name="America/Los_Angeles",
            )

        assert "triage" in result.lower() or "Sophie" in result
        # Verify search was called with correct filters
        mock_logger.search.assert_called_once_with(
            text="Sophie",
            call_type="triage",
            hours=2,
            status=None,
            limit=10,
        )
        # Verify the Opus call includes the log data
        opus_call = mock_complete.call_args
        assert "triage" in opus_call.kwargs.get("messages", [{}])[0].get("content", "")

    @pytest.mark.asyncio
    async def test_handle_debug_session_context_included(self):
        """Session context is passed to the Opus debug response call."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.get_llm_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.search.return_value = [{
                "id": "abc",
                "timestamp": "2026-03-13T10:00:00+00:00",
                "context_id": None,
                "call_type": "response",
                "model": "claude-opus-4-6",
                "model_used": "claude-opus-4-6",
                "system_prompt": "sys",
                "user_prompt": "usr",
                "response": "resp",
                "input_tokens": 100,
                "output_tokens": 50,
                "latency_ms": 300,
                "status": "success",
                "error": None,
            }]
            mock_get_logger.return_value = mock_logger
            mock_complete.return_value = "Here's what happened..."

            await _handle_debug(
                plan={"hours": 1},
                user_text="what happened?",
                session_context="[10:00 User]: what's new?\n[10:01 You (Diplo)]: All quiet!",
                tz_name="UTC",
            )

        opus_call = mock_complete.call_args
        user_content = opus_call.kwargs["messages"][0]["content"]
        assert "what's new?" in user_content  # session context is included


class TestDebugHintsInErrors:
    """Tests that error messages offer to debug."""

    @pytest.mark.asyncio
    async def test_send_failure_offers_debug(self, cache, contacts):
        """When a Beeper send fails, the error message offers to investigate."""
        contacts.update("Sophie", "whatsapp", "chat_sophie", "Sophie", "2026-03-13T10:00:00+00:00")
        cache.store(make_msg("m1", sender_name="Sophie", text="hey", chat_title="Sophie", chat_id="chat_sophie"))

        client = MagicMock()
        client.messages = MagicMock()

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.send_message", new_callable=AsyncMock, return_value=False):
            mock_complete.side_effect = [
                "reply",  # router
                '{"recipient": "Sophie", "message": "hey"}',  # reply extractor
                '{"action": "send", "text": "Hey!"}',  # compose decision
            ]
            result, queried = await handle_user_message(
                "tell Sophie hey", cache, contacts=contacts, beeper_client=client,
            )

        assert "look into why" in result.lower()

    @pytest.mark.asyncio
    async def test_compose_failure_offers_debug(self):
        """When compose returns unexpected action, offer to check logs."""
        from src.assistant import _compose_and_decide

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            # Return something that isn't valid JSON
            mock_complete.return_value = "I can't compose that"

            result = await _compose_and_decide(
                user_text="tell Sophie something weird",
                recipient={"sender_name": "Sophie", "network": "whatsapp", "chat_title": "Sophie"},
                intent="something weird",
                chat_context="no recent messages",
                convo_context="",
            )

        # The fallback response should mention the ability to dig into logs
        assert "dig into" in result.get("response", "").lower() or "verbatim" in result.get("response", "").lower()

    @pytest.mark.asyncio
    async def test_confirmed_send_failure_offers_debug(self, cache):
        """When sending a confirmed message fails, offer to investigate."""
        _set_pending({
            "chat_id": "chat_sophie",
            "chat_title": "Sophie",
            "text": "hey there",
            "recipient_name": "Sophie",
            "network": "whatsapp",
        })

        client = MagicMock()
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete, \
             patch("src.assistant.send_message", new_callable=AsyncMock, return_value=False):
            mock_complete.return_value = '{"action": "confirm", "text": "hey there"}'

            from src.assistant import _handle_pending_confirmation
            result = await _handle_pending_confirmation(
                _get_pending(), "yes", "", client, cache,
            )

        assert "look into why" in result.lower()


class TestSystemPromptDebugSection:
    """Tests that the assistant system prompt documents the debug capability."""

    def test_system_prompt_mentions_debug(self):
        from pathlib import Path
        prompt = (Path(__file__).parent.parent / "prompts" / "assistant_system.md").read_text()
        assert "investigate" in prompt.lower() or "debug" in prompt.lower()
        assert "past decisions" in prompt.lower() or "errors" in prompt.lower()
        assert "logs" in prompt.lower()

    def test_system_prompt_has_debug_examples(self):
        from pathlib import Path
        prompt = (Path(__file__).parent.parent / "prompts" / "assistant_system.md").read_text()
        assert "why wasn't that urgent" in prompt.lower() or "what went wrong" in prompt.lower()

    def test_system_prompt_has_proactive_debug_section(self):
        from pathlib import Path
        prompt = (Path(__file__).parent.parent / "prompts" / "assistant_system.md").read_text()
        assert "offer" in prompt.lower() or "proactiv" in prompt.lower()
        assert "want me to look into" in prompt.lower() or "want me to dig" in prompt.lower()
