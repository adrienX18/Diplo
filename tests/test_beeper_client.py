"""Tests for the Beeper polling client."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from src.beeper_client import BeeperPoller


def make_chat(chat_id, title, account_id, preview_sort_key, preview_text="hi", chat_type="single"):
    preview = SimpleNamespace(
        id=f"msg_{preview_sort_key}",
        sort_key=str(preview_sort_key),
        text=preview_text,
        is_sender=False,
        sender_name="Someone",
        timestamp="2026-03-05T00:00:00Z",
        attachments=None,
    )
    return SimpleNamespace(
        id=chat_id,
        title=title,
        account_id=account_id,
        preview=preview,
        participants=None,
        type=chat_type,
    )


def make_message(sort_key, text="hello", sender_name="Alice", is_sender=False):
    return SimpleNamespace(
        id=f"msg_{sort_key}",
        sort_key=str(sort_key),
        text=text,
        sender_name=sender_name,
        is_sender=is_sender,
        timestamp="2026-03-05T00:00:00Z",
        attachments=None,
    )


@pytest.fixture
def poller():
    with patch("src.beeper_client.BeeperDesktop"):
        p = BeeperPoller()
        p.client = MagicMock()
    return p


class TestSeedWatermarks:
    def test_seeds_from_preview_sort_keys(self, poller):
        chats = [
            make_chat("chat1", "Alice", "whatsapp", 100),
            make_chat("chat2", "Bob", "imessage", 200),
        ]
        poller.client.chats.list.return_value = iter(chats)

        poller.seed_watermarks()

        assert poller._seen == {"chat1": 100, "chat2": 200}

    def test_skips_chats_without_preview(self, poller):
        chat_no_preview = SimpleNamespace(
            id="chat1", title="Empty", account_id="whatsapp", preview=None
        )
        poller.client.chats.list.return_value = iter([chat_no_preview])

        poller.seed_watermarks()

        assert poller._seen == {}


class TestPollOnce:
    def test_detects_new_messages(self, poller):
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "Alice", "whatsapp", 200, "new msg")]
        msgs = [make_message(200, "new msg", "Alice"), make_message(100, "old")]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        result = poller.poll_once()

        assert len(result) == 1
        assert result[0]["text"] == "new msg"
        assert result[0]["sender_name"] == "Alice"
        assert result[0]["chat_title"] == "Alice"

    def test_skips_unchanged_chats(self, poller):
        poller._seen = {"chat1": 200}
        chats = [make_chat("chat1", "Alice", "whatsapp", 200)]

        poller.client.chats.list.return_value = iter(chats)

        result = poller.poll_once()

        assert result == []
        poller.client.messages.list.assert_not_called()

    def test_updates_watermark_after_new_messages(self, poller):
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "Alice", "whatsapp", 300)]
        msgs = [
            make_message(300, "newest"),
            make_message(200, "middle"),
            make_message(100, "old"),
        ]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        poller.poll_once()

        assert poller._seen["chat1"] == 300

    def test_handles_new_unseen_chat(self, poller):
        poller._seen = {}
        chats = [make_chat("new_chat", "Bob", "telegram", 500)]
        msgs = [make_message(500, "first msg", "Bob")]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        result = poller.poll_once()

        assert len(result) == 1
        assert result[0]["chat_title"] == "Bob"
        assert poller._seen["new_chat"] == 500

    def test_includes_own_messages(self, poller):
        """Own messages are not filtered out (user preference)."""
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "Self Chat", "whatsapp", 200)]
        msgs = [make_message(200, "my msg", "Me", is_sender=True)]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        result = poller.poll_once()

        assert len(result) == 1
        assert result[0]["text"] == "my msg"

    def test_multiple_new_messages_in_chronological_order(self, poller):
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "Alice", "whatsapp", 300)]
        # API returns newest first
        msgs = [
            make_message(300, "third"),
            make_message(200, "second"),
            make_message(100, "old"),
        ]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        result = poller.poll_once()

        assert len(result) == 2
        assert result[0]["text"] == "second"
        assert result[1]["text"] == "third"

    def test_message_with_attachment(self, poller):
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "Alice", "whatsapp", 200)]
        msg = make_message(200, None, "Alice")
        msg.attachments = [{"type": "img"}]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter([msg])

        result = poller.poll_once()

        assert result[0]["has_attachments"] is True
        assert result[0]["text"] is None


class TestSortKeyComparison:
    """Sort keys must be compared as integers, not strings."""

    def test_small_sort_key_not_greater_than_large(self, poller):
        """WhatsApp uses small sort keys (389869), iMessage uses large ones (1772670457888).
        String comparison would incorrectly say "389869" > "1772670457888"."""
        poller._seen = {"chat1": 1772670457888}
        chats = [make_chat("chat1", "Alice", "imessage", 1772670457888)]

        poller.client.chats.list.return_value = iter(chats)

        result = poller.poll_once()

        assert result == []
        poller.client.messages.list.assert_not_called()

    def test_numerically_larger_sort_key_detected(self, poller):
        poller._seen = {"chat1": 389869}
        chats = [make_chat("chat1", "Self", "whatsapp", 389870)]
        msgs = [make_message(389870, "new")]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        result = poller.poll_once()

        assert len(result) == 1


class TestFetchErrorWatermark:
    """When fetching messages fails (e.g. iMessage 404), the watermark
    should still advance so we don't retry the same chat every poll."""

    def test_watermark_advances_on_fetch_error(self, poller):
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "+1 555-1234", "imessage", 200)]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.side_effect = Exception("Chat not found: imsg")

        result = poller.poll_once()

        assert result == []
        # Watermark should advance to prevent retrying every 5s
        assert poller._seen["chat1"] == 200

    def test_fetch_error_chat_uses_group_type(self, poller):
        """Fetch error chats default to group type in make_chat for backwards compat."""
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "+1 555-1234", "imessage", 200, chat_type="group")]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.side_effect = Exception("Chat not found: imsg")

        result = poller.poll_once()
        assert result == []

    def test_no_retry_after_fetch_error(self, poller):
        poller._seen = {"chat1": 100}

        # First poll — chat has activity at sort_key 200, fetch fails
        chats1 = [make_chat("chat1", "+1 555-1234", "imessage", 200)]
        poller.client.chats.list.return_value = iter(chats1)
        poller.client.messages.list.side_effect = Exception("Chat not found: imsg")
        poller.poll_once()

        # Second poll — same sort_key, should skip (watermark already at 200)
        chats2 = [make_chat("chat1", "+1 555-1234", "imessage", 200)]
        poller.client.chats.list.return_value = iter(chats2)
        poller.client.messages.list.reset_mock()
        poller.poll_once()

        poller.client.messages.list.assert_not_called()


class TestResolveSenderName:
    """In iMessage DMs, sender_name is often a phone number. For DM chats,
    _resolve_sender_name should swap it for the chat title (the display name)."""

    def test_phone_number_replaced_in_dm(self):
        result = BeeperPoller._resolve_sender_name("+15551234567", "Sophie Martin", "single")
        assert result == "Sophie Martin"

    def test_phone_number_kept_in_group_chat(self):
        result = BeeperPoller._resolve_sender_name("+15551234567", "Family Chat", "group")
        assert result == "+15551234567"

    def test_normal_name_unchanged_in_dm(self):
        result = BeeperPoller._resolve_sender_name("Sophie", "Sophie Martin", "single")
        assert result == "Sophie"

    def test_unknown_sender_unchanged(self):
        result = BeeperPoller._resolve_sender_name("Unknown", "Sophie", "single")
        assert result == "Unknown"

    def test_short_phone_number_not_matched(self):
        """Very short numbers (< 7 digits) shouldn't match the phone pattern."""
        result = BeeperPoller._resolve_sender_name("+123", "Sophie", "single")
        assert result == "+123"

    def test_international_phone_replaced(self):
        result = BeeperPoller._resolve_sender_name("+33612345678", "Pierre", "single")
        assert result == "Pierre"

    def test_empty_chat_title_keeps_phone(self):
        result = BeeperPoller._resolve_sender_name("+15551234567", "", "single")
        assert result == "+15551234567"

    def test_poll_once_resolves_phone_in_dm(self, poller):
        """End-to-end: poll_once should use chat title for phone-number senders in DMs."""
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "Sophie Martin", "imessagego", 200, chat_type="single")]
        msgs = [make_message(200, "hey!", "+15551234567")]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        result = poller.poll_once()

        assert len(result) == 1
        assert result[0]["sender_name"] == "Sophie Martin"

    def test_poll_once_keeps_phone_in_group(self, poller):
        """End-to-end: poll_once should NOT replace phone numbers in group chats."""
        poller._seen = {"chat1": 100}
        chats = [make_chat("chat1", "Family Chat", "imessagego", 200, chat_type="group")]
        msgs = [make_message(200, "hey!", "+15551234567")]

        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        result = poller.poll_once()

        assert len(result) == 1
        assert result[0]["sender_name"] == "+15551234567"
