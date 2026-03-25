"""Tests for iMessage support — raw HTTP workaround for '#' in chat IDs.

The Beeper Python SDK doesn't URL-encode chat_id path parameters. iMessage
chat IDs contain '##' (e.g. 'imsg##thread:...'), which httpx interprets as
URL fragment separators, causing 404s. These tests verify the raw HTTP
fallback and network ID normalization that fix this.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, Mock

import pytest

from src.beeper_client import BeeperPoller
from src.message_cache import MessageCache, resolve_network, normalize_network
from src.assistant import _display_network, _execute_search


# ---------- helpers ----------

IMSG_CHAT_ID = "imsg##thread:b15b1c139676a6993a05e22a7b49a1d13b61da5bcac4f8b1"
IMSG_NETWORK_UUID = "imessage_df461d39ed5545ed025fcd30942f27e8"


def make_msg(message_id, sender_name="Alice", text="hello", chat_title="Alice",
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


def make_chat(chat_id, title, account_id, preview_sort_key, preview_text="hi"):
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
    )


@pytest.fixture
def cache(tmp_path):
    c = MessageCache(db_path=tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def poller():
    with patch("src.beeper_client.BeeperDesktop"):
        p = BeeperPoller()
        p.client = MagicMock()
    return p


# ---------- _needs_raw_http ----------

class TestNeedsRawHttp:
    def test_imessage_chat_id(self):
        assert BeeperPoller._needs_raw_http(IMSG_CHAT_ID) is True

    def test_normal_chat_id(self):
        assert BeeperPoller._needs_raw_http("!abc:beeper.local") is False

    def test_single_hash(self):
        assert BeeperPoller._needs_raw_http("something#else") is True

    def test_no_hash(self):
        assert BeeperPoller._needs_raw_http("whatsapp_chat_123") is False


# ---------- normalize_network ----------

class TestNormalizeNetwork:
    def test_strips_uuid_suffix(self):
        assert normalize_network(IMSG_NETWORK_UUID) == "imessage"

    def test_leaves_normal_network_unchanged(self):
        assert normalize_network("whatsapp") == "whatsapp"
        assert normalize_network("telegram") == "telegram"
        assert normalize_network("twitter") == "twitter"

    def test_strips_go_suffix(self):
        """'go' suffix is stripped: facebookgo -> facebook, imessagego -> imessage."""
        assert normalize_network("facebookgo") == "facebook"
        assert normalize_network("instagramgo") == "instagram"
        assert normalize_network("imessagego") == "imessage"

    def test_leaves_short_suffix_unchanged(self):
        """Only strips suffixes that look like hex UUIDs (8+ hex chars)."""
        assert normalize_network("slack_abc") == "slack_abc"

    def test_case_insensitive(self):
        assert normalize_network("IMESSAGE_DF461D39ED5545ED025FCD30942F27E8") == "imessage"


# ---------- resolve_network with UUID-suffixed IDs ----------

class TestResolveNetworkImessage:
    def test_imessage_resolves(self):
        """'imessage' should resolve to 'imessagego' via alias."""
        assert resolve_network("imessage") == "imessagego"

    def test_uuid_suffixed_resolves(self):
        """UUID-suffixed network ID should resolve via normalization."""
        resolved = resolve_network(IMSG_NETWORK_UUID)
        assert resolved == "imessagego"


# ---------- _display_network ----------

class TestDisplayNetworkImessage:
    def test_imessagego(self):
        assert _display_network("imessagego") == "iMessage"

    def test_imessage_plain(self):
        assert _display_network("imessage") == "iMessage"

    def test_uuid_suffixed(self):
        assert _display_network(IMSG_NETWORK_UUID) == "iMessage"

    def test_other_networks_unchanged(self):
        assert _display_network("whatsapp") == "whatsapp"
        assert _display_network("facebookgo") == "Messenger"
        assert _display_network("twitter") == "X"


# ---------- MessageCache.by_network with iMessage ----------

class TestByNetworkImessage:
    def test_finds_uuid_suffixed_messages(self, cache):
        """by_network('imessage') should find messages stored with UUID-suffixed network."""
        cache.store(make_msg("m1", network=IMSG_NETWORK_UUID, sender_name="Sophie",
                             chat_id=IMSG_CHAT_ID))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Bob"))

        results = cache.by_network("imessage")
        assert len(results) == 1
        assert results[0]["sender_name"] == "Sophie"

    def test_imsg_alias(self, cache):
        cache.store(make_msg("m1", network=IMSG_NETWORK_UUID, chat_id=IMSG_CHAT_ID))

        results = cache.by_network("imsg")
        assert len(results) == 1

    def test_imessagego_still_works(self, cache):
        """Messages stored with 'imessagego' should still be found."""
        cache.store(make_msg("m1", network="imessagego"))

        results = cache.by_network("imessage")
        assert len(results) == 1

    def test_both_variants_found(self, cache):
        """Both 'imessagego' and UUID-suffixed should be found."""
        cache.store(make_msg("m1", network="imessagego"))
        cache.store(make_msg("m2", network=IMSG_NETWORK_UUID, chat_id=IMSG_CHAT_ID))

        results = cache.by_network("imessage")
        assert len(results) == 2

    def test_no_false_positives(self, cache):
        """by_network('imessage') should NOT match unrelated networks."""
        cache.store(make_msg("m1", network="whatsapp"))
        cache.store(make_msg("m2", network="facebookgo"))

        results = cache.by_network("imessage")
        assert results == []


# ---------- _execute_search with iMessage network ----------

class TestExecuteSearchImessage:
    def test_network_filter_finds_uuid_suffixed(self, cache):
        cache.store(make_msg("m1", network=IMSG_NETWORK_UUID, sender_name="Sophie",
                             chat_id=IMSG_CHAT_ID))
        cache.store(make_msg("m2", network="whatsapp", sender_name="Bob"))

        results = _execute_search({"network": "imessage"}, cache)
        assert len(results) == 1
        assert results[0]["sender_name"] == "Sophie"

    def test_network_post_filter_with_sender(self, cache):
        """Network + sender should intersect correctly for iMessage."""
        cache.store(make_msg("m1", network=IMSG_NETWORK_UUID, sender_name="Sophie",
                             chat_id="chat_sophie"))
        cache.store(make_msg("m2", network=IMSG_NETWORK_UUID, sender_name="Marc",
                             chat_id="chat_marc"))
        cache.store(make_msg("m3", network="whatsapp", sender_name="Sophie",
                             chat_id="chat_sophie_wa"))

        results = _execute_search({"network": "imessage", "sender": "Sophie"}, cache)
        assert all(r["network"] == IMSG_NETWORK_UUID for r in results)


# ---------- Polling iMessage chats via raw HTTP ----------

class TestPollImessageRawHttp:
    def test_poll_uses_raw_http_for_imessage(self, poller):
        """poll_once should use raw HTTP for iMessage chats instead of the SDK."""
        poller._seen = {IMSG_CHAT_ID: 100}
        chats = [make_chat(IMSG_CHAT_ID, "+1 555-1234", IMSG_NETWORK_UUID, 200)]
        poller.client.chats.list.return_value = iter(chats)

        # Mock the raw HTTP path
        fake_msg = SimpleNamespace(
            id="msg_200",
            chat_id=IMSG_CHAT_ID,
            account_id=IMSG_NETWORK_UUID,
            sender_name="+1 555-1234",
            text="hey there",
            timestamp=datetime(2026, 3, 7, 20, 0, 0, tzinfo=timezone.utc),
            sort_key="200",
            attachments=[],
        )
        with patch.object(poller, "_raw_list_messages", return_value=[fake_msg]) as mock_raw:
            result = poller.poll_once()

        mock_raw.assert_called_once_with(IMSG_CHAT_ID)
        # SDK messages.list should NOT be called for this chat
        poller.client.messages.list.assert_not_called()
        assert len(result) == 1
        assert result[0]["text"] == "hey there"
        assert result[0]["chat_title"] == "+1 555-1234"

    def test_poll_uses_sdk_for_normal_chats(self, poller):
        """Normal chats should still use the SDK, not raw HTTP."""
        poller._seen = {"!abc:beeper.local": 100}
        chats = [make_chat("!abc:beeper.local", "Alice", "whatsapp", 200)]
        msgs = [SimpleNamespace(
            id="msg_200", sort_key="200", text="hi", sender_name="Alice",
            timestamp=datetime(2026, 3, 7, 20, 0, 0, tzinfo=timezone.utc),
            attachments=None, is_sender=False,
        )]
        poller.client.chats.list.return_value = iter(chats)
        poller.client.messages.list.return_value = iter(msgs)

        with patch.object(poller, "_raw_list_messages") as mock_raw:
            poller.poll_once()

        mock_raw.assert_not_called()
        poller.client.messages.list.assert_called_once()


# ---------- _resolve_chat_metadata with iMessage ----------

class TestResolveChatMetadataImessage:
    def test_uses_raw_http_for_imessage(self, poller):
        with patch.object(poller, "_raw_retrieve_chat",
                          return_value=SimpleNamespace(title="+1 555-1234", account_id=None, type="single")):
            metadata = poller._resolve_chat_metadata({IMSG_CHAT_ID})

        assert metadata[IMSG_CHAT_ID]["title"] == "+1 555-1234"
        assert metadata[IMSG_CHAT_ID]["type"] == "single"

    def test_uses_sdk_for_normal_chats(self, poller):
        poller.client.chats.retrieve.return_value = SimpleNamespace(title="Alice", account_id="whatsapp", type="single")

        metadata = poller._resolve_chat_metadata({"!abc:beeper.local"})

        assert metadata["!abc:beeper.local"]["title"] == "Alice"
        poller.client.chats.retrieve.assert_called_once()


# ---------- Watermark deletion ----------

class TestDeleteWatermarks:
    def test_deletes_specific_watermarks(self, cache):
        cache.save_watermarks({IMSG_CHAT_ID: 100, "!abc:beeper.local": 200})

        cache.delete_watermarks([IMSG_CHAT_ID])

        remaining = cache.load_watermarks()
        assert IMSG_CHAT_ID not in remaining
        assert remaining["!abc:beeper.local"] == 200

    def test_delete_nonexistent_is_noop(self, cache):
        cache.save_watermarks({"chat1": 100})
        cache.delete_watermarks(["nonexistent"])
        assert cache.load_watermarks() == {"chat1": 100}


# ---------- Backfill includes iMessage ----------

class TestBackfillImessage:
    def test_backfill_includes_raw_http_chats(self, poller):
        """backfill_recent should fetch iMessage chats via raw HTTP."""
        # Mock messages.search to return nothing (as it does for iMessage)
        poller.client.messages.search.return_value = iter([])

        # Mock _get_recent_chats to return one iMessage chat
        imsg_chat = make_chat(IMSG_CHAT_ID, "+1 555-1234", IMSG_NETWORK_UUID, 200)
        normal_chat = make_chat("!abc:beeper.local", "Alice", "whatsapp", 300)

        now = datetime.now(timezone.utc)
        fake_msg = SimpleNamespace(
            id="msg_1",
            chat_id=IMSG_CHAT_ID,
            account_id=IMSG_NETWORK_UUID,
            sender_name="+1 555-1234",
            text="hey",
            timestamp=now - timedelta(hours=1),
            sort_key="200",
            attachments=[],
        )

        with patch.object(poller, "_get_recent_chats", return_value=[imsg_chat, normal_chat]):
            with patch.object(poller, "_raw_list_messages", return_value=[fake_msg]):
                results = poller.backfill_recent(hours=48)

        assert len(results) == 1
        assert results[0]["chat_title"] == "+1 555-1234"
        assert results[0]["network"] == IMSG_NETWORK_UUID

    def test_backfill_filters_old_messages(self, poller):
        """Backfill should only include messages within the time window."""
        poller.client.messages.search.return_value = iter([])

        imsg_chat = make_chat(IMSG_CHAT_ID, "+1 555-1234", IMSG_NETWORK_UUID, 200)
        now = datetime.now(timezone.utc)

        recent_msg = SimpleNamespace(
            id="msg_recent", chat_id=IMSG_CHAT_ID, account_id=IMSG_NETWORK_UUID,
            sender_name="+1 555-1234", text="recent",
            timestamp=now - timedelta(hours=1), sort_key="200", attachments=[],
        )
        old_msg = SimpleNamespace(
            id="msg_old", chat_id=IMSG_CHAT_ID, account_id=IMSG_NETWORK_UUID,
            sender_name="+1 555-1234", text="old",
            timestamp=now - timedelta(hours=72), sort_key="100", attachments=[],
        )

        with patch.object(poller, "_get_recent_chats", return_value=[imsg_chat]):
            with patch.object(poller, "_raw_list_messages", return_value=[recent_msg, old_msg]):
                results = poller.backfill_recent(hours=48)

        assert len(results) == 1
        assert results[0]["text"] == "recent"
