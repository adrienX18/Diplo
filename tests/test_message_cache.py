"""Tests for the SQLite message cache."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.message_cache import MessageCache


def make_msg(message_id="msg1", chat_id="chat1", chat_title="Alice", network="whatsapp",
             sender_name="Alice", text="hello", timestamp=None, has_attachments=False):
    return {
        "message_id": message_id,
        "chat_id": chat_id,
        "chat_title": chat_title,
        "network": network,
        "sender_name": sender_name,
        "text": text,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "has_attachments": has_attachments,
    }


@pytest.fixture
def cache(tmp_path):
    db_path = tmp_path / "test.db"
    c = MessageCache(db_path=db_path)
    yield c
    c.close()


class TestStore:
    def test_stores_and_retrieves_message(self, cache):
        msg = make_msg()
        cache.store(msg)
        results = cache._query()
        assert len(results) == 1
        assert results[0]["sender_name"] == "Alice"
        assert results[0]["text"] == "hello"

    def test_ignores_duplicates(self, cache):
        msg = make_msg()
        cache.store(msg)
        cache.store(msg)
        results = cache._query()
        assert len(results) == 1

    def test_stores_message_with_no_text(self, cache):
        msg = make_msg(text=None, has_attachments=True)
        cache.store(msg)
        results = cache._query()
        assert results[0]["text"] is None
        assert results[0]["has_attachments"] == 1

    def test_stores_multiple_messages(self, cache):
        cache.store(make_msg(message_id="m1", sender_name="Alice"))
        cache.store(make_msg(message_id="m2", sender_name="Bob"))
        cache.store(make_msg(message_id="m3", sender_name="Charlie"))
        results = cache._query()
        assert len(results) == 3


class TestSearch:
    def test_search_text(self, cache):
        cache.store(make_msg(message_id="m1", text="let's discuss the fundraising deck"))
        cache.store(make_msg(message_id="m2", text="want to grab coffee?"))
        cache.store(make_msg(message_id="m3", text="fundraising round is closing"))

        results = cache.search_text("fundraising")
        assert len(results) == 2

    def test_by_sender(self, cache):
        cache.store(make_msg(message_id="m1", sender_name="Sophie Martin"))
        cache.store(make_msg(message_id="m2", sender_name="Bob Smith"))
        cache.store(make_msg(message_id="m3", sender_name="sophie jones"))

        results = cache.by_sender("sophie")
        assert len(results) == 2

    def test_by_chat(self, cache):
        cache.store(make_msg(message_id="m1", chat_title="Team Chat"))
        cache.store(make_msg(message_id="m2", chat_title="Sophie"))
        cache.store(make_msg(message_id="m3", chat_title="Team Standup"))

        results = cache.by_chat("team")
        assert len(results) == 2


class TestByChatId:
    def test_returns_messages_for_chat(self, cache):
        cache.store(make_msg(message_id="m1", chat_id="chat1", sender_name="Alice"))
        cache.store(make_msg(message_id="m2", chat_id="chat2", sender_name="Bob"))
        cache.store(make_msg(message_id="m3", chat_id="chat1", sender_name="Alice"))

        results = cache.by_chat_id("chat1")
        assert len(results) == 2
        assert all(r["chat_id"] == "chat1" for r in results)

    def test_returns_oldest_first(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg(message_id="m1", chat_id="chat1", timestamp=(now - timedelta(hours=3)).isoformat()))
        cache.store(make_msg(message_id="m2", chat_id="chat1", timestamp=(now - timedelta(hours=1)).isoformat()))

        results = cache.by_chat_id("chat1")
        assert results[0]["message_id"] == "m1"
        assert results[1]["message_id"] == "m2"

    def test_respects_limit(self, cache):
        for i in range(10):
            cache.store(make_msg(message_id=f"m{i}", chat_id="chat1"))

        results = cache.by_chat_id("chat1", limit=3)
        assert len(results) == 3

    def test_empty_for_unknown_chat(self, cache):
        cache.store(make_msg(message_id="m1", chat_id="chat1"))
        assert cache.by_chat_id("unknown") == []


class TestRecent:
    def test_recent_filters_by_hours(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg(message_id="m1", timestamp=(now - timedelta(hours=2)).isoformat()))
        cache.store(make_msg(message_id="m2", timestamp=(now - timedelta(hours=30)).isoformat()))

        results = cache.recent(hours=24)
        assert len(results) == 1
        assert results[0]["message_id"] == "m1"

    def test_recent_returns_newest_first(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg(message_id="m1", timestamp=(now - timedelta(hours=3)).isoformat()))
        cache.store(make_msg(message_id="m2", timestamp=(now - timedelta(hours=1)).isoformat()))

        results = cache.recent(hours=24)
        assert results[0]["message_id"] == "m2"
        assert results[1]["message_id"] == "m1"


class TestPrune:
    def test_prune_deletes_old_messages(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg(message_id="m1", timestamp=(now - timedelta(days=1)).isoformat()))
        cache.store(make_msg(message_id="m2", timestamp=(now - timedelta(days=20)).isoformat()))

        cache.prune()

        results = cache._query()
        assert len(results) == 1
        assert results[0]["message_id"] == "m1"

    def test_prune_keeps_recent_messages(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg(message_id="m1", timestamp=(now - timedelta(days=1)).isoformat()))
        cache.store(make_msg(message_id="m2", timestamp=(now - timedelta(days=10)).isoformat()))

        cache.prune()

        results = cache._query()
        assert len(results) == 2


class TestLastSeen:
    def test_initially_none(self, cache):
        assert cache.get_last_seen() is None

    def test_touch_and_get(self, cache):
        cache.touch_last_seen()
        last_seen = cache.get_last_seen()
        assert last_seen is not None
        # Should be a valid ISO timestamp
        datetime.fromisoformat(last_seen)

    def test_touch_updates_value(self, cache):
        cache.touch_last_seen()
        first = cache.get_last_seen()

        import time
        time.sleep(0.01)

        cache.touch_last_seen()
        second = cache.get_last_seen()
        assert second >= first

    def test_since_last_seen_with_no_prior_interaction(self, cache):
        """First time user — falls back to last 24h."""
        now = datetime.now(timezone.utc)
        cache.store(make_msg(message_id="m1", timestamp=(now - timedelta(hours=2)).isoformat()))
        cache.store(make_msg(message_id="m2", timestamp=(now - timedelta(hours=30)).isoformat()))

        results = cache.since_last_seen()
        assert len(results) == 1
        assert results[0]["message_id"] == "m1"

    def test_since_last_seen_only_returns_new_messages(self, cache):
        now = datetime.now(timezone.utc)
        # Old message — before last_seen
        cache.store(make_msg(message_id="m1", timestamp=(now - timedelta(hours=2)).isoformat()))

        # User checks in
        cache.touch_last_seen()

        import time
        time.sleep(0.01)

        # New message — after last_seen
        new_ts = datetime.now(timezone.utc).isoformat()
        cache.store(make_msg(message_id="m2", timestamp=new_ts))

        results = cache.since_last_seen()
        assert len(results) == 1
        assert results[0]["message_id"] == "m2"

    def test_since_last_seen_returns_empty_when_no_new_messages(self, cache):
        now = datetime.now(timezone.utc)
        cache.store(make_msg(message_id="m1", timestamp=(now - timedelta(hours=2)).isoformat()))

        cache.touch_last_seen()

        results = cache.since_last_seen()
        assert len(results) == 0


class TestTimezone:
    def test_default_timezone(self, cache):
        assert cache.get_timezone() == "America/Los_Angeles"

    def test_set_and_get_timezone(self, cache):
        cache.set_timezone("Europe/Paris")
        assert cache.get_timezone() == "Europe/Paris"

    def test_update_timezone(self, cache):
        cache.set_timezone("Europe/Paris")
        cache.set_timezone("Asia/Tokyo")
        assert cache.get_timezone() == "Asia/Tokyo"


class TestWatermarks:
    def test_load_empty(self, cache):
        assert cache.load_watermarks() == {}

    def test_save_and_load(self, cache):
        watermarks = {"chat1": 100, "chat2": 200}
        cache.save_watermarks(watermarks)
        assert cache.load_watermarks() == watermarks

    def test_save_updates_existing(self, cache):
        cache.save_watermarks({"chat1": 100})
        cache.save_watermarks({"chat1": 300, "chat2": 200})
        loaded = cache.load_watermarks()
        assert loaded["chat1"] == 300
        assert loaded["chat2"] == 200

    def test_preserves_across_reconnect(self, tmp_path):
        """Watermarks survive closing and reopening the database."""
        db_path = tmp_path / "test.db"
        c1 = MessageCache(db_path=db_path)
        c1.save_watermarks({"chat1": 500})
        c1.close()

        c2 = MessageCache(db_path=db_path)
        assert c2.load_watermarks() == {"chat1": 500}
        c2.close()


