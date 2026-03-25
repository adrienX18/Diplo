"""Tests for the contact registry."""

import pytest
from datetime import datetime, timedelta, timezone

from src.contacts import ContactRegistry


@pytest.fixture
def contacts(tmp_path):
    c = ContactRegistry(db_path=tmp_path / "test.db")
    yield c
    c.close()


class TestUpdate:
    def test_inserts_new_contact(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie Chat", "2026-03-05T10:00:00+00:00")
        result = contacts.resolve("Sophie")
        assert result is not None
        assert result["chat_id"] == "chat1"
        assert result["network"] == "whatsapp"

    def test_updates_when_newer(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie Chat", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie", "whatsapp", "chat2", "Sophie New Chat", "2026-03-05T12:00:00+00:00")
        result = contacts.resolve("Sophie")
        assert result["chat_id"] == "chat2"

    def test_does_not_update_when_older(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie Chat", "2026-03-05T12:00:00+00:00")
        contacts.update("Sophie", "whatsapp", "chat_old", "Old Chat", "2026-03-05T10:00:00+00:00")
        result = contacts.resolve("Sophie")
        assert result["chat_id"] == "chat1"

    def test_same_person_different_networks(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat_wa", "Sophie WA", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat_tg", "Sophie TG", "2026-03-05T12:00:00+00:00")
        results = contacts.lookup("Sophie")
        assert len(results) == 2

    def test_different_name_variants(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie Chat", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie Martin", "whatsapp", "chat2", "Sophie Martin Chat", "2026-03-05T12:00:00+00:00")
        results = contacts.lookup("Sophie")
        assert len(results) == 2


class TestLookup:
    def test_case_insensitive(self, contacts):
        contacts.update("Sophie Martin", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")
        assert len(contacts.lookup("sophie")) == 1
        assert len(contacts.lookup("SOPHIE")) == 1
        assert len(contacts.lookup("Sophie")) == 1

    def test_substring_match(self, contacts):
        contacts.update("Sophie Martin", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")
        assert len(contacts.lookup("Martin")) == 1
        assert len(contacts.lookup("Sophie Martin")) == 1

    def test_filter_by_network(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat_wa", "Sophie WA", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat_tg", "Sophie TG", "2026-03-05T12:00:00+00:00")
        results = contacts.lookup("Sophie", network="whatsapp")
        assert len(results) == 1
        assert results[0]["network"] == "whatsapp"

    def test_matches_chat_title(self, contacts):
        contacts.update("Alice", "whatsapp", "chat1", "Team Chat", "2026-03-05T10:00:00+00:00")
        results = contacts.lookup("team")
        assert len(results) == 1
        assert results[0]["chat_title"] == "Team Chat"

    def test_no_match_returns_empty(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")
        assert contacts.lookup("Bob") == []

    def test_sorted_by_last_seen(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat_wa", "Sophie WA", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat_tg", "Sophie TG", "2026-03-05T14:00:00+00:00")
        contacts.update("Sophie", "signal", "chat_sg", "Sophie SG", "2026-03-05T12:00:00+00:00")
        results = contacts.lookup("Sophie")
        assert results[0]["network"] == "telegram"
        assert results[1]["network"] == "signal"
        assert results[2]["network"] == "whatsapp"


class TestResolve:
    def test_returns_most_recent(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat_wa", "Sophie WA", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat_tg", "Sophie TG", "2026-03-05T14:00:00+00:00")
        result = contacts.resolve("Sophie")
        assert result["network"] == "telegram"
        assert result["chat_id"] == "chat_tg"

    def test_with_network_filter(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat_wa", "Sophie WA", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat_tg", "Sophie TG", "2026-03-05T14:00:00+00:00")
        result = contacts.resolve("Sophie", network="whatsapp")
        assert result["network"] == "whatsapp"

    def test_returns_none_when_not_found(self, contacts):
        assert contacts.resolve("Nobody") is None

    def test_persists_across_reconnect(self, tmp_path):
        db_path = tmp_path / "test.db"
        c1 = ContactRegistry(db_path=db_path)
        c1.update("Sophie", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")
        c1.close()

        c2 = ContactRegistry(db_path=db_path)
        result = c2.resolve("Sophie")
        assert result is not None
        assert result["chat_id"] == "chat1"
        c2.close()


class TestFuzzyResolve:
    def test_exact_single_match(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")
        result = contacts.fuzzy_resolve("Sophie")
        assert isinstance(result, dict)
        assert result["sender_name"] == "Sophie"

    def test_typo_matches(self, contacts):
        contacts.update("Pierre-Louis", "whatsapp", "chat1", "Pierre-Louis", "2026-03-05T10:00:00+00:00")
        result = contacts.fuzzy_resolve("Pierre Louis")
        assert isinstance(result, dict)
        assert result["sender_name"] == "Pierre-Louis"

    def test_close_typo(self, contacts):
        contacts.update("Pierre-Louis", "whatsapp", "chat1", "Pierre-Louis", "2026-03-05T10:00:00+00:00")
        result = contacts.fuzzy_resolve("Pierre-Loui")
        assert isinstance(result, dict)
        assert result["sender_name"] == "Pierre-Louis"

    def test_no_match_returns_none(self, contacts):
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie", "2026-03-05T10:00:00+00:00")
        assert contacts.fuzzy_resolve("Zzzznotaname") is None

    def test_ambiguous_returns_list(self, contacts):
        contacts.update("Paul Martin", "whatsapp", "chat1", "Paul M", "2026-03-05T10:00:00+00:00")
        contacts.update("Paul Dupont", "whatsapp", "chat2", "Paul D", "2026-03-05T10:00:00+00:00")
        result = contacts.fuzzy_resolve("Paul")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_most_recent_wins_when_large_gap(self, contacts):
        """If one match is >24h more recent, return it directly."""
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie", "2026-03-01T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat2", "Sophie TG", "2026-03-05T14:00:00+00:00")
        result = contacts.fuzzy_resolve("Sophie")
        assert isinstance(result, dict)
        assert result["network"] == "telegram"

    def test_ambiguous_when_small_gap(self, contacts):
        """If two matches are close in time (<24h), disambiguate."""
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie WA", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat2", "Sophie TG", "2026-03-05T14:00:00+00:00")
        result = contacts.fuzzy_resolve("Sophie")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_same_chat_id_returns_single(self, contacts):
        """Multiple name variants pointing to same chat should resolve to one."""
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie Chat", "2026-03-05T10:00:00+00:00")
        contacts.update("Sophie Martin", "whatsapp", "chat1", "Sophie Chat", "2026-03-05T12:00:00+00:00")
        result = contacts.fuzzy_resolve("Sophie")
        assert isinstance(result, dict)

    def test_fuzzy_prefers_recent_paul_with_large_gap(self, contacts):
        """'reply to Paul' picks the recent Paul only if gap > 24h."""
        contacts.update("Paul Dupont", "whatsapp", "chat1", "Paul D", "2026-03-01T10:00:00+00:00")
        contacts.update("Paul Martin", "whatsapp", "chat2", "Paul M", "2026-03-05T14:00:00+00:00")
        result = contacts.fuzzy_resolve("Paul")
        # 4-day gap — most recent Paul wins
        assert isinstance(result, dict)
        assert result["sender_name"] == "Paul Martin"

    def test_fuzzy_disambiguates_paul_with_small_gap(self, contacts):
        """'reply to Paul' asks for disambiguation if both Pauls are recent."""
        contacts.update("Paul Dupont", "whatsapp", "chat1", "Paul D", "2026-03-05T10:00:00+00:00")
        contacts.update("Paul Martin", "whatsapp", "chat2", "Paul M", "2026-03-05T14:00:00+00:00")
        result = contacts.fuzzy_resolve("Paul")
        # 4-hour gap — should disambiguate
        assert isinstance(result, list)
        assert len(result) == 2

    def test_exactly_24h_gap_auto_resolves(self, contacts):
        """Exactly 24h gap should auto-resolve (>= threshold)."""
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie WA", "2026-03-04T10:00:00+00:00")
        contacts.update("Sophie", "telegram", "chat2", "Sophie TG", "2026-03-05T10:00:00+00:00")
        result = contacts.fuzzy_resolve("Sophie")
        assert isinstance(result, dict)
        assert result["network"] == "telegram"

    def test_just_under_24h_gap_disambiguates(self, contacts):
        """23h59m gap should still disambiguate."""
        contacts.update("Sophie", "whatsapp", "chat1", "Sophie WA", "2026-03-04T10:01:00+00:00")
        contacts.update("Sophie", "telegram", "chat2", "Sophie TG", "2026-03-05T10:00:00+00:00")
        result = contacts.fuzzy_resolve("Sophie")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_same_timestamp_disambiguates(self, contacts):
        """Identical timestamps should disambiguate, not pick arbitrarily."""
        contacts.update("Paul Dupont", "whatsapp", "chat1", "Paul D", "2026-03-05T10:00:00+00:00")
        contacts.update("Paul Martin", "whatsapp", "chat2", "Paul M", "2026-03-05T10:00:00+00:00")
        result = contacts.fuzzy_resolve("Paul")
        assert isinstance(result, list)
        assert len(result) == 2


class TestSeedFromCache:
    def test_seeds_contacts(self, tmp_path):
        from src.message_cache import MessageCache

        cache = MessageCache(db_path=tmp_path / "cache.db")
        cache.store({
            "message_id": "m1", "chat_id": "chat1", "chat_title": "Sophie",
            "network": "whatsapp", "sender_name": "Sophie",
            "text": "hello", "timestamp": "2026-03-05T10:00:00+00:00",
            "has_attachments": False,
        })
        cache.store({
            "message_id": "m2", "chat_id": "chat2", "chat_title": "Bob",
            "network": "telegram", "sender_name": "Bob",
            "text": "hey", "timestamp": "2026-03-05T12:00:00+00:00",
            "has_attachments": False,
        })

        contacts = ContactRegistry(db_path=tmp_path / "contacts.db")
        contacts.seed_from_cache(cache_db_path=tmp_path / "cache.db")

        assert contacts.resolve("Sophie") is not None
        assert contacts.resolve("Bob") is not None
        assert contacts.resolve("Nobody") is None

        cache.close()
        contacts.close()
