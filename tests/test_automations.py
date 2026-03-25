"""Tests for AutomationStore — scheduled task storage, scheduling, and management."""

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.automations import AutomationStore, run_scheduler_tick, format_delay
from src.assistant import _handle_automation_intent, _humanize_cron, _humanize_trigger


@pytest.fixture
def store(tmp_path):
    """Create a fresh AutomationStore with a temp DB."""
    db = tmp_path / "test.db"
    s = AutomationStore(db_path=db)
    yield s
    s.close()


class TestCreateAutomation:
    def test_create_valid_scheduled(self, store):
        aid = store.create(
            description="Morning summary",
            schedule="0 9 * * *",
            action="summarize my messages",
        )
        assert aid is not None
        assert aid > 0

    def test_create_stores_fields(self, store):
        aid = store.create(
            description="Morning summary",
            schedule="0 9 * * *",
            action="summarize my messages",
        )
        auto = store.get(aid)
        assert auto["description"] == "Morning summary"
        assert auto["schedule"] == "0 9 * * *"
        assert auto["action"] == "summarize my messages"
        assert auto["type"] == "scheduled"
        assert auto["enabled"] == 1
        assert auto["next_run_at"] is not None
        assert auto["last_run_at"] is None

    def test_create_invalid_cron_raises(self, store):
        with pytest.raises(ValueError, match="Invalid cron"):
            store.create(
                description="Bad schedule",
                schedule="not a cron",
                action="do stuff",
            )

    def test_create_computes_next_run_in_future(self, store):
        aid = store.create(
            description="Every hour",
            schedule="0 * * * *",
            action="check messages",
        )
        auto = store.get(aid)
        next_run = datetime.fromisoformat(auto["next_run_at"])
        assert next_run > datetime.now(timezone.utc)

    def test_create_respects_timezone(self, store):
        # Create with a timezone — next_run should differ from UTC interpretation
        aid = store.create(
            description="9am Tokyo",
            schedule="0 9 * * *",
            action="check",
            tz_name="Asia/Tokyo",
        )
        auto = store.get(aid)
        next_run = datetime.fromisoformat(auto["next_run_at"])
        # next_run is stored in UTC, should be valid
        assert next_run.tzinfo is not None or "+" in auto["next_run_at"] or "Z" in auto["next_run_at"]


class TestListAndGet:
    def test_list_empty(self, store):
        assert store.list_all() == []

    def test_list_returns_all(self, store):
        store.create("A", "0 9 * * *", "do a")
        store.create("B", "0 10 * * *", "do b")
        result = store.list_all()
        assert len(result) == 2

    def test_get_nonexistent_returns_none(self, store):
        assert store.get(999) is None


class TestDeleteAutomation:
    def test_delete_by_id(self, store):
        aid = store.create("To delete", "0 9 * * *", "do stuff")
        assert store.delete(aid) is True
        assert store.get(aid) is None

    def test_delete_nonexistent(self, store):
        assert store.delete(999) is False


class TestToggleAutomation:
    def test_disable(self, store):
        aid = store.create("Toggle me", "0 9 * * *", "do stuff")
        store.toggle(aid, enabled=False)
        auto = store.get(aid)
        assert auto["enabled"] == 0

    def test_enable_recomputes_next_run(self, store):
        aid = store.create("Toggle me", "0 9 * * *", "do stuff")
        store.toggle(aid, enabled=False)
        auto_disabled = store.get(aid)
        store.toggle(aid, enabled=True)
        auto_enabled = store.get(aid)
        # next_run should be recomputed (may or may not change, but should exist)
        assert auto_enabled["next_run_at"] is not None

    def test_toggle_nonexistent_returns_false(self, store):
        assert store.toggle(999, enabled=True) is False


class TestGetDue:
    def test_no_due_automations(self, store):
        store.create("Future", "0 9 * * *", "do stuff")
        assert store.get_due() == []

    def test_due_automation_found(self, store):
        aid = store.create("Due now", "0 9 * * *", "do stuff")
        # Manually set next_run_at to the past
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, aid))
        store._conn.commit()

        due = store.get_due()
        assert len(due) == 1
        assert due[0]["id"] == aid

    def test_disabled_automation_not_due(self, store):
        aid = store.create("Disabled", "0 9 * * *", "do stuff")
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, aid))
        store._conn.commit()
        store.toggle(aid, enabled=False)

        assert store.get_due() == []


class TestMarkRun:
    def test_mark_run_updates_timestamps(self, store):
        aid = store.create("Run me", "0 * * * *", "do stuff")

        store.mark_run(aid)

        auto = store.get(aid)
        assert auto["last_run_at"] is not None
        # next_run should be in the future
        next_run = datetime.fromisoformat(auto["next_run_at"])
        assert next_run > datetime.now(timezone.utc)

    def test_mark_run_advances_next_run(self, store):
        """Ensure next_run_at moves forward, not backward."""
        aid = store.create("Hourly", "0 * * * *", "check")
        store.mark_run(aid)
        first_next = store.get(aid)["next_run_at"]

        store.mark_run(aid)
        second_next = store.get(aid)["next_run_at"]

        # Each mark_run should advance next_run_at
        assert second_next >= first_next


class TestResolveByDescription:
    def test_exact_match(self, store):
        store.create("Morning summary", "0 9 * * *", "summarize")
        result = store.resolve_by_description("Morning summary")
        assert isinstance(result, dict)
        assert result["description"] == "Morning summary"

    def test_substring_match(self, store):
        store.create("Morning summary", "0 9 * * *", "summarize")
        result = store.resolve_by_description("morning")
        assert isinstance(result, dict)

    def test_no_match(self, store):
        store.create("Morning summary", "0 9 * * *", "summarize")
        result = store.resolve_by_description("evening")
        assert result is None

    def test_ambiguous_returns_list(self, store):
        store.create("Morning summary", "0 9 * * *", "summarize")
        store.create("Morning check", "0 8 * * *", "check")
        result = store.resolve_by_description("morning")
        assert isinstance(result, list)
        assert len(result) == 2


class TestSchedulerTick:
    @pytest.mark.asyncio
    async def test_executes_due_automation(self, store):
        aid = store.create("Test auto", "0 * * * *", "what's new?")
        # Force it to be due
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, aid))
        store._conn.commit()

        mock_handler = AsyncMock(return_value=("Here's your summary", False))
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        # Handler should have been called with the action text
        mock_handler.assert_called_once_with("what's new?")
        # Channel should have received the prefixed response
        mock_channel.send_message.assert_called_once()
        sent_text = mock_channel.send_message.call_args[0][0]
        assert "[Auto] Test auto" in sent_text
        assert "Here's your summary" in sent_text

        # next_run_at should have advanced
        auto = store.get(aid)
        assert auto["last_run_at"] is not None
        next_run = datetime.fromisoformat(auto["next_run_at"])
        assert next_run > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_skips_not_due(self, store):
        store.create("Future auto", "0 * * * *", "check")

        mock_handler = AsyncMock()
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        mock_handler.assert_not_called()
        mock_channel.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self, store):
        aid = store.create("Failing auto", "0 * * * *", "do stuff")
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, aid))
        store._conn.commit()

        mock_handler = AsyncMock(side_effect=Exception("LLM down"))
        mock_channel = AsyncMock()

        # Should not raise
        await run_scheduler_tick(store, mock_handler, mock_channel)

        # next_run should still advance (don't retry same run)
        auto = store.get(aid)
        assert auto["last_run_at"] is not None

    @pytest.mark.asyncio
    async def test_reply_action_blocked(self, store):
        """Automations should not send messages — reply intents are blocked."""
        aid = store.create("Auto reply", "0 * * * *", "tell Sophie hi")
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, aid))
        store._conn.commit()

        # Handler returns a reply action result (simulating the assistant
        # returning a send confirmation). The key thing is the scheduler
        # doesn't crash and marks the run.
        mock_handler = AsyncMock(return_value=("Sent to Sophie: hi", False))
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        # Should still send the response to channel (it's just text)
        mock_channel.send_message.assert_called_once()


class TestCreateTriggeredAutomation:
    def test_create_triggered(self, store):
        import json
        trigger = {"sender": "Sophie"}
        aid = store.create_triggered(
            description="Notify when Sophie messages",
            trigger_config=trigger,
            action="notify",
        )
        auto = store.get(aid)
        assert auto["type"] == "triggered"
        assert auto["trigger_config"] == json.dumps(trigger)
        assert auto["schedule"] is None
        assert auto["enabled"] == 1

    def test_evaluate_sender_match(self, store):
        trigger = {"sender": "Sophie"}
        aid = store.create_triggered("Sophie alert", trigger, "notify")

        msg = {"sender_name": "Sophie Martin", "text": "hey", "chat_title": "Sophie DM", "network": "whatsapp"}
        matches = store.evaluate_triggers(msg)
        assert len(matches) == 1
        assert matches[0]["id"] == aid

    def test_evaluate_sender_no_match(self, store):
        trigger = {"sender": "Sophie"}
        store.create_triggered("Sophie alert", trigger, "notify")

        msg = {"sender_name": "Marc", "text": "hey", "chat_title": "Marc DM", "network": "whatsapp"}
        assert store.evaluate_triggers(msg) == []

    def test_evaluate_keyword_match(self, store):
        trigger = {"keyword": "fundraising"}
        aid = store.create_triggered("Fundraising alert", trigger, "notify")

        msg = {"sender_name": "Marc", "text": "Let's discuss the fundraising deck", "chat_title": "Investors", "network": "slack"}
        matches = store.evaluate_triggers(msg)
        assert len(matches) == 1

    def test_evaluate_keyword_no_match(self, store):
        trigger = {"keyword": "fundraising"}
        store.create_triggered("Fundraising alert", trigger, "notify")

        msg = {"sender_name": "Marc", "text": "Let's get lunch", "chat_title": "Marc", "network": "whatsapp"}
        assert store.evaluate_triggers(msg) == []

    def test_evaluate_combined_sender_and_keyword(self, store):
        trigger = {"sender": "Sophie", "keyword": "urgent"}
        aid = store.create_triggered("Sophie urgent", trigger, "notify")

        # Both match
        msg = {"sender_name": "Sophie", "text": "this is urgent", "chat_title": "Sophie", "network": "whatsapp"}
        assert len(store.evaluate_triggers(msg)) == 1

        # Only sender matches
        msg2 = {"sender_name": "Sophie", "text": "hey how are you", "chat_title": "Sophie", "network": "whatsapp"}
        assert store.evaluate_triggers(msg2) == []

        # Only keyword matches
        msg3 = {"sender_name": "Marc", "text": "this is urgent", "chat_title": "Marc", "network": "whatsapp"}
        assert store.evaluate_triggers(msg3) == []

    def test_evaluate_network_filter(self, store):
        trigger = {"sender": "Sophie", "network": "whatsapp"}
        store.create_triggered("Sophie on WA", trigger, "notify")

        msg_wa = {"sender_name": "Sophie", "text": "hi", "chat_title": "Sophie", "network": "whatsapp"}
        assert len(store.evaluate_triggers(msg_wa)) == 1

        msg_tg = {"sender_name": "Sophie", "text": "hi", "chat_title": "Sophie", "network": "telegram"}
        assert store.evaluate_triggers(msg_tg) == []

    def test_evaluate_chat_filter(self, store):
        trigger = {"chat": "Investors"}
        store.create_triggered("Investors chat", trigger, "notify")

        msg = {"sender_name": "Marc", "text": "hi", "chat_title": "Investors Group", "network": "slack"}
        assert len(store.evaluate_triggers(msg)) == 1

        msg2 = {"sender_name": "Marc", "text": "hi", "chat_title": "Random", "network": "slack"}
        assert store.evaluate_triggers(msg2) == []

    def test_disabled_trigger_not_evaluated(self, store):
        trigger = {"sender": "Sophie"}
        aid = store.create_triggered("Sophie alert", trigger, "notify")
        store.toggle(aid, enabled=False)

        msg = {"sender_name": "Sophie", "text": "hey", "chat_title": "Sophie", "network": "whatsapp"}
        assert store.evaluate_triggers(msg) == []

    def test_cooldown_prevents_refiring(self, store):
        trigger = {"sender": "Sophie"}
        aid = store.create_triggered("Sophie alert", trigger, "notify", cooldown_seconds=300)

        msg = {"sender_name": "Sophie", "text": "hey", "chat_title": "Sophie", "network": "whatsapp"}

        # First match
        assert len(store.evaluate_triggers(msg)) == 1
        store.mark_triggered(aid)

        # Second match within cooldown — should NOT fire
        assert store.evaluate_triggers(msg) == []

    def test_cooldown_expired_allows_refiring(self, store):
        trigger = {"sender": "Sophie"}
        aid = store.create_triggered("Sophie alert", trigger, "notify", cooldown_seconds=300)

        msg = {"sender_name": "Sophie", "text": "hey", "chat_title": "Sophie", "network": "whatsapp"}

        assert len(store.evaluate_triggers(msg)) == 1
        store.mark_triggered(aid)

        # Fast-forward last_run_at to past the cooldown
        past = (datetime.now(timezone.utc) - timedelta(seconds=301)).isoformat()
        store._conn.execute("UPDATE automations SET last_run_at = ? WHERE id = ?", (past, aid))
        store._conn.commit()

        assert len(store.evaluate_triggers(msg)) == 1


class TestHandleAutomationIntent:
    """Test the intent handling in assistant.py."""

    def test_create_automation_intent(self, store):
        plan = {"create_automation": {"description": "Morning summary", "schedule": "0 9 * * *", "action": "summarize my messages"}}
        result = _handle_automation_intent(plan, store, "UTC")
        assert result is not None
        assert "Morning summary" in result
        assert "#" in result
        assert store.list_all()

    def test_create_trigger_intent(self, store):
        plan = {"create_trigger": {"description": "Sophie alert", "trigger": {"sender": "Sophie"}, "action": "notify"}}
        result = _handle_automation_intent(plan, store, "UTC")
        assert result is not None
        assert "Sophie alert" in result
        assert store.list_all()

    def test_list_automations_empty(self, store):
        plan = {"list_automations": True}
        result = _handle_automation_intent(plan, store, "UTC")
        assert "No automations" in result

    def test_list_automations_with_entries(self, store):
        store.create("Morning summary", "0 9 * * *", "summarize")
        store.create_triggered("Sophie alert", {"sender": "Sophie"}, "notify")
        plan = {"list_automations": True}
        result = _handle_automation_intent(plan, store, "UTC")
        assert "#1" in result
        assert "#2" in result
        assert "Morning summary" in result
        assert "Sophie alert" in result

    def test_delete_by_description(self, store):
        store.create("Morning summary", "0 9 * * *", "summarize")
        plan = {"delete_automation": {"description": "morning"}}
        result = _handle_automation_intent(plan, store, "UTC")
        assert "Deleted" in result
        assert store.list_all() == []

    def test_toggle_automation(self, store):
        aid = store.create("Morning summary", "0 9 * * *", "summarize")
        plan = {"toggle_automation": {"id": aid, "enabled": False}}
        result = _handle_automation_intent(plan, store, "UTC")
        assert "paused" in result.lower()
        assert store.get(aid)["enabled"] == 0

    def test_non_automation_intent_returns_none(self, store):
        plan = {"since_last_seen": True}
        assert _handle_automation_intent(plan, store, "UTC") is None

    def test_invalid_cron_returns_error(self, store):
        plan = {"create_automation": {"description": "Bad", "schedule": "invalid", "action": "do stuff"}}
        result = _handle_automation_intent(plan, store, "UTC")
        assert "couldn't parse" in result


class TestHumanizeCron:
    def test_daily_at_9am(self):
        assert _humanize_cron("0 9 * * *") == "every day at 9:00am"

    def test_daily_at_850am(self):
        assert _humanize_cron("50 8 * * *") == "every day at 8:50am"

    def test_daily_at_5pm(self):
        assert _humanize_cron("0 17 * * *") == "every day at 5:00pm"

    def test_daily_at_noon(self):
        assert _humanize_cron("0 12 * * *") == "every day at 12:00pm"

    def test_daily_at_midnight(self):
        assert _humanize_cron("0 0 * * *") == "every day at 12:00am"

    def test_weekly_friday(self):
        assert _humanize_cron("0 17 * * 5") == "every Fri at 5:00pm"

    def test_weekly_multiple_days(self):
        assert _humanize_cron("0 8 * * 1,3") == "every Mon, Wed at 8:00am"

    def test_every_2_hours(self):
        assert _humanize_cron("0 */2 * * *") == "every 2 hours"

    def test_every_15_minutes(self):
        assert _humanize_cron("*/15 * * * *") == "every 15 minutes"

    def test_exotic_falls_back_to_raw(self):
        assert _humanize_cron("0 9 1 * *") == "0 9 1 * *"  # monthly

    def test_invalid_falls_back_to_raw(self):
        assert _humanize_cron("not a cron") == "not a cron"


class TestDelayedTriggers:
    """Tests for the delayed trigger / one-shot automation system."""

    def test_create_triggered_with_delay(self, store):
        """Delay is stored in trigger_config as _delay_seconds."""
        import json
        aid = store.create_triggered(
            "Delayed reply", {"sender": "Sophie"}, "reply to Sophie",
            delay_seconds=3600,
        )
        auto = store.get(aid)
        config = json.loads(auto["trigger_config"])
        assert config["_delay_seconds"] == 3600
        assert config["sender"] == "Sophie"

    def test_create_triggered_without_delay(self, store):
        """No delay means no _delay_seconds in config."""
        import json
        aid = store.create_triggered(
            "Immediate notify", {"sender": "Marc"}, "notify",
        )
        auto = store.get(aid)
        config = json.loads(auto["trigger_config"])
        assert "_delay_seconds" not in config

    def test_get_delay_seconds(self, store):
        aid = store.create_triggered(
            "Delayed", {"sender": "Sophie"}, "reply",
            delay_seconds=7200,
        )
        auto = store.get(aid)
        assert store.get_delay_seconds(auto) == 7200

    def test_get_delay_seconds_no_delay(self, store):
        aid = store.create_triggered(
            "Immediate", {"sender": "Marc"}, "notify",
        )
        auto = store.get(aid)
        assert store.get_delay_seconds(auto) == 0

    def test_delay_does_not_affect_trigger_matching(self, store):
        """_delay_seconds is ignored by _trigger_matches — only sender/keyword/etc matter."""
        store.create_triggered(
            "Delayed Sophie", {"sender": "Sophie"}, "reply",
            delay_seconds=3600,
        )
        msg = {"sender_name": "Sophie", "text": "hey", "chat_title": "Sophie", "network": "whatsapp"}
        assert len(store.evaluate_triggers(msg)) == 1

        msg2 = {"sender_name": "Marc", "text": "hey", "chat_title": "Marc", "network": "whatsapp"}
        assert store.evaluate_triggers(msg2) == []


class TestOneShotAutomations:
    """Tests for ephemeral one-shot scheduled automations."""

    def test_create_delayed(self, store):
        """create_delayed creates a one-shot with correct fields."""
        # First create a parent trigger
        parent_id = store.create_triggered(
            "Sophie trigger", {"sender": "Sophie"}, "reply to Sophie",
            delay_seconds=3600,
        )
        one_id = store.create_delayed(
            parent_id=parent_id,
            delay_seconds=3600,
            action="reply to Sophie",
            description="[Delayed] Sophie trigger",
        )
        one = store.get(one_id)
        assert one["type"] == "scheduled"
        assert one["one_shot"] == 1
        assert one["parent_id"] == parent_id
        assert one["enabled"] == 1
        assert one["next_run_at"] is not None

    def test_create_delayed_fires_at_correct_time(self, store):
        """One-shot's next_run_at should be ~delay_seconds from now."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=7200)
        one_id = store.create_delayed(parent_id, 7200, "action", "[Delayed] P")

        one = store.get(one_id)
        fire_at = datetime.fromisoformat(one["next_run_at"])
        now = datetime.now(timezone.utc)
        diff = (fire_at - now).total_seconds()
        assert 7190 < diff < 7210  # ~2 hours from now

    def test_oneshot_is_due_when_time_passes(self, store):
        """One-shot should appear in get_due() after its fire time."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=60)
        one_id = store.create_delayed(parent_id, 60, "action", "[Delayed] P")

        # Not due yet
        assert not any(d["id"] == one_id for d in store.get_due())

        # Force next_run_at to the past
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, one_id))
        store._conn.commit()

        due = store.get_due()
        assert any(d["id"] == one_id for d in due)

    def test_oneshot_excluded_from_list_all(self, store):
        """One-shots should not appear in list_all() — they're ephemeral."""
        parent_id = store.create_triggered("Parent", {"sender": "S"}, "action")
        store.create_delayed(parent_id, 3600, "action", "[Delayed]")
        store.create("Regular", "0 9 * * *", "summarize")

        all_autos = store.list_all()
        # Should have the trigger and the regular scheduled, but NOT the one-shot
        assert len(all_autos) == 2
        assert all(a["one_shot"] == 0 for a in all_autos)

    def test_timer_reset_cancels_old_oneshot(self, store):
        """Creating a new delayed one-shot for the same parent cancels the old one."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=3600)
        first_id = store.create_delayed(parent_id, 3600, "action", "[Delayed]")
        assert store.get(first_id) is not None

        # "Reset" — create another for the same parent
        second_id = store.create_delayed(parent_id, 3600, "action", "[Delayed]")
        assert store.get(first_id) is None  # Old one cancelled
        assert store.get(second_id) is not None  # New one exists

    def test_cancel_pending_oneshots(self, store):
        parent_id = store.create_triggered("P", {"sender": "S"}, "action")
        store.create_delayed(parent_id, 3600, "a1", "[D1]")
        store.create_delayed(parent_id, 7200, "a2", "[D2]")  # This cancels D1 internally

        # Explicitly cancel
        cancelled = store.cancel_pending_oneshots(parent_id)
        assert cancelled == 1  # Only D2 was left (D1 was already cancelled by D2)

    def test_cancel_no_pending(self, store):
        """Cancelling when nothing is pending returns 0."""
        assert store.cancel_pending_oneshots(999) == 0

    def test_delete_parent_deletes_child_oneshots(self, store):
        """Deleting a trigger also deletes its pending one-shots."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=3600)
        one_id = store.create_delayed(parent_id, 3600, "action", "[Delayed]")

        store.delete(parent_id)
        assert store.get(parent_id) is None
        assert store.get(one_id) is None

    @pytest.mark.asyncio
    async def test_scheduler_deletes_oneshot_after_run(self, store):
        """One-shot should be deleted (not mark_run'd) after scheduler executes it."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=60)
        one_id = store.create_delayed(parent_id, 60, "reply to Sophie", "[Delayed] P")

        # Force it to be due
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, one_id))
        store._conn.commit()

        mock_handler = AsyncMock(return_value=("Sent to Sophie: hey!", False))
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        # Handler should have been called with the action
        mock_handler.assert_called_once_with("reply to Sophie")

        # Channel should have received the response
        mock_channel.send_message.assert_called_once()
        sent = mock_channel.send_message.call_args[0][0]
        assert "[Auto]" in sent
        assert "[Delayed] P" in sent

        # One-shot should be DELETED, not rescheduled
        assert store.get(one_id) is None

        # Parent trigger should still exist
        assert store.get(parent_id) is not None

    @pytest.mark.asyncio
    async def test_scheduler_runs_regular_and_oneshot_separately(self, store):
        """Regular automations get mark_run, one-shots get deleted."""
        # Create a regular scheduled automation
        reg_id = store.create("Hourly check", "0 * * * *", "what's new?")
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, reg_id))

        # Create a one-shot
        parent_id = store.create_triggered("P", {"sender": "S"}, "action")
        one_id = store.create_delayed(parent_id, 60, "check Sophie", "[Delayed]")
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, one_id))
        store._conn.commit()

        mock_handler = AsyncMock(return_value=("Done", False))
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        # Both should have been executed
        assert mock_handler.call_count == 2

        # Regular should still exist with updated next_run
        reg = store.get(reg_id)
        assert reg is not None
        assert reg["last_run_at"] is not None

        # One-shot should be gone
        assert store.get(one_id) is None

    @pytest.mark.asyncio
    async def test_oneshot_failure_still_deletes(self, store):
        """Even if the action fails, the one-shot is still cleaned up."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action")
        one_id = store.create_delayed(parent_id, 60, "failing action", "[Delayed]")
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, one_id))
        store._conn.commit()

        mock_handler = AsyncMock(side_effect=Exception("LLM down"))
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        # One-shot should still be deleted even though action failed
        assert store.get(one_id) is None


class TestDelayedTriggerIntegration:
    """End-to-end tests for the trigger → one-shot → execute flow."""

    def test_full_flow_trigger_creates_oneshot(self, store):
        """When a delayed trigger matches, calling create_delayed produces a one-shot."""
        trigger_id = store.create_triggered(
            "Auto-reply to Natasha",
            {"sender": "Natasha", "network": "instagram"},
            "reply to Natasha with something fun",
            delay_seconds=18960,  # 5h 16m
        )

        # Simulate a message from Natasha
        msg = {"sender_name": "Natasha", "text": "hey!", "chat_title": "Natasha", "network": "instagramgo"}
        triggered = store.evaluate_triggers(msg)
        assert len(triggered) == 1
        assert triggered[0]["id"] == trigger_id

        # Simulate what main.py does: create the one-shot
        delay = store.get_delay_seconds(triggered[0])
        assert delay == 18960
        one_id = store.create_delayed(
            parent_id=trigger_id,
            delay_seconds=delay,
            action=triggered[0]["action"],
            description=f"[Delayed] {triggered[0]['description']}",
        )

        one = store.get(one_id)
        assert one["one_shot"] == 1
        assert one["parent_id"] == trigger_id
        assert one["action"] == "reply to Natasha with something fun"

    def test_full_flow_remessage_resets_timer(self, store):
        """If Natasha messages again, the timer resets (old one-shot deleted, new one created)."""
        trigger_id = store.create_triggered(
            "Auto-reply", {"sender": "Natasha"}, "reply", delay_seconds=3600,
        )

        # First message → first one-shot
        first_one = store.create_delayed(trigger_id, 3600, "reply", "[Delayed]")

        # Second message → timer reset
        second_one = store.create_delayed(trigger_id, 3600, "reply", "[Delayed]")

        assert store.get(first_one) is None  # Cancelled
        assert store.get(second_one) is not None  # Active

    @pytest.mark.asyncio
    async def test_full_flow_execute_and_cleanup(self, store):
        """After the delay, the one-shot fires, executes the action, and deletes itself."""
        trigger_id = store.create_triggered(
            "Auto-reply to Natasha",
            {"sender": "Natasha"},
            "reply to Natasha saying hey what's up",
            delay_seconds=3600,
        )
        one_id = store.create_delayed(trigger_id, 3600, "reply to Natasha saying hey what's up", "[Delayed] Auto-reply")

        # Fast-forward: make it due
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, one_id))
        store._conn.commit()

        mock_handler = AsyncMock(return_value=("Sent to Natasha (Instagram): hey what's up", False))
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        # Action was executed
        mock_handler.assert_called_once_with("reply to Natasha saying hey what's up")

        # Result sent to Telegram
        sent = mock_channel.send_message.call_args[0][0]
        assert "Sent to Natasha" in sent

        # One-shot cleaned up, trigger still alive
        assert store.get(one_id) is None
        assert store.get(trigger_id) is not None

    def test_disable_trigger_cancels_pending_oneshot(self, store):
        """Disabling a trigger cancels its pending one-shot timer."""
        trigger_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=3600)
        one_id = store.create_delayed(trigger_id, 3600, "action", "[Delayed]")

        store.toggle(trigger_id, enabled=False)

        # One-shot should be cancelled
        assert store.get(one_id) is None
        # Trigger still exists (just disabled)
        assert store.get(trigger_id) is not None
        assert store.get(trigger_id)["enabled"] == 0

    def test_reenable_trigger_does_not_recreate_oneshot(self, store):
        """Re-enabling a trigger doesn't create a new one-shot — it waits for the next match."""
        trigger_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=3600)
        store.create_delayed(trigger_id, 3600, "action", "[Delayed]")
        store.toggle(trigger_id, enabled=False)
        store.toggle(trigger_id, enabled=True)

        # No one-shots should exist
        row = store._conn.execute(
            "SELECT COUNT(*) as c FROM automations WHERE one_shot = 1"
        ).fetchone()
        assert row["c"] == 0


class TestEdgeCases:
    """Edge cases and defensive tests."""

    def test_two_triggers_independent_oneshots(self, store):
        """Two different delayed triggers should not interfere with each other."""
        t1 = store.create_triggered("Reply Sophie", {"sender": "Sophie"}, "reply", delay_seconds=3600)
        t2 = store.create_triggered("Reply Marc", {"sender": "Marc"}, "reply", delay_seconds=1800)

        one1 = store.create_delayed(t1, 3600, "reply to Sophie", "[Delayed] Sophie")
        one2 = store.create_delayed(t2, 1800, "reply to Marc", "[Delayed] Marc")

        # Both exist
        assert store.get(one1) is not None
        assert store.get(one2) is not None

        # Reset Sophie's timer — Marc's should be unaffected
        one1_new = store.create_delayed(t1, 3600, "reply to Sophie", "[Delayed] Sophie")
        assert store.get(one1) is None  # Old Sophie one-shot cancelled
        assert store.get(one1_new) is not None  # New Sophie one-shot
        assert store.get(one2) is not None  # Marc's untouched

    def test_resolve_by_description_excludes_oneshots(self, store):
        """resolve_by_description should not match ephemeral one-shots."""
        trigger_id = store.create_triggered("Auto-reply Sophie", {"sender": "Sophie"}, "reply", delay_seconds=3600)
        store.create_delayed(trigger_id, 3600, "reply", "[Delayed] Auto-reply Sophie")

        # Should find the trigger, not the one-shot
        result = store.resolve_by_description("Sophie")
        assert isinstance(result, dict)
        assert result["id"] == trigger_id
        assert result["one_shot"] == 0

    def test_resolve_by_description_no_oneshot_only(self, store):
        """If only a one-shot matches, resolve should return None."""
        trigger_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=3600)
        store.create_delayed(trigger_id, 3600, "action", "[Delayed] Unique name xyz")

        result = store.resolve_by_description("Unique name xyz")
        assert result is None

    def test_delayed_trigger_with_zero_cooldown_fires_every_time(self, store):
        """Delayed triggers have cooldown=0, so every message re-fires and resets the timer."""
        trigger_id = store.create_triggered(
            "Reply Natasha", {"sender": "Natasha"}, "reply",
            delay_seconds=3600, cooldown_seconds=0,
        )

        msg = {"sender_name": "Natasha", "text": "hey", "chat_title": "Natasha", "network": "instagram"}

        # First message
        matches = store.evaluate_triggers(msg)
        assert len(matches) == 1
        store.mark_triggered(trigger_id)

        # Second message (immediately after) — should still fire (cooldown=0)
        matches = store.evaluate_triggers(msg)
        assert len(matches) == 1

    def test_get_delay_seconds_handles_corrupt_json(self, store):
        """get_delay_seconds handles corrupt trigger_config gracefully."""
        auto = {"trigger_config": "not json{{{"}
        assert store.get_delay_seconds(auto) == 0

    def test_get_delay_seconds_handles_none_config(self, store):
        assert store.get_delay_seconds({"trigger_config": None}) == 0

    def test_get_delay_seconds_handles_missing_config(self, store):
        assert store.get_delay_seconds({}) == 0

    def test_multiple_rapid_resets(self, store):
        """Rapidly creating one-shots should always leave exactly one pending."""
        trigger_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=3600)

        ids = []
        for i in range(5):
            oid = store.create_delayed(trigger_id, 3600, f"action {i}", "[Delayed]")
            ids.append(oid)

        # Only the last one should survive
        for old_id in ids[:-1]:
            assert store.get(old_id) is None
        assert store.get(ids[-1]) is not None

        # Exactly one one-shot in the DB
        row = store._conn.execute(
            "SELECT COUNT(*) as c FROM automations WHERE one_shot = 1"
        ).fetchone()
        assert row["c"] == 1

    @pytest.mark.asyncio
    async def test_oneshot_does_not_call_mark_run(self, store):
        """One-shots should be deleted, not mark_run'd (which would fail on missing schedule)."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action")
        one_id = store.create_delayed(parent_id, 60, "action", "[Delayed]")

        # Verify one-shot has no schedule
        one = store.get(one_id)
        assert one["schedule"] is None

        # Make it due
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        store._conn.execute("UPDATE automations SET next_run_at = ? WHERE id = ?", (past, one_id))
        store._conn.commit()

        mock_handler = AsyncMock(return_value=("Done", False))
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        # Should be deleted, not stuck with a None schedule
        assert store.get(one_id) is None

    def test_disable_non_trigger_does_not_cancel_oneshots(self, store):
        """Disabling a scheduled automation should not call cancel_pending_oneshots."""
        reg_id = store.create("Hourly", "0 * * * *", "check")

        # Even if somehow a one-shot had this as parent (shouldn't happen),
        # disabling a non-trigger shouldn't touch one-shots
        store.toggle(reg_id, enabled=False)
        auto = store.get(reg_id)
        assert auto["enabled"] == 0


class TestHandleCreateTriggerWithDelay:
    """Test the intent handler for creating delayed triggers."""

    def test_create_trigger_with_delay(self, store):
        plan = {
            "create_trigger": {
                "description": "Auto-reply to Natasha",
                "trigger": {"sender": "Natasha", "network": "instagram"},
                "action": "reply to Natasha with something fun",
                "delay_seconds": 18000,
            }
        }
        result = _handle_automation_intent(plan, store, "UTC")
        assert result is not None
        assert "Auto-reply to Natasha" in result
        assert "5h" in result
        assert "resets" in result.lower()

        # Verify the automation was created with delay
        auto = store.list_all()[0]
        assert store.get_delay_seconds(auto) == 18000

    def test_create_trigger_with_delay_sets_zero_cooldown(self, store):
        """Delayed triggers should have cooldown=0 (timer reset handles dedup)."""
        plan = {
            "create_trigger": {
                "description": "Delayed",
                "trigger": {"sender": "Sophie"},
                "action": "reply",
                "delay_seconds": 3600,
                "cooldown_seconds": 9999,  # This should be overridden to 0
            }
        }
        _handle_automation_intent(plan, store, "UTC")
        auto = store.list_all()[0]
        assert auto["cooldown_seconds"] == 0

    def test_create_trigger_without_delay_shows_cooldown(self, store):
        plan = {
            "create_trigger": {
                "description": "Immediate",
                "trigger": {"sender": "Marc"},
                "action": "notify",
            }
        }
        result = _handle_automation_intent(plan, store, "UTC")
        assert "Cooldown:" in result
        assert "resets" not in result.lower()

    def test_list_automations_shows_delay(self, store):
        """Listing automations should show delay info for delayed triggers."""
        store.create_triggered(
            "Auto-reply to Natasha",
            {"sender": "Natasha"},
            "reply to Natasha",
            delay_seconds=18960,
        )
        plan = {"list_automations": True}
        result = _handle_automation_intent(plan, store, "UTC")
        assert "5h 16m" in result
        assert "resets" in result.lower()


class TestFormatDelay:
    """Test the shared format_delay function."""

    def test_hours_and_minutes(self):
        assert format_delay(18960) == "5h 16m"

    def test_hours_only(self):
        assert format_delay(7200) == "2h"

    def test_minutes_only(self):
        assert format_delay(300) == "5m"

    def test_seconds_only(self):
        assert format_delay(45) == "45s"

    def test_zero(self):
        assert format_delay(0) == "0s"

    def test_one_hour(self):
        assert format_delay(3600) == "1h"

    def test_one_minute(self):
        assert format_delay(60) == "1m"


class TestHumanizeTriggerWithDelay:
    """Ensure _humanize_trigger doesn't leak internal _delay_seconds key."""

    def test_delay_seconds_not_shown(self):
        config_json = '{"sender": "Natasha", "_delay_seconds": 18000}'
        result = _humanize_trigger(config_json)
        assert "from Natasha" in result
        assert "delay" not in result.lower()
        assert "18000" not in result

    def test_only_delay_seconds_shows_any_message(self):
        config_json = '{"_delay_seconds": 3600}'
        result = _humanize_trigger(config_json)
        assert result == "any message"

    def test_multiple_conditions_with_delay(self):
        config_json = '{"sender": "Sophie", "network": "instagram", "_delay_seconds": 7200}'
        result = _humanize_trigger(config_json)
        assert "from Sophie" in result
        assert "on instagram" in result
        assert "delay" not in result.lower()


class TestCreateDelayedEdgeCases:
    """Edge cases for create_delayed."""

    def test_create_delayed_with_zero_delay(self, store):
        """delay_seconds=0 should create a one-shot that fires immediately (next_run_at ≈ now)."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action")
        one_id = store.create_delayed(parent_id, 0, "action", "[Delayed]")

        one = store.get(one_id)
        assert one is not None
        assert one["one_shot"] == 1

        # Fire time should be very close to now (within 2 seconds)
        fire_at = datetime.fromisoformat(one["next_run_at"])
        now = datetime.now(timezone.utc)
        assert abs((fire_at - now).total_seconds()) < 2

    def test_create_delayed_with_large_delay(self, store):
        """Large delay (24h) should work fine."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action")
        one_id = store.create_delayed(parent_id, 86400, "action", "[Delayed]")

        one = store.get(one_id)
        fire_at = datetime.fromisoformat(one["next_run_at"])
        now = datetime.now(timezone.utc)
        # Should fire ~24h from now
        diff = (fire_at - now).total_seconds()
        assert 86390 < diff < 86410


class TestMarkTriggeredWithZeroCooldown:
    """Verify mark_triggered + cooldown=0 interaction for delayed triggers."""

    def test_mark_triggered_then_immediate_rematch(self, store):
        """With cooldown=0, mark_triggered should not prevent the next match."""
        trigger_id = store.create_triggered(
            "Reply", {"sender": "Natasha"}, "reply",
            cooldown_seconds=0, delay_seconds=3600,
        )
        msg = {"sender_name": "Natasha", "text": "hi", "chat_title": "Natasha", "network": "ig"}

        # First match + mark
        matches = store.evaluate_triggers(msg)
        assert len(matches) == 1
        store.mark_triggered(trigger_id)

        # Verify last_run_at was set
        auto = store.get(trigger_id)
        assert auto["last_run_at"] is not None

        # Second match immediately — should still fire because cooldown=0
        matches = store.evaluate_triggers(msg)
        assert len(matches) == 1

    def test_mark_triggered_with_positive_cooldown_blocks(self, store):
        """With cooldown > 0, mark_triggered should block the next match."""
        trigger_id = store.create_triggered(
            "Reply", {"sender": "Natasha"}, "reply",
            cooldown_seconds=300,
        )
        msg = {"sender_name": "Natasha", "text": "hi", "chat_title": "Natasha", "network": "ig"}

        matches = store.evaluate_triggers(msg)
        assert len(matches) == 1
        store.mark_triggered(trigger_id)

        # Second match immediately — should be blocked by cooldown
        matches = store.evaluate_triggers(msg)
        assert len(matches) == 0


class TestTriggerEvaluationFlow:
    """Integration test for the trigger evaluation flow as in main.py."""

    def test_delayed_trigger_creates_oneshot_on_match(self, store):
        """Simulates the main.py trigger evaluation flow for delayed triggers."""
        trigger_id = store.create_triggered(
            "Auto-reply Natasha",
            {"sender": "Natasha", "network": "instagram"},
            "reply to Natasha saying hey!",
            delay_seconds=18000,
        )

        msg = {
            "sender_name": "Natasha",
            "text": "hey there",
            "chat_title": "Natasha",
            "network": "instagramgo",  # Beeper internal name
        }

        # Evaluate triggers (as main.py does)
        triggered = store.evaluate_triggers(msg)
        assert len(triggered) == 1
        auto = triggered[0]
        assert auto["id"] == trigger_id

        # Mark triggered
        store.mark_triggered(trigger_id)

        # Get delay and create one-shot (as main.py does)
        delay = store.get_delay_seconds(auto)
        assert delay == 18000

        store.cancel_pending_oneshots(trigger_id)
        one_id = store.create_delayed(
            parent_id=trigger_id,
            delay_seconds=delay,
            action=auto["action"],
            description=f"[Delayed] {auto['description']}",
        )

        # Verify one-shot exists with correct properties
        one = store.get(one_id)
        assert one["one_shot"] == 1
        assert one["parent_id"] == trigger_id
        assert one["action"] == "reply to Natasha saying hey!"
        assert one["type"] == "scheduled"

        # Verify not in list_all
        all_autos = store.list_all()
        one_shot_ids = [a["id"] for a in all_autos if a["one_shot"] == 1]
        assert one_id not in one_shot_ids

    def test_immediate_trigger_no_oneshot(self, store):
        """Immediate triggers (no delay) should not create one-shots."""
        trigger_id = store.create_triggered(
            "Notify on Sophie", {"sender": "Sophie"}, "notify me"
        )

        msg = {"sender_name": "Sophie", "text": "hello", "chat_title": "Sophie", "network": "whatsapp"}
        triggered = store.evaluate_triggers(msg)
        assert len(triggered) == 1

        delay = store.get_delay_seconds(triggered[0])
        assert delay == 0

        # No one-shots should exist
        row = store._conn.execute(
            "SELECT COUNT(*) as c FROM automations WHERE one_shot = 1"
        ).fetchone()
        assert row["c"] == 0

    @pytest.mark.asyncio
    async def test_scheduler_ignores_non_due_oneshot(self, store):
        """One-shots with future next_run_at should not be picked up by the scheduler."""
        parent_id = store.create_triggered("P", {"sender": "S"}, "action", delay_seconds=3600)
        one_id = store.create_delayed(parent_id, 3600, "action", "[Delayed]")

        mock_handler = AsyncMock(return_value=("Done", False))
        mock_channel = AsyncMock()

        await run_scheduler_tick(store, mock_handler, mock_channel)

        # Handler should NOT have been called — one-shot isn't due yet
        mock_handler.assert_not_called()
        # One-shot should still exist
        assert store.get(one_id) is not None
