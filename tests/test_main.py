"""Tests for the main loop — control channel filtering and urgent notification batching."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.main import _is_control_channel, _buffer_urgent, _flush_urgent, _urgent_buffer, _urgent_timers, _startup_summary, _run_periodic_tasks


class TestIsControlChannel:
    """Messages from the bot's own Telegram chat must be skipped for triage
    to prevent a feedback loop: bot sends notification → Beeper picks it up →
    classified as urgent → bot sends another notification → infinite loop."""

    def test_bot_message_in_bot_chat(self):
        """Bot's own message in its own chat — must skip."""
        msg = {"sender_name": "Diplo", "chat_title": "Diplo"}
        assert _is_control_channel(msg) is True

    def test_user_message_in_bot_chat(self):
        """User's message in the bot chat — must also skip (same chat)."""
        msg = {"sender_name": "@alemercier:beeper.com", "chat_title": "Diplo"}
        assert _is_control_channel(msg) is True

    def test_normal_whatsapp_message(self):
        """Regular message from another chat — must NOT skip."""
        msg = {"sender_name": "Sophie", "chat_title": "Sophie"}
        assert _is_control_channel(msg) is False

    def test_group_chat(self):
        """Group chat message — must NOT skip."""
        msg = {"sender_name": "Bob", "chat_title": "Team Chat"}
        assert _is_control_channel(msg) is False

    def test_case_insensitive(self):
        """Match should be case-insensitive."""
        msg = {"sender_name": "POKEBEEPER", "chat_title": "POKEBEEPER"}
        assert _is_control_channel(msg) is True

    def test_message_mentioning_bot_in_text_not_filtered(self):
        """A message that mentions the bot by name in text but is from a different chat."""
        msg = {"sender_name": "Sophie", "chat_title": "Sophie", "text": "hey Diplo do something"}
        assert _is_control_channel(msg) is False


class TestStartupSummary:
    def test_groups_by_chat(self):
        msgs = [
            {"chat_title": "Sophie"},
            {"chat_title": "Sophie"},
            {"chat_title": "Team Chat"},
        ]
        result = _startup_summary(msgs)
        assert "Sophie (2)" in result
        assert "Team Chat (1)" in result

    def test_top_5_with_overflow(self):
        msgs = [{"chat_title": f"Chat {i}"} for i in range(8)]
        result = _startup_summary(msgs)
        assert "+3 more" in result

    def test_single_chat(self):
        msgs = [{"chat_title": "Sophie"}]
        result = _startup_summary(msgs)
        assert result == "Sophie (1)"


@pytest.fixture(autouse=True)
def _clear_urgent_state():
    """Reset global urgent buffer/timers between tests."""
    _urgent_buffer.clear()
    for task in _urgent_timers.values():
        task.cancel()
    _urgent_timers.clear()
    yield
    _urgent_buffer.clear()
    for task in _urgent_timers.values():
        task.cancel()
    _urgent_timers.clear()


class TestUrgentBatching:
    """Urgent notifications are buffered per chat and sent after a delay."""

    @pytest.mark.asyncio
    async def test_single_message_sent_after_delay(self):
        channel = MagicMock()
        channel.send_notification = AsyncMock()
        convo = MagicMock()
        convo.add_turn = MagicMock()

        _buffer_urgent("chat1", "Sophie in Chat", "sign this now", channel, convo)

        # Not sent yet
        channel.send_notification.assert_not_called()

        # Wait for flush (use short delay for test)
        with patch("src.main.URGENT_BATCH_DELAY_SECONDS", 0.05):
            # Re-buffer with patched delay
            _urgent_buffer.clear()
            for t in _urgent_timers.values():
                t.cancel()
            _urgent_timers.clear()
            _buffer_urgent("chat1", "Sophie in Chat", "sign this now", channel, convo)
            await asyncio.sleep(0.1)

        channel.send_notification.assert_called_once()
        assert "Sophie in Chat" in channel.send_notification.call_args[0][0]

    @pytest.mark.asyncio
    async def test_multiple_messages_batched(self):
        channel = MagicMock()
        channel.send_notification = AsyncMock()
        convo = MagicMock()
        convo.add_turn = MagicMock()

        with patch("src.main.URGENT_BATCH_DELAY_SECONDS", 0.1):
            _buffer_urgent("chat1", "Sophie in Chat", "msg 1", channel, convo)
            await asyncio.sleep(0.02)
            _buffer_urgent("chat1", "Sophie in Chat", "msg 2", channel, convo)
            await asyncio.sleep(0.02)
            _buffer_urgent("chat1", "Sophie in Chat", "msg 3", channel, convo)

            # Wait for flush
            await asyncio.sleep(0.15)

        # Should send ONE notification with all 3 messages
        channel.send_notification.assert_called_once()
        title, body = channel.send_notification.call_args[0]
        assert "3 messages" in title
        assert "msg 1" in body
        assert "msg 2" in body
        assert "msg 3" in body

    @pytest.mark.asyncio
    async def test_different_chats_sent_separately(self):
        channel = MagicMock()
        channel.send_notification = AsyncMock()
        convo = MagicMock()
        convo.add_turn = MagicMock()

        with patch("src.main.URGENT_BATCH_DELAY_SECONDS", 0.05):
            _buffer_urgent("chat1", "Sophie", "urgent 1", channel, convo)
            _buffer_urgent("chat2", "Bob", "urgent 2", channel, convo)

            await asyncio.sleep(0.1)

        assert channel.send_notification.call_count == 2

    @pytest.mark.asyncio
    async def test_timer_resets_on_new_message(self):
        channel = MagicMock()
        channel.send_notification = AsyncMock()
        convo = MagicMock()
        convo.add_turn = MagicMock()

        with patch("src.main.URGENT_BATCH_DELAY_SECONDS", 0.1):
            _buffer_urgent("chat1", "Sophie", "msg 1", channel, convo)
            await asyncio.sleep(0.07)
            # New message resets the timer
            _buffer_urgent("chat1", "Sophie", "msg 2", channel, convo)
            await asyncio.sleep(0.07)

            # Should NOT have sent yet (timer was reset)
            channel.send_notification.assert_not_called()

            # Now wait for the full delay
            await asyncio.sleep(0.05)

        channel.send_notification.assert_called_once()
        title, body = channel.send_notification.call_args[0]
        assert "2 messages" in title


class TestPeriodicTasks:
    """Periodic tasks (prune + consolidation) run independently of the poller."""

    @pytest.mark.asyncio
    async def test_runs_prune_and_consolidation(self):
        """Both prune and consolidation are called on each cycle."""
        cache = MagicMock()
        convo = MagicMock()
        mock_llm_log = MagicMock()

        with patch("src.main.PRUNE_INTERVAL_SECONDS", 0.01), \
             patch("src.main.run_consolidation", new_callable=AsyncMock) as mock_consolidation, \
             patch("src.main.get_llm_logger", return_value=mock_llm_log):
            task = asyncio.create_task(_run_periodic_tasks(cache, convo))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        cache.prune.assert_called()
        convo.prune.assert_called()
        mock_consolidation.assert_called()
        mock_llm_log.prune.assert_called()

    @pytest.mark.asyncio
    async def test_prune_failure_does_not_block_consolidation(self):
        """If prune fails, consolidation still runs."""
        cache = MagicMock()
        cache.prune.side_effect = RuntimeError("db locked")
        convo = MagicMock()

        with patch("src.main.PRUNE_INTERVAL_SECONDS", 0.01), \
             patch("src.main.run_consolidation", new_callable=AsyncMock) as mock_consolidation:
            task = asyncio.create_task(_run_periodic_tasks(cache, convo))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        mock_consolidation.assert_called()

    @pytest.mark.asyncio
    async def test_consolidation_failure_does_not_crash(self):
        """If consolidation fails, the task keeps running."""
        cache = MagicMock()
        convo = MagicMock()

        call_count = 0

        async def failing_consolidation():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("API down")

        with patch("src.main.PRUNE_INTERVAL_SECONDS", 0.01), \
             patch("src.main.run_consolidation", side_effect=failing_consolidation):
            task = asyncio.create_task(_run_periodic_tasks(cache, convo))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have been called multiple times (task didn't crash after first failure)
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_llm_log_prune_called(self):
        """LLM logger prune is called via get_llm_logger(), not a local variable."""
        cache = MagicMock()
        convo = MagicMock()
        mock_llm_log = MagicMock()

        with patch("src.main.PRUNE_INTERVAL_SECONDS", 0.01), \
             patch("src.main.run_consolidation", new_callable=AsyncMock), \
             patch("src.main.get_llm_logger", return_value=mock_llm_log):
            task = asyncio.create_task(_run_periodic_tasks(cache, convo))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        mock_llm_log.prune.assert_called()

    @pytest.mark.asyncio
    async def test_llm_log_prune_skipped_when_no_logger(self):
        """When LLM logger is not initialized, prune is skipped without error."""
        cache = MagicMock()
        convo = MagicMock()

        with patch("src.main.PRUNE_INTERVAL_SECONDS", 0.01), \
             patch("src.main.run_consolidation", new_callable=AsyncMock), \
             patch("src.main.get_llm_logger", return_value=None):
            task = asyncio.create_task(_run_periodic_tasks(cache, convo))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # No exception — task ran fine without a logger
        cache.prune.assert_called()

    @pytest.mark.asyncio
    async def test_llm_log_prune_failure_does_not_crash(self):
        """If LLM log prune fails, the periodic task keeps running."""
        cache = MagicMock()
        convo = MagicMock()
        mock_llm_log = MagicMock()
        mock_llm_log.prune.side_effect = RuntimeError("db error")

        prune_call_count = 0
        original_prune = mock_llm_log.prune.side_effect

        def counting_prune():
            nonlocal prune_call_count
            prune_call_count += 1
            raise RuntimeError("db error")

        mock_llm_log.prune.side_effect = counting_prune

        with patch("src.main.PRUNE_INTERVAL_SECONDS", 0.01), \
             patch("src.main.run_consolidation", new_callable=AsyncMock), \
             patch("src.main.get_llm_logger", return_value=mock_llm_log):
            task = asyncio.create_task(_run_periodic_tasks(cache, convo))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have been called multiple times (didn't crash after first failure)
        assert prune_call_count >= 2
