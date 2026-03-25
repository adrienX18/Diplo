"""Tests for control channel base class and Telegram adapter."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from src.channels.base import ControlChannel
from src.channels.telegram import TelegramChannel, _escape_md, _split_message


class TestControlChannelInterface:
    """Verify the abstract base class enforces the interface."""

    def test_cannot_instantiate_base_class(self):
        with pytest.raises(TypeError):
            ControlChannel()

    def test_subclass_must_implement_all_methods(self):
        class IncompleteChannel(ControlChannel):
            async def send_notification(self, title, body):
                pass

        with pytest.raises(TypeError):
            IncompleteChannel()

    def test_complete_subclass_can_be_instantiated(self):
        class CompleteChannel(ControlChannel):
            async def send_notification(self, title, body): pass
            async def send_message(self, text): pass
            async def start(self, on_user_message, on_reply_sent=None): pass
            async def stop(self): pass

        channel = CompleteChannel()
        assert isinstance(channel, ControlChannel)


class TestEscapeMarkdown:
    def test_escapes_special_characters(self):
        assert _escape_md("hello *world*") == "hello \\*world\\*"
        assert _escape_md("foo_bar") == "foo\\_bar"
        assert _escape_md("test (1)") == "test \\(1\\)"

    def test_no_change_for_plain_text(self):
        assert _escape_md("hello world") == "hello world"


def _make_channel():
    """Create a TelegramChannel with a mocked bot for typing indicator."""
    channel = TelegramChannel()
    channel._app = MagicMock()
    channel._app.bot.send_chat_action = AsyncMock()
    return channel


class TestTelegramChannel:
    def test_rejects_unauthorized_chat(self):
        """Messages from unauthorized chat IDs should be ignored."""
        channel = _make_channel()
        channel._on_user_message = AsyncMock(return_value="response")

        update = MagicMock()
        update.message.text = "hello"
        update.effective_chat.id = 999999  # wrong chat ID

        import asyncio
        asyncio.run(channel._handle_message(update, MagicMock()))

        channel._on_user_message.assert_not_called()

    def test_processes_authorized_message(self):
        """Messages from the authorized chat ID should be processed."""
        channel = _make_channel()
        channel._on_user_message = AsyncMock(return_value=("here's your summary", True))

        update = MagicMock()
        update.message.text = "what did I miss?"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545  # authorized chat ID

        import asyncio
        asyncio.run(channel._handle_message(update, MagicMock()))

        # on_chunk kwarg is now passed for streaming support
        channel._on_user_message.assert_called_once()
        args, kwargs = channel._on_user_message.call_args
        assert args == ("what did I miss?",)
        assert "on_chunk" in kwargs
        update.message.reply_text.assert_called_once_with("here's your summary")

    def test_handles_error_gracefully(self):
        """If the callback raises, the user gets an error message."""
        channel = _make_channel()
        channel._on_user_message = AsyncMock(side_effect=RuntimeError("boom"))

        update = MagicMock()
        update.message.text = "hello"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        import asyncio
        asyncio.run(channel._handle_message(update, MagicMock()))

        update.message.reply_text.assert_called_once_with("Something went wrong. Try again in a moment.")

    def test_on_reply_sent_called_after_reply(self):
        """last_seen_at hook must fire AFTER the reply is sent, not before."""
        channel = _make_channel()
        call_order = []

        async def fake_reply(text):
            call_order.append("reply_sent")

        async def fake_on_user_message(text, on_chunk=None):
            call_order.append("response_generated")
            return "here's your summary", True

        async def fake_on_reply_sent():
            call_order.append("last_seen_updated")

        channel._on_user_message = fake_on_user_message
        channel._on_reply_sent = fake_on_reply_sent

        update = MagicMock()
        update.message.text = "what's new?"
        update.message.reply_text = fake_reply
        update.effective_chat.id = 5100237545

        import asyncio
        asyncio.run(channel._handle_message(update, MagicMock()))

        assert call_order == ["response_generated", "reply_sent", "last_seen_updated"]

    def test_on_reply_sent_not_called_for_casual_chat(self):
        """Casual greetings (no_query) should NOT advance last_seen_at."""
        channel = _make_channel()

        async def fake_on_user_message(text, on_chunk=None):
            return "Hey! What's up?", False  # no_query = True -> queried_cache = False

        channel._on_user_message = fake_on_user_message
        channel._on_reply_sent = AsyncMock()

        update = MagicMock()
        update.message.text = "hey!"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        import asyncio
        asyncio.run(channel._handle_message(update, MagicMock()))

        channel._on_reply_sent.assert_not_called()

    def test_on_reply_sent_not_called_on_error(self):
        """If the response fails, last_seen_at should NOT be updated."""
        channel = _make_channel()
        channel._on_user_message = AsyncMock(side_effect=RuntimeError("boom"))
        channel._on_reply_sent = AsyncMock()
        # Note: error happens before unpacking, so on_reply_sent is never reached

        update = MagicMock()
        update.message.text = "hello"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        import asyncio
        asyncio.run(channel._handle_message(update, MagicMock()))

        channel._on_reply_sent.assert_not_called()

    def test_ignores_empty_messages(self):
        """Messages with no text should be ignored."""
        channel = _make_channel()
        channel._on_user_message = AsyncMock()

        update = MagicMock()
        update.message = None

        import asyncio
        asyncio.run(channel._handle_message(update, MagicMock()))

        channel._on_user_message.assert_not_called()

    def test_long_response_split_into_chunks(self):
        """Responses over 500 chars should be sent as multiple messages."""
        channel = _make_channel()
        long_response = "First paragraph here.\n\n" + "Second paragraph. " * 30 + "\n\nThird paragraph."
        channel._on_user_message = AsyncMock(return_value=(long_response, True))

        update = MagicMock()
        update.message.text = "summarize everything"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        import asyncio
        asyncio.run(channel._handle_message(update, MagicMock()))

        assert update.message.reply_text.call_count > 1

    def test_typing_indicator_sent(self):
        """Bot should send typing action while processing."""
        channel = _make_channel()

        import asyncio

        async def slow_response(text, on_chunk=None):
            await asyncio.sleep(0.05)  # Give typing task a chance to fire
            return "quick reply", True

        channel._on_user_message = slow_response

        update = MagicMock()
        update.message.text = "hey"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        asyncio.run(channel._handle_message(update, MagicMock()))

        channel._app.bot.send_chat_action.assert_called()


class TestSplitMessage:
    def test_short_message_not_split(self):
        assert _split_message("hello") == ["hello"]

    def test_splits_at_paragraph_boundary(self):
        text = ("A" * 300) + "\n\n" + ("B" * 300)
        chunks = _split_message(text)
        assert len(chunks) == 2
        assert chunks[0] == "A" * 300
        assert chunks[1] == "B" * 300

    def test_keeps_short_paragraphs_together(self):
        text = "Short one.\n\nShort two.\n\nShort three."
        chunks = _split_message(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_splits(self):
        text = "\n\n".join(["X" * 400 for _ in range(3)])
        chunks = _split_message(text)
        assert len(chunks) == 3

    def test_exactly_at_limit(self):
        text = "A" * 500
        chunks = _split_message(text)
        assert len(chunks) == 1


def _make_voice_update(duration=5, mime_type=None, is_audio_file=False):
    """Create a mock Telegram Update with a voice or audio message."""
    update = MagicMock()
    update.effective_chat.id = 5100237545  # authorized chat ID
    update.message.reply_text = AsyncMock()

    tg_file = MagicMock()
    tg_file.download_to_drive = AsyncMock()

    if is_audio_file:
        update.message.voice = None
        update.message.audio = MagicMock()
        update.message.audio.duration = duration
        update.message.audio.mime_type = mime_type or "audio/mp3"
        update.message.audio.get_file = AsyncMock(return_value=tg_file)
    else:
        update.message.voice = MagicMock()
        update.message.voice.duration = duration
        update.message.voice.get_file = AsyncMock(return_value=tg_file)
        update.message.audio = None

    return update


class TestVoiceMessages:
    def test_rejects_unauthorized_voice(self):
        """Voice messages from unauthorized chat IDs should be ignored."""
        channel = _make_channel()
        channel._on_user_message = AsyncMock()

        update = _make_voice_update()
        update.effective_chat.id = 999999  # wrong chat ID

        asyncio.run(channel._handle_voice(update, MagicMock()))
        channel._on_user_message.assert_not_called()

    @patch("src.channels.telegram.transcribe_audio", new_callable=AsyncMock)
    def test_voice_message_transcribed_and_prefixed(self, mock_transcribe):
        """Voice messages should be transcribed and passed with [voice message] prefix."""
        mock_transcribe.return_value = "check my messages"

        channel = _make_channel()
        channel._on_user_message = AsyncMock(return_value=("Here's what's new", True))

        update = _make_voice_update()
        asyncio.run(channel._handle_voice(update, MagicMock()))

        channel._on_user_message.assert_called_once()
        args, kwargs = channel._on_user_message.call_args
        assert args == ("[voice message] check my messages",)
        assert "on_chunk" in kwargs
        update.message.reply_text.assert_called_once_with("Here's what's new")

    @patch("src.channels.telegram.transcribe_audio", new_callable=AsyncMock)
    def test_voice_on_reply_sent_called(self, mock_transcribe):
        """Voice messages that query the cache should advance last_seen_at."""
        mock_transcribe.return_value = "what's new"

        channel = _make_channel()
        channel._on_user_message = AsyncMock(return_value=("Nothing new", True))
        channel._on_reply_sent = AsyncMock()

        update = _make_voice_update()
        asyncio.run(channel._handle_voice(update, MagicMock()))

        channel._on_reply_sent.assert_called_once()

    @patch("src.channels.telegram.transcribe_audio", new_callable=AsyncMock)
    def test_voice_on_reply_sent_not_called_for_casual(self, mock_transcribe):
        """Voice greetings should not advance last_seen_at."""
        mock_transcribe.return_value = "hey!"

        channel = _make_channel()
        channel._on_user_message = AsyncMock(return_value=("Hey boss!", False))
        channel._on_reply_sent = AsyncMock()

        update = _make_voice_update()
        asyncio.run(channel._handle_voice(update, MagicMock()))

        channel._on_reply_sent.assert_not_called()

    @patch("src.channels.telegram.transcribe_audio", new_callable=AsyncMock)
    def test_empty_transcript_prompts_retry(self, mock_transcribe):
        """Empty transcription should ask the user to try again."""
        mock_transcribe.return_value = "   "

        channel = _make_channel()
        channel._on_user_message = AsyncMock()

        update = _make_voice_update()
        asyncio.run(channel._handle_voice(update, MagicMock()))

        channel._on_user_message.assert_not_called()
        update.message.reply_text.assert_called_once()
        assert "couldn't make out" in update.message.reply_text.call_args[0][0]

    @patch("src.channels.telegram.transcribe_audio", new_callable=AsyncMock)
    def test_transcription_failure_handled_gracefully(self, mock_transcribe):
        """If transcription fails, the user gets a clear error message."""
        mock_transcribe.side_effect = RuntimeError("No OpenAI API key")

        channel = _make_channel()
        channel._on_user_message = AsyncMock()

        update = _make_voice_update()
        asyncio.run(channel._handle_voice(update, MagicMock()))

        channel._on_user_message.assert_not_called()
        assert "can't process voice" in update.message.reply_text.call_args[0][0]

    @patch("src.channels.telegram.transcribe_audio", new_callable=AsyncMock)
    def test_audio_file_handled(self, mock_transcribe):
        """Audio file attachments (not just voice notes) should also be transcribed."""
        mock_transcribe.return_value = "tell Sophie I'll be late"

        channel = _make_channel()
        channel._on_user_message = AsyncMock(return_value=("Sent!", False))

        update = _make_voice_update(is_audio_file=True, mime_type="audio/mp3")
        asyncio.run(channel._handle_voice(update, MagicMock()))

        channel._on_user_message.assert_called_once()
        args, kwargs = channel._on_user_message.call_args
        assert args == ("[voice message] tell Sophie I'll be late",)
        assert "on_chunk" in kwargs


class TestStreamingChunks:
    """Tests for progressive streaming of response chunks."""

    def test_streaming_sends_chunks_progressively(self):
        """When on_chunk is called by the assistant, chunks are sent immediately."""
        channel = _make_channel()
        sent_chunks = []

        async def fake_on_user_message(text, on_chunk=None):
            # Simulate the assistant calling on_chunk progressively
            if on_chunk:
                await on_chunk("First paragraph here.")
                await on_chunk("Second paragraph with more detail.")
            return "First paragraph here.\n\nSecond paragraph with more detail.", True

        channel._on_user_message = fake_on_user_message
        channel._on_reply_sent = AsyncMock()

        update = MagicMock()
        update.message.text = "what's new?"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        asyncio.run(channel._handle_message(update, MagicMock()))

        # Two chunks sent via on_chunk — no additional calls from split fallback
        assert update.message.reply_text.call_count == 2
        assert update.message.reply_text.call_args_list[0][0][0] == "First paragraph here."
        assert update.message.reply_text.call_args_list[1][0][0] == "Second paragraph with more detail."
        channel._on_reply_sent.assert_called_once()

    def test_fallback_to_split_when_no_chunks_sent(self):
        """Short responses (no streaming) fall back to normal split-and-send."""
        channel = _make_channel()

        async def fake_on_user_message(text, on_chunk=None):
            # on_chunk is available but not called (short response)
            return "Quick answer.", False

        channel._on_user_message = fake_on_user_message

        update = MagicMock()
        update.message.text = "hey"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        asyncio.run(channel._handle_message(update, MagicMock()))

        update.message.reply_text.assert_called_once_with("Quick answer.")

    def test_typing_indicator_restarts_between_chunks(self):
        """Typing indicator should restart after each streamed chunk.

        We verify that multiple typing tasks are created — one initially,
        then a new one after each chunk is sent (cancel + recreate).
        """
        channel = _make_channel()

        async def fake_on_user_message(text, on_chunk=None):
            if on_chunk:
                await on_chunk("Chunk one.")
                # After on_chunk, send_chunk cancels the old typing task
                # and starts a new one — so _keep_typing is called 3 times:
                # initial + restart after chunk 1 + restart after chunk 2
                await on_chunk("Chunk two.")
            return "Chunk one.\n\nChunk two.", True

        channel._on_user_message = fake_on_user_message
        channel._on_reply_sent = AsyncMock()

        update = MagicMock()
        update.message.text = "summarize"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        asyncio.run(channel._handle_message(update, MagicMock()))

        # Two chunks sent
        assert update.message.reply_text.call_count == 2

    def test_on_reply_sent_not_called_when_not_queried_with_streaming(self):
        """Even with streaming, on_reply_sent should respect queried_cache=False."""
        channel = _make_channel()

        async def fake_on_user_message(text, on_chunk=None):
            if on_chunk:
                await on_chunk("Hey boss!")
            return "Hey boss!", False

        channel._on_user_message = fake_on_user_message
        channel._on_reply_sent = AsyncMock()

        update = MagicMock()
        update.message.text = "hey"
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 5100237545

        asyncio.run(channel._handle_message(update, MagicMock()))

        channel._on_reply_sent.assert_not_called()
