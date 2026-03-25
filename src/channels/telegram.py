"""Telegram control channel adapter."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Callable, Awaitable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from src.channels.base import ControlChannel
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.llm import transcribe_audio, describe_image

logger = logging.getLogger(__name__)


class TelegramChannel(ControlChannel):
    """Telegram bot adapter for the control channel."""

    def __init__(self):
        self._app: Application | None = None
        self._on_user_message: Callable[[str], Awaitable[str]] | None = None
        self._on_reply_sent: Callable[[], Awaitable[None]] | None = None

    async def send_notification(self, title: str, body: str) -> None:
        """Push an urgent notification to the user."""
        text = f"*{_escape_md(title)}*\n\n{_escape_md(body)}"
        await self._app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode="MarkdownV2",
        )

    async def send_message(self, text: str) -> None:
        """Send a plain message to the user."""
        await self._app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
        )

    async def start(
        self,
        on_user_message: Callable[[str], Awaitable[str]],
        on_reply_sent: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Start the Telegram bot polling loop."""
        self._on_user_message = on_user_message
        self._on_reply_sent = on_reply_sent
        self._app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        self._app.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice)
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO, self._handle_photo)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Shut down the Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an incoming text message from Telegram."""
        if not update.message or not update.message.text:
            return

        # Only respond to the authorized user
        if update.effective_chat.id != TELEGRAM_CHAT_ID:
            logger.warning("Ignoring message from unauthorized chat_id=%s", update.effective_chat.id)
            return

        user_text = update.message.text
        logger.info("Received from user: %s", user_text[:100])

        await self._process_and_reply(update, user_text)

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle a voice message or audio file from Telegram."""
        if not update.message:
            return

        if update.effective_chat.id != TELEGRAM_CHAT_ID:
            logger.warning("Ignoring voice from unauthorized chat_id=%s", update.effective_chat.id)
            return

        # Get the file object — voice notes use .voice, audio files use .audio
        voice = update.message.voice or update.message.audio
        if not voice:
            return

        logger.info("Received voice message from user (%ds)", voice.duration or 0)

        # Show typing while we download + transcribe
        typing_task = asyncio.create_task(
            self._keep_typing(update.effective_chat.id)
        )

        try:
            # Download to a temp file
            tg_file = await voice.get_file()
            suffix = ".ogg"  # Telegram voice notes are ogg/opus
            if update.message.audio and update.message.audio.mime_type:
                mime = update.message.audio.mime_type
                ext_map = {"audio/mp3": ".mp3", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
                           "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/webm": ".webm",
                           "audio/ogg": ".ogg", "audio/m4a": ".m4a"}
                suffix = ext_map.get(mime, ".ogg")

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)

            await tg_file.download_to_drive(str(tmp_path))

            # Transcribe
            transcript = await transcribe_audio(tmp_path)
            logger.info("Transcribed voice message: %s", transcript[:100])

            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

            if not transcript.strip():
                typing_task.cancel()
                await update.message.reply_text("I couldn't make out anything in that voice message. Try again?")
                return

            # Pass to the assistant with annotation
            user_text = f"[voice message] {transcript}"
            typing_task.cancel()
            await self._process_and_reply(update, user_text)

        except RuntimeError as e:
            typing_task.cancel()
            # No OpenAI key or transcription failed
            logger.warning("Voice transcription error: %s", e)
            await update.message.reply_text("I can't process voice messages right now. Type it out for me?")
            # Clean up temp file on error
            if 'tmp_path' in locals():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            typing_task.cancel()
            logger.exception("Error handling voice message")
            await update.message.reply_text("Something went wrong with that voice message. Try again in a moment.")
            if 'tmp_path' in locals():
                tmp_path.unlink(missing_ok=True)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle a photo message from Telegram."""
        if not update.message or not update.message.photo:
            return

        if update.effective_chat.id != TELEGRAM_CHAT_ID:
            logger.warning("Ignoring photo from unauthorized chat_id=%s", update.effective_chat.id)
            return

        logger.info("Received photo from user")

        typing_task = asyncio.create_task(
            self._keep_typing(update.effective_chat.id)
        )

        try:
            # Get the highest-resolution version (last in the list)
            photo = update.message.photo[-1]
            tg_file = await photo.get_file()

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            await tg_file.download_to_drive(str(tmp_path))

            # Describe via Claude vision
            try:
                description = await describe_image(tmp_path)
                image_tag = f"[image: {description}]"
            except Exception:
                logger.warning("Failed to describe photo from Telegram", exc_info=True)
                image_tag = "[image]"

            tmp_path.unlink(missing_ok=True)

            # Combine with caption (if any), same format as Beeper path
            caption = update.message.caption or ""
            user_text = f"{image_tag} {caption}".strip() if caption else image_tag

            typing_task.cancel()
            await self._process_and_reply(update, user_text)

        except Exception:
            typing_task.cancel()
            logger.exception("Error handling photo message")
            await update.message.reply_text("Something went wrong with that photo. Try again in a moment.")
            if 'tmp_path' in locals():
                tmp_path.unlink(missing_ok=True)

    async def _process_and_reply(self, update: Update, user_text: str) -> None:
        """Shared logic: send text through the assistant pipeline and reply.

        Uses streaming: chunks are sent as they become ready during generation,
        with a typing indicator shown between chunks.
        """
        chat_id = update.effective_chat.id
        typing_task = asyncio.create_task(self._keep_typing(chat_id))
        chunks_sent = 0

        async def send_chunk(chunk: str) -> None:
            """Send a chunk and restart the typing indicator."""
            nonlocal typing_task, chunks_sent
            typing_task.cancel()
            await update.message.reply_text(chunk)
            chunks_sent += 1
            # Restart typing for the next chunk
            typing_task = asyncio.create_task(self._keep_typing(chat_id))

        try:
            response, queried_cache = await self._on_user_message(
                user_text, on_chunk=send_chunk,
            )
            typing_task.cancel()

            # If no chunks were sent (non-streaming path, e.g. short replies),
            # fall back to the normal split-and-send behavior.
            if chunks_sent == 0:
                for chunk in _split_message(response):
                    await update.message.reply_text(chunk)

            if self._on_reply_sent and queried_cache:
                await self._on_reply_sent()
        except Exception:
            typing_task.cancel()
            logger.exception("Error handling user message")
            await update.message.reply_text("Something went wrong. Try again in a moment.")

    async def _keep_typing(self, chat_id: int) -> None:
        """Send typing action repeatedly until cancelled."""
        try:
            while True:
                await self._app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                await asyncio.sleep(4)  # Telegram typing indicator lasts ~5s
        except asyncio.CancelledError:
            pass


_MAX_CHUNK_CHARS = 500


def _split_message(text: str) -> list[str]:
    """Split a message into chunks of ~500 chars, breaking at paragraph boundaries."""
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > _MAX_CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = para
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


# MarkdownV2 requires escaping these characters
_MD_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"


def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    for char in _MD_ESCAPE_CHARS:
        text = text.replace(char, f"\\{char}")
    return text
