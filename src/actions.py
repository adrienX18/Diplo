"""Actions — sending messages through Beeper on the user's behalf."""

import asyncio
import logging

from beeper_desktop_api import BeeperDesktop

logger = logging.getLogger(__name__)


async def send_message(client: BeeperDesktop, chat_id: str, text: str) -> bool:
    """Send a message to a chat via the Beeper Desktop API.

    The Beeper SDK is synchronous, so this runs in an executor.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    if not text or not text.strip():
        logger.warning("Refusing to send empty/whitespace message to %s", chat_id)
        return False

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None, lambda: client.messages.send(chat_id, text=text)
        )
        logger.info("Sent message to %s: %s", chat_id, text[:100])
        return True
    except Exception:
        logger.exception("Failed to send message to %s", chat_id)
        return False
