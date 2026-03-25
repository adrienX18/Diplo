"""Live test: run urgency triage on the most recent messages from Beeper.

Fetches the latest messages across recent chats and classifies each one,
logging the results for manual review. Not a unit test — meant to be run
interactively to sanity-check triage behavior.

Usage:
    python -m tests.test_triage_live
"""

import asyncio
import logging

from src.beeper_client import BeeperPoller
from src.triage import classify_urgency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

NUM_CHATS = 10
MESSAGES_PER_CHAT = 5
CONTEXT_LIMIT = 20


async def main():
    poller = BeeperPoller()
    chats = poller._get_recent_chats(limit=NUM_CHATS)

    logger.info("Running triage on last %d messages from %d most recent chats...\n", MESSAGES_PER_CHAT, NUM_CHATS)

    for chat in chats:
        chat_id = chat.id
        title = chat.title
        network = chat.account_id

        # Get conversation context
        context = poller.get_recent_messages(chat_id, limit=CONTEXT_LIMIT)
        if not context:
            continue

        # Triage the last N messages from this chat
        messages_to_test = context[-MESSAGES_PER_CHAT:]

        logger.info("=== %s [%s] ===", title, network)

        for msg_dict in messages_to_test:
            msg_for_triage = {
                **msg_dict,
                "chat_title": title,
                "network": network,
            }

            is_urgent = await classify_urgency(msg_for_triage, conversation_context=context)
            tag = "URGENT" if is_urgent else "not urgent"
            text_preview = (msg_dict.get("text") or "(no text)")[:100]

            logger.info(
                "  [%s] %s: %s",
                tag,
                msg_dict["sender_name"],
                text_preview,
            )

        logger.info("")

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
