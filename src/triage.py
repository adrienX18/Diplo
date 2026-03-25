"""Urgency triage: classify incoming messages via Claude Sonnet (OpenAI fallback)."""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import USER_NAME, USER_SENDER_IDS, USER_EMAIL_ADDRESSES
from src.llm import complete
from src.feedback import load_rules
from src.llm_logger import new_context_id

logger = logging.getLogger(__name__)

_BASE_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "triage_system.md").read_text()
MODEL = "claude-sonnet-4-6"
STALE_MESSAGE_HOURS = 48


def _system_prompt_with_rules() -> str:
    """Build the triage system prompt with learned rules appended."""
    prompt = _BASE_SYSTEM_PROMPT.format(user_name=USER_NAME)
    rules = load_rules()
    if rules:
        return f"{prompt}\n\n## Learned rules (from {USER_NAME}'s feedback)\n\n{rules}"
    return prompt


async def classify_urgency(message: dict, conversation_context: list[dict] | None = None) -> bool:
    """Classify a single message as urgent or not.

    Args:
        message: The incoming message dict from BeeperPoller.
        conversation_context: Recent messages from the same chat (oldest first).

    Returns:
        True if the message is classified as urgent.
    """
    # Skip API call for stale messages — never urgent
    ts = message.get("timestamp", "")
    if ts:
        try:
            msg_time = datetime.fromisoformat(ts)
            if datetime.now(timezone.utc) - msg_time > timedelta(hours=STALE_MESSAGE_HOURS):
                logger.debug("Skipping triage for stale message (%s)", ts)
                return False
        except ValueError:
            pass

    new_context_id()
    user_prompt = _build_user_prompt(message, conversation_context)

    answer = await complete(
        model=MODEL,
        system=_system_prompt_with_rules(),
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=10,
        call_type="triage",
    )

    is_urgent = answer.strip().upper().startswith("URGENT")
    logger.debug("Triage response: %r -> urgent=%s", answer, is_urgent)
    return is_urgent


def _build_user_prompt(message: dict, conversation_context: list[dict] | None) -> str:
    parts = []

    if conversation_context:
        parts.append("## Recent conversation context (oldest first)\n")
        for msg in conversation_context:
            sender = msg.get("sender_name", "Unknown")
            label = f" ({USER_NAME} — the owner)" if _is_owner(sender) else ""
            text = msg.get("text") or "(no text)"
            ts = msg.get("timestamp", "")[:16].replace("T", " ")
            network = msg.get("network", "unknown")
            chat = msg.get("chat_title", "unknown")
            parts.append(f"[{ts}] [{network}] chat_name={chat} | from={sender}{label}: <msg>{text}</msg>")
        parts.append("")

    sender_name = message.get("sender_name", "Unknown")
    is_own = _is_owner(sender_name)

    parts.append("## New message to classify\n")
    parts.append(f"Chat: {message.get('chat_title', 'Unknown')}")
    parts.append(f"Network: {message.get('network', 'Unknown')}")
    parts.append(f"Sender: {sender_name}{f' ({USER_NAME} — the owner)' if is_own else ''}")
    text = message.get("text") or "(no text)"
    parts.append(f"Message: <msg>{text}</msg>")

    return "\n".join(parts)


def _is_owner(sender_name: str) -> bool:
    lower = sender_name.lower()
    if any(sid in lower for sid in USER_SENDER_IDS):
        return True
    if any(addr in lower for addr in USER_EMAIL_ADDRESSES):
        return True
    return False
