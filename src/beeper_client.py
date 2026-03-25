"""Beeper Desktop API client — polling for new messages."""

import json
import logging
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
from beeper_desktop_api import BeeperDesktop

from src.config import BEEPER_ACCESS_TOKEN, POLL_INTERVAL_SECONDS

_BEEPER_BASE_URL = "http://localhost:23373"

logger = logging.getLogger(__name__)

# Phone number pattern: starts with + followed by digits (E.164 format)
import re
_PHONE_RE = re.compile(r"^\+\d{7,15}$")


def normalize_message_text(text: str | None) -> str | None:
    """Extract plain text from JSON-encoded iMessage content.

    iMessage messages sometimes arrive as JSON objects like:
        {"text": "actual message", "textEntities": [...]}
    This extracts the inner "text" field. Non-JSON text passes through unchanged.
    """
    if not text:
        return text
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "text" in parsed and isinstance(parsed["text"], str):
            return parsed["text"]
    except (json.JSONDecodeError, TypeError):
        pass
    return text


@dataclass
class BeeperPoller:
    """Polls Beeper Desktop API for new incoming messages across all chats."""

    client: BeeperDesktop = field(init=False)
    # chat_id -> sort_key (as int) of the last message we've seen
    _seen: dict[str, int] = field(default_factory=dict)
    # Chats where we failed to fetch messages: chat_id -> {title, network, error, timestamp}
    fetch_errors: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        self.client = BeeperDesktop(access_token=BEEPER_ACCESS_TOKEN)

    @staticmethod
    def _needs_raw_http(chat_id: str) -> bool:
        """Check if a chat_id contains characters that break the SDK's URL construction.

        The Beeper Python SDK interpolates chat_id directly into the URL path
        without encoding. Characters like '#' (common in iMessage IDs such as
        'imsg##thread:...') are treated as URL fragment separators by httpx,
        causing the server to receive a truncated path and return 404.
        """
        return "#" in chat_id

    def _raw_list_messages(self, chat_id: str) -> list:
        """Fetch messages via raw HTTP with URL-encoded chat_id.

        Returns a list of SimpleNamespace objects with the same attributes as
        the SDK's Message objects so callers don't need to branch on type.
        """
        encoded = urllib.parse.quote(chat_id, safe="")
        url = f"{_BEEPER_BASE_URL}/v1/chats/{encoded}/messages"
        headers = {"Authorization": f"Bearer {BEEPER_ACCESS_TOKEN}"}
        resp = httpx.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            SimpleNamespace(
                id=m["id"],
                chat_id=m["chatID"],
                account_id=m.get("accountID", ""),
                sender_name=m.get("senderName") or "Unknown",
                text=normalize_message_text(m.get("text")),
                timestamp=datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")),
                sort_key=m["sortKey"],
                attachments=m.get("attachments") or [],
                type=m.get("type"),
            )
            for m in items
        ]

    def _raw_retrieve_chat(self, chat_id: str) -> SimpleNamespace | None:
        """Retrieve chat metadata via raw HTTP with URL-encoded chat_id."""
        encoded = urllib.parse.quote(chat_id, safe="")
        url = f"{_BEEPER_BASE_URL}/v1/chats/{encoded}"
        headers = {"Authorization": f"Bearer {BEEPER_ACCESS_TOKEN}"}
        resp = httpx.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return SimpleNamespace(title=data.get("title"), account_id=data.get("account_id"), type=data.get("type", ""))

    def _get_recent_chats(self, limit: int = 30) -> list[dict]:
        """Return the most recently active chats as raw dicts."""
        chats = []
        for chat in self.client.chats.list():
            chats.append(chat)
            if len(chats) >= limit:
                break
        return chats

    def get_recent_messages(self, chat_id: str, limit: int = 20) -> list[dict]:
        """Fetch the most recent messages from a chat as dicts (oldest first)."""
        try:
            if self._needs_raw_http(chat_id):
                raw = self._raw_list_messages(chat_id)
            else:
                raw = []
                for msg in self.client.messages.list(chat_id=chat_id):
                    raw.append(msg)
                    if len(raw) >= limit:
                        break
            msgs = []
            for msg in raw[:limit]:
                msgs.append({
                    "chat_id": chat_id,
                    "message_id": msg.id,
                    "sender_name": msg.sender_name or "Unknown",
                    "text": normalize_message_text(msg.text),
                    "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp),
                    "has_attachments": bool(msg.attachments),
                })
            msgs.sort(key=lambda m: m["timestamp"])
            return msgs
        except Exception as e:
            self._record_fetch_error(chat_id, chat_id, "unknown", e)
            return []

    def _record_fetch_error(self, chat_id: str, chat_title: str, network: str, error: Exception):
        """Track a chat where message fetching failed."""
        self.fetch_errors[chat_id] = {
            "chat_title": chat_title,
            "network": network,
            "error": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning("Failed to fetch messages for '%s' [%s]: %s", chat_title, network, error)

    def get_fetch_error_summary(self) -> str | None:
        """Return a human-readable summary of chats with fetch errors, or None if all clear."""
        if not self.fetch_errors:
            return None
        lines = [f"I couldn't read messages from {len(self.fetch_errors)} chat(s):"]
        for info in self.fetch_errors.values():
            lines.append(f"  - {info['chat_title']} [{info['network']}]: {info['error']}")
        lines.append("You may want to check if these chats are accessible in Beeper Desktop.")
        return "\n".join(lines)

    def _get_new_messages(self, chat_id: str, after_sort_key: int | None) -> list:
        """Fetch messages in a chat newer than `after_sort_key`."""
        if self._needs_raw_http(chat_id):
            all_msgs = self._raw_list_messages(chat_id)
        else:
            all_msgs = []
            for msg in self.client.messages.list(chat_id=chat_id):
                all_msgs.append(msg)
                if len(all_msgs) >= 50:
                    break

        # Filter to only messages newer than the watermark
        new_msgs = []
        for msg in all_msgs:
            if after_sort_key is not None and int(msg.sort_key) <= after_sort_key:
                continue
            new_msgs.append(msg)
            if len(new_msgs) >= 50:
                break

        # Sort chronologically (oldest first) — raw HTTP returns newest first
        new_msgs.sort(key=lambda m: int(m.sort_key))
        return new_msgs

    @staticmethod
    def _resolve_sender_name(sender_name: str, chat_title: str, chat_type: str) -> str:
        """Use chat title as sender name for DM chats where sender is a phone number.

        In iMessage DMs, sender_name is often a raw phone number (e.g. "+15551234567")
        while the chat title is the contact's display name. For DMs (type="single"),
        if sender_name looks like a phone number, swap it for the chat title.
        """
        if not sender_name or sender_name == "Unknown":
            return sender_name
        if chat_type == "single" and _PHONE_RE.match(sender_name) and chat_title:
            return chat_title
        return sender_name

    def poll_once(self) -> list[dict]:
        """Check all recent chats for new messages. Returns list of new messages."""
        all_new = []
        chats = self._get_recent_chats()

        for chat in chats:
            chat_id = chat.id
            # Use the preview message sort_key as a quick check
            preview = chat.preview
            if not preview:
                continue

            last_seen = self._seen.get(chat_id)
            if last_seen is not None and int(preview.sort_key) <= last_seen:
                continue

            # This chat has new activity — fetch the new messages
            try:
                new_msgs = self._get_new_messages(chat_id, last_seen)
            except Exception as e:
                self._record_fetch_error(chat_id, chat.title, chat.account_id, e)
                # Advance watermark so we don't retry the same messages every poll
                if preview:
                    self._seen[chat_id] = int(preview.sort_key)
                continue

            chat_type = getattr(chat, "type", "")

            for msg in new_msgs:
                # Skip reactions/tapbacks — they have no text content
                msg_type = getattr(msg, "type", None)
                if msg_type and msg_type.upper() == "REACTION":
                    continue

                raw_name = msg.sender_name or "Unknown"
                all_new.append({
                    "chat_id": chat_id,
                    "chat_title": chat.title,
                    "network": chat.account_id,
                    "message_id": msg.id,
                    "sender_name": self._resolve_sender_name(raw_name, chat.title, chat_type),
                    "text": normalize_message_text(msg.text),
                    "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp),
                    "has_attachments": bool(msg.attachments),
                    "attachments_raw": msg.attachments or [],
                })

            # Update watermark to the latest message in this chat
            if new_msgs:
                self._seen[chat_id] = int(new_msgs[-1].sort_key)
            elif preview:
                self._seen[chat_id] = int(preview.sort_key)

        return all_new

    def _resolve_chat_metadata(self, chat_ids: set[str]) -> dict[str, dict]:
        """Look up chat titles and types for a set of chat_ids via the Beeper API.

        Uses raw HTTP for chat_ids containing '#' (e.g. iMessage) to work
        around the SDK's URL encoding bug. Returns a dict mapping
        chat_id -> {"title": str, "type": str}. Failed lookups are silently skipped.
        """
        metadata = {}
        for chat_id in chat_ids:
            try:
                if self._needs_raw_http(chat_id):
                    chat = self._raw_retrieve_chat(chat_id)
                    if chat and chat.title:
                        metadata[chat_id] = {
                            "title": chat.title,
                            "type": getattr(chat, "type", ""),
                        }
                else:
                    chat = self.client.chats.retrieve(chat_id)
                    metadata[chat_id] = {
                        "title": chat.title or chat_id,
                        "type": getattr(chat, "type", ""),
                    }
            except Exception:
                logger.debug("Could not resolve metadata for chat %s", chat_id)
        return metadata

    def backfill_recent(self, hours: int = 48, max_messages: int = 2000) -> list[dict]:
        """Fetch all messages from the last N hours via messages.search().

        Uses the Beeper search API to pull messages across all chats and
        platforms within the time window. This is meant to run once at
        startup so the local SQLite cache has recent history even on a
        fresh start or after a long downtime.

        Note: messages.search() does not index iMessage. iMessage chats
        are backfilled separately via _backfill_raw_http_chats().

        Args:
            hours: How far back to look (default 48h).
            max_messages: Safety cap to avoid unbounded iteration.

        Returns:
            List of message dicts in the same format as poll_once(),
            sorted chronologically (oldest first).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        # Step 1: Fetch all messages from the last N hours.
        # The SDK returns a paginated iterator — we consume it up to max_messages.
        raw_messages = []
        for msg in self.client.messages.search(date_after=cutoff):
            raw_messages.append(msg)
            if len(raw_messages) >= max_messages:
                break

        # Step 2: Collect unique chat_ids, then resolve their titles and types
        # via individual chats.retrieve() calls. Each Message has account_id
        # (network) but no chat_title — we need this lookup for display.
        unique_chat_ids = {msg.chat_id for msg in raw_messages}
        chat_metadata = self._resolve_chat_metadata(unique_chat_ids)

        # Step 3: Convert each Message object to our standard dict format.
        messages = []
        for msg in raw_messages:
            # Skip reactions/tapbacks
            msg_type = getattr(msg, "type", None)
            if msg_type and msg_type.upper() == "REACTION":
                continue

            meta = chat_metadata.get(msg.chat_id, {})
            chat_title = meta.get("title", msg.chat_id)
            chat_type = meta.get("type", "")
            raw_name = msg.sender_name or "Unknown"
            messages.append({
                "chat_id": msg.chat_id,
                "chat_title": chat_title,
                "network": msg.account_id,
                "message_id": msg.id,
                "sender_name": self._resolve_sender_name(raw_name, chat_title, chat_type),
                "text": normalize_message_text(msg.text),
                "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp),
                "has_attachments": bool(msg.attachments),
            })

        # Step 4: Backfill chats that messages.search() misses (e.g. iMessage).
        # These require raw HTTP due to '#' in their chat IDs, and the search
        # API doesn't index them at all.
        messages.extend(self._backfill_raw_http_chats(cutoff))

        # Sort oldest first (search returns newest first by default)
        messages.sort(key=lambda m: m["timestamp"])
        return messages

    def _backfill_raw_http_chats(self, cutoff: str) -> list[dict]:
        """Backfill messages from chats that require raw HTTP (e.g. iMessage).

        Finds all chats needing raw HTTP in the recent chat list, fetches
        their messages, and filters to those after the cutoff timestamp.
        """
        messages = []
        chats = self._get_recent_chats(limit=50)
        for chat in chats:
            if not self._needs_raw_http(chat.id):
                continue
            chat_type = getattr(chat, "type", "")
            try:
                raw_msgs = self._raw_list_messages(chat.id)
                for msg in raw_msgs:
                    # Skip reactions/tapbacks
                    msg_type = getattr(msg, "type", None)
                    if msg_type and msg_type.upper() == "REACTION":
                        continue

                    ts = msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp)
                    if ts < cutoff:
                        continue
                    raw_name = msg.sender_name or "Unknown"
                    messages.append({
                        "chat_id": chat.id,
                        "chat_title": chat.title or chat.id,
                        "network": chat.account_id or "",
                        "message_id": msg.id,
                        "sender_name": self._resolve_sender_name(raw_name, chat.title or chat.id, chat_type),
                        "text": normalize_message_text(msg.text),
                        "timestamp": ts,
                        "has_attachments": bool(msg.attachments),
                    })
            except Exception:
                logger.warning("Failed to backfill chat %s (%s)", chat.title, chat.id[:30])
        if messages:
            logger.info("Backfilled %d messages from %d raw-HTTP chat(s)",
                        len(messages), len({m["chat_id"] for m in messages}))
        return messages

    def seed_watermarks(self, persisted: dict[str, int] | None = None):
        """Load persisted watermarks, then seed any new chats from Beeper.

        If persisted watermarks are provided, they are loaded first so the poller
        picks up where it left off. Chats not in the persisted set are seeded
        from their current preview sort key (same as before).
        """
        if persisted:
            self._seen.update(persisted)
            logger.info("Loaded %d persisted watermarks", len(persisted))

        chats = self._get_recent_chats(limit=50)
        new_count = 0
        for chat in chats:
            if chat.preview and chat.id not in self._seen:
                self._seen[chat.id] = int(chat.preview.sort_key)
                new_count += 1
        if new_count:
            logger.info("Seeded %d new chat watermarks from Beeper", new_count)
        logger.info("Tracking %d chats total", len(self._seen))

