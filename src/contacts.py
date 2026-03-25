"""Contact registry — maps (sender_name, network) to chat_id.

Unlike the message cache, this table is never pruned. It provides a persistent
mapping of who the user talks to, on which platform, and in which chat. Updated
on every polled message.

Known limitation: the same person can appear under multiple sender_name variants
(e.g. "Sophie", "Sophie Martin", "sophie.martin@gmail.com"). Each variant is
stored as a separate row. Deduplication (merging variants into a canonical
contact) is deferred to a future milestone.
"""

import sqlite3
import logging
from difflib import SequenceMatcher
from datetime import datetime, timezone

from src.config import DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)


class ContactRegistry:
    """Persistent mapping of (sender_name, network) -> chat_id."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                sender_name TEXT NOT NULL,
                network TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                chat_title TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                PRIMARY KEY (sender_name, network)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_contacts_sender ON contacts(sender_name)
        """)
        self._conn.commit()

    def update(self, sender_name: str, network: str, chat_id: str, chat_title: str, timestamp: str):
        """Update or insert a contact entry.

        Called on every polled message. If the (sender_name, network) pair already
        exists, updates chat_id, chat_title, and last_seen_at only if the new
        timestamp is more recent.
        """
        existing = self._conn.execute(
            "SELECT last_seen_at FROM contacts WHERE sender_name = ? AND network = ?",
            (sender_name, network),
        ).fetchone()

        if existing and existing["last_seen_at"] >= timestamp:
            return

        self._conn.execute(
            """INSERT OR REPLACE INTO contacts
               (sender_name, network, chat_id, chat_title, last_seen_at)
               VALUES (?, ?, ?, ?, ?)""",
            (sender_name, network, chat_id, chat_title, timestamp),
        )
        self._conn.commit()

    def lookup(self, name: str, network: str | None = None) -> list[dict]:
        """Find contacts matching a name or chat title (case-insensitive substring).

        Searches both sender_name and chat_title so that "tell the team chat"
        resolves correctly even if no sender has "team" in their name.

        Args:
            name: The name to search for (fuzzy, case-insensitive).
            network: Optional network filter (e.g. "whatsapp", "telegram").

        Returns:
            List of matching contacts sorted by last_seen_at (most recent first).
        """
        pattern = f"%{name.lower()}%"
        if network:
            cursor = self._conn.execute(
                """SELECT * FROM contacts
                   WHERE (LOWER(sender_name) LIKE ? OR LOWER(chat_title) LIKE ?)
                   AND LOWER(network) LIKE ?
                   ORDER BY last_seen_at DESC""",
                (pattern, pattern, f"%{network.lower()}%"),
            )
        else:
            cursor = self._conn.execute(
                """SELECT * FROM contacts
                   WHERE LOWER(sender_name) LIKE ? OR LOWER(chat_title) LIKE ?
                   ORDER BY last_seen_at DESC""",
                (pattern, pattern),
            )
        return [dict(row) for row in cursor.fetchall()]

    def resolve(self, name: str, network: str | None = None) -> dict | None:
        """Resolve a name to a single contact — the most recently active match.

        If network is specified, only matches on that network.
        If multiple matches exist, returns the one with the most recent last_seen_at.
        """
        matches = self.lookup(name, network)
        return matches[0] if matches else None

    # Minimum gap (in seconds) between the top two matches for auto-resolution.
    # If the gap is smaller, we ask the user to disambiguate.
    _RECENCY_GAP_SECONDS = 86400  # 24 hours

    def fuzzy_resolve(self, name: str, network: str | None = None) -> dict | list[dict] | None:
        """Resolve a name with fuzzy matching and typo tolerance.

        Returns:
            - A single dict if there's one clear match (exact or fuzzy)
            - A list of dicts if ambiguous (multiple equally good matches)
            - None if nothing matches at all
        """
        # Try exact substring match first (existing behavior)
        exact = self.lookup(name, network)
        if len(exact) == 1:
            return exact[0]
        if exact:
            # Multiple matches — try to narrow down
            # If all matches share the same chat_id, it's the same conversation
            chat_ids = {m["chat_id"] for m in exact}
            if len(chat_ids) == 1:
                return exact[0]  # same chat, just return most recent

            # Auto-resolve only if the most recent match is significantly newer (>24h gap)
            if self._has_clear_recency_winner(exact[0], exact[1]):
                return exact[0]

            # Genuinely ambiguous — return the list
            return exact

        # No exact match — try fuzzy matching against all known names
        all_contacts = self._all_names()
        if not all_contacts:
            return None

        name_lower = name.lower()
        scored = []
        for contact_name in all_contacts:
            ratio = SequenceMatcher(None, name_lower, contact_name.lower()).ratio()
            if ratio >= 0.6:  # threshold for "close enough"
                scored.append((contact_name, ratio))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        best_ratio = scored[0][1]

        # If the best match is clearly better than others, use it
        close_matches = [s for s in scored if s[1] >= best_ratio - 0.1]

        if len(close_matches) == 1:
            return self.resolve(close_matches[0][0], network)

        # Multiple close matches — look them up and return for disambiguation
        candidates = []
        seen_chat_ids = set()
        for match_name, _ in close_matches:
            results = self.lookup(match_name, network)
            for r in results:
                if r["chat_id"] not in seen_chat_ids:
                    seen_chat_ids.add(r["chat_id"])
                    candidates.append(r)

        if len(candidates) == 1:
            return candidates[0]
        return candidates if candidates else None

    @classmethod
    def _has_clear_recency_winner(cls, first: dict, second: dict) -> bool:
        """Check if the first match is clearly more recent than the second (>24h gap)."""
        try:
            ts1 = datetime.fromisoformat(first["last_seen_at"])
            ts2 = datetime.fromisoformat(second["last_seen_at"])
            return (ts1 - ts2).total_seconds() >= cls._RECENCY_GAP_SECONDS
        except (ValueError, KeyError):
            return False

    def _all_names(self) -> list[str]:
        """Return all unique sender_name values."""
        cursor = self._conn.execute("SELECT DISTINCT sender_name FROM contacts")
        return [row["sender_name"] for row in cursor.fetchall()]

    def seed_from_cache(self, cache_db_path=None):
        """Populate contacts from the message cache.

        Reads all messages from the cache and calls update() for each unique
        (sender_name, network, chat_id) combination. This ensures the contact
        registry is populated even if the bot just started.
        """
        from src.config import DB_PATH as default_cache_path
        cache_path = cache_db_path or default_cache_path

        try:
            cache_conn = sqlite3.connect(str(cache_path))
            cache_conn.row_factory = sqlite3.Row
            cursor = cache_conn.execute("""
                SELECT sender_name, network, chat_id, chat_title, MAX(timestamp) as timestamp
                FROM messages
                GROUP BY sender_name, network, chat_id
            """)
            count = 0
            for row in cursor.fetchall():
                self.update(
                    sender_name=row["sender_name"],
                    network=row["network"],
                    chat_id=row["chat_id"],
                    chat_title=row["chat_title"],
                    timestamp=row["timestamp"],
                )
                count += 1
            cache_conn.close()
            logger.info("Seeded %d contacts from message cache", count)
        except Exception:
            logger.exception("Failed to seed contacts from cache")

    def close(self):
        self._conn.close()
