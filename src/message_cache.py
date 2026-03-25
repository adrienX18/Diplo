"""SQLite message cache — stores all polled messages for fast local queries."""

import sqlite3
import logging
from datetime import datetime, timedelta, timezone

from src.config import DB_PATH, DATA_DIR, MESSAGE_CACHE_RETENTION_DAYS, DEFAULT_TIMEZONE

logger = logging.getLogger(__name__)

# Maps natural/common names to Beeper's internal network IDs.
# Keys are lowercase. Multiple aliases can point to the same network.
NETWORK_ALIASES: dict[str, str] = {
    "facebook": "facebookgo",
    "messenger": "facebookgo",
    "facebook messenger": "facebookgo",
    "fb": "facebookgo",
    "fb messenger": "facebookgo",
    "instagram": "instagramgo",
    "insta": "instagramgo",
    "ig": "instagramgo",
    "imessage": "imessagego",
    "imsg": "imessagego",
    "whatsapp": "whatsapp",
    "wa": "whatsapp",
    "telegram": "telegram",
    "tg": "telegram",
    "signal": "signal",
    "twitter": "twitter",
    "x": "twitter",
    "slack": "slack",
    "discord": "discord",
    "linkedin": "linkedin",
    "sms": "sms",
    "hungryserv": "hungryserv",
    "beeper": "hungryserv",
}


def normalize_network(raw: str) -> str:
    """Normalize a Beeper network ID by stripping suffixes.

    Beeper uses two conventions for the same network:
    - 'go' suffix: 'imessagego', 'facebookgo', 'instagramgo'
    - UUID suffix: 'imessage_df461d39ed5545ed...'

    This strips both to get the base name (e.g. 'imessage', 'facebook').
    """
    import re
    lower = raw.lower().strip()
    # Strip UUID suffix: 'imessage_df461d39ed5545ed...' -> 'imessage'
    normalized = re.sub(r"_[0-9a-f]{8,}$", "", lower)
    # Strip 'go' suffix: 'imessagego' -> 'imessage', 'facebookgo' -> 'facebook'
    if normalized.endswith("go") and len(normalized) > 2:
        normalized = normalized[:-2]
    return normalized


def resolve_network(name: str) -> str:
    """Resolve a natural network name to the Beeper internal network ID.

    Returns the resolved ID if found in aliases, otherwise returns the
    input lowercased (in case it's already a valid Beeper network ID).
    """
    lower = name.lower().strip()
    if lower in NETWORK_ALIASES:
        return NETWORK_ALIASES[lower]
    # Try after normalizing (strips UUID suffixes like 'imessage_abc123...')
    normalized = normalize_network(lower)
    if normalized in NETWORK_ALIASES:
        return NETWORK_ALIASES[normalized]
    return lower


class MessageCache:
    """Write-optimized SQLite cache for incoming messages."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                chat_title TEXT NOT NULL,
                network TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                text TEXT,
                timestamp TEXT NOT NULL,
                has_attachments INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_name)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS watermarks (
                chat_id TEXT PRIMARY KEY,
                sort_key INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def store(self, msg: dict):
        """Store a single message. Ignores duplicates."""
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO messages
                   (message_id, chat_id, chat_title, network, sender_name, text, timestamp, has_attachments)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    msg["message_id"],
                    msg["chat_id"],
                    msg["chat_title"],
                    msg["network"],
                    msg["sender_name"],
                    msg.get("text"),
                    msg["timestamp"],
                    1 if msg.get("has_attachments") else 0,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to store message %s", msg.get("message_id"))

    def _query(self, sql_where: str = "1=1", params: tuple = (), limit: int = 1000) -> list[dict]:
        """Run a filtered query against the cache. Returns list of dicts, newest first."""
        cursor = self._conn.execute(
            f"SELECT * FROM messages WHERE {sql_where} ORDER BY timestamp DESC LIMIT ?",
            (*params, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def search_text(self, query: str, limit: int = 100) -> list[dict]:
        """Full-text search on message text."""
        return self._query("text LIKE ?", (f"%{query}%",), limit=limit)

    def recent(self, hours: int = 24, limit: int = 1000) -> list[dict]:
        """Get messages from the last N hours."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        return self._query("timestamp >= ?", (cutoff,), limit=limit)

    def by_sender(self, sender: str, limit: int = 100) -> list[dict]:
        """Get messages from a specific sender (case-insensitive substring match)."""
        return self._query("LOWER(sender_name) LIKE ?", (f"%{sender.lower()}%",), limit=limit)

    def by_network(self, network: str, limit: int = 1000) -> list[dict]:
        """Get messages from a specific network (e.g. 'whatsapp', 'facebookgo').

        Accepts both natural names ('messenger', 'instagram') and Beeper
        internal IDs ('facebookgo', 'instagramgo') via resolve_network().
        Uses LIKE prefix matching to handle UUID-suffixed network IDs
        (e.g. 'imessage_df461d39...' matches when querying 'imessage').
        """
        # Collect all prefixes that could match the stored network ID.
        # Example for "imsg": resolved="imessagego", normalized_resolved="imessage"
        # This matches both 'imessagego' and 'imessage_df461d39...'
        resolved = resolve_network(network)
        prefixes = {resolved, normalize_network(network), normalize_network(resolved)}
        prefixes.discard("")
        clauses = " OR ".join("LOWER(network) LIKE ?" for _ in prefixes)
        params = tuple(f"{p}%" for p in prefixes)
        return self._query(clauses, params, limit=limit)

    def by_chat(self, chat_title: str, limit: int = 100) -> list[dict]:
        """Get messages from a specific chat (case-insensitive substring match)."""
        return self._query("LOWER(chat_title) LIKE ?", (f"%{chat_title.lower()}%",), limit=limit)

    def by_chat_id(self, chat_id: str, limit: int = 20) -> list[dict]:
        """Get recent messages from a chat by ID, oldest first. Used for triage context."""
        rows = self._query("chat_id = ?", (chat_id,), limit=limit)
        rows.reverse()  # query() returns newest first, we want oldest first
        return rows

    def prune(self):
        """Delete messages older than the retention period."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=MESSAGE_CACHE_RETENTION_DAYS)).isoformat()
        cursor = self._conn.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
        self._conn.commit()
        if cursor.rowcount > 0:
            logger.info("Pruned %d messages older than %d days", cursor.rowcount, MESSAGE_CACHE_RETENTION_DAYS)

    def touch_last_seen(self):
        """Update last_seen_at to now. Called when the user interacts with the bot."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES ('last_seen_at', ?)",
            (now,),
        )
        self._conn.commit()

    def get_last_seen(self) -> str | None:
        """Return the last_seen_at timestamp, or None if never set."""
        row = self._conn.execute("SELECT value FROM state WHERE key = 'last_seen_at'").fetchone()
        return row["value"] if row else None

    def since_last_seen(self, limit: int = 1000) -> list[dict]:
        """Get messages since the user last interacted with the bot."""
        last_seen = self.get_last_seen()
        if last_seen:
            return self._query("timestamp > ?", (last_seen,), limit=limit)
        # First time — fall back to last 24 hours
        return self.recent(hours=24, limit=limit)

    def get_timezone(self) -> str:
        """Return the configured timezone name, or the default."""
        row = self._conn.execute("SELECT value FROM state WHERE key = 'timezone'").fetchone()
        return row["value"] if row else DEFAULT_TIMEZONE

    def set_timezone(self, tz_name: str):
        """Set the timezone (IANA name like 'America/Los_Angeles')."""
        self._conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES ('timezone', ?)",
            (tz_name,),
        )
        self._conn.commit()

    def load_watermarks(self) -> dict[str, int]:
        """Load persisted poller watermarks."""
        cursor = self._conn.execute("SELECT chat_id, sort_key FROM watermarks")
        return {row["chat_id"]: row["sort_key"] for row in cursor.fetchall()}

    def delete_watermarks(self, chat_ids: list[str]):
        """Delete persisted watermarks for specific chat_ids."""
        self._conn.executemany(
            "DELETE FROM watermarks WHERE chat_id = ?",
            [(cid,) for cid in chat_ids],
        )
        self._conn.commit()

    def save_watermarks(self, seen: dict[str, int]):
        """Persist poller watermarks."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO watermarks (chat_id, sort_key) VALUES (?, ?)",
            seen.items(),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
