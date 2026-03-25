"""SQLite email cache — stores polled emails for fast local queries.

Separate database from the message cache (data/emails.db) to keep concerns
clean and avoid schema bloat.
"""

import sqlite3
import logging
from datetime import datetime, timedelta, timezone

from src.config import EMAIL_DB_PATH, DATA_DIR, MESSAGE_CACHE_RETENTION_DAYS, DEFAULT_TIMEZONE

logger = logging.getLogger(__name__)


class EmailCache:
    """Write-optimized SQLite cache for incoming emails."""

    def __init__(self, db_path=None):
        self.db_path = db_path or EMAIL_DB_PATH
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                email_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                mailbox TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                from_name TEXT NOT NULL DEFAULT '',
                from_address TEXT NOT NULL DEFAULT '',
                to_addresses TEXT NOT NULL DEFAULT '',
                cc_addresses TEXT NOT NULL DEFAULT '',
                body_text TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL,
                has_attachments INTEGER NOT NULL DEFAULT 0,
                attachment_names TEXT NOT NULL DEFAULT '',
                is_read INTEGER NOT NULL DEFAULT 0,
                is_from_adrien INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_emails_timestamp ON emails(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_emails_thread ON emails(thread_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_emails_mailbox ON emails(mailbox)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_emails_from ON emails(from_name)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS mailboxes (
                name TEXT PRIMARY KEY,
                provider TEXT NOT NULL DEFAULT 'gmail',
                email_address TEXT NOT NULL DEFAULT '',
                token_path TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                history_id TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        """)
        self._conn.commit()

    def store(self, email: dict):
        """Store a single email. Ignores duplicates."""
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO emails
                   (email_id, thread_id, mailbox, subject, from_name, from_address,
                    to_addresses, cc_addresses, body_text, timestamp,
                    has_attachments, attachment_names, is_read, is_from_adrien)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    email["email_id"],
                    email["thread_id"],
                    email["mailbox"],
                    email.get("subject", ""),
                    email.get("from_name", ""),
                    email.get("from_address", ""),
                    ",".join(email.get("to", [])) if isinstance(email.get("to"), list) else email.get("to", ""),
                    ",".join(email.get("cc", [])) if isinstance(email.get("cc"), list) else email.get("cc", ""),
                    email.get("body_text", ""),
                    email["timestamp"],
                    1 if email.get("has_attachments") else 0,
                    ",".join(email.get("attachment_names", [])) if isinstance(email.get("attachment_names"), list) else email.get("attachment_names", ""),
                    1 if email.get("is_read") else 0,
                    1 if email.get("is_from_adrien") else 0,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to store email %s", email.get("email_id"))

    def _query(self, sql_where: str = "1=1", params: tuple = (), limit: int = 500) -> list[dict]:
        """Run a filtered query. Returns list of dicts, newest first."""
        cursor = self._conn.execute(
            f"SELECT * FROM emails WHERE {sql_where} ORDER BY timestamp DESC LIMIT ?",
            (*params, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def by_sender(self, sender: str, limit: int = 100) -> list[dict]:
        """Get emails from a sender (case-insensitive substring on name or address)."""
        pattern = f"%{sender.lower()}%"
        return self._query(
            "LOWER(from_name) LIKE ? OR LOWER(from_address) LIKE ?",
            (pattern, pattern),
            limit=limit,
        )

    def search_text(self, query: str, limit: int = 100) -> list[dict]:
        """Full-text search on subject and body."""
        pattern = f"%{query}%"
        return self._query(
            "subject LIKE ? OR body_text LIKE ?",
            (pattern, pattern),
            limit=limit,
        )

    def by_mailbox(self, mailbox: str, limit: int = 500) -> list[dict]:
        """Get emails from a specific mailbox."""
        return self._query("mailbox = ?", (mailbox,), limit=limit)

    def by_thread(self, thread_id: str, limit: int = 20) -> list[dict]:
        """Get emails in a thread, oldest first (for context)."""
        rows = self._query("thread_id = ?", (thread_id,), limit=limit)
        rows.reverse()
        return rows

    def recent(self, hours: int = 24, limit: int = 500) -> list[dict]:
        """Get emails from the last N hours."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        return self._query("timestamp >= ?", (cutoff,), limit=limit)

    def since_last_seen(self, limit: int = 500) -> list[dict]:
        """Get emails since the user last checked."""
        last_seen = self.get_last_seen()
        if last_seen:
            return self._query("timestamp > ?", (last_seen,), limit=limit)
        return self.recent(hours=24, limit=limit)

    def get_last_seen(self) -> str | None:
        """Return the email last_seen_at timestamp, or None."""
        row = self._conn.execute("SELECT value FROM state WHERE key = 'email_last_seen_at'").fetchone()
        return row["value"] if row else None

    def touch_last_seen(self):
        """Update email last_seen_at to now."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES ('email_last_seen_at', ?)",
            (now,),
        )
        self._conn.commit()

    def get_timezone(self) -> str:
        """Return the configured timezone (shares with message cache setting)."""
        row = self._conn.execute("SELECT value FROM state WHERE key = 'timezone'").fetchone()
        return row["value"] if row else DEFAULT_TIMEZONE

    # -- Mailbox management --

    def add_mailbox(self, name: str, provider: str, email_address: str, token_path: str):
        """Register a new mailbox."""
        self._conn.execute(
            """INSERT OR REPLACE INTO mailboxes (name, provider, email_address, token_path)
               VALUES (?, ?, ?, ?)""",
            (name, provider, email_address, token_path),
        )
        self._conn.commit()

    def remove_mailbox(self, name: str):
        """Remove a mailbox."""
        self._conn.execute("DELETE FROM mailboxes WHERE name = ?", (name,))
        self._conn.commit()

    def list_mailboxes(self) -> list[dict]:
        """Return all configured mailboxes."""
        cursor = self._conn.execute("SELECT * FROM mailboxes ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]

    def get_mailbox(self, name: str) -> dict | None:
        """Get a single mailbox config."""
        row = self._conn.execute("SELECT * FROM mailboxes WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def get_history_id(self, mailbox_name: str) -> str | None:
        """Get the last history_id checkpoint for a mailbox."""
        row = self._conn.execute(
            "SELECT history_id FROM mailboxes WHERE name = ?", (mailbox_name,)
        ).fetchone()
        return row["history_id"] if row and row["history_id"] else None

    def set_history_id(self, mailbox_name: str, history_id: str):
        """Update the history_id checkpoint for a mailbox."""
        self._conn.execute(
            "UPDATE mailboxes SET history_id = ? WHERE name = ?",
            (history_id, mailbox_name),
        )
        self._conn.commit()

    def prune(self):
        """Delete emails older than the retention period."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=MESSAGE_CACHE_RETENTION_DAYS)).isoformat()
        cursor = self._conn.execute("DELETE FROM emails WHERE timestamp < ?", (cutoff,))
        self._conn.commit()
        if cursor.rowcount > 0:
            logger.info("Pruned %d emails older than %d days", cursor.rowcount, MESSAGE_CACHE_RETENTION_DAYS)

    def close(self):
        self._conn.close()
