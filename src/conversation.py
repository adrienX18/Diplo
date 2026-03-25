"""Conversation history — tracks the dialogue between the user and the assistant."""

import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.config import DB_PATH, DATA_DIR, USER_NAME

logger = logging.getLogger(__name__)

PRUNE_HOURS = 48
DEFAULT_LIMIT = 30
SESSION_GAP_SECONDS = 300  # 5 minutes of user inactivity = new session


class ConversationHistory:
    """Stores and retrieves the assistant-user conversation turns."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_convo_timestamp ON conversation_history(timestamp)
        """)
        self._conn.commit()

    def add_turn(self, role: str, text: str):
        """Store a conversation turn. Role is 'user' or 'assistant'."""
        self._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            (role, text, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def recent(self, limit: int = DEFAULT_LIMIT) -> list[dict]:
        """Get the last N turns, oldest first."""
        cursor = self._conn.execute(
            "SELECT role, text, timestamp FROM conversation_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        rows.reverse()
        return rows

    def format_for_prompt(self, limit: int = DEFAULT_LIMIT, max_chars: int = 500, tz_name: str | None = None) -> str:
        """Format recent conversation history for injection into a prompt.

        Long assistant responses are truncated to max_chars to keep the context lean.
        Timestamps are converted to local time if tz_name is provided.
        """
        turns = self.recent(limit)
        if not turns:
            return ""

        lines = [f"## Recent conversation with {USER_NAME}\n"]
        for turn in turns:
            role = USER_NAME if turn["role"] == "user" else "You (Diplo)"
            ts = _to_local_hhmm(turn["timestamp"], tz_name) if tz_name else turn["timestamp"][11:16]
            text = turn["text"]
            if turn["role"] == "assistant" and len(text) > max_chars:
                text = text[:max_chars] + "..."
            lines.append(f"[{ts} {role}]: {text}")

        return "\n".join(lines)

    def format_session_for_prompt(self, tz_name: str | None = None) -> str:
        """Format the current session's conversation for the response prompt.

        A session boundary is a >5min gap between consecutive user messages.
        Assistant responses are NOT truncated (unlike format_for_prompt) since
        sessions are short and Opus needs to see its own full responses to
        avoid repetition and self-correct.
        """
        turns = self.recent(limit=DEFAULT_LIMIT)
        if not turns:
            return ""

        # Find session boundary: walk backwards through user turns,
        # stop when gap between consecutive user messages > 5 min.
        prev_user_ts = None
        first_session_user_idx = 0

        for i in range(len(turns) - 1, -1, -1):
            turn = turns[i]
            if turn["role"] == "user":
                ts = datetime.fromisoformat(turn["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if prev_user_ts is not None:
                    gap = (prev_user_ts - ts).total_seconds()
                    if gap > SESSION_GAP_SECONDS:
                        break
                first_session_user_idx = i
                prev_user_ts = ts

        # Include assistant turns before the first user message of this
        # session if they're within the gap window (e.g. urgent notifications
        # that prompted the session). Exclude ones that belong to the old
        # session (e.g. responses to old questions).
        session_start_idx = first_session_user_idx
        if first_session_user_idx > 0:
            first_user_ts = datetime.fromisoformat(turns[first_session_user_idx]["timestamp"])
            if first_user_ts.tzinfo is None:
                first_user_ts = first_user_ts.replace(tzinfo=timezone.utc)
            for j in range(first_session_user_idx - 1, -1, -1):
                turn_ts = datetime.fromisoformat(turns[j]["timestamp"])
                if turn_ts.tzinfo is None:
                    turn_ts = turn_ts.replace(tzinfo=timezone.utc)
                if (first_user_ts - turn_ts).total_seconds() <= SESSION_GAP_SECONDS:
                    session_start_idx = j
                else:
                    break

        session_turns = turns[session_start_idx:]
        if not session_turns:
            return ""

        lines = []
        for turn in session_turns:
            role = USER_NAME if turn["role"] == "user" else "You (Diplo)"
            ts = _to_local_hhmm(turn["timestamp"], tz_name) if tz_name else turn["timestamp"][11:16]
            lines.append(f"[{ts} {role}]: {turn['text']}")

        return "\n".join(lines)

    def prune(self):
        """Delete conversation turns older than PRUNE_HOURS."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=PRUNE_HOURS)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM conversation_history WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            logger.info("Pruned %d conversation turns older than %dh", cursor.rowcount, PRUNE_HOURS)

    def close(self):
        self._conn.close()


def _to_local_hhmm(iso_ts: str, tz_name: str) -> str:
    """Convert an ISO UTC timestamp to local HH:MM."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone(ZoneInfo(tz_name))
        return local_dt.strftime("%H:%M")
    except Exception:
        return iso_ts[11:16]
