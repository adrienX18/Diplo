"""LLM call logger — records every LLM interaction for observability and debugging.

Every call through llm.complete() is logged to a SQLite table (llm_calls) with:
- Full prompts and responses (for debugging what the model saw)
- Token usage, latency, model info (for cost/performance tracking)
- Context IDs (to correlate multi-step interactions like route→extract→respond)
- Status tracking (success, retry, fallback, error)

Auto-pruned at 7 days. Same DB as the message cache.
"""

import contextvars
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from src.config import DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)

# Context variable for threading context_id through async call chains.
# Each entry point (handle_user_message, classify_urgency, run_consolidation)
# sets a fresh context_id. All LLM calls within that flow share it.
_current_context_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_context_id", default=None
)

LLM_LOG_RETENTION_DAYS = 7


def get_context_id() -> str | None:
    """Get the current context ID (set by the active entry point)."""
    return _current_context_id.get()


def set_context_id(ctx_id: str | None) -> contextvars.Token:
    """Set the current context ID."""
    return _current_context_id.set(ctx_id)


def new_context_id() -> str:
    """Generate a fresh context ID and set it as current. Returns the ID."""
    ctx_id = uuid.uuid4().hex[:16]
    set_context_id(ctx_id)
    return ctx_id


class LLMLogger:
    """SQLite-backed logger for all LLM API calls."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                context_id TEXT,
                call_type TEXT NOT NULL,
                model TEXT NOT NULL,
                model_used TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                response TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                latency_ms INTEGER,
                status TEXT NOT NULL,
                error TEXT
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_timestamp ON llm_calls(timestamp)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_context ON llm_calls(context_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_type ON llm_calls(call_type)"
        )
        self._conn.commit()

    def log(
        self,
        *,
        call_type: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
        status: str = "success",
        error: str | None = None,
        model_used: str | None = None,
        context_id: str | None = None,
    ):
        """Record an LLM call. Fails silently (logs warning) to never block the main flow."""
        call_id = uuid.uuid4().hex[:16]
        ctx = context_id or get_context_id()
        try:
            self._conn.execute(
                """INSERT INTO llm_calls
                   (id, timestamp, context_id, call_type, model, model_used,
                    system_prompt, user_prompt, response, input_tokens, output_tokens,
                    latency_ms, status, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    call_id,
                    datetime.now(timezone.utc).isoformat(),
                    ctx,
                    call_type,
                    model,
                    model_used or model,
                    system_prompt,
                    user_prompt,
                    response,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    status,
                    error,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to log LLM call")

    def query_recent(self, limit: int = 20, call_type: str | None = None) -> list[dict]:
        """Get recent LLM calls, optionally filtered by call type."""
        if call_type:
            cursor = self._conn.execute(
                "SELECT * FROM llm_calls WHERE call_type = ? ORDER BY timestamp DESC LIMIT ?",
                (call_type, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM llm_calls ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]

    def query_by_context(self, context_id: str) -> list[dict]:
        """Get all LLM calls for a given context ID, in chronological order."""
        cursor = self._conn.execute(
            "SELECT * FROM llm_calls WHERE context_id = ? ORDER BY timestamp ASC",
            (context_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def search(
        self,
        *,
        text: str | None = None,
        call_type: str | None = None,
        hours: int | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search LLM calls with flexible filters. All conditions are ANDed."""
        conditions: list[str] = []
        params: list = []

        if call_type:
            conditions.append("call_type = ?")
            params.append(call_type)
        if hours:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            conditions.append("timestamp >= ?")
            params.append(cutoff)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if text:
            conditions.append(
                "(user_prompt LIKE ? OR response LIKE ? OR error LIKE ?)"
            )
            params.extend([f"%{text}%"] * 3)

        where = " AND ".join(conditions) if conditions else "1=1"
        cursor = self._conn.execute(
            f"SELECT * FROM llm_calls WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            (*params, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def prune(self):
        """Delete LLM call logs older than retention period."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=LLM_LOG_RETENTION_DAYS)
        ).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM llm_calls WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            logger.info(
                "Pruned %d LLM call logs older than %d days",
                cursor.rowcount,
                LLM_LOG_RETENTION_DAYS,
            )

    def close(self):
        self._conn.close()


# Module-level singleton — initialized once by main.py at startup.
_logger_instance: LLMLogger | None = None


def init_logger(db_path=None) -> LLMLogger:
    """Initialize the global LLM logger. Called once at startup."""
    global _logger_instance
    _logger_instance = LLMLogger(db_path=db_path)
    return _logger_instance


def get_logger() -> LLMLogger | None:
    """Get the global LLM logger, or None if not initialized."""
    return _logger_instance
