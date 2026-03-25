"""Automations — scheduled and triggered tasks.

Scheduled tasks fire on a cron schedule (e.g. "every morning at 9am").
Triggered tasks fire when message conditions are met (e.g. "when Sophie messages").

Both execute their action through the assistant pipeline and send
results to the user via the control channel.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from croniter import croniter

from src.config import DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes


class AutomationStore:
    """SQLite-backed storage for automations (scheduled + triggered)."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS automations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL DEFAULT 'scheduled',
                description TEXT NOT NULL,
                schedule TEXT,
                trigger_config TEXT,
                action TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                cooldown_seconds INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_run_at TEXT,
                next_run_at TEXT,
                one_shot INTEGER NOT NULL DEFAULT 0,
                parent_id INTEGER
            )
        """)
        self._migrate_columns()
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_automations_next_run
            ON automations(next_run_at)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_automations_enabled
            ON automations(enabled)
        """)
        self._conn.commit()

    def _migrate_columns(self):
        """Add new columns to existing databases (safe no-op if already present)."""
        for col, default in [("one_shot INTEGER NOT NULL", "0"), ("parent_id INTEGER", None)]:
            try:
                default_clause = f" DEFAULT {default}" if default is not None else ""
                self._conn.execute(f"ALTER TABLE automations ADD COLUMN {col}{default_clause}")
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

    # ── Scheduled automations ──

    def create(
        self,
        description: str,
        schedule: str,
        action: str,
        tz_name: str = "UTC",
    ) -> int:
        """Create a scheduled automation. Returns the new ID.

        Raises ValueError if the cron expression is invalid.
        """
        if not croniter.is_valid(schedule):
            raise ValueError(f"Invalid cron expression: '{schedule}'")

        now = datetime.now(timezone.utc)
        next_run = self._compute_next_run(schedule, now, tz_name)

        cursor = self._conn.execute(
            """INSERT INTO automations
               (type, description, schedule, action, enabled, created_at, next_run_at)
               VALUES (?, ?, ?, ?, 1, ?, ?)""",
            ("scheduled", description, schedule, action,
             now.isoformat(), next_run.isoformat()),
        )
        self._conn.commit()
        logger.info("Created automation #%d: %s (%s)", cursor.lastrowid, description, schedule)
        return cursor.lastrowid

    def get_due(self) -> list[dict]:
        """Get all scheduled automations that are due to run now."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """SELECT * FROM automations
               WHERE type = 'scheduled'
               AND enabled = 1
               AND next_run_at <= ?
               ORDER BY next_run_at""",
            (now,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def mark_run(self, automation_id: int, tz_name: str = "UTC"):
        """Mark a scheduled automation as just run. Updates last_run_at and next_run_at."""
        auto = self.get(automation_id)
        if not auto or not auto["schedule"]:
            return

        now = datetime.now(timezone.utc)
        next_run = self._compute_next_run(auto["schedule"], now, tz_name)

        self._conn.execute(
            """UPDATE automations
               SET last_run_at = ?, next_run_at = ?
               WHERE id = ?""",
            (now.isoformat(), next_run.isoformat(), automation_id),
        )
        self._conn.commit()

    # ── Triggered automations ──

    def create_triggered(
        self,
        description: str,
        trigger_config: dict,
        action: str,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        delay_seconds: int = 0,
    ) -> int:
        """Create a triggered automation. Returns the new ID.

        Args:
            delay_seconds: If > 0, the trigger creates ephemeral one-shot
                scheduled automations instead of acting immediately.
                Stored in the trigger_config as "_delay_seconds".
        """
        if delay_seconds > 0:
            trigger_config = {**trigger_config, "_delay_seconds": delay_seconds}

        now = datetime.now(timezone.utc)
        cursor = self._conn.execute(
            """INSERT INTO automations
               (type, description, trigger_config, action, enabled, cooldown_seconds, created_at)
               VALUES (?, ?, ?, ?, 1, ?, ?)""",
            ("triggered", description, json.dumps(trigger_config), action,
             cooldown_seconds, now.isoformat()),
        )
        self._conn.commit()
        logger.info("Created triggered automation #%d: %s", cursor.lastrowid, description)
        return cursor.lastrowid

    def evaluate_triggers(self, message: dict) -> list[dict]:
        """Find all triggered automations that match a message.

        Returns a list of matching automation dicts.
        """
        cursor = self._conn.execute(
            "SELECT * FROM automations WHERE type = 'triggered' AND enabled = 1"
        )
        matches = []
        now = datetime.now(timezone.utc)

        for row in cursor.fetchall():
            auto = dict(row)
            config = json.loads(auto["trigger_config"]) if auto["trigger_config"] else {}

            # Check cooldown
            if auto["cooldown_seconds"] and auto["last_run_at"]:
                last_run = datetime.fromisoformat(auto["last_run_at"])
                if (now - last_run).total_seconds() < auto["cooldown_seconds"]:
                    continue

            if self._trigger_matches(config, message):
                matches.append(auto)

        return matches

    def mark_triggered(self, automation_id: int):
        """Mark a triggered automation as just fired. Updates last_run_at."""
        now = datetime.now(timezone.utc)
        self._conn.execute(
            "UPDATE automations SET last_run_at = ? WHERE id = ?",
            (now.isoformat(), automation_id),
        )
        self._conn.commit()

    def create_delayed(
        self,
        parent_id: int,
        delay_seconds: int,
        action: str,
        description: str,
    ) -> int:
        """Create a one-shot scheduled automation that fires after a delay.

        Cancels any existing pending one-shot for the same parent trigger
        (timer reset). Returns the new automation ID.
        """
        self.cancel_pending_oneshots(parent_id)

        now = datetime.now(timezone.utc)
        fire_at = now + timedelta(seconds=delay_seconds)

        cursor = self._conn.execute(
            """INSERT INTO automations
               (type, description, action, enabled, one_shot, parent_id,
                created_at, next_run_at)
               VALUES ('scheduled', ?, ?, 1, 1, ?, ?, ?)""",
            (description, action, parent_id,
             now.isoformat(), fire_at.isoformat()),
        )
        self._conn.commit()
        logger.info(
            "Created one-shot #%d (parent=#%d, fires in %ds): %s",
            cursor.lastrowid, parent_id, delay_seconds, description,
        )
        return cursor.lastrowid

    def cancel_pending_oneshots(self, parent_id: int) -> int:
        """Cancel (delete) any pending one-shot automations for a parent trigger.

        Returns the number of cancelled one-shots.
        """
        cursor = self._conn.execute(
            "DELETE FROM automations WHERE parent_id = ? AND one_shot = 1",
            (parent_id,),
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            logger.info("Cancelled %d pending one-shot(s) for parent #%d",
                        cursor.rowcount, parent_id)
        return cursor.rowcount

    def get_delay_seconds(self, automation: dict) -> int:
        """Extract delay_seconds from a triggered automation's config. Returns 0 if none."""
        config_str = automation.get("trigger_config")
        if not config_str:
            return 0
        try:
            config = json.loads(config_str) if isinstance(config_str, str) else config_str
            return config.get("_delay_seconds", 0)
        except (json.JSONDecodeError, TypeError):
            return 0

    @staticmethod
    def _trigger_matches(config: dict, message: dict) -> bool:
        """Check if a message matches a trigger configuration.

        All conditions in the config must match (AND logic).
        """
        sender = message.get("sender_name", "")
        text = message.get("text", "") or ""
        chat_title = message.get("chat_title", "")
        network = message.get("network", "")

        if "sender" in config:
            if config["sender"].lower() not in sender.lower():
                return False

        if "keyword" in config:
            if config["keyword"].lower() not in text.lower():
                return False

        if "chat" in config:
            if config["chat"].lower() not in chat_title.lower():
                return False

        if "network" in config:
            if config["network"].lower() not in network.lower():
                return False

        return True

    # ── Shared CRUD ──

    def get(self, automation_id: int) -> dict | None:
        """Get a single automation by ID."""
        row = self._conn.execute(
            "SELECT * FROM automations WHERE id = ?", (automation_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_all(self) -> list[dict]:
        """List all automations (enabled and disabled), excluding ephemeral one-shots."""
        cursor = self._conn.execute(
            "SELECT * FROM automations WHERE one_shot = 0 ORDER BY created_at"
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete(self, automation_id: int) -> bool:
        """Delete an automation by ID. Also deletes any child one-shots.

        Returns True if found and deleted.
        """
        # Delete child one-shots (if this is a trigger with pending delayed actions)
        self._conn.execute(
            "DELETE FROM automations WHERE parent_id = ? AND one_shot = 1",
            (automation_id,),
        )
        cursor = self._conn.execute(
            "DELETE FROM automations WHERE id = ?", (automation_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def toggle(self, automation_id: int, enabled: bool, tz_name: str = "UTC") -> bool:
        """Enable or disable an automation. Returns True if found.

        When re-enabling a scheduled automation, recomputes next_run_at.
        When disabling a trigger, cancels any pending one-shot timers.
        """
        auto = self.get(automation_id)
        if not auto:
            return False

        updates = {"enabled": 1 if enabled else 0}
        if enabled and auto["schedule"]:
            next_run = self._compute_next_run(
                auto["schedule"], datetime.now(timezone.utc), tz_name
            )
            updates["next_run_at"] = next_run.isoformat()

        # Disabling a trigger should cancel any pending delayed one-shots
        if not enabled and auto["type"] == "triggered":
            self.cancel_pending_oneshots(automation_id)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        self._conn.execute(
            f"UPDATE automations SET {set_clause} WHERE id = ?",
            (*updates.values(), automation_id),
        )
        self._conn.commit()
        return True

    def resolve_by_description(self, text: str) -> dict | list[dict] | None:
        """Fuzzy-match an automation by description substring.

        Excludes ephemeral one-shots — those are internal and shouldn't be
        matched by user-facing commands.

        Returns:
            - A single dict if exactly one match
            - A list if ambiguous (multiple matches)
            - None if no match
        """
        pattern = f"%{text.lower()}%"
        cursor = self._conn.execute(
            "SELECT * FROM automations WHERE LOWER(description) LIKE ? AND one_shot = 0",
            (pattern,),
        )
        matches = [dict(row) for row in cursor.fetchall()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches
        return None

    @staticmethod
    def _compute_next_run(
        schedule: str,
        after: datetime,
        tz_name: str = "UTC",
    ) -> datetime:
        """Compute the next fire time for a cron schedule.

        Cron is interpreted in the given timezone, but the result is
        returned in UTC for storage.
        """
        tz = ZoneInfo(tz_name)
        local_now = after.astimezone(tz)
        cron = croniter(schedule, local_now)
        local_next = cron.get_next(datetime)
        return local_next.astimezone(timezone.utc)

    def close(self):
        self._conn.close()


def format_delay(seconds: int) -> str:
    """Format a delay in seconds to a human-readable string like '5h 16m'."""
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours and minutes:
        return f"{hours}h {minutes}m"
    elif hours:
        return f"{hours}h"
    elif minutes:
        return f"{minutes}m"
    return f"{seconds}s"


async def run_scheduler_tick(
    store: AutomationStore,
    handle_action: Callable[[str], Awaitable[tuple[str, bool]]],
    channel,
    tz_name: str = "UTC",
):
    """Check for due scheduled automations and execute them.

    Args:
        store: The automation store to query.
        handle_action: Async function that processes the action text through
            the assistant pipeline. Same signature as handle_user_message
            but without cache/convo (those are wired in main.py).
        channel: Control channel to send results to the user.
        tz_name: User's timezone for computing next run times.
    """
    due = store.get_due()
    for auto in due:
        auto_id = auto["id"]
        description = auto["description"]
        action = auto["action"]
        is_one_shot = auto.get("one_shot", 0)

        logger.info("Running %sautomation #%d: %s",
                     "one-shot " if is_one_shot else "", auto_id, description)

        try:
            response, _ = await handle_action(action)
            await channel.send_message(f"[Auto] {description}\n\n{response}")
        except Exception:
            logger.exception("Automation #%d failed: %s", auto_id, description)

        if is_one_shot:
            store.delete(auto_id)
        else:
            # Always advance next_run_at, even on failure (don't retry same run)
            store.mark_run(auto_id, tz_name)
