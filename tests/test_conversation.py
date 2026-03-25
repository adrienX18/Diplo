"""Tests for the conversation history."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from src.conversation import ConversationHistory
from src.config import USER_NAME


@pytest.fixture
def convo(tmp_path):
    c = ConversationHistory(db_path=tmp_path / "test.db")
    yield c
    c.close()


class TestAddAndRetrieve:
    def test_stores_and_retrieves_turn(self, convo):
        convo.add_turn("user", "what's new?")
        turns = convo.recent()
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert turns[0]["text"] == "what's new?"

    def test_preserves_order(self, convo):
        convo.add_turn("user", "first")
        convo.add_turn("assistant", "response")
        convo.add_turn("user", "second")

        turns = convo.recent()
        assert len(turns) == 3
        assert turns[0]["text"] == "first"
        assert turns[1]["text"] == "response"
        assert turns[2]["text"] == "second"

    def test_respects_limit(self, convo):
        for i in range(10):
            convo.add_turn("user", f"msg {i}")

        turns = convo.recent(limit=3)
        assert len(turns) == 3
        # Should be the 3 most recent, oldest first
        assert turns[0]["text"] == "msg 7"
        assert turns[2]["text"] == "msg 9"

    def test_empty_history(self, convo):
        turns = convo.recent()
        assert turns == []


class TestFormatForPrompt:
    def test_formats_turns(self, convo):
        convo.add_turn("user", "what did Sophie say?")
        convo.add_turn("assistant", "Sophie said she has a meeting at 3pm.")

        formatted = convo.format_for_prompt()
        assert f"{USER_NAME}]:" in formatted
        assert "You (Diplo)]:" in formatted
        assert "Sophie" in formatted
        # Should include HH:MM timestamps
        import re
        assert re.search(rf"\[\d{{2}}:\d{{2}} {re.escape(USER_NAME)}\]", formatted)

    def test_truncates_long_assistant_responses(self, convo):
        convo.add_turn("user", "summarize everything")
        convo.add_turn("assistant", "x" * 1000)

        formatted = convo.format_for_prompt(max_chars=100)
        # The assistant response should be truncated + "..."
        assert "..." in formatted
        assert "x" * 1000 not in formatted

    def test_does_not_truncate_user_messages(self, convo):
        long_msg = "y" * 1000
        convo.add_turn("user", long_msg)

        formatted = convo.format_for_prompt(max_chars=100)
        assert long_msg in formatted

    def test_empty_returns_empty_string(self, convo):
        assert convo.format_for_prompt() == ""

    def test_includes_header(self, convo):
        convo.add_turn("user", "hello")
        formatted = convo.format_for_prompt()
        assert f"Recent conversation with {USER_NAME}" in formatted


class TestFormatTimezone:
    def test_converts_to_local_timezone(self, convo):
        """Timestamps should be converted to local time when tz_name is provided."""
        # Insert a turn with a known UTC timestamp
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "hello", "2026-03-05T06:00:00+00:00"),
        )
        convo._conn.commit()

        # Pacific is UTC-8 (before DST) -> 22:00 on March 4
        formatted = convo.format_for_prompt(tz_name="America/Los_Angeles")
        assert "22:00" in formatted

    def test_converts_to_paris(self, convo):
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "bonjour", "2026-03-05T06:00:00+00:00"),
        )
        convo._conn.commit()

        # Paris is UTC+1 -> 07:00
        formatted = convo.format_for_prompt(tz_name="Europe/Paris")
        assert "07:00" in formatted

    def test_without_tz_uses_utc(self, convo):
        """Without tz_name, timestamps remain in UTC."""
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "hello", "2026-03-05T06:00:00+00:00"),
        )
        convo._conn.commit()

        formatted = convo.format_for_prompt()
        assert "06:00" in formatted


class TestPrune:
    def test_prunes_old_turns(self, convo):
        # Insert an old turn directly
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "old message", old_ts),
        )
        convo._conn.commit()

        convo.add_turn("user", "recent message")

        convo.prune()

        turns = convo.recent()
        assert len(turns) == 1
        assert turns[0]["text"] == "recent message"

    def test_keeps_recent_turns(self, convo):
        convo.add_turn("user", "just now")
        convo.prune()

        turns = convo.recent()
        assert len(turns) == 1


class TestFormatSessionForPrompt:
    """Tests for session-aware conversation formatting."""

    def test_single_session_returns_all_turns(self, convo):
        """All turns within 5 min of each other = one session."""
        now = datetime.now(timezone.utc)
        for i, (role, text) in enumerate([
            ("user", "hey"),
            ("assistant", "hey boss"),
            ("user", "what's new?"),
            ("assistant", "Nothing new since last check."),
        ]):
            ts = (now + timedelta(seconds=i * 30)).isoformat()
            convo._conn.execute(
                "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
                (role, text, ts),
            )
        convo._conn.commit()

        result = convo.format_session_for_prompt()
        assert "hey" in result
        assert "hey boss" in result
        assert "what's new?" in result
        assert "Nothing new" in result

    def test_session_boundary_splits_at_5min_gap(self, convo):
        """A >5min gap between user messages starts a new session."""
        now = datetime.now(timezone.utc)
        # Old session
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "old question", (now - timedelta(minutes=10)).isoformat()),
        )
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("assistant", "old answer", (now - timedelta(minutes=9, seconds=50)).isoformat()),
        )
        # 6+ min gap -> new session
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "new question", (now - timedelta(minutes=2)).isoformat()),
        )
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("assistant", "new answer", (now - timedelta(minutes=1, seconds=50)).isoformat()),
        )
        convo._conn.commit()

        result = convo.format_session_for_prompt()
        assert "old question" not in result
        assert "old answer" not in result
        assert "new question" in result
        assert "new answer" in result

    def test_includes_assistant_turns_within_session(self, convo):
        """Assistant turns between the gap and first user turn are included."""
        now = datetime.now(timezone.utc)
        # Old session user message
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "old msg", (now - timedelta(minutes=15)).isoformat()),
        )
        # Urgent notification from Diplo (assistant turn, before user's new session)
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("assistant", "[Urgent] Sophie needs you", (now - timedelta(minutes=3)).isoformat()),
        )
        # User responds (new session starts — >5min since last user message)
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "what did sophie say?", (now - timedelta(minutes=2)).isoformat()),
        )
        convo._conn.commit()

        result = convo.format_session_for_prompt()
        assert "old msg" not in result
        # The urgent notification is between the gap and the new user msg,
        # so it's part of the current session block
        assert "[Urgent] Sophie needs you" in result
        assert "what did sophie say?" in result

    def test_does_not_truncate_assistant_responses(self, convo):
        """Session format should NOT truncate assistant responses."""
        now = datetime.now(timezone.utc)
        long_response = "x" * 1000
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "summarize", now.isoformat()),
        )
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("assistant", long_response, (now + timedelta(seconds=5)).isoformat()),
        )
        convo._conn.commit()

        result = convo.format_session_for_prompt()
        assert long_response in result
        assert "..." not in result

    def test_empty_returns_empty_string(self, convo):
        assert convo.format_session_for_prompt() == ""

    def test_timezone_applied(self, convo):
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "hello", "2026-03-05T06:00:00+00:00"),
        )
        convo._conn.commit()

        result = convo.format_session_for_prompt(tz_name="Europe/Paris")
        assert "07:00" in result

    def test_only_one_user_message_returns_it(self, convo):
        """Single user message = entire session."""
        convo.add_turn("user", "solo question")
        result = convo.format_session_for_prompt()
        assert "solo question" in result

    def test_rapid_back_and_forth_stays_one_session(self, convo):
        """Many turns under 5 min apart = one session."""
        now = datetime.now(timezone.utc)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            ts = (now + timedelta(seconds=i * 60)).isoformat()  # 1 min apart
            convo._conn.execute(
                "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
                (role, f"turn {i}", ts),
            )
        convo._conn.commit()

        result = convo.format_session_for_prompt()
        for i in range(10):
            assert f"turn {i}" in result

    def test_multiple_sessions_returns_only_latest(self, convo):
        """Three sessions — only the last one should be returned."""
        now = datetime.now(timezone.utc)
        # Session 1
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "session one", (now - timedelta(minutes=30)).isoformat()),
        )
        # Session 2 (>5min gap)
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "session two", (now - timedelta(minutes=15)).isoformat()),
        )
        # Session 3 (>5min gap)
        convo._conn.execute(
            "INSERT INTO conversation_history (role, text, timestamp) VALUES (?, ?, ?)",
            ("user", "session three", (now - timedelta(minutes=1)).isoformat()),
        )
        convo._conn.commit()

        result = convo.format_session_for_prompt()
        assert "session one" not in result
        assert "session two" not in result
        assert "session three" in result

    def test_no_header_line(self, convo):
        """Session format should not include the '## Recent conversation' header."""
        convo.add_turn("user", "hi")
        result = convo.format_session_for_prompt()
        assert "## Recent conversation" not in result


class TestNotificationInHistory:
    def test_notification_stored_as_assistant_turn(self, convo):
        convo.add_turn("assistant", "[Urgent notification] Sophie in Team Chat: sign the contract by EOD")
        convo.add_turn("user", "ok reply to her saying I'll sign tonight")

        turns = convo.recent()
        assert len(turns) == 2
        assert turns[0]["role"] == "assistant"
        assert "Urgent notification" in turns[0]["text"]
        assert turns[1]["role"] == "user"
