"""Tests for the observability toolkit — LLM call logging, context IDs, and debug intent."""

import json
import time
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm_logger import (
    LLMLogger,
    get_context_id,
    set_context_id,
    new_context_id,
    init_logger,
    get_logger,
    LLM_LOG_RETENTION_DAYS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_log(tmp_path):
    """Create a fresh LLMLogger with a temp DB."""
    log = LLMLogger(db_path=tmp_path / "test.db")
    yield log
    log.close()


@pytest.fixture(autouse=True)
def _reset_context_id():
    """Reset the context ID before each test."""
    set_context_id(None)
    yield
    set_context_id(None)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global logger singleton before/after each test."""
    import src.llm_logger as mod
    old = mod._logger_instance
    mod._logger_instance = None
    yield
    mod._logger_instance = old


# ---------------------------------------------------------------------------
# LLMLogger — basic CRUD
# ---------------------------------------------------------------------------

class TestLLMLoggerLog:
    def test_log_stores_call(self, llm_log):
        llm_log.log(
            call_type="triage",
            model="claude-sonnet-4-6",
            system_prompt="You classify messages.",
            user_prompt="Is this urgent?",
            response="NOT URGENT",
            input_tokens=100,
            output_tokens=5,
            latency_ms=230,
            status="success",
        )
        rows = llm_log.query_recent(limit=1)
        assert len(rows) == 1
        row = rows[0]
        assert row["call_type"] == "triage"
        assert row["model"] == "claude-sonnet-4-6"
        assert row["model_used"] == "claude-sonnet-4-6"
        assert row["system_prompt"] == "You classify messages."
        assert row["user_prompt"] == "Is this urgent?"
        assert row["response"] == "NOT URGENT"
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 5
        assert row["latency_ms"] == 230
        assert row["status"] == "success"
        assert row["error"] is None

    def test_log_with_error(self, llm_log):
        llm_log.log(
            call_type="response",
            model="claude-opus-4-6",
            system_prompt="sys",
            user_prompt="user",
            status="error",
            error="Both Claude and OpenAI failed",
        )
        rows = llm_log.query_recent(limit=1)
        assert rows[0]["status"] == "error"
        assert rows[0]["error"] == "Both Claude and OpenAI failed"
        assert rows[0]["response"] is None

    def test_log_with_fallback_model(self, llm_log):
        llm_log.log(
            call_type="triage",
            model="claude-sonnet-4-6",
            model_used="gpt-4.1",
            system_prompt="sys",
            user_prompt="user",
            response="NOT URGENT",
            status="fallback",
        )
        row = llm_log.query_recent(limit=1)[0]
        assert row["model"] == "claude-sonnet-4-6"
        assert row["model_used"] == "gpt-4.1"
        assert row["status"] == "fallback"

    def test_log_uses_context_id_from_contextvar(self, llm_log):
        set_context_id("ctx_abc123")
        llm_log.log(
            call_type="route_intent",
            model="claude-sonnet-4-6",
            system_prompt="sys",
            user_prompt="user",
            response="query",
        )
        row = llm_log.query_recent(limit=1)[0]
        assert row["context_id"] == "ctx_abc123"

    def test_log_explicit_context_id_overrides_contextvar(self, llm_log):
        set_context_id("from_contextvar")
        llm_log.log(
            call_type="triage",
            model="claude-sonnet-4-6",
            system_prompt="sys",
            user_prompt="user",
            context_id="explicit_override",
        )
        row = llm_log.query_recent(limit=1)[0]
        assert row["context_id"] == "explicit_override"

    def test_log_no_context_id(self, llm_log):
        llm_log.log(
            call_type="triage",
            model="claude-sonnet-4-6",
            system_prompt="sys",
            user_prompt="user",
        )
        row = llm_log.query_recent(limit=1)[0]
        assert row["context_id"] is None

    def test_log_failure_does_not_raise(self, llm_log):
        """Logging should never crash the main flow."""
        llm_log._conn.close()  # break the connection
        # Should not raise
        llm_log.log(
            call_type="triage",
            model="model",
            system_prompt="sys",
            user_prompt="user",
        )


# ---------------------------------------------------------------------------
# LLMLogger — query methods
# ---------------------------------------------------------------------------

class TestLLMLoggerQuery:
    def _insert_calls(self, llm_log, n=5):
        """Insert n test calls with predictable data."""
        for i in range(n):
            llm_log.log(
                call_type="triage" if i % 2 == 0 else "response",
                model="claude-sonnet-4-6" if i % 2 == 0 else "claude-opus-4-6",
                system_prompt=f"sys_{i}",
                user_prompt=f"user_{i} Sophie" if i == 2 else f"user_{i}",
                response=f"resp_{i}",
                input_tokens=100 + i * 10,
                output_tokens=5 + i,
                latency_ms=200 + i * 50,
                status="success",
                context_id=f"ctx_{i // 3}",  # Groups: 0,0,0,1,1
            )

    def test_query_recent_returns_desc_order(self, llm_log):
        self._insert_calls(llm_log, 3)
        rows = llm_log.query_recent(limit=10)
        assert len(rows) == 3
        # Most recent first
        assert rows[0]["user_prompt"] == "user_2 Sophie"
        assert rows[2]["user_prompt"] == "user_0"

    def test_query_recent_limit(self, llm_log):
        self._insert_calls(llm_log, 5)
        rows = llm_log.query_recent(limit=2)
        assert len(rows) == 2

    def test_query_recent_by_call_type(self, llm_log):
        self._insert_calls(llm_log, 5)
        triage = llm_log.query_recent(call_type="triage")
        response = llm_log.query_recent(call_type="response")
        assert all(r["call_type"] == "triage" for r in triage)
        assert all(r["call_type"] == "response" for r in response)
        assert len(triage) == 3  # indices 0, 2, 4
        assert len(response) == 2  # indices 1, 3

    def test_query_by_context(self, llm_log):
        self._insert_calls(llm_log, 5)
        ctx0 = llm_log.query_by_context("ctx_0")
        ctx1 = llm_log.query_by_context("ctx_1")
        assert len(ctx0) == 3  # indices 0, 1, 2
        assert len(ctx1) == 2  # indices 3, 4
        # Ascending order
        assert ctx0[0]["user_prompt"] == "user_0"
        assert ctx0[2]["user_prompt"] == "user_2 Sophie"

    def test_query_by_context_empty(self, llm_log):
        assert llm_log.query_by_context("nonexistent") == []


class TestLLMLoggerSearch:
    def _insert_mixed(self, llm_log):
        """Insert a mix of call types and statuses for search testing."""
        llm_log.log(
            call_type="triage", model="sonnet", system_prompt="sys",
            user_prompt="Message from Sophie: urgent meeting",
            response="URGENT", status="success",
        )
        llm_log.log(
            call_type="response", model="opus", system_prompt="sys",
            user_prompt="What did Sophie say?",
            response="Sophie said she needs a meeting", status="success",
        )
        llm_log.log(
            call_type="triage", model="sonnet", system_prompt="sys",
            user_prompt="Message from Marc: hey",
            response="NOT URGENT", status="fallback",
        )
        llm_log.log(
            call_type="compose", model="opus", system_prompt="sys",
            user_prompt="Tell Marc I'm busy",
            status="error", error="Both Claude and OpenAI failed",
        )

    def test_search_by_text(self, llm_log):
        self._insert_mixed(llm_log)
        results = llm_log.search(text="Sophie")
        assert len(results) == 2
        assert all("Sophie" in (r["user_prompt"] or "") or "Sophie" in (r["response"] or "")
                    for r in results)

    def test_search_by_call_type(self, llm_log):
        self._insert_mixed(llm_log)
        results = llm_log.search(call_type="triage")
        assert len(results) == 2
        assert all(r["call_type"] == "triage" for r in results)

    def test_search_by_status(self, llm_log):
        self._insert_mixed(llm_log)
        errors = llm_log.search(status="error")
        assert len(errors) == 1
        assert errors[0]["call_type"] == "compose"

        fallbacks = llm_log.search(status="fallback")
        assert len(fallbacks) == 1
        assert fallbacks[0]["user_prompt"] == "Message from Marc: hey"

    def test_search_by_hours(self, llm_log):
        self._insert_mixed(llm_log)
        # All recent calls should be within last 1 hour
        results = llm_log.search(hours=1)
        assert len(results) == 4

    def test_search_combined_filters(self, llm_log):
        self._insert_mixed(llm_log)
        results = llm_log.search(call_type="triage", text="Sophie")
        assert len(results) == 1
        assert results[0]["response"] == "URGENT"

    def test_search_no_results(self, llm_log):
        self._insert_mixed(llm_log)
        results = llm_log.search(text="nonexistent_person")
        assert results == []

    def test_search_limit(self, llm_log):
        self._insert_mixed(llm_log)
        results = llm_log.search(limit=2)
        assert len(results) == 2

    def test_search_text_in_error(self, llm_log):
        self._insert_mixed(llm_log)
        results = llm_log.search(text="OpenAI failed")
        assert len(results) == 1
        assert results[0]["status"] == "error"

    def test_search_no_filters_returns_all(self, llm_log):
        self._insert_mixed(llm_log)
        results = llm_log.search()
        assert len(results) == 4


# ---------------------------------------------------------------------------
# LLMLogger — prune
# ---------------------------------------------------------------------------

class TestLLMLoggerPrune:
    def test_prune_removes_old_entries(self, llm_log):
        # Insert an "old" entry by directly manipulating the DB
        old_ts = (
            datetime.now(timezone.utc) - timedelta(days=LLM_LOG_RETENTION_DAYS + 1)
        ).isoformat()
        llm_log._conn.execute(
            """INSERT INTO llm_calls
               (id, timestamp, call_type, model, model_used, system_prompt, user_prompt, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("old_id", old_ts, "triage", "sonnet", "sonnet", "sys", "user", "success"),
        )
        llm_log._conn.commit()

        # Insert a recent entry
        llm_log.log(
            call_type="triage", model="sonnet",
            system_prompt="sys", user_prompt="recent",
        )

        assert len(llm_log.query_recent(limit=100)) == 2
        llm_log.prune()
        remaining = llm_log.query_recent(limit=100)
        assert len(remaining) == 1
        assert remaining[0]["user_prompt"] == "recent"

    def test_prune_keeps_recent_entries(self, llm_log):
        llm_log.log(
            call_type="triage", model="sonnet",
            system_prompt="sys", user_prompt="fresh",
        )
        llm_log.prune()
        assert len(llm_log.query_recent()) == 1


# ---------------------------------------------------------------------------
# Context ID management
# ---------------------------------------------------------------------------

class TestContextId:
    def test_default_is_none(self):
        assert get_context_id() is None

    def test_set_and_get(self):
        set_context_id("test_ctx")
        assert get_context_id() == "test_ctx"

    def test_new_context_id_generates_and_sets(self):
        ctx = new_context_id()
        assert ctx is not None
        assert len(ctx) == 16
        assert get_context_id() == ctx

    def test_new_context_id_is_unique(self):
        ids = {new_context_id() for _ in range(100)}
        assert len(ids) == 100

    def test_set_resets_token(self):
        new_context_id()
        token = set_context_id("override")
        assert get_context_id() == "override"
        assert token is not None


# ---------------------------------------------------------------------------
# Singleton init/get
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_logger_returns_none_before_init(self):
        assert get_logger() is None

    def test_init_and_get(self, tmp_path):
        log = init_logger(db_path=tmp_path / "singleton.db")
        assert get_logger() is log
        log.close()


# ---------------------------------------------------------------------------
# Integration: llm.complete() logs calls
# ---------------------------------------------------------------------------

def _claude_response(text, input_tokens=50, output_tokens=10):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


def _openai_response(text, prompt_tokens=60, completion_tokens=12):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return resp


class TestCompleteLogging:
    @pytest.mark.asyncio
    async def test_successful_call_is_logged(self, tmp_path):
        log = init_logger(db_path=tmp_path / "log.db")
        try:
            set_context_id("test_ctx_1")
            with patch("src.llm._claude") as mock_claude:
                mock_claude.messages.create = AsyncMock(
                    return_value=_claude_response("URGENT")
                )
                from src.llm import complete
                result = await complete(
                    "claude-sonnet-4-6", "sys prompt", [{"role": "user", "content": "is this urgent?"}],
                    max_tokens=10, call_type="triage",
                )

            assert result == "URGENT"
            rows = log.query_recent(limit=1)
            assert len(rows) == 1
            row = rows[0]
            assert row["call_type"] == "triage"
            assert row["model"] == "claude-sonnet-4-6"
            assert row["model_used"] == "claude-sonnet-4-6"
            assert row["system_prompt"] == "sys prompt"
            assert row["user_prompt"] == "is this urgent?"
            assert row["response"] == "URGENT"
            assert row["input_tokens"] == 50
            assert row["output_tokens"] == 10
            assert row["status"] == "success"
            assert row["context_id"] == "test_ctx_1"
            assert row["latency_ms"] is not None
            assert row["latency_ms"] >= 0
        finally:
            log.close()

    @pytest.mark.asyncio
    async def test_retry_success_is_logged(self, tmp_path):
        log = init_logger(db_path=tmp_path / "log.db")
        try:
            with patch("src.llm._claude") as mock_claude, \
                 patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
                mock_claude.messages.create = AsyncMock(
                    side_effect=[RuntimeError("503"), _claude_response("recovered")]
                )
                from src.llm import complete
                result = await complete(
                    "claude-sonnet-4-6", "sys", [{"role": "user", "content": "hi"}],
                    call_type="response",
                )

            assert result == "recovered"
            rows = log.query_recent(limit=1)
            assert rows[0]["status"] == "retry_success"
        finally:
            log.close()

    @pytest.mark.asyncio
    async def test_fallback_is_logged(self, tmp_path):
        log = init_logger(db_path=tmp_path / "log.db")
        try:
            with patch("src.llm._claude") as mock_claude, \
                 patch("src.llm._openai") as mock_openai, \
                 patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
                mock_claude.messages.create = AsyncMock(side_effect=RuntimeError("down"))
                mock_openai.chat.completions.create = AsyncMock(
                    return_value=_openai_response("fallback answer")
                )
                from src.llm import complete
                result = await complete(
                    "claude-opus-4-6", "sys", [{"role": "user", "content": "hi"}],
                    call_type="compose",
                )

            assert result == "fallback answer"
            rows = log.query_recent(limit=1)
            row = rows[0]
            assert row["status"] == "fallback"
            assert row["model"] == "claude-opus-4-6"
            assert row["model_used"] == "gpt-4.1"
            assert row["input_tokens"] == 60
            assert row["output_tokens"] == 12
        finally:
            log.close()

    @pytest.mark.asyncio
    async def test_error_is_logged_no_openai(self, tmp_path):
        log = init_logger(db_path=tmp_path / "log.db")
        try:
            with patch("src.llm._claude") as mock_claude, \
                 patch("src.llm._openai", None), \
                 patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
                mock_claude.messages.create = AsyncMock(side_effect=RuntimeError("down"))
                from src.llm import complete
                with pytest.raises(RuntimeError, match="no OpenAI API key"):
                    await complete(
                        "claude-sonnet-4-6", "sys", [{"role": "user", "content": "hi"}],
                        call_type="triage",
                    )

            rows = log.query_recent(limit=1)
            assert rows[0]["status"] == "error"
            assert "no OpenAI API key" in rows[0]["error"]
        finally:
            log.close()

    @pytest.mark.asyncio
    async def test_error_is_logged_both_fail(self, tmp_path):
        log = init_logger(db_path=tmp_path / "log.db")
        try:
            with patch("src.llm._claude") as mock_claude, \
                 patch("src.llm._openai") as mock_openai, \
                 patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
                mock_claude.messages.create = AsyncMock(side_effect=RuntimeError("claude down"))
                mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("openai down"))
                from src.llm import complete
                with pytest.raises(RuntimeError, match="Both Claude and OpenAI failed"):
                    await complete(
                        "claude-sonnet-4-6", "sys", [{"role": "user", "content": "hi"}],
                        call_type="triage",
                    )

            rows = log.query_recent(limit=1)
            assert rows[0]["status"] == "error"
            assert "openai down" in rows[0]["error"]
            assert rows[0]["model_used"] == "gpt-4.1"
        finally:
            log.close()

    @pytest.mark.asyncio
    async def test_no_crash_without_logger(self):
        """When logger is not initialized, complete() still works."""
        assert get_logger() is None
        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.create = AsyncMock(
                return_value=_claude_response("hello")
            )
            from src.llm import complete
            result = await complete(
                "claude-sonnet-4-6", "sys", [{"role": "user", "content": "hi"}],
                call_type="triage",
            )
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_call_type_defaults_to_unknown(self, tmp_path):
        log = init_logger(db_path=tmp_path / "log.db")
        try:
            with patch("src.llm._claude") as mock_claude:
                mock_claude.messages.create = AsyncMock(
                    return_value=_claude_response("hi")
                )
                from src.llm import complete
                await complete(
                    "claude-sonnet-4-6", "sys", [{"role": "user", "content": "hi"}],
                )

            rows = log.query_recent(limit=1)
            assert rows[0]["call_type"] == "unknown"
        finally:
            log.close()

    @pytest.mark.asyncio
    async def test_context_id_threaded_through(self, tmp_path):
        """Multiple complete() calls in the same context share the same context_id."""
        log = init_logger(db_path=tmp_path / "log.db")
        try:
            ctx = new_context_id()
            with patch("src.llm._claude") as mock_claude:
                mock_claude.messages.create = AsyncMock(
                    return_value=_claude_response("ok")
                )
                from src.llm import complete
                await complete("sonnet", "sys", [{"role": "user", "content": "q1"}], call_type="route_intent")
                await complete("sonnet", "sys", [{"role": "user", "content": "q2"}], call_type="query_plan")
                await complete("opus", "sys", [{"role": "user", "content": "q3"}], call_type="response")

            calls = log.query_by_context(ctx)
            assert len(calls) == 3
            assert [c["call_type"] for c in calls] == ["route_intent", "query_plan", "response"]
            assert all(c["context_id"] == ctx for c in calls)
        finally:
            log.close()


# ---------------------------------------------------------------------------
# Debug intent — routing and handling
# ---------------------------------------------------------------------------

from src.message_cache import MessageCache
from src.conversation import ConversationHistory
from src.assistant import (
    _format_debug_entries,
    _extract_debug_plan,
    _handle_debug,
    handle_user_message,
    _set_pending,
)


def make_msg(message_id, sender_name="Alice", text="hello", chat_title="Alice Chat",
             network="whatsapp", hours_ago=1, chat_id=None):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "message_id": message_id,
        "chat_id": chat_id or f"chat_{message_id}",
        "chat_title": chat_title,
        "network": network,
        "sender_name": sender_name,
        "text": text,
        "timestamp": ts,
        "has_attachments": False,
    }


@pytest.fixture
def cache(tmp_path):
    c = MessageCache(db_path=tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def convo(tmp_path):
    c = ConversationHistory(db_path=tmp_path / "test_convo.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _clear_pending():
    _set_pending(None)
    yield
    _set_pending(None)


class TestDebugRouting:
    @pytest.mark.asyncio
    async def test_debug_intent_routed(self, cache, convo):
        """'why wasn't that urgent?' should route to the debug handler."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            # Route intent → "debug"
            # Debug plan extraction → {"hours": 2, "call_type": "triage"}
            # Debug response → "Here's what happened..."
            mock_complete.side_effect = [
                "debug",
                '{"hours": 2, "call_type": "triage", "text": "Sophie"}',
                "The triage call classified Sophie's message as NOT URGENT because...",
            ]
            with patch("src.assistant.get_llm_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_logger.search.return_value = [{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "call_type": "triage",
                    "model": "claude-sonnet-4-6",
                    "model_used": "claude-sonnet-4-6",
                    "system_prompt": "You classify messages.",
                    "user_prompt": "Message from Sophie: can we talk?",
                    "response": "NOT URGENT",
                    "input_tokens": 100,
                    "output_tokens": 5,
                    "latency_ms": 230,
                    "status": "success",
                    "error": None,
                    "context_id": "abc123",
                }]
                mock_get_logger.return_value = mock_logger

                response, queried = await handle_user_message(
                    "why wasn't Sophie's message marked urgent?",
                    cache, convo,
                )

        assert queried is False
        assert "NOT URGENT" in response or "triage" in response.lower() or "Sophie" in response

    @pytest.mark.asyncio
    async def test_debug_no_logger_returns_message(self, cache, convo):
        """When LLM logger is not initialized, debug returns a helpful message."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "debug",
                '{"hours": 1}',
            ]
            with patch("src.assistant.get_llm_logger", return_value=None):
                response, queried = await handle_user_message(
                    "what went wrong?", cache, convo,
                )

        assert "logging isn't enabled" in response.lower()

    @pytest.mark.asyncio
    async def test_debug_no_results_returns_message(self, cache, convo):
        """When no matching log entries, returns helpful message."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "debug",
                '{"hours": 1, "call_type": "triage", "text": "Sophie"}',
            ]
            with patch("src.assistant.get_llm_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_logger.search.return_value = []
                mock_get_logger.return_value = mock_logger

                response, queried = await handle_user_message(
                    "why wasn't Sophie's message urgent?", cache, convo,
                )

        assert "no llm calls found" in response.lower()
        assert "Sophie" in response


class TestExtractDebugPlan:
    @pytest.mark.asyncio
    async def test_parses_valid_json(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"hours": 3, "call_type": "triage", "text": "Marc"}'
            plan = await _extract_debug_plan("why wasn't Marc's message urgent?")
        assert plan == {"hours": 3, "call_type": "triage", "text": "Marc"}

    @pytest.mark.asyncio
    async def test_defaults_on_bad_json(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "I don't understand"
            plan = await _extract_debug_plan("what happened?")
        assert plan == {"hours": 2}

    @pytest.mark.asyncio
    async def test_call_type_is_debug_plan(self):
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = '{"hours": 1}'
            await _extract_debug_plan("what happened?")
        assert mock_complete.call_args[1]["call_type"] == "debug_plan"


class TestHandleDebug:
    @pytest.mark.asyncio
    async def test_queries_logger_with_plan(self):
        mock_logger = MagicMock()
        mock_logger.search.return_value = [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_type": "triage",
            "model": "claude-sonnet-4-6",
            "model_used": "claude-sonnet-4-6",
            "system_prompt": "sys",
            "user_prompt": "user",
            "response": "NOT URGENT",
            "input_tokens": 100,
            "output_tokens": 5,
            "latency_ms": 200,
            "status": "success",
            "error": None,
            "context_id": "ctx1",
        }]

        with patch("src.assistant.get_llm_logger", return_value=mock_logger), \
             patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "The triage decided NOT URGENT."
            result = await _handle_debug(
                {"hours": 2, "call_type": "triage", "text": "Sophie"},
                "why wasn't Sophie's message urgent?",
                "", "UTC",
            )

        mock_logger.search.assert_called_once_with(
            text="Sophie", call_type="triage", hours=2, status=None, limit=10,
        )
        assert mock_complete.call_args[1]["call_type"] == "debug_response"
        assert result == "The triage decided NOT URGENT."

    @pytest.mark.asyncio
    async def test_errors_only_filter(self):
        mock_logger = MagicMock()
        mock_logger.search.return_value = []

        with patch("src.assistant.get_llm_logger", return_value=mock_logger):
            result = await _handle_debug(
                {"hours": 24, "errors_only": True},
                "show me recent errors", "", "UTC",
            )

        mock_logger.search.assert_called_once_with(
            text=None, call_type=None, hours=24, status="error", limit=10,
        )


class TestFormatDebugEntries:
    def test_formats_single_entry(self):
        entries = [{
            "timestamp": "2026-03-13T10:00:00+00:00",
            "call_type": "triage",
            "model": "claude-sonnet-4-6",
            "model_used": "claude-sonnet-4-6",
            "system_prompt": "You classify messages.",
            "user_prompt": "Is this urgent? Sophie says hi",
            "response": "NOT URGENT",
            "input_tokens": 100,
            "output_tokens": 5,
            "latency_ms": 230,
            "status": "success",
            "error": None,
            "context_id": "abc123",
        }]
        result = _format_debug_entries(entries, "UTC")
        assert "triage" in result
        assert "claude-sonnet-4-6" in result
        assert "NOT URGENT" in result
        assert "success" in result
        assert "230ms" in result
        assert "100in/5out" in result
        assert "abc123" in result

    def test_formats_error_entry(self):
        entries = [{
            "timestamp": "2026-03-13T10:00:00+00:00",
            "call_type": "compose",
            "model": "claude-opus-4-6",
            "model_used": "gpt-4.1",
            "system_prompt": "sys",
            "user_prompt": "tell Marc hi",
            "response": None,
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": 5000,
            "status": "error",
            "error": "Both Claude and OpenAI failed",
            "context_id": None,
        }]
        result = _format_debug_entries(entries, "UTC")
        assert "error" in result.lower()
        assert "Both Claude and OpenAI failed" in result
        assert "gpt-4.1" in result

    def test_formats_multiple_entries(self):
        entries = [
            {
                "timestamp": "2026-03-13T10:00:00+00:00",
                "call_type": "route_intent",
                "model": "sonnet", "model_used": "sonnet",
                "system_prompt": "sys", "user_prompt": "what's new?",
                "response": "query", "input_tokens": 50, "output_tokens": 3,
                "latency_ms": 100, "status": "success", "error": None,
                "context_id": "ctx1",
            },
            {
                "timestamp": "2026-03-13T10:00:01+00:00",
                "call_type": "query_plan",
                "model": "sonnet", "model_used": "sonnet",
                "system_prompt": "sys", "user_prompt": "what's new?",
                "response": '{"since_last_seen": true}', "input_tokens": 80,
                "output_tokens": 10, "latency_ms": 150, "status": "success",
                "error": None, "context_id": "ctx1",
            },
        ]
        result = _format_debug_entries(entries, "UTC")
        assert "route_intent" in result
        assert "query_plan" in result
        assert "---" in result  # separator

    def test_truncates_long_prompts(self):
        entries = [{
            "timestamp": "2026-03-13T10:00:00+00:00",
            "call_type": "response",
            "model": "opus", "model_used": "opus",
            "system_prompt": "x" * 2000,
            "user_prompt": "y" * 2000,
            "response": "z" * 2000,
            "input_tokens": None, "output_tokens": None,
            "latency_ms": None, "status": "success",
            "error": None, "context_id": None,
        }]
        result = _format_debug_entries(entries, "UTC")
        # System truncated to 500, user to 800, response to 500
        assert len(result) < 2500


# ---------------------------------------------------------------------------
# Integration: triage call_type label
# ---------------------------------------------------------------------------

class TestTriageCallType:
    @pytest.mark.asyncio
    async def test_triage_passes_call_type(self):
        from src.triage import classify_urgency
        msg = make_msg("m1", sender_name="Sophie", text="urgent meeting NOW")
        with patch("src.triage.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "NOT URGENT"
            await classify_urgency(msg)
        assert mock_complete.call_args[1]["call_type"] == "triage"

    @pytest.mark.asyncio
    async def test_triage_sets_context_id(self):
        from src.triage import classify_urgency
        set_context_id(None)
        msg = make_msg("m1", sender_name="Sophie", text="hi")
        with patch("src.triage.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "NOT URGENT"
            await classify_urgency(msg)
        # After classify_urgency, a new context_id should have been set
        assert get_context_id() is not None


# ---------------------------------------------------------------------------
# Integration: feedback consolidation call_type label
# ---------------------------------------------------------------------------

class TestFeedbackCallType:
    @pytest.mark.asyncio
    async def test_consolidation_passes_call_type(self, tmp_path):
        from src.feedback import run_consolidation, FEEDBACK_FILE, RULES_FILE
        # Write some feedback
        FEEDBACK_FILE.write_text("-- test feedback\n")
        try:
            with patch("src.feedback.complete", new_callable=AsyncMock) as mock_complete:
                mock_complete.return_value = "- Rule one\n- Rule two"
                await run_consolidation()
            assert mock_complete.call_args[1]["call_type"] == "consolidation"
        finally:
            # Clean up
            if FEEDBACK_FILE.exists():
                FEEDBACK_FILE.write_text("")
            if RULES_FILE.exists():
                RULES_FILE.write_text("")


# ---------------------------------------------------------------------------
# Integration: assistant call_type labels
# ---------------------------------------------------------------------------

class TestAssistantCallTypes:
    @pytest.mark.asyncio
    async def test_query_flow_labels(self, cache, convo):
        """A query flow should label: route_intent, query_plan, response."""
        cache.store(make_msg("m1", text="hi from alice"))

        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "query",                         # route_intent
                '{"since_last_seen": true}',     # query_plan
                "You have 1 message from Alice",  # response
            ]
            await handle_user_message("what's new?", cache, convo)

        call_types = [c[1].get("call_type") or c[0][4] if len(c[0]) > 4 else None
                      for c in [mock_complete.call_args_list[i] for i in range(3)]]
        # Check via kwargs
        assert mock_complete.call_args_list[0][1]["call_type"] == "route_intent"
        assert mock_complete.call_args_list[1][1]["call_type"] == "query_plan"
        assert mock_complete.call_args_list[2][1]["call_type"] == "response"

    @pytest.mark.asyncio
    async def test_casual_flow_labels(self, cache, convo):
        """Casual flow: route_intent → response (casual)."""
        with patch("src.assistant.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = [
                "casual",
                "Hey boss! 🚀",
            ]
            await handle_user_message("hey!", cache, convo)

        assert mock_complete.call_args_list[0][1]["call_type"] == "route_intent"
        assert mock_complete.call_args_list[1][1]["call_type"] == "response"

    @pytest.mark.asyncio
    async def test_context_id_shared_across_calls(self, cache, convo):
        """All complete() calls within handle_user_message share one context_id."""
        captured_context_ids = []

        original_complete = AsyncMock(side_effect=[
            "query",
            '{"since_last_seen": true}',
            "Summary here",
        ])

        async def capture_ctx(*args, **kwargs):
            captured_context_ids.append(get_context_id())
            return await original_complete(*args, **kwargs)

        cache.store(make_msg("m1", text="hi"))

        with patch("src.assistant.complete", side_effect=capture_ctx):
            await handle_user_message("what's new?", cache, convo)

        assert len(captured_context_ids) == 3
        # All should be the same non-None context_id
        assert captured_context_ids[0] is not None
        assert len(set(captured_context_ids)) == 1
