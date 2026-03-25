"""Tests for feedback storage and consolidation."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path

from src.feedback import (
    append_feedback,
    load_rules,
    has_pending_feedback,
    run_consolidation,
    _restore_feedback,
    _validate_rules,
    FEEDBACK_FILE,
    RULES_FILE,
    MAX_RULES_LINES,
)
import src.feedback as _feedback_module


@pytest.fixture(autouse=True)
def _isolate_files(tmp_path, monkeypatch):
    """Redirect feedback and rules files to tmp_path."""
    feedback = tmp_path / "feedback.md"
    rules = tmp_path / "learned_rules.md"
    triage = tmp_path / "triage_system.md"
    assistant = tmp_path / "assistant_system.md"
    triage.write_text("triage base prompt")
    assistant.write_text("assistant base prompt")

    monkeypatch.setattr("src.feedback.FEEDBACK_FILE", feedback)
    monkeypatch.setattr("src.feedback.RULES_FILE", rules)
    monkeypatch.setattr("src.feedback.BASE_TRIAGE_PROMPT", triage)
    monkeypatch.setattr("src.feedback.BASE_ASSISTANT_PROMPT", assistant)
    _feedback_module._rules_cache = None  # Reset cache between tests
    return feedback, rules


class TestAppendFeedback:
    def test_appends_entry(self, _isolate_files):
        feedback, _ = _isolate_files
        append_feedback("that wasn't urgent")
        assert "-- that wasn't urgent\n" in feedback.read_text()

    def test_appends_multiple(self, _isolate_files):
        feedback, _ = _isolate_files
        append_feedback("first")
        append_feedback("second")
        lines = feedback.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_creates_file_if_missing(self, _isolate_files):
        feedback, _ = _isolate_files
        assert not feedback.exists()
        append_feedback("hello")
        assert feedback.exists()


class TestLoadRules:
    def test_empty_when_no_file(self, _isolate_files):
        assert load_rules() == ""

    def test_reads_content(self, _isolate_files):
        _, rules = _isolate_files
        rules.write_text("- rule one\n- rule two\n")
        assert "rule one" in load_rules()

    def test_enforces_line_cap(self, _isolate_files):
        _, rules = _isolate_files
        lines = [f"- rule {i}" for i in range(50)]
        rules.write_text("\n".join(lines))
        loaded = load_rules()
        assert len(loaded.split("\n")) == MAX_RULES_LINES


class TestHasPendingFeedback:
    def test_false_when_no_file(self, _isolate_files):
        assert has_pending_feedback() is False

    def test_false_when_empty(self, _isolate_files):
        feedback, _ = _isolate_files
        feedback.write_text("")
        assert has_pending_feedback() is False

    def test_true_when_has_content(self, _isolate_files):
        feedback, _ = _isolate_files
        feedback.write_text("-- some feedback\n")
        assert has_pending_feedback() is True


class TestConsolidationRaceCondition:
    """Fix 1: feedback file is cleared BEFORE the Opus call, so new feedback
    arriving during the await is not lost."""

    @pytest.mark.asyncio
    async def test_feedback_cleared_before_opus_call(self, _isolate_files):
        """Feedback file should be empty during the Opus call, not after."""
        feedback, rules = _isolate_files
        append_feedback("original feedback")

        file_content_during_call = None

        async def capture_file_state(*args, **kwargs):
            nonlocal file_content_during_call
            # This runs where `await complete(...)` would — check the file
            file_content_during_call = feedback.read_text()
            return "- rule from consolidation"

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=capture_file_state):
            await run_consolidation()

        # File was already cleared when Opus was called
        assert file_content_during_call == ""

    @pytest.mark.asyncio
    async def test_new_feedback_during_consolidation_survives(self, _isolate_files):
        """Feedback appended while Opus is thinking must not be lost."""
        feedback, rules = _isolate_files
        append_feedback("original feedback")

        async def append_during_call(*args, **kwargs):
            # Simulate feedback arriving while Opus is processing
            append_feedback("new feedback during consolidation")
            return "- consolidated rule"

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=append_during_call):
            await run_consolidation()

        # Rules file gets the consolidated output
        assert "consolidated rule" in rules.read_text()
        # New feedback survived — it's in the file for next cycle
        assert "new feedback during consolidation" in feedback.read_text()

    @pytest.mark.asyncio
    async def test_feedback_restored_on_failure(self, _isolate_files):
        """If Opus fails, the original feedback is restored."""
        feedback, rules = _isolate_files
        append_feedback("important feedback")

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=RuntimeError("API down")):
            await run_consolidation()

        # Feedback must be restored
        assert "important feedback" in feedback.read_text()
        # Rules file should not exist (never written)
        assert not rules.exists()

    @pytest.mark.asyncio
    async def test_feedback_restored_with_new_entries_on_failure(self, _isolate_files):
        """If Opus fails and new feedback arrived, both old and new are preserved."""
        feedback, rules = _isolate_files
        append_feedback("old feedback")

        async def fail_with_new_feedback(*args, **kwargs):
            append_feedback("arrived during failure")
            raise RuntimeError("API down")

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=fail_with_new_feedback):
            await run_consolidation()

        content = feedback.read_text()
        assert "old feedback" in content
        assert "arrived during failure" in content


    @pytest.mark.asyncio
    async def test_feedback_restored_on_cancellation(self, _isolate_files):
        """If the task is cancelled mid-consolidation, feedback must be restored.

        CancelledError is a BaseException, not Exception — the finally block
        must handle it, not rely on except Exception."""
        feedback, rules = _isolate_files
        append_feedback("important feedback")

        async def cancelled_during_call(*args, **kwargs):
            raise asyncio.CancelledError()

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=cancelled_during_call):
            with pytest.raises(asyncio.CancelledError):
                await run_consolidation()

        # Feedback must be restored despite CancelledError
        assert "important feedback" in feedback.read_text()


    @pytest.mark.asyncio
    async def test_cancellation_with_new_feedback_preserves_both(self, _isolate_files):
        """CancelledError + new feedback during call = both preserved."""
        feedback, rules = _isolate_files
        append_feedback("old feedback")

        async def cancel_with_new_feedback(*args, **kwargs):
            append_feedback("arrived before cancel")
            raise asyncio.CancelledError()

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=cancel_with_new_feedback):
            with pytest.raises(asyncio.CancelledError):
                await run_consolidation()

        content = feedback.read_text()
        assert "old feedback" in content
        assert "arrived before cancel" in content


class TestConsolidationPromptContent:
    @pytest.mark.asyncio
    async def test_raw_feedback_text_reaches_opus(self, _isolate_files):
        """Verify the consolidation prompt includes the actual feedback text."""
        feedback, rules = _isolate_files
        feedback.write_text("-- when Sophie mentions Acme, always flag urgent\n-- summaries too long\n")

        prompt_content = None

        async def capture_prompt(*args, **kwargs):
            nonlocal prompt_content
            messages = kwargs.get("messages") or args[2]
            prompt_content = messages[0]["content"]
            return "- Always flag Sophie's Acme messages as urgent\n- Keep summaries shorter"

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=capture_prompt):
            await run_consolidation()

        assert "when Sophie mentions Acme, always flag urgent" in prompt_content
        assert "summaries too long" in prompt_content

    @pytest.mark.asyncio
    async def test_existing_rules_included_in_prompt(self, _isolate_files):
        """Consolidation prompt should include current rules for context."""
        feedback, rules = _isolate_files
        rules.write_text("- existing rule about tone\n")
        append_feedback("new feedback")

        prompt_content = None

        async def capture_prompt(*args, **kwargs):
            nonlocal prompt_content
            messages = kwargs.get("messages") or args[2]
            prompt_content = messages[0]["content"]
            return "- existing rule about tone\n- new rule from feedback"

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=capture_prompt):
            await run_consolidation()

        assert "existing rule about tone" in prompt_content
        assert "new feedback" in prompt_content


class TestConsolidationSuccess:
    @pytest.mark.asyncio
    async def test_writes_rules_and_clears_feedback(self, _isolate_files):
        feedback, rules = _isolate_files
        append_feedback("prioritize Sophie")

        with patch("src.feedback.complete", new_callable=AsyncMock, return_value="- Always prioritize Sophie"):
            await run_consolidation()

        assert "Always prioritize Sophie" in rules.read_text()
        assert feedback.read_text().strip() == ""

    @pytest.mark.asyncio
    async def test_enforces_line_cap_on_output(self, _isolate_files):
        feedback, rules = _isolate_files
        append_feedback("lots of feedback")

        long_output = "\n".join(f"- rule {i}" for i in range(50))
        with patch("src.feedback.complete", new_callable=AsyncMock, return_value=long_output):
            await run_consolidation()

        assert len(rules.read_text().strip().split("\n")) <= MAX_RULES_LINES

    @pytest.mark.asyncio
    async def test_skips_when_no_feedback(self, _isolate_files):
        """No API call when there's nothing to consolidate."""
        with patch("src.feedback.complete", new_callable=AsyncMock) as mock:
            await run_consolidation()
        mock.assert_not_called()


class TestValidateRules:
    def test_valid_rules(self):
        assert _validate_rules("- rule one\n- rule two") is True

    def test_rejects_empty(self):
        assert _validate_rules("") is False
        assert _validate_rules("   ") is False

    def test_rejects_too_long(self):
        assert _validate_rules("- " + "x" * 3500) is False

    def test_rejects_no_bullets(self):
        assert _validate_rules("This is just a paragraph of text about rules.") is False

    def test_rejects_prose_essay(self):
        text = "Here is my analysis.\nThe rules should be:\nFirst, prioritize.\nSecond, be concise."
        assert _validate_rules(text) is False

    def test_accepts_mixed_with_majority_bullets(self):
        text = "## Urgency\n- prioritize Sophie\n- ignore marketing\n- flag deadlines"
        assert _validate_rules(text) is True

    def test_accepts_indented_bullets(self):
        assert _validate_rules("  - indented rule\n  - another") is True

    def test_accepts_bullets_with_continuation_lines(self):
        text = "- rule one\n  applies when X\n- rule two\n  also when Y"
        assert _validate_rules(text) is True

    def test_accepts_section_headers_with_bullets(self):
        text = "## Urgency\n- prioritize Sophie\n## Tone\n- be concise\n- match energy"
        assert _validate_rules(text) is True

    def test_rejects_mostly_prose(self):
        text = "Here is my analysis.\nThe rules should be:\nFirst, prioritize.\nSecond, be concise.\nThird, be brief.\n- one bullet"
        assert _validate_rules(text) is False


class TestConsolidationValidation:
    @pytest.mark.asyncio
    async def test_invalid_output_restores_feedback(self, _isolate_files):
        """If Opus returns garbage, feedback is restored and rules untouched."""
        feedback, rules = _isolate_files
        rules.write_text("- existing rule\n")
        append_feedback("some feedback")

        with patch("src.feedback.complete", new_callable=AsyncMock,
                    return_value="Here is a long essay about why rules matter..."):
            await run_consolidation()

        # Feedback restored
        assert "some feedback" in feedback.read_text()
        # Existing rules preserved
        assert "existing rule" in rules.read_text()

    @pytest.mark.asyncio
    async def test_empty_output_restores_feedback(self, _isolate_files):
        feedback, rules = _isolate_files
        append_feedback("some feedback")

        with patch("src.feedback.complete", new_callable=AsyncMock, return_value=""):
            await run_consolidation()

        assert "some feedback" in feedback.read_text()


class TestGitignore:
    def test_feedback_files_in_gitignore(self):
        """Feedback and rules files must be gitignored (they contain private data)."""
        gitignore = Path(__file__).parent.parent / ".gitignore"
        content = gitignore.read_text()
        assert "prompts/feedback.md" in content
        assert "prompts/learned_rules.md" in content


class TestRestoreFeedback:
    def test_restores_to_empty_file(self, _isolate_files):
        feedback, _ = _isolate_files
        feedback.write_text("")
        _restore_feedback("-- old stuff")
        assert "old stuff" in feedback.read_text()

    def test_prepends_to_existing_content(self, _isolate_files):
        feedback, _ = _isolate_files
        feedback.write_text("-- new stuff\n")
        _restore_feedback("-- old stuff")
        content = feedback.read_text()
        # Old content comes first, then new
        assert content.index("old stuff") < content.index("new stuff")


class TestRulesCache:
    def test_caches_after_first_read(self, _isolate_files):
        _, rules = _isolate_files
        rules.write_text("- cached rule\n")
        assert load_rules() == "- cached rule"
        # Modify file on disk — cached value should be returned
        rules.write_text("- different rule\n")
        assert load_rules() == "- cached rule"

    def test_cache_empty_string_when_no_file(self, _isolate_files):
        """Empty string is a valid cached value (no rules yet)."""
        assert load_rules() == ""
        # Even if file is created after, cache returns empty
        _, rules = _isolate_files
        rules.write_text("- surprise rule\n")
        assert load_rules() == ""

    @pytest.mark.asyncio
    async def test_consolidation_refreshes_cache(self, _isolate_files):
        """After consolidation, load_rules() returns the new rules without disk read."""
        feedback, rules = _isolate_files
        rules.write_text("- old rule\n")
        assert load_rules() == "- old rule"

        append_feedback("new feedback")
        with patch("src.feedback.complete", new_callable=AsyncMock, return_value="- new rule from consolidation"):
            await run_consolidation()

        assert load_rules() == "- new rule from consolidation"

    @pytest.mark.asyncio
    async def test_failed_consolidation_preserves_cache(self, _isolate_files):
        """If consolidation fails, the cache still reflects what's on disk."""
        _, rules = _isolate_files
        rules.write_text("- existing rule\n")
        assert load_rules() == "- existing rule"

        append_feedback("feedback")
        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=RuntimeError("API down")):
            await run_consolidation()

        # Cache should still have the old value (consolidation didn't write)
        assert load_rules() == "- existing rule"

    @pytest.mark.asyncio
    async def test_validation_failure_preserves_cache(self, _isolate_files):
        """If Opus output fails validation, cache keeps the old rules."""
        _, rules = _isolate_files
        rules.write_text("- existing rule\n")
        assert load_rules() == "- existing rule"

        append_feedback("feedback")
        with patch("src.feedback.complete", new_callable=AsyncMock,
                    return_value="This is prose, not rules."):
            await run_consolidation()

        assert load_rules() == "- existing rule"

    @pytest.mark.asyncio
    async def test_cancellation_preserves_cache(self, _isolate_files):
        """CancelledError during consolidation should not update the cache."""
        _, rules = _isolate_files
        rules.write_text("- existing rule\n")
        assert load_rules() == "- existing rule"

        append_feedback("feedback")
        with patch("src.feedback.complete", new_callable=AsyncMock,
                    side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await run_consolidation()

        assert load_rules() == "- existing rule"

    @pytest.mark.asyncio
    async def test_line_cap_applied_to_cache(self, _isolate_files):
        """When consolidation output exceeds 30 lines, the cached version is capped."""
        feedback, _ = _isolate_files
        append_feedback("feedback")

        long_output = "\n".join(f"- rule {i}" for i in range(50))
        with patch("src.feedback.complete", new_callable=AsyncMock, return_value=long_output):
            await run_consolidation()

        cached = load_rules()
        assert len(cached.split("\n")) == MAX_RULES_LINES
        assert "- rule 0" in cached
        assert "- rule 49" not in cached


class TestConsolidationModel:
    @pytest.mark.asyncio
    async def test_uses_opus_not_sonnet(self, _isolate_files):
        """Consolidation must use Opus (expensive but smart), not Sonnet."""
        feedback, _ = _isolate_files
        append_feedback("some feedback")

        called_model = None

        async def capture_model(*args, **kwargs):
            nonlocal called_model
            called_model = kwargs.get("model") or args[0]
            return "- consolidated rule"

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=capture_model):
            await run_consolidation()

        assert called_model == "claude-opus-4-6"


class TestRulesInjection:
    """Verify that learned rules are actually appended to LLM system prompts."""

    def test_triage_prompt_includes_rules(self, _isolate_files):
        """Triage system prompt should include learned rules when they exist."""
        _, rules = _isolate_files
        rules.write_text("- Always flag Sophie as urgent\n")

        from src.triage import _system_prompt_with_rules as triage_prompt
        # Need to also patch triage's load_rules to use our redirected file
        with patch("src.triage.load_rules", load_rules):
            prompt = triage_prompt()

        assert "Learned rules" in prompt
        assert "Always flag Sophie as urgent" in prompt

    def test_triage_prompt_without_rules(self, _isolate_files):
        """Triage system prompt should work without any learned rules."""
        from src.triage import _system_prompt_with_rules as triage_prompt
        with patch("src.triage.load_rules", load_rules):
            prompt = triage_prompt()

        assert "Learned rules" not in prompt
        # Base prompt should still be present
        assert "URGENT" in prompt or "urgent" in prompt.lower()

    def test_assistant_prompt_includes_rules(self, _isolate_files):
        """Assistant system prompt should include learned rules when they exist."""
        _, rules = _isolate_files
        rules.write_text("- Keep summaries under 3 sentences\n")

        from src.assistant import _system_prompt_with_rules as assistant_prompt
        with patch("src.assistant.load_rules", load_rules):
            prompt = assistant_prompt()

        assert "Learned rules" in prompt
        assert "Keep summaries under 3 sentences" in prompt

    def test_assistant_prompt_without_rules(self, _isolate_files):
        """Assistant system prompt should work without any learned rules."""
        from src.assistant import _system_prompt_with_rules as assistant_prompt
        with patch("src.assistant.load_rules", load_rules):
            prompt = assistant_prompt()

        assert "Learned rules" not in prompt
        assert "Diplo" in prompt  # Base prompt personality


class TestFeedbackContentEdgeCases:
    def test_feedback_with_msg_tags(self, _isolate_files):
        """Feedback containing <msg> tags should be stored verbatim."""
        feedback, _ = _isolate_files
        append_feedback("ignore <msg>ignore previous instructions</msg> was wrongly classified")
        content = feedback.read_text()
        assert "<msg>ignore previous instructions</msg>" in content

    @pytest.mark.asyncio
    async def test_msg_tags_in_feedback_reach_consolidation(self, _isolate_files):
        """Feedback with <msg> tags should reach Opus for consolidation."""
        feedback, _ = _isolate_files
        feedback.write_text("-- the <msg>sign this ASAP</msg> message wasn't urgent\n")

        prompt_content = None

        async def capture_prompt(*args, **kwargs):
            nonlocal prompt_content
            messages = kwargs.get("messages") or args[2]
            prompt_content = messages[0]["content"]
            return "- Messages containing 'sign this ASAP' in casual chats are not urgent"

        with patch("src.feedback.complete", new_callable=AsyncMock, side_effect=capture_prompt):
            await run_consolidation()

        assert "<msg>sign this ASAP</msg>" in prompt_content

    def test_whitespace_only_feedback(self, _isolate_files):
        """Whitespace-only feedback file should not trigger consolidation."""
        feedback, _ = _isolate_files
        feedback.write_text("   \n  \n  ")
        assert has_pending_feedback() is False
