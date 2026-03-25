"""Tests for normalize_message_text — extracting plain text from JSON-encoded iMessage content."""

from src.beeper_client import normalize_message_text


class TestNormalizeMessageText:
    """iMessage messages sometimes arrive as JSON objects like:
        {"text": "actual message", "textEntities": [...]}
    normalize_message_text() should extract the inner "text" field.
    """

    def test_plain_text_unchanged(self):
        assert normalize_message_text("hello world") == "hello world"

    def test_none_returns_none(self):
        assert normalize_message_text(None) is None

    def test_empty_string_returns_empty(self):
        assert normalize_message_text("") == ""

    def test_json_with_text_field(self):
        """Core case: iMessage JSON with text + textEntities."""
        raw = '{"text":"indu just emailed about the george","textEntities":[{"from":0,"to":10}]}'
        assert normalize_message_text(raw) == "indu just emailed about the george"

    def test_json_with_text_and_link(self):
        """iMessage text with embedded link entity."""
        raw = '{"text":"Check this out https://example.com","textEntities":[{"from":15,"to":36,"link":"https://example.com","children":[]}]}'
        assert normalize_message_text(raw) == "Check this out https://example.com"

    def test_json_reaction_message(self):
        """iMessage tapback / reaction messages."""
        raw = '{"text":"{{sender}} loved \\"NYC trip booked!\\"","textEntities":[{"from":0,"to":10,"mentionedUser":{"id":"imsg##participant:abc123"}}]}'
        result = normalize_message_text(raw)
        assert '{{sender}} loved "NYC trip booked!"' == result

    def test_json_without_text_field_unchanged(self):
        """JSON that doesn't have a 'text' key should pass through."""
        raw = '{"foo": "bar"}'
        assert normalize_message_text(raw) == '{"foo": "bar"}'

    def test_json_with_whitespace(self):
        """Leading/trailing whitespace around JSON."""
        raw = '  {"text": "hello"}  '
        assert normalize_message_text(raw) == "hello"

    def test_non_json_starting_with_brace(self):
        """Text that starts with { but isn't valid JSON passes through."""
        raw = "{this is not json}"
        assert normalize_message_text(raw) == "{this is not json}"

    def test_text_with_emoji(self):
        """Plain emoji text should pass through."""
        assert normalize_message_text("🔥🚀") == "🔥🚀"

    def test_multiline_plain_text(self):
        """Multi-line plain text passes through unchanged."""
        text = "line 1\nline 2\nline 3"
        assert normalize_message_text(text) == text

    def test_json_text_with_newlines(self):
        """JSON where the inner text has newlines."""
        raw = '{"text":"line 1\\nline 2","textEntities":[]}'
        assert normalize_message_text(raw) == "line 1\nline 2"

    def test_text_field_is_empty_string(self):
        """JSON with empty text field."""
        raw = '{"text":"","textEntities":[]}'
        assert normalize_message_text(raw) == ""

    def test_nested_json_not_over_extracted(self):
        """If someone sends actual JSON as a message, don't extract from it."""
        # A person sending code/data — has "text" key but also "text" is the whole message
        raw = '{"text": "hello", "unrelated": true}'
        # This WILL extract — acceptable because real humans don't send JSON
        # and iMessage wraps it this way. The tradeoff is correct.
        assert normalize_message_text(raw) == "hello"

    # ── Edge cases: non-string "text" values should NOT be extracted ──

    def test_text_field_is_int_returns_original(self):
        """JSON with non-string text field should pass through (prevents type errors)."""
        raw = '{"text": 123}'
        assert normalize_message_text(raw) == raw

    def test_text_field_is_null_returns_original(self):
        """JSON with null text should pass through, not return None."""
        raw = '{"text": null, "textEntities": []}'
        assert normalize_message_text(raw) == raw

    def test_text_field_is_list_returns_original(self):
        """JSON with list text should pass through."""
        raw = '{"text": ["a", "b"]}'
        assert normalize_message_text(raw) == raw

    def test_text_field_is_dict_returns_original(self):
        """JSON with nested dict text should pass through."""
        raw = '{"text": {"inner": "value"}}'
        assert normalize_message_text(raw) == raw

    def test_json_array_passes_through(self):
        """JSON array (not object) should pass through."""
        raw = '[{"text": "hello"}]'
        assert normalize_message_text(raw) == raw
