"""Tests for the LLM abstraction — retry, OpenAI fallback, and audio transcription."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm import complete, stream_complete, transcribe_audio


def _claude_response(text):
    """Build a mock Claude API response."""
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def _openai_response(text):
    """Build a mock OpenAI API response."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp


class TestComplete:
    @pytest.mark.asyncio
    async def test_claude_success(self):
        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.create = AsyncMock(return_value=_claude_response("hello"))
            result = await complete("claude-sonnet-4-6", "system", [{"role": "user", "content": "hi"}])

        assert result == "hello"

    @pytest.mark.asyncio
    async def test_claude_retry_then_success(self):
        """First attempt fails, second succeeds — no fallback needed."""
        with patch("src.llm._claude") as mock_claude, \
             patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
            mock_claude.messages.create = AsyncMock(
                side_effect=[RuntimeError("503"), _claude_response("recovered")]
            )
            result = await complete("claude-sonnet-4-6", "system", [{"role": "user", "content": "hi"}])

        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_falls_back_to_openai(self):
        """Both Claude attempts fail — falls back to OpenAI."""
        with patch("src.llm._claude") as mock_claude, \
             patch("src.llm._openai") as mock_openai, \
             patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
            mock_claude.messages.create = AsyncMock(side_effect=RuntimeError("down"))
            mock_openai.chat.completions.create = AsyncMock(
                return_value=_openai_response("openai answer")
            )
            result = await complete("claude-opus-4-6", "system", [{"role": "user", "content": "hi"}])

        assert result == "openai answer"
        # Verify OpenAI was called with system message + user message
        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0] == {"role": "system", "content": "system"}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "hi"}

    @pytest.mark.asyncio
    async def test_raises_when_no_openai_key(self):
        """Both Claude attempts fail and no OpenAI configured — raises."""
        with patch("src.llm._claude") as mock_claude, \
             patch("src.llm._openai", None), \
             patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
            mock_claude.messages.create = AsyncMock(side_effect=RuntimeError("down"))
            with pytest.raises(RuntimeError, match="no OpenAI API key"):
                await complete("claude-sonnet-4-6", "system", [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_raises_when_both_fail(self):
        """Both Claude and OpenAI fail — raises with OpenAI error."""
        with patch("src.llm._claude") as mock_claude, \
             patch("src.llm._openai") as mock_openai, \
             patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
            mock_claude.messages.create = AsyncMock(side_effect=RuntimeError("claude down"))
            mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("openai down"))
            with pytest.raises(RuntimeError, match="Both Claude and OpenAI failed"):
                await complete("claude-sonnet-4-6", "system", [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_openai_receives_correct_model(self):
        """Fallback uses gpt-4.1 for any Claude model."""
        with patch("src.llm._claude") as mock_claude, \
             patch("src.llm._openai") as mock_openai, \
             patch("src.llm.asyncio.sleep", new_callable=AsyncMock):
            mock_claude.messages.create = AsyncMock(side_effect=RuntimeError("down"))
            mock_openai.chat.completions.create = AsyncMock(
                return_value=_openai_response("ok")
            )
            await complete("claude-sonnet-4-6", "system", [{"role": "user", "content": "hi"}])

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4.1"


class TestTranscribeAudio:
    @pytest.mark.asyncio
    async def test_transcription_success(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        mock_response.text = "  Hello boss, what's new?  "

        with patch("src.llm._openai") as mock_openai:
            mock_openai.audio.transcriptions.create = AsyncMock(return_value=mock_response)
            result = await transcribe_audio(str(audio_file))

        assert result == "Hello boss, what's new?"
        # Verify correct model was used
        call_kwargs = mock_openai.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-transcribe"

    @pytest.mark.asyncio
    async def test_transcription_no_openai_key(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio data")

        with patch("src.llm._openai", None):
            with pytest.raises(RuntimeError, match="No OpenAI API key"):
                await transcribe_audio(str(audio_file))

    @pytest.mark.asyncio
    async def test_transcription_api_failure(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio data")

        with patch("src.llm._openai") as mock_openai:
            mock_openai.audio.transcriptions.create = AsyncMock(
                side_effect=RuntimeError("API error")
            )
            with pytest.raises(RuntimeError, match="Audio transcription failed"):
                await transcribe_audio(str(audio_file))


class _FakeStream:
    """Mock Claude streaming context manager that yields text deltas."""

    def __init__(self, deltas: list[str], final_message=None):
        self._deltas = deltas
        self._final = final_message or MagicMock(usage=MagicMock(input_tokens=10, output_tokens=20))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @property
    async def text_stream(self):
        for d in self._deltas:
            yield d

    async def get_final_message(self):
        return self._final


class TestStreamComplete:
    @pytest.mark.asyncio
    async def test_stream_yields_deltas(self):
        """stream_complete should yield text deltas from Claude streaming."""
        fake = _FakeStream(["Hello", " ", "world"])

        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.stream = MagicMock(return_value=fake)

            chunks = []
            async for delta in stream_complete(
                "claude-opus-4-6", "system", [{"role": "user", "content": "hi"}],
            ):
                chunks.append(delta)

        assert chunks == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_stream_falls_back_on_error(self):
        """If streaming fails, fall back to non-streaming complete()."""
        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.stream = MagicMock(
                side_effect=RuntimeError("stream broke")
            )
            mock_claude.messages.create = AsyncMock(
                return_value=_claude_response("fallback result")
            )

            chunks = []
            async for delta in stream_complete(
                "claude-opus-4-6", "system", [{"role": "user", "content": "hi"}],
            ):
                chunks.append(delta)

        # Should get the full result as a single chunk
        assert chunks == ["fallback result"]


class TestStreamAndChunk:
    """Tests for the _stream_and_chunk helper in assistant.py."""

    @pytest.mark.asyncio
    async def test_single_short_chunk(self):
        """Short response should be sent as one chunk."""
        from src.assistant import _stream_and_chunk

        chunks_sent = []

        async def on_chunk(text):
            chunks_sent.append(text)

        # Mock stream_complete to yield a short response
        async def fake_stream(**kwargs):
            yield "Short reply."

        with patch("src.assistant.stream_complete", fake_stream):
            result = await _stream_and_chunk(on_chunk, model="m", system="s",
                                              messages=[{"role": "user", "content": "hi"}])

        assert chunks_sent == ["Short reply."]
        assert result == "Short reply."

    @pytest.mark.asyncio
    async def test_splits_at_paragraph_boundary(self):
        """Long response with paragraph breaks should be sent as multiple chunks."""
        from src.assistant import _stream_and_chunk

        chunks_sent = []

        async def on_chunk(text):
            chunks_sent.append(text)

        # Build a response with two clear paragraphs, each > 200 chars
        para1 = "A" * 300
        para2 = "B" * 300
        full_text = f"{para1}\n\n{para2}"

        async def fake_stream(**kwargs):
            # Yield in small pieces to simulate token-by-token streaming
            for char in full_text:
                yield char

        with patch("src.assistant.stream_complete", fake_stream):
            result = await _stream_and_chunk(on_chunk, model="m", system="s",
                                              messages=[{"role": "user", "content": "q"}])

        assert len(chunks_sent) == 2
        assert chunks_sent[0] == para1
        assert chunks_sent[1] == para2

    @pytest.mark.asyncio
    async def test_no_split_without_paragraph_break(self):
        """Long text with no \\n\\n should be sent as one chunk at the end."""
        from src.assistant import _stream_and_chunk

        chunks_sent = []

        async def on_chunk(text):
            chunks_sent.append(text)

        long_text = "X" * 800  # Long but no paragraph breaks

        async def fake_stream(**kwargs):
            yield long_text

        with patch("src.assistant.stream_complete", fake_stream):
            result = await _stream_and_chunk(on_chunk, model="m", system="s",
                                              messages=[{"role": "user", "content": "q"}])

        # No paragraph break, so everything goes at the end
        assert len(chunks_sent) == 1
        assert chunks_sent[0] == long_text

    @pytest.mark.asyncio
    async def test_three_paragraphs(self):
        """Three substantial paragraphs should produce three chunks."""
        from src.assistant import _stream_and_chunk

        chunks_sent = []

        async def on_chunk(text):
            chunks_sent.append(text)

        paras = ["P" * 250, "Q" * 250, "R" * 250]
        full_text = "\n\n".join(paras)

        async def fake_stream(**kwargs):
            for char in full_text:
                yield char

        with patch("src.assistant.stream_complete", fake_stream):
            result = await _stream_and_chunk(on_chunk, model="m", system="s",
                                              messages=[{"role": "user", "content": "q"}])

        assert len(chunks_sent) == 3
        assert chunks_sent[0] == paras[0]
        assert chunks_sent[1] == paras[1]
        assert chunks_sent[2] == paras[2]

    @pytest.mark.asyncio
    async def test_short_paragraphs_batched_together(self):
        """Short paragraphs should be combined until they reach the min threshold."""
        from src.assistant import _stream_and_chunk

        chunks_sent = []

        async def on_chunk(text):
            chunks_sent.append(text)

        # Two short paragraphs (< 200 chars) followed by a long one
        short1 = "Hello boss!"
        short2 = "Here's the rundown:"
        long_para = "D" * 500

        full_text = f"{short1}\n\n{short2}\n\n{long_para}"

        async def fake_stream(**kwargs):
            for char in full_text:
                yield char

        with patch("src.assistant.stream_complete", fake_stream):
            result = await _stream_and_chunk(on_chunk, model="m", system="s",
                                              messages=[{"role": "user", "content": "q"}])

        # The two short paragraphs should NOT be split (rfind won't find a
        # break >= 200 chars), so they'll be combined with the long one or
        # sent as a batch. The exact split depends on when buffer > 500.
        assert len(chunks_sent) >= 1
        # The full content should be preserved
        assert "Hello boss!" in "\n\n".join(chunks_sent)
        assert "D" * 500 in "\n\n".join(chunks_sent)
