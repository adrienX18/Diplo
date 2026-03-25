"""LLM abstraction — Claude with retry and OpenAI fallback, plus audio transcription."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import anthropic
import openai

from src.config import ANTHROPIC_API_KEY, OPENAI_API_KEY
from src.llm_logger import get_logger

logger = logging.getLogger(__name__)

_claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_openai = openai.AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

OPENAI_FALLBACK_MODEL = "gpt-4.1"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"


async def complete(
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int = 1500,
    call_type: str = "unknown",
) -> str:
    """Call Claude with one retry, then fall back to OpenAI.

    Args:
        model: Claude model name (e.g. "claude-sonnet-4-6").
        system: System prompt.
        messages: List of {"role": ..., "content": ...} dicts.
        max_tokens: Max tokens for the response.
        call_type: Label for this call (triage, response, compose, etc.) — used in logs.

    Returns:
        The text content of the response.
    """
    llm_log = get_logger()
    user_prompt = messages[0]["content"] if messages else ""
    t0 = time.perf_counter()

    def _log(*, response=None, input_tokens=None, output_tokens=None,
             status="success", error=None, model_used=None):
        if not llm_log:
            return
        latency = int((time.perf_counter() - t0) * 1000)
        llm_log.log(
            call_type=call_type,
            model=model,
            model_used=model_used,
            system_prompt=system,
            user_prompt=user_prompt,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            status=status,
            error=error,
        )

    # Try Claude (with one retry)
    for attempt in range(2):
        try:
            response = await _claude.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            text = response.content[0].text.strip()
            usage = getattr(response, "usage", None)
            _log(
                response=text,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                status="success" if attempt == 0 else "retry_success",
            )
            return text
        except Exception as e:
            if attempt == 0:
                logger.warning("Claude API failed (attempt 1), retrying in 2s: %s", e)
                await asyncio.sleep(2)
            else:
                logger.warning("Claude API failed (attempt 2): %s", e)

    # Fall back to OpenAI
    if not _openai:
        error_msg = "Claude API failed and no OpenAI API key configured for fallback"
        _log(status="error", error=error_msg)
        raise RuntimeError(error_msg)

    openai_model = OPENAI_FALLBACK_MODEL
    logger.info("Falling back to OpenAI (%s)", openai_model)

    try:
        response = await _openai.chat.completions.create(
            model=openai_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                *messages,
            ],
        )
        text = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        _log(
            response=text,
            model_used=openai_model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            status="fallback",
        )
        return text
    except Exception as e:
        error_msg = f"Both Claude and OpenAI failed. OpenAI error: {e}"
        _log(status="error", error=error_msg, model_used=openai_model)
        raise RuntimeError(error_msg) from e


async def stream_complete(
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int = 1500,
    call_type: str = "unknown",
) -> AsyncGenerator[str, None]:
    """Stream Claude response, yielding text deltas as they arrive.

    On any error, falls back to non-streaming complete() and yields the full
    result as a single chunk — callers always get at least one yield.
    """
    llm_log = get_logger()
    user_prompt = messages[0]["content"] if messages else ""
    t0 = time.perf_counter()

    try:
        async with _claude.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            full_text = ""
            async for text in stream.text_stream:
                full_text += text
                yield text

            # Log after stream completes (must be inside async with)
            final = await stream.get_final_message()
            usage = getattr(final, "usage", None)
            if llm_log:
                latency = int((time.perf_counter() - t0) * 1000)
                llm_log.log(
                    call_type=call_type, model=model,
                    system_prompt=system, user_prompt=user_prompt,
                    response=full_text.strip(),
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                    latency_ms=latency, status="success",
                )
    except Exception as e:
        logger.warning("Streaming failed, falling back to non-streaming: %s", e)
        result = await complete(model, system, messages, max_tokens, call_type)
        yield result


async def describe_image(file_path: str | Path) -> str:
    """Describe an image using Claude's vision capability.

    Args:
        file_path: Path to the image file (jpg, png, gif, webp, etc.).

    Returns:
        A concise 1-2 sentence description of the image.

    Raises:
        RuntimeError: If description fails.
    """
    import base64
    import mimetypes

    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"  # safe default

    image_data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

    try:
        response = await _claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system="Describe this image in 1-2 concise sentences. Focus on what's visually important — people, text, objects, context. Be factual. Do not start with 'This image shows' or 'The image depicts'.",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_data,
                        },
                    },
                ],
            }],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.error("Image description failed: %s", e)
        raise RuntimeError(f"Image description failed: {e}") from e


def _guess_audio_extension(file_path: Path) -> str:
    """Guess audio file extension from magic bytes when the filename has none."""
    if file_path.suffix:
        return file_path.suffix

    try:
        header = file_path.read_bytes()[:16]
        if header[:4] == b"OggS":
            return ".ogg"
        if header[:4] == b"fLaC":
            return ".flac"
        if header[:3] == b"ID3" or header[:2] == b"\xff\xfb":
            return ".mp3"
        if header[4:8] == b"ftyp":
            return ".m4a"
        if header[:4] == b"RIFF":
            return ".wav"
        if header[:4] == b"\x1aE\xdf\xa3":
            return ".webm"
    except Exception:
        pass
    return ".ogg"  # safe default — WhatsApp voice notes are OGG/Opus


async def transcribe_audio(file_path: str | Path) -> str:
    """Transcribe an audio file using OpenAI's transcription API.

    Args:
        file_path: Path to the audio file (mp3, mp4, m4a, wav, webm, ogg, etc.).
            Files without an extension (e.g. Beeper asset downloads) are auto-detected
            via magic bytes and given a proper filename for the API.

    Returns:
        The transcribed text.

    Raises:
        RuntimeError: If no OpenAI API key is configured or transcription fails.
    """
    if not _openai:
        raise RuntimeError("No OpenAI API key configured — cannot transcribe audio")

    path = Path(file_path)
    ext = _guess_audio_extension(path)
    filename = f"audio{ext}"

    try:
        audio_bytes = path.read_bytes()
        response = await _openai.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=(filename, audio_bytes),
        )
        return response.text.strip()
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise RuntimeError(f"Audio transcription failed: {e}") from e
