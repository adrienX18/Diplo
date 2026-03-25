"""Tests for media attachment processing — llm.describe_image(), llm.transcribe_audio(), and main._process_media_attachments()."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Tests for llm.describe_image()
# ---------------------------------------------------------------------------

class TestDescribeImage:
    @pytest.mark.asyncio
    async def test_describe_image_success(self, tmp_path):
        """Happy path: image file is read, sent to Claude, description returned."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # fake JPEG header

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="  A cat sitting on a windowsill.  ")]

        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            from src.llm import describe_image
            result = await describe_image(str(img))

        assert result == "A cat sitting on a windowsill."

        # Verify correct model and vision payload
        call_kwargs = mock_claude.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["max_tokens"] == 300
        # Check the message has an image content block
        content = call_kwargs["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_describe_image_png_mime(self, tmp_path):
        """PNG file should get image/png mime type."""
        img = tmp_path / "screenshot.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="A screenshot of a chat window.")]

        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            from src.llm import describe_image
            result = await describe_image(str(img))

        call_kwargs = mock_claude.messages.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert content[0]["source"]["media_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_describe_image_unknown_extension_defaults_to_jpeg(self, tmp_path):
        """Files with unrecognized extensions default to image/jpeg."""
        img = tmp_path / "photo.xyz"
        img.write_bytes(b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Something.")]

        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            from src.llm import describe_image
            result = await describe_image(str(img))

        call_kwargs = mock_claude.messages.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert content[0]["source"]["media_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_describe_image_api_failure(self, tmp_path):
        """API failure raises RuntimeError."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
            from src.llm import describe_image
            with pytest.raises(RuntimeError, match="Image description failed"):
                await describe_image(str(img))

    @pytest.mark.asyncio
    async def test_describe_image_webp(self, tmp_path):
        """WebP gets correct mime type."""
        img = tmp_path / "sticker.webp"
        img.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="A sticker.")]

        with patch("src.llm._claude") as mock_claude:
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            from src.llm import describe_image
            await describe_image(str(img))

        call_kwargs = mock_claude.messages.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert content[0]["source"]["media_type"] == "image/webp"


# ---------------------------------------------------------------------------
# Tests for main._process_media_attachments()
# ---------------------------------------------------------------------------

def _make_msg(text=None, attachments_raw=None):
    """Build a message dict like the poller produces."""
    return {
        "chat_id": "!abc:beeper.local",
        "chat_title": "Sophie",
        "network": "whatsapp",
        "message_id": "msg_123",
        "sender_name": "Sophie",
        "text": text,
        "timestamp": "2026-03-13T10:00:00+00:00",
        "has_attachments": bool(attachments_raw),
        "attachments_raw": attachments_raw or [],
    }


def _sdk_attachment(att_type="img", att_id="mxc://beeper.local/abc123"):
    """Build a mock SDK attachment object."""
    return SimpleNamespace(type=att_type, id=att_id)


def _dict_attachment(att_type="img", att_id="mxc://beeper.local/abc123"):
    """Build a raw dict attachment (from raw HTTP path)."""
    return {"type": att_type, "id": att_id}


def _mock_download(src_url="file:///tmp/fake_asset"):
    """Return a mock httpx.post response for Beeper asset download."""
    return MagicMock(
        status_code=200,
        json=lambda: {"src_url": src_url},
        raise_for_status=lambda: None,
    )


class TestProcessMediaAttachments:
    """Tests for _process_media_attachments (images)."""

    @pytest.mark.asyncio
    async def test_single_image_with_caption(self):
        """Image + caption text → [image: desc] caption."""
        msg = _make_msg(text="check this out!", attachments_raw=[_sdk_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock, return_value="A sunset over the ocean.") as mock_desc, \
             patch("httpx.post", return_value=_mock_download("file:///tmp/fake_image.jpg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[image: A sunset over the ocean.] check this out!"
        mock_desc.assert_called_once_with("/tmp/fake_image.jpg")

    @pytest.mark.asyncio
    async def test_single_image_no_caption(self):
        """Image with no text → just [image: desc]."""
        msg = _make_msg(text=None, attachments_raw=[_sdk_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock, return_value="A selfie of two people."), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/img.jpg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[image: A selfie of two people.]"

    @pytest.mark.asyncio
    async def test_multiple_images(self):
        """Two images → both described, concatenated."""
        msg = _make_msg(
            text="look at these",
            attachments_raw=[
                _sdk_attachment(att_id="mxc://1"),
                _sdk_attachment(att_id="mxc://2"),
            ],
        )

        descriptions = iter(["First photo: a dog.", "Second photo: a cat."])

        with patch("src.main.describe_image", new_callable=AsyncMock, side_effect=lambda _: next(descriptions)), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/img.jpg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[image: First photo: a dog.] [image: Second photo: a cat.] look at these"

    @pytest.mark.asyncio
    async def test_video_and_unknown_attachments_ignored(self):
        """Video/unknown attachments are not processed."""
        msg = _make_msg(text="something", attachments_raw=[
            _sdk_attachment(att_type="video", att_id="mxc://video1"),
            _sdk_attachment(att_type="unknown", att_id="mxc://file1"),
        ])

        with patch("src.main.describe_image", new_callable=AsyncMock) as mock_desc, \
             patch("src.main.transcribe_audio", new_callable=AsyncMock) as mock_trans:
            from src.main import _process_media_attachments
            await _process_media_attachments(msg)

        mock_desc.assert_not_called()
        mock_trans.assert_not_called()
        assert msg["text"] == "something"

    @pytest.mark.asyncio
    async def test_description_failure_falls_back_to_bare_tag(self):
        """If describe_image fails, use [image] with no description."""
        msg = _make_msg(text="pic", attachments_raw=[_sdk_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock, side_effect=RuntimeError("API down")), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/img.jpg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[image] pic"

    @pytest.mark.asyncio
    async def test_download_failure_falls_back_to_bare_tag(self):
        """If Beeper asset download fails, use [image] with no description."""
        msg = _make_msg(text=None, attachments_raw=[_sdk_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock) as mock_desc, \
             patch("httpx.post", side_effect=Exception("Connection refused")):
            from src.main import _process_media_attachments
            await _process_media_attachments(msg)

        mock_desc.assert_not_called()
        assert msg["text"] == "[image]"

    @pytest.mark.asyncio
    async def test_invalid_download_path_falls_back(self):
        """If downloaded path doesn't exist, use [image]."""
        msg = _make_msg(text="hmm", attachments_raw=[_sdk_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock) as mock_desc, \
             patch("httpx.post", return_value=_mock_download("file:///nonexistent/path.jpg")):
            from src.main import _process_media_attachments
            await _process_media_attachments(msg)

        mock_desc.assert_not_called()
        assert msg["text"] == "[image] hmm"

    @pytest.mark.asyncio
    async def test_attachment_without_id_skipped(self):
        """Attachments with no id (mxc URL) are skipped."""
        msg = _make_msg(text="sticker", attachments_raw=[
            SimpleNamespace(type="img", id=None),
        ])

        with patch("src.main.describe_image", new_callable=AsyncMock) as mock_desc:
            from src.main import _process_media_attachments
            await _process_media_attachments(msg)

        mock_desc.assert_not_called()
        assert msg["text"] == "sticker"

    @pytest.mark.asyncio
    async def test_dict_attachment_from_raw_http(self):
        """Dict-style attachments (from raw HTTP path) work the same as SDK objects."""
        msg = _make_msg(text=None, attachments_raw=[_dict_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock, return_value="A meme."), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/meme.jpg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[image: A meme.]"

    @pytest.mark.asyncio
    async def test_empty_attachments_raw_noop(self):
        """No attachments → text unchanged."""
        msg = _make_msg(text="just text", attachments_raw=[])

        from src.main import _process_media_attachments
        await _process_media_attachments(msg)

        assert msg["text"] == "just text"

    @pytest.mark.asyncio
    async def test_attachments_raw_key_removed_after_processing(self):
        """The attachments_raw key is consumed (popped) during processing."""
        msg = _make_msg(text="pic", attachments_raw=[_sdk_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock, return_value="Desc."), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/img.jpg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert "attachments_raw" not in msg

    @pytest.mark.asyncio
    async def test_src_url_without_file_prefix(self):
        """If src_url is a plain path (no file://), it still works."""
        msg = _make_msg(text=None, attachments_raw=[_sdk_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock, return_value="A photo.") as mock_desc, \
             patch("httpx.post", return_value=_mock_download("/tmp/direct_path.jpg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        mock_desc.assert_called_once_with("/tmp/direct_path.jpg")
        assert msg["text"] == "[image: A photo.]"

    @pytest.mark.asyncio
    async def test_empty_src_url_falls_back(self):
        """If src_url is empty, fall back to [image]."""
        msg = _make_msg(text=None, attachments_raw=[_sdk_attachment()])

        with patch("src.main.describe_image", new_callable=AsyncMock) as mock_desc, \
             patch("httpx.post", return_value=_mock_download("")):
            from src.main import _process_media_attachments
            await _process_media_attachments(msg)

        mock_desc.assert_not_called()
        assert msg["text"] == "[image]"


# ---------------------------------------------------------------------------
# Tests for _process_media_attachments (audio)
# ---------------------------------------------------------------------------

class TestProcessMediaAttachmentsAudio:
    @pytest.mark.asyncio
    async def test_single_audio_with_caption(self):
        """Audio + caption → [voice message: transcript] caption."""
        msg = _make_msg(text="listen to this", attachments_raw=[
            _sdk_attachment(att_type="audio", att_id="mxc://beeper.local/audio1"),
        ])

        with patch("src.main.transcribe_audio", new_callable=AsyncMock, return_value="Hey, call me back when you can.") as mock_trans, \
             patch("httpx.post", return_value=_mock_download("file:///tmp/voice.ogg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[voice message: Hey, call me back when you can.] listen to this"
        mock_trans.assert_called_once_with("/tmp/voice.ogg")

    @pytest.mark.asyncio
    async def test_single_audio_no_caption(self):
        """Audio with no text → just [voice message: transcript]."""
        msg = _make_msg(text=None, attachments_raw=[
            _sdk_attachment(att_type="audio", att_id="mxc://beeper.local/audio1"),
        ])

        with patch("src.main.transcribe_audio", new_callable=AsyncMock, return_value="On my way."), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/voice.ogg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[voice message: On my way.]"

    @pytest.mark.asyncio
    async def test_audio_transcription_failure_falls_back(self):
        """If transcription fails, use [voice message] with no transcript."""
        msg = _make_msg(text="note", attachments_raw=[
            _sdk_attachment(att_type="audio", att_id="mxc://beeper.local/audio1"),
        ])

        with patch("src.main.transcribe_audio", new_callable=AsyncMock, side_effect=RuntimeError("No OpenAI key")), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/voice.ogg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[voice message] note"

    @pytest.mark.asyncio
    async def test_audio_download_failure_falls_back(self):
        """If Beeper asset download fails for audio, use [voice message]."""
        msg = _make_msg(text=None, attachments_raw=[
            _sdk_attachment(att_type="audio", att_id="mxc://beeper.local/audio1"),
        ])

        with patch("src.main.transcribe_audio", new_callable=AsyncMock) as mock_trans, \
             patch("httpx.post", side_effect=Exception("Connection refused")):
            from src.main import _process_media_attachments
            await _process_media_attachments(msg)

        mock_trans.assert_not_called()
        assert msg["text"] == "[voice message]"

    @pytest.mark.asyncio
    async def test_audio_invalid_path_falls_back(self):
        """If downloaded audio path doesn't exist, use [voice message]."""
        msg = _make_msg(text=None, attachments_raw=[
            _sdk_attachment(att_type="audio", att_id="mxc://beeper.local/audio1"),
        ])

        with patch("src.main.transcribe_audio", new_callable=AsyncMock) as mock_trans, \
             patch("httpx.post", return_value=_mock_download("file:///nonexistent/voice.ogg")):
            from src.main import _process_media_attachments
            await _process_media_attachments(msg)

        mock_trans.assert_not_called()
        assert msg["text"] == "[voice message]"

    @pytest.mark.asyncio
    async def test_multiple_audio(self):
        """Two audio attachments → both transcribed."""
        msg = _make_msg(text=None, attachments_raw=[
            _sdk_attachment(att_type="audio", att_id="mxc://1"),
            _sdk_attachment(att_type="audio", att_id="mxc://2"),
        ])

        transcripts = iter(["First message.", "Second message."])

        with patch("src.main.transcribe_audio", new_callable=AsyncMock, side_effect=lambda _: next(transcripts)), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/voice.ogg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[voice message: First message.] [voice message: Second message.]"

    @pytest.mark.asyncio
    async def test_dict_audio_attachment(self):
        """Dict-style audio attachments (from raw HTTP) work the same."""
        msg = _make_msg(text=None, attachments_raw=[
            _dict_attachment(att_type="audio", att_id="mxc://beeper.local/audio1"),
        ])

        with patch("src.main.transcribe_audio", new_callable=AsyncMock, return_value="Hello!"), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/voice.ogg")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[voice message: Hello!]"


# ---------------------------------------------------------------------------
# Tests for mixed image + audio attachments
# ---------------------------------------------------------------------------

class TestProcessMediaAttachmentsMixed:
    @pytest.mark.asyncio
    async def test_mixed_image_and_audio(self):
        """Image + audio in same message → both processed, images first."""
        msg = _make_msg(text="multimedia", attachments_raw=[
            _sdk_attachment(att_type="audio", att_id="mxc://audio1"),
            _sdk_attachment(att_type="img", att_id="mxc://img1"),
        ])

        with patch("src.main.describe_image", new_callable=AsyncMock, return_value="A beach.") as mock_desc, \
             patch("src.main.transcribe_audio", new_callable=AsyncMock, return_value="Wish you were here.") as mock_trans, \
             patch("httpx.post", return_value=_mock_download("file:///tmp/asset")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert mock_desc.call_count == 1
        assert mock_trans.call_count == 1
        # Images processed first, then audio
        assert msg["text"] == "[image: A beach.] [voice message: Wish you were here.] multimedia"

    @pytest.mark.asyncio
    async def test_mixed_with_partial_failure(self):
        """Image succeeds, audio fails → image described, audio gets bare tag."""
        msg = _make_msg(text=None, attachments_raw=[
            _sdk_attachment(att_type="img", att_id="mxc://img1"),
            _sdk_attachment(att_type="audio", att_id="mxc://audio1"),
        ])

        with patch("src.main.describe_image", new_callable=AsyncMock, return_value="A photo."), \
             patch("src.main.transcribe_audio", new_callable=AsyncMock, side_effect=RuntimeError("fail")), \
             patch("httpx.post", return_value=_mock_download("file:///tmp/asset")):
            with patch("os.path.exists", return_value=True):
                from src.main import _process_media_attachments
                await _process_media_attachments(msg)

        assert msg["text"] == "[image: A photo.] [voice message]"


# ---------------------------------------------------------------------------
# Tests for poller integration — attachments_raw is passed through
# ---------------------------------------------------------------------------

class TestPollerAttachmentsRaw:
    def test_poll_once_includes_attachments_raw(self):
        """poll_once() message dicts include the raw attachments list."""
        from src.beeper_client import BeeperPoller

        with patch("src.beeper_client.BeeperDesktop"):
            poller = BeeperPoller()
            poller.client = MagicMock()

        poller._seen = {"chat1": 100}

        chat = SimpleNamespace(
            id="chat1", title="Alice", account_id="whatsapp",
            preview=SimpleNamespace(id="msg_200", sort_key="200", text="hi",
                                     is_sender=False, sender_name="Alice",
                                     timestamp="2026-03-13T00:00:00Z", attachments=None),
            participants=None, type="single",
        )
        img_att = SimpleNamespace(type="img", id="mxc://beeper.local/abc")
        msg = SimpleNamespace(
            id="msg_200", sort_key="200", text="look!",
            sender_name="Alice", is_sender=False,
            timestamp="2026-03-13T00:00:00Z",
            attachments=[img_att],
        )

        poller.client.chats.list.return_value = iter([chat])
        poller.client.messages.list.return_value = iter([msg])

        result = poller.poll_once()

        assert len(result) == 1
        assert result[0]["has_attachments"] is True
        assert result[0]["attachments_raw"] == [img_att]

    def test_poll_once_empty_attachments_raw(self):
        """Messages without attachments have empty attachments_raw."""
        from src.beeper_client import BeeperPoller

        with patch("src.beeper_client.BeeperDesktop"):
            poller = BeeperPoller()
            poller.client = MagicMock()

        poller._seen = {"chat1": 100}

        chat = SimpleNamespace(
            id="chat1", title="Alice", account_id="whatsapp",
            preview=SimpleNamespace(id="msg_200", sort_key="200", text="hi",
                                     is_sender=False, sender_name="Alice",
                                     timestamp="2026-03-13T00:00:00Z", attachments=None),
            participants=None, type="single",
        )
        msg = SimpleNamespace(
            id="msg_200", sort_key="200", text="just text",
            sender_name="Alice", is_sender=False,
            timestamp="2026-03-13T00:00:00Z",
            attachments=None,
        )

        poller.client.chats.list.return_value = iter([chat])
        poller.client.messages.list.return_value = iter([msg])

        result = poller.poll_once()

        assert len(result) == 1
        assert result[0]["attachments_raw"] == []
        assert result[0]["has_attachments"] is False
