"""Tests for the actions module — sending messages through Beeper."""

import pytest
from unittest.mock import MagicMock, patch

from src.actions import send_message


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_success(self):
        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        result = await send_message(client, "chat123", "hello!")

        assert result is True
        client.messages.send.assert_called_once_with("chat123", text="hello!")

    @pytest.mark.asyncio
    async def test_failure(self):
        client = MagicMock()
        client.messages.send = MagicMock(side_effect=Exception("network error"))

        result = await send_message(client, "chat123", "hello!")

        assert result is False

    @pytest.mark.asyncio
    async def test_empty_text_rejected(self):
        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        result = await send_message(client, "chat123", "")

        assert result is False
        client.messages.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_rejected(self):
        client = MagicMock()
        client.messages.send = MagicMock(return_value=None)

        result = await send_message(client, "chat123", "   \n  ")

        assert result is False
        client.messages.send.assert_not_called()
