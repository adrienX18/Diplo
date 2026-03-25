"""Abstract base class for control channels."""

from abc import ABC, abstractmethod
from typing import Callable, Awaitable


class ControlChannel(ABC):
    """Base class for control channel adapters (Telegram, SMS, WhatsApp, etc.)."""

    @abstractmethod
    async def send_notification(self, title: str, body: str) -> None:
        """Push an urgent notification to the user."""

    @abstractmethod
    async def send_message(self, text: str) -> None:
        """Send a plain message to the user."""

    @abstractmethod
    async def start(
        self,
        on_user_message: Callable[..., Awaitable[tuple[str, bool]]],
        on_reply_sent: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Start listening for incoming messages from the user.

        Args:
            on_user_message: async callback that receives the user's text
                and returns (response, queried_cache). queried_cache indicates
                whether the message cache was consulted — if False (e.g. casual
                greeting), the last_seen_at watermark should NOT be advanced.

                Accepts an optional ``on_chunk`` keyword argument: an async
                callback the adapter provides so the assistant can send response
                chunks progressively as they're generated (streaming).

                Voice messages are transcribed by the channel adapter and passed
                as text with a "[voice message]" prefix so the downstream models
                know the input was spoken, not typed.

            on_reply_sent: async callback invoked after the reply is successfully
                sent to the user. Used to update last_seen_at. Only called when
                queried_cache is True.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""
