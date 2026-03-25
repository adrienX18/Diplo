"""Abstract base class for email providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EmailMessage:
    """A single email message, normalized across providers."""

    email_id: str
    thread_id: str
    mailbox: str
    subject: str
    from_name: str
    from_address: str
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    body_text: str = ""
    timestamp: str = ""  # ISO 8601 UTC
    has_attachments: bool = False
    attachment_names: list[str] = field(default_factory=list)
    is_read: bool = False
    is_from_adrien: bool = False
    in_reply_to: str | None = None
    references: str | None = None


class EmailProvider(ABC):
    """Base class for email providers (Gmail, Outlook, IMAP, etc.)."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the connection / authenticate."""

    @abstractmethod
    async def poll_new(self, since_history_id: str | None = None) -> tuple[list[EmailMessage], str | None]:
        """Fetch new emails since last poll.

        Args:
            since_history_id: Provider-specific checkpoint (e.g. Gmail historyId).
                If None, does an initial fetch of recent emails.

        Returns:
            Tuple of (list of new emails, new checkpoint value).
            The checkpoint should be passed to the next poll_new() call.
        """

    @abstractmethod
    async def send_reply(
        self,
        thread_id: str,
        to: str,
        body: str,
        subject: str | None = None,
    ) -> bool:
        """Send a threaded reply to an existing email thread.

        Args:
            thread_id: The thread to reply to.
            to: Recipient email address.
            body: Plain text body of the reply.
            subject: Optional subject override (defaults to Re: original subject).

        Returns:
            True if sent successfully.
        """

    @abstractmethod
    async def get_thread(self, thread_id: str, max_messages: int = 5) -> list[EmailMessage]:
        """Fetch the last N messages in a thread for context.

        Args:
            thread_id: The thread to fetch.
            max_messages: Maximum number of messages to return.

        Returns:
            List of messages in chronological order (oldest first).
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up resources."""
