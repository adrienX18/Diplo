"""EmailManager — orchestrates multiple email providers and their mailboxes."""

import logging

from src.email.base import EmailProvider, EmailMessage
from src.email.cache import EmailCache

logger = logging.getLogger(__name__)


class EmailManager:
    """Manages multiple email mailboxes, each backed by an EmailProvider."""

    def __init__(self, cache: EmailCache):
        self._cache = cache
        self._providers: dict[str, EmailProvider] = {}

    @property
    def has_mailboxes(self) -> bool:
        return len(self._providers) > 0

    def add_provider(self, name: str, provider: EmailProvider) -> None:
        """Register a provider for a named mailbox."""
        self._providers[name] = provider

    async def connect_all(self) -> None:
        """Connect all registered providers. Logs errors but doesn't crash."""
        for name, provider in self._providers.items():
            try:
                await provider.connect()
                logger.info("Email provider '%s' connected", name)
            except Exception:
                logger.exception("Failed to connect email provider '%s'", name)

    async def poll_all(self) -> list[dict]:
        """Poll all mailboxes for new emails.

        Returns a list of email dicts (ready for cache.store()).
        Updates history_id checkpoints per mailbox.
        """
        all_emails = []

        for name, provider in self._providers.items():
            try:
                history_id = self._cache.get_history_id(name)
                new_emails, new_history_id = await provider.poll_new(since_history_id=history_id)

                if new_history_id:
                    self._cache.set_history_id(name, new_history_id)

                for email in new_emails:
                    all_emails.append(_email_to_dict(email))

            except Exception:
                logger.exception("Failed to poll mailbox '%s'", name)

        return all_emails

    async def send_reply(self, mailbox_name: str, thread_id: str, to: str, body: str) -> bool:
        """Send a reply through a specific mailbox.

        Args:
            mailbox_name: Which mailbox to send from.
            thread_id: The thread to reply to.
            to: Recipient email address.
            body: Plain text body.

        Returns:
            True if sent successfully.
        """
        provider = self._providers.get(mailbox_name)
        if not provider:
            logger.error("No email provider for mailbox '%s'", mailbox_name)
            return False

        return await provider.send_reply(thread_id=thread_id, to=to, body=body)

    async def get_thread(self, mailbox_name: str, thread_id: str, max_messages: int = 5) -> list[EmailMessage]:
        """Fetch thread context from a specific mailbox."""
        provider = self._providers.get(mailbox_name)
        if not provider:
            logger.error("No email provider for mailbox '%s'", mailbox_name)
            return []

        return await provider.get_thread(thread_id, max_messages)

    def list_mailboxes(self) -> list[str]:
        """Return names of all connected mailboxes."""
        return list(self._providers.keys())

    async def disconnect_all(self) -> None:
        """Disconnect all providers."""
        for name, provider in self._providers.items():
            try:
                await provider.disconnect()
            except Exception:
                logger.exception("Failed to disconnect email provider '%s'", name)


def _email_to_dict(email: EmailMessage) -> dict:
    """Convert an EmailMessage dataclass to a dict suitable for cache.store()."""
    return {
        "email_id": email.email_id,
        "thread_id": email.thread_id,
        "mailbox": email.mailbox,
        "subject": email.subject,
        "from_name": email.from_name,
        "from_address": email.from_address,
        "to": email.to,
        "cc": email.cc,
        "body_text": email.body_text,
        "timestamp": email.timestamp,
        "has_attachments": email.has_attachments,
        "attachment_names": email.attachment_names,
        "is_read": email.is_read,
        "is_from_adrien": email.is_from_adrien,
    }
