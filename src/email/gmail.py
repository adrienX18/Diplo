"""Gmail adapter — implements EmailProvider using Google API.

Setup:
1. Create a Google Cloud project at https://console.cloud.google.com
2. Enable the Gmail API
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download the client_secret JSON and save as email_client_secret.json (project root)
5. Run: python3 -m src.email.setup --name "work"
   This opens a browser for OAuth consent and saves the token.
"""

import asyncio
import base64
import logging
import re
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

from src.config import USER_EMAIL_ADDRESSES, EMAIL_INITIAL_FETCH_LIMIT
from src.email.base import EmailProvider, EmailMessage

logger = logging.getLogger(__name__)
# How many messages to fetch per thread for context
_THREAD_CONTEXT_LIMIT = 5


def _build_service(token_path: str):
    """Build a Gmail API service object from a saved token.

    Handles token refresh automatically. Raises if the token is invalid
    or revoked (caller should notify the user to re-run setup).
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(token_path)
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Save refreshed token
            Path(token_path).write_text(creds.to_json())
            logger.info("Gmail token refreshed for %s", token_path)
        except Exception:
            logger.exception("Failed to refresh Gmail token at %s — re-run setup", token_path)
            raise
    elif not creds.valid:
        raise RuntimeError(f"Gmail token at {token_path} is invalid — re-run setup")

    return build("gmail", "v1", credentials=creds)


def _parse_email_headers(headers: list[dict]) -> dict[str, str]:
    """Extract key headers from a Gmail message's headers list."""
    result = {}
    for h in headers:
        name = h.get("name", "").lower()
        if name in ("from", "to", "cc", "subject", "date", "in-reply-to", "references", "message-id"):
            result[name] = h.get("value", "")
    return result


def _parse_from(from_str: str) -> tuple[str, str]:
    """Parse 'Display Name <email@example.com>' into (name, address)."""
    match = re.match(r'^"?([^"<]*?)"?\s*<([^>]+)>', from_str)
    if match:
        return match.group(1).strip(), match.group(2).strip().lower()
    # Bare email address
    email = from_str.strip().lower()
    return email, email


def _extract_body_text(payload: dict) -> str:
    """Extract plain text body from a Gmail message payload.

    Walks the MIME tree looking for text/plain parts. Falls back to
    stripping HTML via simple regex if only text/html is available.
    """
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if mime_type == "text/plain" and body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    if mime_type == "text/html" and body_data:
        return _strip_html(base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace"))

    # Multipart — recurse into parts
    parts = payload.get("parts", [])
    # Prefer text/plain
    for part in parts:
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    # Fall back to text/html
    for part in parts:
        if part.get("mimeType") == "text/html":
            data = part.get("body", {}).get("data")
            if data:
                return _strip_html(base64.urlsafe_b64decode(data).decode("utf-8", errors="replace"))
    # Recurse into nested multipart
    for part in parts:
        text = _extract_body_text(part)
        if text:
            return text

    return ""


def _strip_html(html: str) -> str:
    """Basic HTML to plain text conversion.

    Uses html2text if available, otherwise falls back to regex stripping.
    """
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        return h.handle(html).strip()
    except ImportError:
        # Fallback: strip tags and decode entities
        import html as html_module
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = html_module.unescape(text)
        return text.strip()


def _extract_attachments(payload: dict) -> list[str]:
    """Extract attachment filenames from a Gmail message payload."""
    names = []
    parts = payload.get("parts", [])
    for part in parts:
        filename = part.get("filename")
        if filename:
            names.append(filename)
        # Recurse into nested parts
        names.extend(_extract_attachments(part))
    return names


def _is_from_owner(address: str) -> bool:
    """Check if an email address belongs to the user."""
    return address.lower() in USER_EMAIL_ADDRESSES


def _gmail_msg_to_email(msg_data: dict, mailbox: str) -> EmailMessage:
    """Convert a raw Gmail API message to an EmailMessage.

    This is a pure data transformation — no API calls. Errors here
    mean the Gmail API response format changed.
    """
    payload = msg_data.get("payload", {})
    headers = _parse_email_headers(payload.get("headers", []))
    from_name, from_address = _parse_from(headers.get("from", ""))

    # Parse timestamp from internalDate (milliseconds since epoch)
    internal_date = msg_data.get("internalDate", "0")
    ts = datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc).isoformat()

    # Parse To and CC
    to_raw = headers.get("to", "")
    cc_raw = headers.get("cc", "")
    to_list = [a.strip() for a in to_raw.split(",") if a.strip()] if to_raw else []
    cc_list = [a.strip() for a in cc_raw.split(",") if a.strip()] if cc_raw else []

    # Attachments
    attachment_names = _extract_attachments(payload)

    # Body text
    body = _extract_body_text(payload)
    # Truncate very long emails to save context window
    if len(body) > 3000:
        body = body[:3000] + "\n\n[... truncated]"

    label_ids = msg_data.get("labelIds", [])

    return EmailMessage(
        email_id=msg_data["id"],
        thread_id=msg_data.get("threadId", msg_data["id"]),
        mailbox=mailbox,
        subject=headers.get("subject", "(no subject)"),
        from_name=from_name,
        from_address=from_address,
        to=to_list,
        cc=cc_list,
        body_text=body,
        timestamp=ts,
        has_attachments=len(attachment_names) > 0,
        attachment_names=attachment_names,
        is_read="UNREAD" not in label_ids,
        is_from_adrien=_is_from_owner(from_address),
        in_reply_to=headers.get("in-reply-to"),
        references=headers.get("references"),
    )


class GmailProvider(EmailProvider):
    """Gmail email provider using the Google API."""

    def __init__(self, mailbox_name: str, token_path: str):
        self.mailbox_name = mailbox_name
        self.token_path = token_path
        self._service = None

    async def connect(self) -> None:
        """Build the Gmail service (runs token refresh in executor)."""
        loop = asyncio.get_event_loop()
        self._service = await loop.run_in_executor(None, _build_service, self.token_path)
        logger.info("Gmail provider connected: %s", self.mailbox_name)

    async def poll_new(self, since_history_id: str | None = None) -> tuple[list[EmailMessage], str | None]:
        """Fetch new emails using Gmail history API for incremental sync.

        If since_history_id is None, does an initial fetch of recent emails.
        Returns (new_emails, new_history_id).
        """
        if not self._service:
            raise RuntimeError("Gmail provider not connected — call connect() first")

        loop = asyncio.get_event_loop()

        if since_history_id:
            return await self._poll_incremental(since_history_id, loop)
        else:
            return await self._poll_initial(loop)

    async def _poll_initial(self, loop) -> tuple[list[EmailMessage], str | None]:
        """Initial fetch — get recent emails and current historyId.

        Paginates through Gmail's messages.list API (max 500 per page)
        until EMAIL_INITIAL_FETCH_LIMIT is reached.
        """
        try:
            # Get current profile for historyId baseline
            profile = await loop.run_in_executor(
                None, lambda: self._service.users().getProfile(userId="me").execute()
            )
            history_id = str(profile.get("historyId", ""))

            # Paginate through messages.list — Gmail caps maxResults at 500
            message_ids: list[str] = []
            page_token = None
            remaining = EMAIL_INITIAL_FETCH_LIMIT

            while remaining > 0:
                page_size = min(remaining, 500)

                def _list_page(pt=page_token, ps=page_size):
                    kwargs = {"userId": "me", "maxResults": ps, "labelIds": ["INBOX"]}
                    if pt:
                        kwargs["pageToken"] = pt
                    return self._service.users().messages().list(**kwargs).execute()

                result = await loop.run_in_executor(None, _list_page)
                page_msgs = result.get("messages", [])
                message_ids.extend(m["id"] for m in page_msgs)
                remaining -= len(page_msgs)

                page_token = result.get("nextPageToken")
                if not page_token or not page_msgs:
                    break

            emails = await self._fetch_messages(message_ids, loop)

            logger.info("Gmail initial fetch: %d emails, historyId=%s", len(emails), history_id)
            return emails, history_id

        except Exception:
            logger.exception("Gmail initial fetch failed for %s", self.mailbox_name)
            return [], None

    async def _poll_incremental(self, since_history_id: str, loop) -> tuple[list[EmailMessage], str | None]:
        """Incremental fetch using Gmail history API (historyId-based)."""
        try:
            history_response = await loop.run_in_executor(
                None,
                lambda: self._service.users().history().list(
                    userId="me",
                    startHistoryId=since_history_id,
                    historyTypes=["messageAdded"],
                    labelId="INBOX",
                ).execute(),
            )

            new_history_id = str(history_response.get("historyId", since_history_id))

            # Collect all newly added message IDs
            message_ids = set()
            for record in history_response.get("history", []):
                for added in record.get("messagesAdded", []):
                    msg = added.get("message", {})
                    # Only inbox messages (skip drafts, spam, etc.)
                    if "INBOX" in msg.get("labelIds", []):
                        message_ids.add(msg["id"])

            if not message_ids:
                return [], new_history_id

            emails = await self._fetch_messages(list(message_ids), loop)
            logger.info("Gmail incremental: %d new emails, historyId %s→%s",
                        len(emails), since_history_id, new_history_id)
            return emails, new_history_id

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "historyId" in error_str.lower():
                # historyId expired — fall back to initial fetch
                logger.warning("Gmail historyId expired for %s, doing initial fetch", self.mailbox_name)
                return await self._poll_initial(loop)
            logger.exception("Gmail incremental poll failed for %s", self.mailbox_name)
            return [], since_history_id

    async def _fetch_messages(self, message_ids: list[str], loop) -> list[EmailMessage]:
        """Fetch full message data for a list of message IDs.

        Each message is fetched individually. Errors on individual messages
        are logged but don't fail the whole batch.
        """
        emails = []
        for msg_id in message_ids:
            try:
                msg_data = await loop.run_in_executor(
                    None,
                    lambda mid=msg_id: self._service.users().messages().get(
                        userId="me", id=mid, format="full"
                    ).execute(),
                )
                email = _gmail_msg_to_email(msg_data, self.mailbox_name)
                emails.append(email)
            except Exception:
                logger.warning("Failed to fetch Gmail message %s", msg_id, exc_info=True)
        return emails

    async def send_reply(
        self,
        thread_id: str,
        to: str,
        body: str,
        subject: str | None = None,
    ) -> bool:
        """Send a reply in an existing email thread."""
        if not self._service:
            raise RuntimeError("Gmail provider not connected")

        loop = asyncio.get_event_loop()

        try:
            # Get the original thread for subject and references
            thread_data = await loop.run_in_executor(
                None,
                lambda: self._service.users().threads().get(
                    userId="me", id=thread_id, format="metadata",
                    metadataHeaders=["Subject", "Message-ID", "References"],
                ).execute(),
            )

            messages = thread_data.get("messages", [])
            if not messages:
                logger.error("Empty thread %s — cannot reply", thread_id)
                return False

            last_msg = messages[-1]
            last_headers = _parse_email_headers(last_msg.get("payload", {}).get("headers", []))

            if not subject:
                orig_subject = last_headers.get("subject", "")
                subject = f"Re: {orig_subject}" if not orig_subject.lower().startswith("re:") else orig_subject

            # Build MIME message
            mime_msg = MIMEText(body)
            mime_msg["to"] = to
            mime_msg["subject"] = subject

            # Thread headers
            msg_id = last_headers.get("message-id", "")
            if msg_id:
                mime_msg["In-Reply-To"] = msg_id
                refs = last_headers.get("references", "")
                mime_msg["References"] = f"{refs} {msg_id}".strip()

            raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()

            await loop.run_in_executor(
                None,
                lambda: self._service.users().messages().send(
                    userId="me",
                    body={"raw": raw, "threadId": thread_id},
                ).execute(),
            )

            logger.info("Sent reply in thread %s to %s", thread_id, to)
            return True

        except Exception:
            logger.exception("Failed to send Gmail reply in thread %s", thread_id)
            return False

    async def get_thread(self, thread_id: str, max_messages: int = _THREAD_CONTEXT_LIMIT) -> list[EmailMessage]:
        """Fetch the last N messages in a thread for triage context."""
        if not self._service:
            raise RuntimeError("Gmail provider not connected")

        loop = asyncio.get_event_loop()

        try:
            thread_data = await loop.run_in_executor(
                None,
                lambda: self._service.users().threads().get(
                    userId="me", id=thread_id, format="full"
                ).execute(),
            )

            messages = thread_data.get("messages", [])
            # Take the last N messages
            messages = messages[-max_messages:]

            result = []
            for msg_data in messages:
                try:
                    email = _gmail_msg_to_email(msg_data, self.mailbox_name)
                    result.append(email)
                except Exception:
                    logger.warning("Failed to parse thread message", exc_info=True)

            return result

        except Exception:
            logger.exception("Failed to fetch Gmail thread %s", thread_id)
            return []

    async def disconnect(self) -> None:
        """Close the Gmail service."""
        self._service = None
        logger.info("Gmail provider disconnected: %s", self.mailbox_name)
