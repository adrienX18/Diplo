"""Diplo — async main loop with polling, triage, and Telegram control channel."""

import asyncio
import logging
import os
import signal
import tempfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from src.beeper_client import BeeperPoller
from src.triage import classify_urgency
from src.message_cache import MessageCache
from src.conversation import ConversationHistory
from src.contacts import ContactRegistry
from src.channels.telegram import TelegramChannel
from src.assistant import handle_user_message, _display_network
from src.config import POLL_INTERVAL_SECONDS, BOT_SENDER_NAMES, URGENT_BATCH_DELAY_SECONDS, PRUNE_INTERVAL_SECONDS, SCHEDULER_INTERVAL_SECONDS, BEEPER_ACCESS_TOKEN, GOOGLE_CALENDAR_CREDENTIALS, GOOGLE_CALENDAR_TOKEN, EMAIL_POLL_INTERVAL, USER_EMAIL_ADDRESSES
from src.feedback import run_consolidation
from src.automations import AutomationStore, run_scheduler_tick, format_delay
from src.llm import describe_image, transcribe_audio, complete
from src.llm_logger import init_logger, get_logger as get_llm_logger
from src.calendar import CalendarManager
from src.email.cache import EmailCache
from src.email.manager import EmailManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)

_BEEPER_BASE_URL = "http://localhost:23373"


async def _download_beeper_asset(mxc_url: str) -> str | None:
    """Download a Beeper asset and return the local file path, or None on failure."""
    import httpx

    headers = {"Authorization": f"Bearer {BEEPER_ACCESS_TOKEN}"}
    resp = httpx.post(
        f"{_BEEPER_BASE_URL}/v1/assets/download",
        json={"url": mxc_url},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    src_url = data.get("srcURL") or data.get("src_url") or data.get("srcUrl") or data.get("url") or ""

    if src_url.startswith("file://"):
        local_path = urllib.parse.unquote(src_url[7:])  # decode %20 etc.
    else:
        local_path = src_url

    if not local_path or not os.path.exists(local_path):
        logger.warning("Asset download returned invalid path: %r (full response: %s)", src_url, data)
        return None
    return local_path


async def _process_media_attachments(msg: dict) -> None:
    """Download and process image/audio attachments, prepending descriptions to msg text.

    Modifies msg["text"] in place.
    - Images get a [image: description] tag (via Claude vision).
    - Audio gets a [voice message: transcript] tag (via OpenAI transcription).
    On failure, falls back to [image] or [voice message] with no content.
    """
    attachments = msg.pop("attachments_raw", [])
    image_ids = []
    audio_ids = []
    for att in attachments:
        att_type = att.get("type") if isinstance(att, dict) else getattr(att, "type", None)
        att_id = att.get("id") if isinstance(att, dict) else getattr(att, "id", None)
        if not att_id:
            continue
        if att_type == "img":
            image_ids.append(att_id)
        elif att_type == "audio":
            audio_ids.append(att_id)

    if not image_ids and not audio_ids:
        return

    descriptions = []

    # Process images
    for mxc_url in image_ids:
        try:
            local_path = await _download_beeper_asset(mxc_url)
            if not local_path:
                descriptions.append("[image]")
                continue
            description = await describe_image(local_path)
            descriptions.append(f"[image: {description}]")
        except Exception:
            logger.warning("Failed to describe image attachment %s", mxc_url, exc_info=True)
            descriptions.append("[image]")

    # Process audio
    for mxc_url in audio_ids:
        try:
            local_path = await _download_beeper_asset(mxc_url)
            if not local_path:
                descriptions.append("[voice message]")
                continue
            transcript = await transcribe_audio(local_path)
            descriptions.append(f"[voice message: {transcript}]")
        except Exception:
            logger.warning("Failed to transcribe audio attachment %s", mxc_url, exc_info=True)
            descriptions.append("[voice message]")

    # Prepend media descriptions to message text
    prefix = " ".join(descriptions)
    existing_text = msg.get("text") or ""
    msg["text"] = f"{prefix} {existing_text}".strip() if existing_text else prefix


async def _summarize_email_thread(thread_context: list[dict], new_email: dict) -> str:
    """Summarize an urgent email thread for the notification.

    Uses Sonnet for a quick 1-2 sentence summary. Falls back to
    subject + body excerpt if the LLM call fails.
    """
    subject = new_email.get("subject", "(no subject)")
    body = new_email.get("body_text", "")
    sender = new_email.get("from_name") or new_email.get("from_address", "")

    # Build thread context for the LLM
    thread_lines = []
    for e in thread_context:
        e_sender = e.get("from_name") or e.get("from_address", "")
        e_body = (e.get("body_text") or "")[:500]
        thread_lines.append(f"From {e_sender}: {e_body}")

    thread_text = "\n---\n".join(thread_lines) if thread_lines else "(no prior thread)"

    try:
        summary = await complete(
            model="claude-sonnet-4-6",
            system="Summarize this urgent email thread in 1-3 concise sentences for a mobile notification. Include what's being asked or what action is needed. No preamble.",
            messages=[{"role": "user", "content": f"Subject: {subject}\nFrom: {sender}\n\nThread:\n{thread_text}"}],
            max_tokens=200,
            call_type="email_urgent_summary",
        )
        return f"{subject}\n\n{summary}"
    except Exception:
        logger.warning("Failed to summarize urgent email — falling back to excerpt", exc_info=True)
        excerpt = body[:500] + ("..." if len(body) > 500 else "")
        return f"{subject}\n\n{excerpt}" if excerpt else subject


# Buffered urgent messages per chat_id, waiting to be sent as a batch
# chat_id -> list of (title, body) tuples
_urgent_buffer: dict[str, list[tuple[str, str]]] = {}
# chat_id -> scheduled asyncio.Task that will flush after the delay
_urgent_timers: dict[str, asyncio.Task] = {}


async def _flush_urgent(chat_id: str, channel: TelegramChannel, convo: ConversationHistory):
    """Wait for the batch delay, then send all buffered urgent messages for a chat."""
    await asyncio.sleep(URGENT_BATCH_DELAY_SECONDS)

    items = _urgent_buffer.pop(chat_id, [])
    _urgent_timers.pop(chat_id, None)

    if not items:
        return

    try:
        if len(items) == 1:
            title, body = items[0]
            await channel.send_notification(title, body)
            convo.add_turn("assistant", f"[Urgent notification] {title}: {body}")
        else:
            # Multiple urgent messages from the same chat — send one combined notification
            title = items[0][0]  # Use first message's title (same chat)
            body = "\n".join(f"• {b}" for _, b in items)
            await channel.send_notification(f"{title} ({len(items)} messages)", body)
            convo.add_turn("assistant", f"[Urgent notification] {title} ({len(items)} messages): {body}")
    except Exception:
        logger.exception("Failed to send urgent notification for chat %s", chat_id)


def _buffer_urgent(chat_id: str, title: str, body: str, channel: TelegramChannel, convo: ConversationHistory):
    """Buffer an urgent message and schedule/reset the flush timer."""
    if chat_id not in _urgent_buffer:
        _urgent_buffer[chat_id] = []
    _urgent_buffer[chat_id].append((title, body))

    # Cancel existing timer for this chat (resets the 20s window)
    existing = _urgent_timers.get(chat_id)
    if existing and not existing.done():
        existing.cancel()

    _urgent_timers[chat_id] = asyncio.create_task(
        _flush_urgent(chat_id, channel, convo)
    )


async def run_email_poller(
    email_manager: EmailManager,
    email_cache: EmailCache,
    contacts: ContactRegistry,
    channel: TelegramChannel,
    convo: ConversationHistory,
):
    """Poll all email mailboxes on a fixed interval."""
    first_run = True
    while True:
        # First poll after 5s (let startup finish), then normal interval
        await asyncio.sleep(5 if first_run else EMAIL_POLL_INTERVAL)
        first_run = False
        try:
            new_emails = await email_manager.poll_all()

            for email in new_emails:
                email_cache.store(email)

                # Update contact registry (sender → email address, not thread_id)
                from_name = email.get("from_name", "")
                from_address = email.get("from_address", "")
                mailbox = email.get("mailbox", "")
                network = f"email:{mailbox}"
                # Use email address as a more meaningful display name if name is empty
                sender_display = from_name or from_address

                contacts.update(
                    sender_name=sender_display,
                    network=network,
                    chat_id=email["thread_id"],
                    chat_title=email.get("subject", "(no subject)"),
                    timestamp=email["timestamp"],
                )

                # Skip triage for the user's own emails
                if email.get("is_from_adrien"):
                    continue

                # Triage — use thread context from email cache
                thread_context = email_cache.by_thread(email["thread_id"], limit=5)
                # Convert email dicts to message-like dicts for triage
                triage_context = [
                    {
                        "sender_name": e.get("from_name") or e.get("from_address", ""),
                        "text": f"[Subject: {e.get('subject', '')}] {e.get('body_text', '')}",
                        "timestamp": e["timestamp"],
                        "network": f"email:{e.get('mailbox', '')}",
                        "chat_title": e.get("subject", ""),
                    }
                    for e in thread_context
                ]

                triage_msg = {
                    "sender_name": sender_display,
                    "text": f"[Subject: {email.get('subject', '')}] {email.get('body_text', '')}",
                    "timestamp": email["timestamp"],
                    "network": network,
                    "chat_title": email.get("subject", ""),
                }

                is_urgent = await classify_urgency(triage_msg, conversation_context=triage_context)

                subject_preview = (email.get("subject") or "(no subject)")[:80]
                body_preview = (email.get("body_text") or "")[:80]
                tag = "URGENT" if is_urgent else "not urgent"
                logger.info("[%s] [%s] %s — %s: %s", tag, network, subject_preview, sender_display, body_preview)

                if is_urgent:
                    title = f"(email:{mailbox}) {sender_display}"
                    body = await _summarize_email_thread(thread_context, email)
                    _buffer_urgent(email["thread_id"], title, body, channel, convo)

        except Exception:
            logger.exception("Email poll failed")


async def _run_scheduler(automations: AutomationStore, handler, channel: TelegramChannel, cache: MessageCache):
    """Run the automation scheduler loop — checks for due automations every 30s."""
    while True:
        await asyncio.sleep(SCHEDULER_INTERVAL_SECONDS)
        try:
            tz_name = cache.get_timezone()
            await run_scheduler_tick(automations, handler, channel, tz_name)
        except Exception:
            logger.exception("Scheduler tick failed")


async def run_poller(poller: BeeperPoller, cache: MessageCache, contacts: ContactRegistry, channel: TelegramChannel, convo: ConversationHistory, automations: AutomationStore | None = None):
    """Run the Beeper poller in a thread, process results async."""
    loop = asyncio.get_event_loop()

    # Load persisted watermarks, then seed new chats from Beeper
    persisted = cache.load_watermarks()
    await loop.run_in_executor(_executor, poller.seed_watermarks, persisted)
    logger.info("Listening for new messages. Press Ctrl+C to stop.\n")

    consecutive_failures = 0
    beeper_down_notified = False
    last_saved_watermarks = dict(poller._seen)

    while True:
        try:
            new_messages = await loop.run_in_executor(_executor, poller.poll_once)

            # Beeper is back — reset failure state
            if consecutive_failures > 0:
                logger.info("Beeper connection restored after %d failed polls", consecutive_failures)
                if beeper_down_notified:
                    try:
                        await channel.send_message("Beeper connection restored. I'm watching your messages again.")
                    except Exception:
                        pass
                consecutive_failures = 0
                beeper_down_notified = False

            for msg in new_messages:
                # Skip control channel messages entirely — they're meta-conversation
                # between the user and the bot, not real messages to cache or triage
                if _is_control_channel(msg):
                    continue

                # Process media attachments before caching (prepends [image: ...] / [voice message: ...] to text)
                if msg.get("attachments_raw"):
                    await _process_media_attachments(msg)
                else:
                    msg.pop("attachments_raw", None)  # clean up the key

                cache.store(msg)
                contacts.update(
                    sender_name=msg["sender_name"],
                    network=msg["network"],
                    chat_id=msg["chat_id"],
                    chat_title=msg["chat_title"],
                    timestamp=msg["timestamp"],
                )

                # Triage — use local cache for context (instant, no Beeper API call)
                context = cache.by_chat_id(msg["chat_id"], limit=20)
                is_urgent = await classify_urgency(msg, conversation_context=context)

                text_preview = (msg["text"] or "(no text)")[:120]
                attachment_tag = " [+attachment]" if msg["has_attachments"] else ""
                tag = "URGENT" if is_urgent else "not urgent"

                logger.info(
                    "[%s] [%s] %s — %s: %s%s",
                    tag, msg["network"], msg["chat_title"],
                    msg["sender_name"], text_preview, attachment_tag,
                )

                # Buffer urgent notification — sends after 20s of quiet from this chat
                if is_urgent:
                    title = f"({_display_network(msg['network'])}) {msg['sender_name']} in {msg['chat_title']}"
                    full_text = msg["text"] or "(no text)"
                    _buffer_urgent(msg["chat_id"], title, full_text, channel, convo)

                # Evaluate triggered automations
                if automations:
                    try:
                        triggered = automations.evaluate_triggers(msg)
                        for auto in triggered:
                            auto_desc = auto["description"]
                            auto_id = auto["id"]
                            logger.info("Trigger fired #%d: %s", auto_id, auto_desc)
                            automations.mark_triggered(auto_id)

                            delay = automations.get_delay_seconds(auto)
                            if delay > 0:
                                # Delayed trigger — create (or reset) a one-shot
                                cancelled = automations.cancel_pending_oneshots(auto_id)
                                automations.create_delayed(
                                    parent_id=auto_id,
                                    delay_seconds=delay,
                                    action=auto["action"],
                                    description=f"[Delayed] {auto_desc}",
                                )
                                verb = "reset" if cancelled > 0 else "set"
                                delay_desc = format_delay(delay)
                                try:
                                    await channel.send_message(
                                        f"[Trigger] {auto_desc} — timer {verb} ({delay_desc})"
                                    )
                                except Exception:
                                    logger.exception("Failed to send trigger timer notification #%d", auto_id)
                            else:
                                # Immediate trigger — just notify
                                try:
                                    await channel.send_message(
                                        f"Heads up — {auto_desc}\n\n"
                                        f"{msg['sender_name']} ({_display_network(msg['network'])}): {text_preview}"
                                    )
                                except Exception:
                                    logger.exception("Failed to send trigger notification #%d", auto_id)
                    except Exception:
                        logger.exception("Trigger evaluation failed")

        except Exception:
            consecutive_failures += 1
            logger.exception("Error during poll cycle (failure #%d)", consecutive_failures)

            # Notify user on 3rd consecutive failure (avoids noise from transient blips)
            if consecutive_failures == 3 and not beeper_down_notified:
                try:
                    await channel.send_message("I lost connection to Beeper. I'll keep retrying and let you know when it's back.")
                    beeper_down_notified = True
                except Exception:
                    logger.exception("Failed to send Beeper-down notification")

        # Persist watermarks only when they've changed
        if poller._seen != last_saved_watermarks:
            cache.save_watermarks(poller._seen)
            last_saved_watermarks = dict(poller._seen)

        # Back off on repeated failures: 5s -> 10s -> 20s -> 30s max
        if consecutive_failures > 0:
            backoff = min(POLL_INTERVAL_SECONDS * (2 ** consecutive_failures), 30)
            await asyncio.sleep(backoff)
        else:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _run_periodic_tasks(cache: MessageCache, convo: ConversationHistory, email_cache: EmailCache | None = None):
    """Run pruning and feedback consolidation on a fixed schedule, independent of the poller."""
    while True:
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
        try:
            cache.prune()
            convo.prune()
            if email_cache:
                email_cache.prune()
        except Exception:
            logger.exception("Prune failed")
        try:
            await run_consolidation()
        except Exception:
            logger.exception("Feedback consolidation failed")
        try:
            _llm_log = get_llm_logger()
            if _llm_log:
                _llm_log.prune()
        except Exception:
            logger.exception("LLM log prune failed")


async def main():
    logger.info("Diplo — Listening + Triage + Telegram + Reply + Feedback + Gmail + Calendar")

    llm_log = init_logger()
    poller = BeeperPoller()
    cache = MessageCache()
    contacts = ContactRegistry()
    contacts.seed_from_cache()  # Populate contacts from existing message cache
    convo = ConversationHistory()
    channel = TelegramChannel()
    automations = AutomationStore()

    # Set up calendar integration (optional — auto-detects credentials)
    calendar = CalendarManager()
    cal_creds = GOOGLE_CALENDAR_CREDENTIALS or ""
    # Auto-detect: try explicit env var, then dedicated file, then shared email client secret
    for candidate in [cal_creds, "google_calendar_credentials.json", "email_client_secret.json"]:
        if candidate and os.path.exists(candidate):
            cal_creds = candidate
            break
    else:
        cal_creds = ""
    if cal_creds:
        try:
            from src.calendar.google import GoogleCalendarProvider
            token_path = GOOGLE_CALENDAR_TOKEN or "google_calendar_token.json"
            google_cal = GoogleCalendarProvider(
                credentials_path=cal_creds,
                token_path=token_path,
            )
            calendar.add_provider(google_cal)
            logger.info("Google Calendar provider loaded (credentials: %s)", cal_creds)
        except Exception:
            logger.exception("Failed to load Google Calendar provider — continuing without calendar")
    else:
        logger.info("No Google Calendar credentials found — calendar features disabled")

    # Set up email integration (optional — only if mailboxes are configured)
    email_cache = EmailCache()
    email_manager = EmailManager(email_cache)
    mailboxes = email_cache.list_mailboxes()
    for mb in mailboxes:
        if not mb.get("enabled"):
            continue
        provider_type = mb.get("provider", "gmail")
        if provider_type == "gmail":
            try:
                from src.email.gmail import GmailProvider
                token_path = mb.get("token_path", "")
                if not token_path or not os.path.exists(token_path):
                    logger.warning("Email mailbox '%s' — token not found at %s, skipping", mb["name"], token_path)
                    continue
                gmail = GmailProvider(
                    mailbox_name=mb["name"],
                    token_path=token_path,
                )
                email_manager.add_provider(mb["name"], gmail)
                logger.info("Email provider loaded: %s (%s)", mb["name"], mb.get("email_address", ""))
            except Exception:
                logger.exception("Failed to load email provider '%s' — continuing without it", mb["name"])
        else:
            logger.warning("Unknown email provider type '%s' for mailbox '%s'", provider_type, mb["name"])

    if email_manager.has_mailboxes:
        try:
            await email_manager.connect_all()
        except Exception:
            logger.exception("Failed to connect email providers")
    else:
        logger.info("No email mailboxes configured — email features disabled")

    # Backfill last 48h of messages from Beeper into the local cache.
    # Runs before the poller so the cache has recent history even after
    # a cold start or long downtime. cache.store() uses INSERT OR IGNORE,
    # so messages already in the cache are silently skipped (no duplicates).
    loop = asyncio.get_event_loop()
    try:
        backfill_msgs = await loop.run_in_executor(_executor, poller.backfill_recent)
        backfill_stored = 0
        for msg in backfill_msgs:
            if _is_control_channel(msg):
                continue
            cache.store(msg)
            contacts.update(
                sender_name=msg["sender_name"],
                network=msg["network"],
                chat_id=msg["chat_id"],
                chat_title=msg["chat_title"],
                timestamp=msg["timestamp"],
            )
            backfill_stored += 1
        logger.info("Backfill complete: %d messages fetched, %d stored (after control channel filter)",
                     len(backfill_msgs), backfill_stored)
    except Exception:
        logger.exception("Backfill failed — continuing without it")

    # Fix stale watermarks for chats that previously failed to fetch (e.g.
    # iMessage 404s). These chats had their watermarks advanced to the
    # preview sort_key despite no messages being cached. Now that backfill
    # has stored their messages, we need to remove the stale watermarks so
    # seed_watermarks() re-seeds them from the current preview, and the
    # poller picks up new messages going forward.
    stale_watermarks = cache.load_watermarks()
    reset_ids = [cid for cid in stale_watermarks if "#" in cid]
    if reset_ids:
        cache.delete_watermarks(reset_ids)
        logger.info("Reset %d stale watermarks for raw-HTTP chats (e.g. iMessage)", len(reset_ids))

    async def on_user_message(text: str, on_chunk=None) -> str:
        return await handle_user_message(
            text, cache, convo, contacts=contacts, beeper_client=poller.client,
            automations=automations, calendar=calendar,
            email_cache=email_cache if email_manager.has_mailboxes else None,
            email_manager=email_manager if email_manager.has_mailboxes else None,
            on_chunk=on_chunk,
        )

    async def on_reply_sent():
        cache.touch_last_seen()

    # Start Telegram bot
    await channel.start(on_user_message, on_reply_sent)

    # Send startup summary
    try:
        new_msgs = cache.since_last_seen()
        count = len(new_msgs)
        active_autos = [a for a in automations.list_all() if a["enabled"]]
        auto_note = f" {len(active_autos)} automation(s) active." if active_autos else ""
        if count > 0:
            summary = _startup_summary(new_msgs)
            await channel.send_message(f"I'm online. {count} messages since we last spoke.{auto_note}\n{summary}")
        else:
            await channel.send_message(f"I'm online.{auto_note}" if auto_note else "I'm online.")
        # Surface any chats where message fetching fails (e.g. iMessage 404s)
        error_summary = poller.get_fetch_error_summary()
        if error_summary:
            await channel.send_message(error_summary)
    except Exception:
        logger.exception("Failed to send startup message")

    # Handle SIGTERM (from launchd/systemd/docker) the same as Ctrl+C
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(channel, cache, convo, contacts, automations, email_cache=email_cache, email_manager=email_manager)))

    # Start periodic tasks (prune + consolidation) as a separate coroutine
    periodic_task = asyncio.create_task(_run_periodic_tasks(cache, convo, email_cache=email_cache))

    # Start automation scheduler — checks for due scheduled automations every 30s
    async def _automation_handler(action_text: str) -> tuple[str, bool]:
        return await handle_user_message(
            action_text, cache, convo=None, contacts=contacts, beeper_client=poller.client,
            automations=automations, calendar=calendar,
            email_cache=email_cache if email_manager.has_mailboxes else None,
            email_manager=email_manager if email_manager.has_mailboxes else None,
        )

    scheduler_task = asyncio.create_task(
        _run_scheduler(automations, _automation_handler, channel, cache)
    )

    # Start email poller if any mailboxes are configured
    email_poller_task = None
    if email_manager.has_mailboxes:
        email_poller_task = asyncio.create_task(
            run_email_poller(email_manager, email_cache, contacts, channel, convo)
        )

    try:
        # Run poller alongside Telegram bot
        await run_poller(poller, cache, contacts, channel, convo, automations)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        periodic_task.cancel()
        scheduler_task.cancel()
        if email_poller_task:
            email_poller_task.cancel()
        await _shutdown(channel, cache, convo, contacts, automations, email_cache=email_cache, email_manager=email_manager)


_shutting_down = False


async def _shutdown(channel: TelegramChannel, cache: MessageCache, convo: ConversationHistory, contacts: ContactRegistry | None = None, automations: AutomationStore | None = None, email_cache: EmailCache | None = None, email_manager: EmailManager | None = None):
    """Gracefully shut down all components."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    logger.info("Shutting down...")
    await channel.stop()
    cache.close()
    convo.close()
    if contacts:
        contacts.close()
    if automations:
        automations.close()
    if email_manager:
        await email_manager.disconnect_all()
    if email_cache:
        email_cache.close()
    _llm_log = get_llm_logger()
    if _llm_log:
        _llm_log.close()
    logger.info("Shutdown complete.")
    # Stop the event loop
    asyncio.get_event_loop().stop()


def _startup_summary(messages: list[dict]) -> str:
    """Build a short tl;dr of who messaged, grouped by chat."""
    from collections import Counter
    chat_counts = Counter(msg["chat_title"] for msg in messages)
    top = chat_counts.most_common(5)
    parts = [f"{title} ({n})" for title, n in top]
    remaining = len(chat_counts) - len(top)
    line = ", ".join(parts)
    if remaining > 0:
        line += f", +{remaining} more"
    return line


def _is_control_channel(msg: dict) -> bool:
    """Check if a message is from the bot's own control channel chat."""
    sender = msg.get("sender_name", "").lower()
    chat = msg.get("chat_title", "").lower()
    return any(name in sender or name in chat for name in BOT_SENDER_NAMES)


if __name__ == "__main__":
    asyncio.run(main())
