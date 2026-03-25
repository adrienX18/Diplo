# Diplo — The Full Guide

Diplo is an always-on Python agent running on a Mac Mini. It connects to Beeper Desktop (which aggregates all your messaging platforms into one local API), classifies every message for urgency using Claude, and talks to you through a Telegram bot. You interact with it in plain English — no commands, no menus.

## Architecture at a glance

```
Beeper Desktop (localhost:23373)
  ↕ polls every 5s
Python agent (single async event loop)
  ├── Poller — watches all chats across all networks
  ├── Triage — Sonnet 4.6 classifies urgency
  ├── Assistant — Opus 4.6 answers questions, composes replies
  ├── Email poller — Gmail API, every 20s
  ├── Scheduler — cron-based automations, every 30s
  ├── Feedback — file-based, hourly Opus consolidation
  └── Control channel — Telegram bot
        ↕
      Your phone
```

Everything runs in one `asyncio` event loop. The Beeper SDK is synchronous, so it's wrapped in `run_in_executor`. All LLM calls go through a single abstraction (`src/llm.py`) that retries once on Claude failure, then falls back to OpenAI gpt-4.1.

---

## Message monitoring & triage

### How messages flow in

Diplo polls Beeper's local API every 5 seconds, checking the 30 most recently active chats. Each chat has a "preview" (the latest message's sort key) — if it hasn't changed since the last poll, that chat is skipped entirely. When new messages are found, they're stored in a local SQLite cache and triaged for urgency.

On startup, Diplo backfills the last 48 hours of messages from Beeper's search API so the cache has history even after a cold start. Beeper's search doesn't index iMessage, so those chats are backfilled separately via raw HTTP.

### Watermarks

Each chat has a watermark (the sort key of the last seen message). These are persisted to SQLite, so restarting Diplo doesn't re-process old messages. iMessage chat IDs contain `#` characters that break the Beeper SDK's URL encoding, so those chats use raw HTTP with `urllib.parse.quote()` instead.

### Urgency classification

Every message gets a one-word verdict from Sonnet 4.6: `URGENT` or `NOT`. The system prompt encodes rules like:

- Explicit urgency language ("ASAP", "need this now") → always urgent, regardless of sender
- Lawyers, investors, close collaborators with action items → urgent
- Casual chat, group banter, marketing, acknowledgments → not urgent
- Messages older than 48 hours → automatically not urgent (no API call)

The last 20 messages from the same chat are included as context, so Sonnet can judge a "yes" that's confirming a time-sensitive deal differently from a "yes" in casual conversation.

Emails are triaged with a higher bar — only fundraising, legal, hard deadlines, or genuine crises count as urgent for email.

### Urgent notifications

Urgent messages push to your Telegram automatically. They're batched per chat with a 20-second delay — if Sophie sends 3 urgent messages in a row, you get one combined notification instead of three. The timer resets on each new message from the same chat.

### Feedback loop protection

Diplo's own Telegram chat appears in Beeper. Without filtering, an urgent notification containing "ASAP" would get re-triaged as urgent → infinite loop. Fix: the control channel is detected by matching sender/chat names against known bot names and excluded from both caching and triage entirely.

---

## Talking to Diplo

### Intent classification (two-stage)

When you message Diplo, it goes through two LLM calls:

1. **Router** (Sonnet, ~10 tokens) — classifies your message into one of 7 intent types: `query`, `reply`, `automation`, `feedback`, `casual`, `timezone`, `debug`
2. **Extractor** (Sonnet, ~200 tokens) — extracts structured parameters for that intent type (e.g., sender name, time range, cron schedule)

Then Opus generates the actual response. This two-stage approach keeps the expensive Opus call focused — it only runs after cheap Sonnet calls have figured out what's being asked.

### Summaries & search

"What's new?" returns messages since your last interaction. "What did Sophie say?" filters by sender. "Anything about fundraising on Slack this week?" combines sender, keyword, network, and time range filters. All queries hit the local SQLite cache — instant, no Beeper API calls.

Key behaviors:
- **"What's new?"** uses a `last_seen_at` watermark that only advances after Diplo sends you a cache-based reply. Casual greetings ("hey!") don't advance it.
- **Insistence handling**: If Diplo says "nothing new" and you say "check again" or "are you sure?", it automatically widens to a 1-hour lookback instead of repeating the same empty query. You can go further: "go back 3 hours."
- **Full conversation context**: Asking about a sender returns both sides of the conversation (their messages + yours), not just theirs.
- **Network filter**: Natural names work — "messenger", "insta", "whatsapp", "imessage" — mapped to Beeper's internal IDs.

### Conversation memory

Diplo stores the last 30 turns of your conversation (auto-pruned at 48 hours). This is injected into both the search plan extractor and response generator, so follow-ups work naturally: "what about him?", "tell me more", "the one from yesterday."

Sessions are detected by gaps: if you haven't messaged for 5+ minutes, it's a new session. The response generator sees the full current session (un-truncated) so Diplo doesn't repeat itself.

### Input formats

- **Text**: plain messages, delivered as-is
- **Voice**: Telegram voice notes or audio files → downloaded, transcribed via OpenAI gpt-4o-transcribe, passed as `[voice message] {transcript}`
- **Photos**: Telegram photos (with optional captions) → described via Claude vision, passed as `[image: {description}] {caption}`

### Streaming

Opus responses are streamed paragraph-by-paragraph. As each ~500-character chunk completes, it's sent to Telegram immediately. You see the first part of a long answer while the rest is still generating.

### Personality

Diplo is your hype man — calls you "boss", "chief", "bro", pushes you toward world domination, cracks jokes. Greetings get greetings back, not unsolicited summaries. Summaries are ultra-concise (important stuff with detail first, then just names for the rest). Limited emoji set: 🚀🔥💡💰👊💪🎯🏆⚡🤝.

---

## Replying on behalf

"Tell Sophie I'll be late" → Diplo composes a natural message matching the conversation's tone and language, then either sends directly or asks you to confirm first.

### How it works

1. **Sonnet** extracts intent: recipient name, message content, optional network
2. **Contact resolution**: fuzzy matches the name in a persistent contact registry (SQLite, updated on every polled message). Picks the most recently used platform unless you specify one ("tell Sophie on WhatsApp")
3. **Opus** composes the message and decides send vs. confirm:
   - Send directly: short, clear, low-stakes, near-exact wording
   - Confirm first: long/complex, sensitive context, significant interpretation needed
4. **Confirmation flow**: you can confirm, modify ("make it shorter"), cancel, or just ignore (pending actions expire after 10 minutes)

### Safety guards

- Diplo never messages you via Beeper — it only talks to you on Telegram. "Tell me", "notify me" are caught and treated as queries, not reply intents.
- Diplo never sends to group chats. If a contact is only found in a group, it searches Beeper for their DM chat first.
- Sent messages are cached so the next compose prompt includes them in conversation context.

---

## Feedback & learning

Diplo gets better the more you use it. Just tell it what you think — in plain language, whenever you want:

- "That wasn't urgent" — adjusts urgency calibration
- "Always prioritize messages from Sophie" — learns sender importance
- "Your summaries are too long" — changes how it formats responses
- "Don't bother me about marketing emails" — updates triage rules
- "When I say 'what's new' I mean since we last talked, not the last 24 hours" — learns your vocabulary

Diplo detects feedback automatically (you don't need to label it — just say it), stores it, and acknowledges with a short "Noted" / "Got it."

Every hour, the feedback is consolidated: Opus reads all pending feedback alongside the current rules and base prompts, and distills everything into a compact, updated ruleset (`prompts/learned_rules.md`, max 30 lines). These learned rules are injected into both triage and response prompts at runtime — so feedback you give about urgency affects triage, and feedback about summaries affects how Diplo talks to you.

The base system prompts are never modified — learned rules supplement them. Old granular feedback gets synthesized into general rules over time, so the context doesn't grow unboundedly. If consolidation fails for any reason, feedback is preserved for the next cycle.

---

## Automations

### Scheduled tasks

"Every morning at 9am, summarize my messages" → Diplo translates to a cron expression, saves it, and runs it automatically. The action ("summarize my messages") goes through the exact same assistant pipeline as if you'd typed it. Results are sent to Telegram prefixed with `[Auto]`.

More examples:
- "Every Friday at 5pm, tell me who I haven't replied to"
- "Every 2 hours, check for messages from Sophie"

Schedules are timezone-aware — "9am" means 9am in whatever timezone you've told Diplo.

### Triggered tasks

"Whenever Sophie messages, notify me" → fires on every message matching the conditions. Supports sender, keyword, chat name, and network filters (AND logic). Default 5-minute cooldown to prevent spam.

Triggers can have delays: "If Sophie messages about the contract and I don't reply within 30 minutes, remind me" → creates a timer that resets each time Sophie messages again.

### Management

All via natural language: "show my automations", "pause the morning summary", "delete automation #3."

Under the hood, both types live in a single SQLite table. The scheduler loop ticks every 30 seconds. Automation runs never advance your `last_seen_at` watermark.

---

## Email integration

Email is built on a provider-based architecture: an abstract `EmailProvider` base class with a concrete Gmail implementation, and an `EmailManager` that orchestrates multiple mailboxes. Only Gmail is implemented today, but adding Outlook, IMAP, or any other email service means writing a single adapter class — the rest of the system (caching, triage, querying, replies) works unchanged.

### Setup (Gmail)

1. Enable the Gmail API in Google Cloud Console
2. Download OAuth credentials as `email_client_secret.json`
3. Run `python3 -m src.email.setup --name "work"` — opens a browser for consent, saves token
4. Add your email addresses to `.env` as `USER_EMAIL_ADDRESSES`

You can add multiple mailboxes (work, personal, etc.), each with its own OAuth token.

### How it works

- Polls each mailbox every 20 seconds using Gmail's `historyId`-based incremental sync (efficient — only fetches changes since the last poll)
- New emails stored in a separate SQLite database (`data/emails.db`)
- Each email is triaged for urgency (higher bar than messages — newsletters and routine updates are never urgent)
- Thread context (last 5 emails in the same thread) is included in triage so Sonnet can judge urgency in context
- Urgent emails get a Sonnet-generated summary in the notification

### Querying email

Email is opt-in in queries — "what's new?" only shows messages unless you explicitly mention email:
- "Check my email"
- "Any emails from Sophie?"
- "Emails about the contract"
- "Last 5 emails"

### Email replies

"Reply to Sophie's email saying I'll review the contract tomorrow" → Opus composes with professional tone (greeting, sign-off, appropriate formality), sends as a threaded reply through the Gmail API. Same confirm-or-send flow as message replies.

---

## Calendar integration

Same provider pattern as email: an abstract `CalendarProvider` base class with a Google Calendar implementation, and a `CalendarManager` that aggregates multiple providers. Adding CalDAV, Outlook, or any other calendar service is just another adapter.

### Setup (Google Calendar)

1. Enable the Google Calendar API in Google Cloud Console (same project as Gmail works)
2. Add to `.env`:
   ```
   GOOGLE_CALENDAR_CREDENTIALS=email_client_secret.json
   GOOGLE_CALENDAR_TOKEN=google_calendar_token.json
   ```
3. First calendar query triggers OAuth consent in the browser

### How it works

- Read-only access — Diplo can see your schedule but can't modify it
- On-demand only — no polling, events fetched when you ask
- Fetches from all visible calendars on your Google account
- Timezone-aware formatting

### What you can ask

- "What's on my calendar today?"
- "Am I free Thursday afternoon?"
- "When am I free next week?" — finds gaps between events
- "Propose 3 times for a 1-hour meeting next week"
- "Do I have a meeting with Sophie?" — combines calendar + message search

---

## Self-awareness & debugging

Diplo knows what it can and can't do. Ask "what can you do?" or "how does triage work?" and it'll explain — its system prompt describes all its own capabilities.

More importantly, when something goes wrong, you can ask Diplo to investigate itself. Every LLM call (triage, intent routing, response generation, compose, etc.) is logged to a SQLite table with full prompts, responses, token counts, latency, and status (retained 7 days). Ask "Why wasn't that urgent?", "What went wrong with the reply?", or "Show me recent errors" — Diplo queries its own logs, sees exactly what the model saw, and explains what happened and why. This makes debugging straightforward: you don't need to dig through logs yourself.

**Important**: Diplo does not search the internet or browse any websites. It only works with your messages, email, and calendar data. This is a deliberate security boundary — external web access would open the door to prompt injection from untrusted content.

---

## Error handling

- **Beeper connection**: retries with exponential backoff (5s → 10s → 20s → 30s max). Notifies you on Telegram after 3 consecutive failures, and again when connection restores.
- **LLM calls**: retry once (2s delay), then fall back to OpenAI gpt-4.1. If both fail, error is surfaced.
- **Gmail tokens**: auto-refresh. If revoked, logs a warning and skips that mailbox until re-setup.
- **Graceful shutdown**: handles SIGTERM/SIGINT for launchd/systemd/docker. Cleans up Telegram polling, SQLite connections, email providers.

---

## Project structure

```
src/
├── main.py             # async entry point, polling loop, startup, shutdown
├── llm.py              # Claude retry + OpenAI fallback + streaming + vision + transcription
├── llm_logger.py       # SQLite logging for every LLM call (7-day retention)
├── beeper_client.py    # Beeper API polling, watermarks, iMessage workarounds
├── triage.py           # urgency classification via Sonnet
├── assistant.py        # two-stage intent routing + Opus response generation
├── message_cache.py    # SQLite message cache, search, network aliases
├── conversation.py     # conversation history (30 turns, 48h prune, sessions)
├── contacts.py         # persistent contact registry with fuzzy resolution
├── actions.py          # send messages via Beeper on your behalf
├── automations.py      # scheduled + triggered automations (SQLite, cron)
├── feedback.py         # feedback storage + Opus consolidation into rules
├── config.py           # env vars, constants, paths
├── channels/
│   ├── base.py         # abstract ControlChannel base class
│   └── telegram.py     # Telegram adapter (typing, splitting, voice, photo)
├── calendar/
│   ├── base.py         # abstract CalendarProvider + CalendarEvent
│   ├── google.py       # Google Calendar (read-only, OAuth2, lazy init)
│   └── manager.py      # aggregates multiple calendar providers
└── email/
    ├── base.py         # abstract EmailProvider + EmailMessage
    ├── gmail.py        # Gmail (OAuth2, historyId incremental sync)
    ├── cache.py        # SQLite email cache + mailbox management
    ├── manager.py      # orchestrates multiple mailboxes
    └── setup.py        # CLI for one-time OAuth setup

prompts/
├── triage_system.md    # urgency classification system prompt (never modified)
├── assistant_system.md # personality + capabilities prompt (never modified)
├── feedback.md         # raw feedback entries (cleared after consolidation)
└── learned_rules.md    # consolidated rules (max 30 lines, injected at runtime)

data/                   # SQLite databases (gitignored)
├── messages.db         # message cache, contacts, watermarks, automations, convo history
└── emails.db           # email cache, mailbox config

tests/                  # test coverage across all modules
```

## Tests

```bash
uv run pytest tests/ -v
```

---

## Known limitations

- **iMessage contacts often show as phone numbers** — Beeper's iMessage bridge doesn't always sync display names from macOS Contacts. Diplo resolves phone senders using the chat title for DMs, but when both are phone numbers, there's nothing to resolve against.
- **Message edits are not detected** — the watermark-based poller skips already-seen messages, and the cache ignores duplicates. Edits are silently missed.
- **LIKE wildcards in search** — `%` and `_` in search terms act as SQL wildcards. No injection risk (parameterized queries), but could cause unexpected results.
- **Pending action race condition** — if you send two messages very quickly while a confirmation is pending, both could trigger the send. Window is small (requires two messages within Opus API latency).
- **WebSocket not yet used** — Beeper has an experimental WebSocket endpoint for real-time events. Currently using 5s polling instead.
