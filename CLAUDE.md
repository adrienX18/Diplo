# Diplo — Personal AI Messaging Assistant

## Vision

An always-on AI assistant ("Diplo") running on Adrien's Mac Mini that monitors all personal messages through Beeper Desktop. It operates in two simultaneous modes:

- **Passive mode**: Always watching, building context. When Adrien reaches out (via Telegram bot), it can summarize important messages, draft/send replies on his behalf, answer questions about conversations, etc.
- **Proactive mode**: When something genuinely urgent comes in, it pushes a notification via Telegram without being asked. The AI decides what's urgent based on message content, sender identity, and conversational context.

The AI learns from Adrien's feedback over time (e.g., "this wasn't urgent" or "you missed this — it was urgent").

## Documentation

- **README.md** — short landing page: what Diplo does, quick setup, pointer to GUIDE.md
- **GUIDE.md** — full guide: every capability explained, architecture, setup details for email/calendar, known limitations
- **CLAUDE.md** — this file: project context for Claude (architecture internals, design decisions, conventions, lessons learned)

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Mac Mini (always on)                       │
│                                                              │
│  ┌──────────────┐    ┌────────────────────────┐              │
│  │Beeper Desktop│───▶│  Python Agent (main)   │              │
│  │  (localhost   │    │                        │              │
│  │   :23373)     │    │  ├─ Poller/WS listener │              │
│  └──────────────┘    │  ├─ Triage (Sonnet)    │              │
│                      │  ├─ Assistant (Opus)   │              │
│  ┌──────────────┐    │  ├─ Conversation hist  │              │
│  │ Gmail API    │───▶│  ├─ Feedback store     │              │
│  │ (polling)    │    │  ├─ Email poller       │              │
│  └──────────────┘    │  ├─ Calendar reader    │              │
│                      │  └─ Control channel    │              │
│  ┌──────────────┐    │     (Telegram v1)      │              │
│  │Google Cal API│───▶└────────────────────────┘              │
│  │ (on-demand)  │            │                               │
│  └──────────────┘  ┌────────┴─────────┐                     │
│                    │  Claude API       │                     │
│                    │  (Sonnet + Opus)  │                     │
│                    │  ↓ fallback       │                     │
│                    │  OpenAI (gpt-4.1) │                     │
│                    └──────────────────┘                      │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │  Control Channel │
               │  (Telegram v1)   │
               │  Adrien's phone  │
               └──────────────────┘
```

### Key components

- **Beeper Desktop API** — local REST API + optional WebSocket for real-time events, running at `http://localhost:23373`. Provides access to all messages across WhatsApp, iMessage, Telegram, Signal, Instagram, Messenger, SMS, Slack, Discord, LinkedIn, X, etc.
- **Python agent** — long-running process that polls or listens for new messages, triages them through Claude Sonnet, and takes action.
- **Claude API** — Sonnet 4.6 for triage and search plan extraction. Opus 4.6 for the assistant (answering questions, summaries, complex reasoning). All LLM calls go through `src/llm.py` which retries once on failure, then falls back to OpenAI (gpt-4.1).
- **OpenAI API** — gpt-4o-transcribe for voice message transcription (via `transcribe_audio()` in `llm.py`), and gpt-4.1 as LLM fallback when Claude is down. If `OPENAI_API_KEY` is not set, voice messages are rejected gracefully and the LLM fallback raises instead of silently degrading.
- **Control channel (Telegram first)** — Adrien's remote interface. Receives urgent notifications, responds to commands like "what did I miss?", "reply to X saying Y", etc. The architecture MUST be channel-agnostic: Telegram is the v1 implementation, but SMS (via Twilio), WhatsApp, or other channels should be trivially addable. All control channel logic should go through an abstract `ControlChannel` base class so adding a new channel means writing a small adapter, nothing more.
- **Feedback store** — SQLite database storing ALL of Adrien's feedback — not just urgency, but also tone preferences, summarization style, reply drafting quality, sender importance, or anything else. The agent should accept freeform feedback via the control channel (e.g., "your summary was too long", "don't bother me about marketing emails", "always prioritize messages from Sophie"). A **feedback consolidation process** should run periodically (e.g., daily or weekly) to distill accumulated feedback into a compact, updated set of instructions/rules. This prevents the context from growing unboundedly — old granular feedback gets synthesized into general rules, and the raw entries can be archived.
- **Gmail integration** — multi-mailbox email monitoring via Gmail API. Each mailbox (e.g., "work", "personal") has its own OAuth token and is polled every 20 seconds using Gmail's historyId-based incremental sync. Emails are stored in a separate SQLite database (`data/emails.db`), triaged for urgency, and queryable by sender, subject, body text, or mailbox. Provider-based architecture (`EmailProvider` base class) so Outlook/IMAP can be added later.
- **Google Calendar** — read-only access to Adrien's calendar via Google Calendar API. On-demand fetching (no polling) — events are fetched when Adrien asks about his schedule. Supports date range queries, text search, and multi-calendar aggregation. Provider-based architecture (`CalendarProvider` base class) for future CalDAV/Outlook support.

### Possible future extensions (not planned)

- **Notion** — look up contacts, projects, and notes for richer understanding of who someone is
- **Email reply by address** — currently email replies require the sender to be in the contact registry (they must have emailed Adrien). Supporting direct email address input ("email john@example.com saying...") would remove this limitation
- **Outlook / CalDAV** — additional calendar/email providers (architecture already supports them via abstract base classes)

## Milestones

- [x] **M1 — The listening loop**: Connect to Beeper Desktop API from Python, receive/poll incoming messages, log them to console. No AI yet. Proves the plumbing works.
- [x] **M2 — Urgency triage**: Wire up Claude Sonnet API to classify each incoming message as urgent or not. System prompt encodes Adrien's rules. Log decisions.
- [x] **M3 — Control channel (Telegram v1)**: Build the abstract `ControlChannel` base class, then implement the Telegram adapter. Two-way communication with Adrien via Telegram. See **M3 design decisions** below for details.
- [x] **M4 — Reply on behalf**: Tell the bot "reply to X saying Y" and have it send through Beeper. See **M4 design decisions** below.
- [x] **M5 — Feedback & learning**: Freeform feedback via Telegram, file-based storage, hourly Opus consolidation into learned rules. See **M5 design decisions** below.
- [x] **M5.5 — Automations**: Scheduled tasks (cron-based) and triggered tasks (event-based). See **M5.5 design decisions** below.
- [x] **M6 — Gmail & Calendar**: Connect Gmail (multi-mailbox) and Google Calendar for email monitoring, schedule queries, and availability checks. See **M6 design decisions** below.

## Conventions

- **Language**: Python 3.12+
- **Package manager**: uv (`uv sync` to install, `uv run` to execute)
- **Beeper SDK**: `beeper_desktop_api` (official Python SDK)
- **Claude API**: `anthropic`
- **OpenAI API**: `openai` (fallback only)
- **Telegram**: `python-telegram-bot`
- **Data storage**: SQLite for feedback, message cache, and conversation state
- **Config**: `.env` file for secrets (Beeper token, Claude API key, OpenAI API key, Telegram bot token)
- **Structure**:
  ```
  diplo/
  ├── CLAUDE.md                     # project context for Claude
  ├── README.md                     # landing page — what Diplo does, quick setup
  ├── GUIDE.md                      # full guide — every capability, architecture, setup details
  ├── .env                          # secrets (gitignored)
  ├── .env.example                  # template
  ├── google_calendar_credentials.json  # Google OAuth client secret for Calendar (gitignored)
  ├── google_calendar_token.json    # saved OAuth token for Calendar (gitignored)
  ├── email_client_secret.json      # Google OAuth client secret for Gmail (gitignored)
  ├── src/
  │   ├── __init__.py
  │   ├── main.py           # entry point, orchestrator, urgent notification batching, email poller
  │   ├── beeper_client.py  # Beeper API connection, polling & startup backfill
  │   ├── triage.py         # urgency classification via Sonnet
  │   ├── llm.py            # LLM abstraction — Claude with retry + OpenAI fallback + audio transcription
  │   ├── llm_logger.py     # LLM call logging (SQLite) — every call recorded for debug introspection, 7-day retention
  │   ├── channels/         # control channel adapters
  │   │   ├── __init__.py
  │   │   ├── base.py       # abstract ControlChannel base class
  │   │   └── telegram.py   # Telegram implementation (v1) — text, voice, audio, photo
  │   ├── calendar/         # calendar integration (M6)
  │   │   ├── __init__.py   # exports CalendarManager
  │   │   ├── base.py       # abstract CalendarProvider + CalendarEvent dataclass
  │   │   ├── google.py     # Google Calendar provider (read-only, OAuth2)
  │   │   └── manager.py    # CalendarManager — aggregates multiple providers
  │   ├── email/            # email integration (M6)
  │   │   ├── __init__.py
  │   │   ├── base.py       # abstract EmailProvider + EmailMessage dataclass
  │   │   ├── gmail.py      # Gmail provider (OAuth2, historyId-based incremental sync)
  │   │   ├── cache.py      # EmailCache — SQLite cache for emails (data/emails.db)
  │   │   ├── manager.py    # EmailManager — orchestrates multiple mailbox providers
  │   │   └── setup.py      # one-time OAuth setup script (python3 -m src.email.setup)
  │   ├── message_cache.py  # SQLite message cache, last_seen_at, timezone setting
  │   ├── conversation.py   # conversation history between Adrien and Diplo (SQLite, 30 turns, 48h prune)
  │   ├── assistant.py      # Opus-powered assistant (search plan via Sonnet, response via Opus, reply flow, calendar + email queries)
  │   ├── contacts.py       # persistent contact registry — (sender_name, network) -> chat_id
  │   ├── actions.py        # reply-on-behalf via Beeper API
  │   ├── feedback.py       # feedback storage (file-based) & Opus consolidation
  │   ├── automations.py    # scheduled + triggered automations (SQLite, cron, trigger eval)
  │   └── config.py         # env loading, constants
  ├── data/
  │   ├── messages.db       # SQLite message cache (gitignored)
  │   ├── emails.db         # SQLite email cache + mailbox config (gitignored)
  │   └── email_tokens/     # per-mailbox OAuth tokens (gitignored)
  │       └── work.json     # e.g., token for "work" mailbox
  ├── prompts/
  │   ├── triage_system.md  # system prompt for urgency classification
  │   ├── assistant_system.md # assistant personality + system prompt
  │   ├── feedback.md       # raw feedback entries (cleared after consolidation)
  │   └── learned_rules.md  # consolidated rules from feedback (max 30 lines)
  ├── pyproject.toml                # project metadata + dependencies (uv)
  └── uv.lock                       # lockfile (committed)
  ```

## M3 design decisions

### Freeform natural language, not slash commands
Adrien's messages to the Telegram bot are passed directly to the assistant, which interprets intent and generates a response. No `/slash` commands, no rigid command parsing. This makes the interface feel like texting an assistant, not operating a CLI. Sonnet extracts a search plan (what to query), then Opus generates the response — "what did I miss?", "any messages about fundraising?", "what did Sophie say yesterday?" — and answers it.

### Local SQLite message cache
Every message the poller picks up is stored in SQLite (sender, timestamp, chat name, network, text). When Adrien asks a question, the agent queries the local cache (instant) to build relevant context, then sends it to Sonnet. It does NOT hit the Beeper API in real time for queries — the cache is the source of truth for answering questions. This keeps responses fast and avoids unnecessary Beeper API calls.

The cache is auto-pruned: messages older than 14 days are deleted to keep it lean.

### Fully async architecture
The entire agent runs in a single asyncio event loop:
- **Telegram bot** uses `python-telegram-bot`'s built-in async loop
- **Beeper poller** runs in `asyncio.run_in_executor()` (the Beeper SDK is synchronous with no async client available — wrapping it in an executor is simpler and more reliable than dropping the SDK to use raw async HTTP calls)
- **Triage** and **assistant calls** use `src/llm.py` (async Claude client with OpenAI fallback)

Alternative considered: separate threads or processes for poller and Telegram bot. Rejected because a single event loop is simpler to reason about, avoids thread-safety issues with shared state (message cache, etc.), and `python-telegram-bot` is already async-native.

### What M3 delivers
1. **Proactive**: Urgent messages (classified by Sonnet in triage) push a notification to Adrien's Telegram automatically, including the network name (e.g., "(whatsapp)"). Notifications are batched per chat with a 20s delay — if multiple urgent messages arrive from the same chat, they're combined into a single notification. The timer resets on each new message from the same chat.
2. **Passive**: Adrien texts the bot anything in natural language → Sonnet extracts a search plan → Opus answers using the SQLite message cache as context
3. **Message cache**: All polled messages stored in SQLite, auto-pruned at 14 days. Poller watermarks are also persisted to SQLite — on restart, the poller resumes from where it left off so no messages are missed.
4. **Startup backfill**: On boot, `backfill_recent()` fetches the last 48h of messages across all platforms via `messages.search(date_after=...)`. Chat titles are resolved via individual `chats.retrieve()` calls per unique chat_id. Messages are stored in the cache (`INSERT OR IGNORE` skips duplicates) and contacts are updated. Control channel messages are filtered out. Backfilled messages are NOT triaged (they're historical). If backfill fails, the bot continues without it.
4. **Conversation memory**: Adrien-Diplo dialogue stored in SQLite (30 turns, 48h auto-prune). Injected into both the search plan extractor and response generator so Diplo can handle follow-ups ("what about him?", "tell me more"), remember urgent notifications it sent, and maintain conversational continuity. Timestamps (HH:MM, local timezone) included in the prompt.
5. **Personality**: Diplo matches Adrien's energy — greetings get greetings, not unsolicited summaries. Summaries are minimal words. Casual messages skip the cache entirely (`no_query` flag in search plan) and do NOT advance the `last_seen_at` watermark.
6. **Voice messages**: Adrien can send voice notes or audio files to Diplo via Telegram. The audio is downloaded to a temp file, transcribed via OpenAI gpt-4o-transcribe (`transcribe_audio()` in `llm.py`), and passed to the assistant pipeline as `[voice message] {transcript}`. The `[voice message]` prefix tells the models the input was spoken, not typed. Temp files are cleaned up immediately after transcription. Handles both Telegram voice notes (`.ogg/opus`) and audio file attachments (`.mp3`, `.m4a`, `.wav`, etc.). Graceful fallback if OpenAI key is missing or transcription fails.
6b. **Photo messages**: Adrien can send photos (with optional captions) to Diplo via Telegram. The photo is downloaded, described via Claude vision (`describe_image()` in `llm.py`), and passed to the assistant pipeline as `[image: description] caption`. Same `[image: description]` format used by the Beeper poller path for image attachments. Falls back to `[image]` if vision fails. Temp files cleaned up immediately.
7. **Typing indicator**: Shows "typing..." in Telegram while Opus thinks. Refreshes every 4s.
8. **Message splitting**: Responses over ~500 chars are split at paragraph boundaries and sent as separate Telegram messages for phone readability.
9. **Error recovery**: Beeper connection failures retry with exponential backoff (5s → 10s → 20s → 30s max). Adrien is notified via Telegram after 3 consecutive failures, and again when connection restores. Claude API failures retry once (2s delay), then fall back to OpenAI gpt-4.1.
10. **Startup summary**: On boot, Diplo sends "I'm online. X messages since we last spoke." with a tl;dr of top chats by message count. Also surfaces any Beeper fetch errors (e.g., iMessage 404s).
11. **Local timezone**: Timestamps shown to Adrien are converted from UTC to his local timezone (default: Pacific). Adrien can change it via chat ("I'm in Paris") — Sonnet maps the location to an IANA timezone, persisted in SQLite.
12. **Graceful shutdown**: SIGTERM/SIGINT handlers ensure clean shutdown of Telegram polling, SQLite connections, and the event loop (for launchd/systemd/docker). Guarded against double-shutdown (signal handler + finally block).
13. **Abstract base class**: `ControlChannel` in `src/channels/base.py` with `send_notification()`, `send_message()`, etc. — Telegram is the v1 adapter, but SMS/WhatsApp/other channels can be added by implementing the same interface
14. **LLM abstraction**: All Claude API calls go through `src/llm.py` which handles retry + OpenAI fallback + audio transcription. Neither `triage.py` nor `assistant.py` import `anthropic` directly.
15. **Debug self-awareness**: Diplo knows it can investigate its own past decisions and errors. The `debug` intent (routed by Sonnet) queries the `llm_calls` SQLite table via `LLMLogger` — every LLM call is recorded with full prompts, responses, tokens, latency, and errors (7-day retention). Opus analyzes the logs and explains what happened. The system prompt documents this capability and instructs Diplo to proactively offer debugging when something fails (e.g., "Failed to send — want me to look into why?"). Error messages in reply/compose flows include debug hints so Adrien knows he can ask.

## M4 design decisions

### Contact registry
A persistent SQLite table `contacts` maps (sender_name, network) → chat_id. Updated on every polled message. Never pruned — unlike the message cache, contacts persist indefinitely. When a chat_id changes for a given (sender_name, network) pair, it gets updated.

Known limitation: the same person can appear under multiple sender_name variants (e.g. "Sophie", "Sophie Martin", "+1 555-1234"). Each variant is stored separately. Deduplication is deferred to a future milestone.

### Recipient resolution
When Adrien says "reply to Sophie":
1. Fuzzy match the name in the contact registry (case-insensitive substring)
2. If multiple networks match, pick the one with the most recent `last_seen_at` (latest-used channel)
3. If Adrien specifies a network ("on whatsapp"), filter by that network
4. If no match, tell Adrien

### Opus composes and decides
Opus writes the message (matching tone, language, and style of the existing conversation) and decides whether to send directly or ask for confirmation:
- **Send directly**: short, clear, low-stakes, near-exact wording from Adrien
- **Confirm first**: long/complex, sensitive context, significant interpretation needed, ambiguous recipient

### Pending action state
When Opus asks for confirmation, a pending action is stored in memory (not SQLite — ephemeral). On Adrien's next message, Opus interprets whether it's a confirmation, modification, cancellation, or unrelated. Unrelated messages clear the pending action and proceed with normal flow.

### Two-step LLM flow for replies
1. **Sonnet** extracts intent: `{"reply": {"recipient": "Sophie", "message": "I'll be late"}}` (or with optional `"network"` field)
2. **Opus** composes the actual message and decides send vs confirm, using recent chat context

### What M4 delivers
1. **Reply/send**: "tell Sophie I'll be late" → Diplo composes a natural message matching the conversation's tone/language and sends via Beeper
2. **Smart confirmation**: Opus decides whether to send directly or confirm first, based on message complexity, sensitivity, and recipient
3. **Network selection**: Automatically picks the most recently used platform for a contact. Overridable: "tell Sophie on whatsapp..."
4. **Confirmation flow**: Adrien can confirm, modify, cancel, or ignore a pending reply
5. **Contact registry**: Persistent mapping of sender names to chat IDs, updated on every polled message
6. **Error handling**: Unknown recipients, Beeper send failures, and missing components all produce clear messages
7. **Sent message caching**: Messages sent on Adrien's behalf are stored in the SQLite cache so the next compose prompt includes them in the conversation context

### Known issues (accepted)

**1. Pending action race condition (double-send risk)**
`_pending_action` in `assistant.py` is a module-level global with no locking. If Adrien sends two messages in rapid succession while a pending confirmation is active, both messages enter `_handle_pending_confirmation` concurrently (the first `await`s the Opus call, yielding control to the second). Both can read the same pending action and both can call `send_message()`, resulting in a duplicate send. Fix would be an `asyncio.Lock` around pending action access, or making it a per-session object. Accepted for now because: (a) Adrien is the only user, (b) the window is small (requires two messages within the Opus API latency), and (c) the consequence (a double message) is embarrassing but not dangerous.

**2. Adrien's conversation with Diplo leaks into compose prompts alongside untrusted messages**
In `_compose_and_decide()`, `convo_context` (Adrien's private dialogue with Diplo — including things like "reply to X saying Y", "don't message Z about this") is concatenated into the user prompt alongside `chat_context` which contains untrusted messages wrapped in `<msg>` tags. A prompt injection in a `<msg>` tag has access to Adrien's private conversation in the same context window. The `<msg>` tag defense mitigates this, but the attack surface is wider than necessary — `convo_context` should ideally be in the system prompt or a clearly separate section. Accepted for now because: (a) the `<msg>` tags provide reasonable protection, (b) convo_context helps Opus understand Adrien's full intent, and (c) restructuring the prompt architecture is a larger refactor best done alongside M5's context management work.

**3. Compose prompt behavior may diverge on OpenAI fallback**
When Claude fails and `llm.py` falls back to gpt-4.1, the compose/confirm prompts (`_compose_and_decide`, `_interpret_confirmation`) were designed and tested against Claude models only. gpt-4.1 may not respect the `<msg>` tag convention the same way, may wrap JSON output in prose (triggering regex fallback in `_parse_json`), or may interpret the compose prompt differently — potentially composing a message with wrong tone/content and auto-sending it (`"action": "send"`). The JSON parse fallback at least falls back to `"action": "confirm"`, but the risk is a miscomposed message being sent directly. Mitigation: force `"action": "confirm"` when running on the OpenAI fallback model, or test prompts against gpt-4.1 and adjust.

**4. Stale pending action can trigger accidental send**
`_pending_action` has no TTL. If Opus asks for confirmation and Adrien never responds (closes Telegram, forgets), the pending action stays in memory indefinitely. Hours or days later, a casual "yes" or "go ahead" (in response to something else entirely) enters `_handle_pending_confirmation`, and Opus may interpret it as a confirmation of the stale action. Opus will *usually* say "unrelated" for non-sequiturs, but short affirmatives are ambiguous. Fix: add a timestamp to pending actions and auto-expire them after ~10 minutes.

**5. Disambiguation list shows opaque Beeper IDs for some contacts**
In `_handle_reply_action`, the disambiguation list (`assistant.py:188-189`) shows `sender_name`, which can be a Beeper internal ID (e.g., `whatsapp:+15551234567@s.whatsapp.net`) for contacts without display names. This makes it hard for Adrien to pick the right option. Fix: prefer `chat_title` over `sender_name` when the sender name contains `@` or looks like an internal ID.

### Env vars
- `BEEPER_ACCESS_TOKEN` — from Beeper Desktop → Settings → Developers
- `ANTHROPIC_API_KEY` — Claude API key
- `OPENAI_API_KEY` — OpenAI API key (optional, for fallback when Claude is down)
- `TELEGRAM_BOT_TOKEN` — bot token from @BotFather
- `TELEGRAM_CHAT_ID` — Adrien's personal chat ID (restricts the bot to only respond to Adrien)
- `GOOGLE_CALENDAR_CREDENTIALS` — path to Google OAuth client credentials JSON for Calendar (optional)
- `GOOGLE_CALENDAR_TOKEN` — path to saved Calendar OAuth token (optional, defaults to `google_calendar_token.json`)
- `USER_EMAIL_ADDRESSES` — comma-separated list of Adrien's email addresses, used to identify his own emails in triage (optional)

## M5 design decisions

### File-based feedback, not SQLite
Feedback is stored as plain markdown (`prompts/feedback.md`), one entry per line prefixed with `-- `. This is simpler than a database table, human-readable, and easy to inspect/edit manually. After consolidation, the file is cleared.

### Consolidated rules as a separate file
`prompts/learned_rules.md` holds the distilled rules (max 30 lines). It supplements the base prompts — `triage_system.md` and `assistant_system.md` are never modified. At runtime, `load_rules()` reads the file and it's appended to the base system prompt under a "Learned rules" section.

### Feedback detection via search plan
Sonnet recognizes feedback as a new intent type (`{"feedback": "..."}`) in the search plan extraction step. This reuses the existing intent routing — no separate classification call needed. The feedback text is Sonnet's concise summary of what Adrien said.

### Consolidation via Opus
Runs on the hourly prune cycle (alongside cache prune and conversation prune). Opus reads all pending feedback + current rules + both base prompts, and produces an updated ruleset that stays true to the base prompts' spirit. If consolidation fails (API error), feedback is preserved for the next cycle.

### What M5 delivers
1. **Feedback intake**: "that wasn't urgent", "always prioritize Sophie" → Sonnet detects feedback intent → appended to `prompts/feedback.md` → Diplo acknowledges naturally
2. **Learned rules**: `prompts/learned_rules.md` — max 30 lines, injected into both triage and assistant prompts at runtime
3. **Hourly consolidation**: Opus reads pending feedback + current rules → produces updated compact ruleset → clears feedback file
4. **Base prompt preservation**: `triage_system.md` and `assistant_system.md` are never modified — learned rules are a supplement
5. **Failure resilience**: If consolidation fails, feedback is preserved for the next cycle
6. **No new dependencies**: File-based, no new SQLite tables, no new packages
7. **Rules caching**: `load_rules()` caches in memory after the first read. Cache is refreshed when `run_consolidation()` writes new rules. Avoids disk I/O on every triage/assistant LLM call.

### Possible improvements (deferred)

**1. No way to inspect or reset rules via control channel**
Adrien can give feedback but can't ask "what are your current rules?" or "forget all rules". He'd have to SSH into the Mac Mini and edit `learned_rules.md`. Fix: add new intent types in the search plan (e.g. `{"show_rules": true}`, `{"clear_rules": true}`) — trivial to wire up since `load_rules()` and `RULES_FILE` are already accessible.

**2. Multiline messages break the `-- ` prefix format**
`append_feedback` writes `-- {text}\n`. If Adrien's message contains newlines, only the first line gets the `-- ` prefix. Doesn't break consolidation today (Opus reads the whole file as a blob), but breaks the implicit per-entry contract and would break any future line-by-line parsing. Fix: escape newlines in feedback text, or use a blank-line delimiter between entries.

**3. Consolidation sends both full base prompts every time**
`run_consolidation()` includes the complete `triage_system.md` and `assistant_system.md` in the Opus prompt (~1-2K tokens of static content). These never change. Could extract key behavioral sections into a shorter reference to save tokens. Acceptable waste since consolidation runs at most once per hour.

**4. Canned acks are context-blind**
`_feedback_ack()` returns a random short response ("Noted.", "Got it.") regardless of what Adrien said. Saves an Opus call, which is the right tradeoff for most feedback. But for longer or emotional feedback, a more contextual ack would feel better. Could use a cheap Sonnet call for feedback messages over ~100 chars.

**5. No feedback throttling or deduplication**
Nothing prevents the same feedback from being stored multiple times. Opus consolidation handles this gracefully (deduplicates semantically), so the impact is just extra tokens in the consolidation prompt. Not worth adding code for unless it becomes a real problem.

## M5.5 design decisions

### Two types of automations

1. **Scheduled tasks** — fire on a cron schedule (e.g. "every morning at 9am, summarize my messages"). Uses `croniter` for cron expression parsing and next-fire-time computation.
2. **Triggered tasks** — fire when a message matches conditions (e.g. "whenever Sophie messages, notify me"). Evaluated on every polled message in the poller loop.

### Single SQLite table for both types
The `automations` table in `messages.db` stores both scheduled and triggered automations. The `type` column distinguishes them. Scheduled tasks use `schedule` (cron string) and `next_run_at` (precomputed UTC). Triggered tasks use `trigger_config` (JSON) and `cooldown_seconds`.

### Actions are natural language
The `action` field is a natural language instruction (e.g. "summarize my messages"). When an automation fires, the action is fed through `handle_user_message()` — the same pipeline Adrien uses. This reuses all existing logic (search plan extraction, cache queries, Opus response generation).

### Scheduled tasks: 30-second scheduler loop
A separate `asyncio.create_task` in `main.py` ticks every 30s. It queries `automations` for due tasks (`next_run_at <= now AND enabled = 1`), executes each action through the assistant pipeline, sends the result to Adrien via the control channel prefixed with `[Auto]`, and advances `next_run_at`.

**Key behaviors:**
- `next_run_at` is advanced immediately (before execution), not after — prevents double-execution if the action is slow
- Automation runs do NOT advance `last_seen_at` — a follow-up "what's new?" still works
- Automations run without conversation context (`convo=None`) — they're standalone
- Cron expressions are timezone-aware: interpreted in Adrien's local timezone, stored as UTC

### Triggered tasks: evaluated in the poller loop
After each message is triaged and stored in the cache, `automations.evaluate_triggers(msg)` checks it against all enabled triggered automations. If conditions match (AND logic across sender, keyword, chat, network filters), a notification is sent to Adrien via the control channel prefixed with `[Trigger]`.

**Key behaviors:**
- Cooldown prevents re-firing within a configurable window (default 5 minutes)
- Disabled triggers are not evaluated
- All trigger conditions must match (AND logic)
- Trigger matching is case-insensitive substring

### NL-based management via existing intent extraction
Automations are created, listed, toggled, and deleted via natural language through the existing search plan extraction (Sonnet). New intent types: `create_automation`, `create_trigger`, `list_automations`, `delete_automation`, `toggle_automation`. Sonnet translates natural language schedules to cron expressions.

### What M5.5 delivers
1. **Scheduled tasks**: "every morning at 9am, summarize my messages" → creates a cron-based automation → runs automatically → sends result to Telegram
2. **Triggered tasks**: "whenever Sophie messages, notify me" → creates an event-based automation → fires when conditions match → sends notification to Telegram
3. **Management**: list, pause/resume, delete automations — all via natural language
4. **Timezone-aware**: cron schedules honor Adrien's configured timezone
5. **Cooldown**: triggered tasks have a configurable cooldown to prevent spam
6. **Safety**: automation runs don't advance `last_seen_at`, don't inject conversation context
7. **Resilience**: failed automations log errors but don't crash the scheduler; `next_run_at` still advances

### Dependencies
- `croniter` — cron expression parsing and next-fire-time computation

## M6 design decisions

### Provider-based architecture for both email and calendar
Both email and calendar use the same pattern: an abstract base class (`EmailProvider`, `CalendarProvider`) with a concrete Google implementation, and a manager that aggregates multiple providers. This means adding Outlook, CalDAV, or IMAP is just writing a new adapter — no changes to the assistant, triage, or main loop.

### Email: separate SQLite database
Emails live in `data/emails.db`, not in the message cache (`data/messages.db`). This keeps concerns clean — email has different fields (subject, from_address, thread_id, mailbox, attachments, is_read) and different query patterns (by thread, by mailbox) than messages. The `EmailCache` class mirrors `MessageCache`'s API shape (by_sender, search_text, recent, since_last_seen, prune) for consistency.

### Email: multi-mailbox with per-mailbox OAuth tokens
Each mailbox (e.g., "work", "personal") gets its own Gmail OAuth token, allowing monitoring of multiple Google accounts. The `mailboxes` table in `emails.db` stores the config (name, provider type, email address, token path, enabled flag, history_id checkpoint). The shared OAuth client secret (`email_client_secret.json`) is reused across mailboxes.

### Email: historyId-based incremental sync
Gmail's History API provides efficient incremental polling: after an initial fetch (last 50 inbox emails), subsequent polls only request changes since the last `historyId`. This avoids re-fetching the entire inbox every 20 seconds. If the historyId expires (Gmail retains ~7 days), the provider falls back to an initial fetch. The `history_id` checkpoint is persisted per-mailbox in the `mailboxes` table.

### Email: opt-in in the assistant, always-on for triage
The email poller runs continuously (every 20 seconds) and triages every new email for urgency — urgent emails push a Telegram notification just like urgent messages. However, when Adrien asks "what's new?", emails are NOT included unless he explicitly mentions email ("check my email", "any emails?"). This prevents email noise from drowning out message summaries. Sonnet extracts `"include_email": true` in the query plan only when email is explicitly requested.

### Email: triage uses thread context
When triaging an incoming email, the last 5 messages from the same thread are fetched from the email cache and provided as conversation context to Sonnet. This mirrors how message triage uses the last 20 messages from the same chat — the model needs context to judge urgency (e.g., a "yes" in an investor thread may be urgent).

### Email: Adrien's own emails skipped in triage
Like messages, emails sent by Adrien are not triaged for urgency (no self-alerts). `USER_EMAIL_ADDRESSES` in config identifies Adrien's email addresses. The `is_from_adrien` flag is set during parsing and checked before triage.

### Calendar: on-demand, not polled
Unlike email and messages, calendar events are fetched on-demand — only when Adrien asks about his schedule. There's no background polling loop. This is appropriate because calendar events don't trigger urgency notifications (they're Adrien's own events, not incoming communication).

### Calendar: read-only
The Google Calendar provider uses the `calendar.readonly` scope — it cannot create, modify, or delete events. This is intentional: calendar writes are a higher-risk action that should be deferred until there's a clear need and a confirmation flow (like the reply confirmation in M4).

### Calendar: lazy service initialization
The Google Calendar API service object is built on first use, not at startup. This means the OAuth consent flow (browser popup) only triggers when Adrien actually asks a calendar question, not every time the bot restarts.

### Calendar: timezone-aware formatting
Calendar events are converted to Adrien's local timezone (from the SQLite-persisted timezone setting) before being formatted for the LLM prompt. All-day events show just the date; timed events show `HH:MM-HH:MM`. The calendar name is included in parentheses for multi-calendar disambiguation.

### Email setup flow
Adding a new Gmail mailbox requires running `python3 -m src.email.setup --name "work"`:
1. Checks for `email_client_secret.json` (Google OAuth credentials) in project root
2. Opens a browser for Google OAuth consent (Gmail readonly + send scopes)
3. Saves the token to `data/email_tokens/{name}.json`
4. Fetches the authenticated email address from the Gmail API
5. Registers the mailbox in the `mailboxes` table in `emails.db`

### What M6 delivers
1. **Email monitoring**: Gmail inbox polled every 20s, new emails stored in SQLite cache, triaged for urgency, urgent emails push Telegram notifications
2. **Email queries**: "check my email", "any emails from Sophie?", "emails about the contract" — Sonnet extracts email search intent, queries the email cache, Opus answers
3. **Calendar queries**: "when am I free next week?", "what's on my calendar tomorrow?", "do I have a meeting with Sophie?" — Sonnet extracts date range + optional search text, events fetched from Google Calendar, Opus answers with availability/schedule info
4. **Multi-mailbox**: supports multiple Gmail accounts (work, personal, etc.) with independent OAuth tokens and polling state
5. **Email triage**: urgency classification for incoming emails, with thread context and lower-priority baseline (most emails are not urgent)
6. **Provider architecture**: abstract base classes for both email and calendar, ready for Outlook/IMAP/CalDAV adapters
7. **Separate email database**: `data/emails.db` with emails table, mailboxes config, and state tracking
8. **Contact registry integration**: email senders are added to the contact registry with `email:{mailbox}` as the network
9. **Email body truncation**: bodies capped at 3000 chars on ingestion, further truncated to 500 chars in LLM prompts to save context
10. **Email pruning**: emails older than 14 days (same retention as messages) are pruned on the hourly cycle

### Known issues (accepted)

**1. Email `last_seen_at` is independent from message `last_seen_at`**
The email cache has its own `email_last_seen_at` in its `state` table, separate from the message cache's `last_seen_at`. This means "what's new?" advances the message watermark but not the email one (since emails are opt-in). This is intentional but could confuse Adrien if he expects email state to track with message state.

**2. No email unread/read tracking beyond initial state**
The `is_read` field is set from Gmail's UNREAD label at fetch time, but never updated afterward. If Adrien reads an email in Gmail, the cache still shows the stale state. This doesn't affect functionality (the cache is for search/context, not inbox management), but could lead to misleading "unread" counts if displayed.

**3. Calendar OAuth consent flow blocks startup**
If the calendar credentials exist but no token has been saved yet, `GoogleCalendarProvider._get_service()` triggers the OAuth browser flow. Since this happens lazily (on first calendar query), it blocks the assistant until Adrien completes the consent flow in the browser. The first calendar query will time out from Adrien's perspective. After that, the token is saved and subsequent queries are fast.

**4. HTML email body conversion is best-effort**
`_strip_html()` uses `html2text` if installed, otherwise falls back to regex tag stripping. The regex fallback loses formatting, links, and may mangle complex HTML layouts. For most emails this is fine, but heavily styled marketing emails may produce garbled text.

### Dependencies
- `google-api-python-client` — Google Calendar and Gmail API client
- `google-auth-oauthlib` — OAuth2 consent flow
- `google-auth-httplib2` — HTTP transport for Google API auth
- `html2text` — HTML to plain text conversion for email bodies (optional, regex fallback)

## Urgency rules (for triage prompt)

These are the initial heuristics. The AI should use judgment based on these principles:

**Likely urgent:**
- Message explicitly says "urgent", "ASAP", "need this now", etc.
- Sender is a lawyer, investor, or close collaborator
- Action required with a deadline (e.g., "sign this", "approve by EOD")
- Someone Adrien is actively working with on a time-sensitive project
- Emergency or crisis language

**Likely NOT urgent:**
- Casual social messages ("what are you up to this weekend?")
- Group chat banter
- Marketing or automated messages
- News or content sharing without action required
- Messages from people Adrien doesn't know well, without urgent content

**The AI should also consider:**
- The full conversation context (not just the latest message)
- Who the sender is (if identifiable)
- Time of day and day of week
- Whether the message is a follow-up to something already handled

## Context window management

The agent runs 24/7, which means we must be deliberate about context windows. Key principles:

- **Every API call to Claude is stateless.** There is no persistent conversation. Each call gets a freshly constructed prompt with only the context it needs.
- **Triage calls (Sonnet)** should be lightweight: system prompt + urgency rules + recent feedback examples + the incoming message + minimal conversation context. Keep it small and fast.
- **Assistant calls (Opus)** are richer: include conversation history with Adrien (last 30 turns), message cache results, sender context, and relevant feedback when doing summaries or drafting replies.
- **Never let context accumulate unboundedly.** The agent does NOT maintain a growing conversation with Claude. Each call is independent.
- **Conversation history with Adrien** (via the control channel) is stored in SQLite (`conversation_history` table). Last 30 turns are injected into Opus calls, with long assistant responses truncated to 500 chars. Auto-pruned at 48 hours. Urgent notifications sent by the bot are also stored as assistant turns so Diplo knows what it already told Adrien.
- **Feedback consolidation** prevents the instruction set from growing without limit. Raw feedback in `prompts/feedback.md` is periodically distilled by Opus into `prompts/learned_rules.md` (max 30 lines), then cleared. Both triage and assistant prompts load rules at call time via `load_rules()`.
- **Message context for triage** should be bounded: e.g., include the last 5-10 messages in a conversation for context, not the entire thread history.
- **Email body truncation**: email bodies are capped at 3000 chars on ingestion (in `gmail.py`), and further truncated to 500 chars when formatting for the LLM response prompt (in `assistant.py`). This keeps email-heavy queries from blowing up the context window.
- **Calendar events** are fetched on-demand (not cached), and formatted as compact one-line strings. Calendar queries typically return 10-50 events, which fits comfortably in the context.
- **Email triage context** uses the last 5 messages from the same thread (from the email cache), not the full thread. Mirrors the message triage pattern of bounded context.

## Beeper Desktop API — TL;DR

### Overview
- Fully local REST API running inside Beeper Desktop at `http://localhost:23373`
- Requires Beeper Desktop v4.1.169+ to be running
- Supports all connected networks: WhatsApp, iMessage (macOS only), Telegram, Signal, Instagram, Messenger, SMS/Google Messages, Slack, Discord, LinkedIn, X, Google Chat, Google Voice
- SDKs available for Python, TypeScript, and Go
- Has a built-in MCP server (useful for Claude Code integration during dev)
- There is an experimental WebSocket endpoint for real-time events (check docs for current status)

### Authentication
- Get access token from: Beeper Desktop → Settings → Developers → API Access Token
- Pass as Bearer token in Authorization header, or set `BEEPER_ACCESS_TOKEN` env var

### Python SDK
Already included as a dependency (`beeper-desktop-api` in pyproject.toml).

```python
from beeper_desktop_api import BeeperDesktop

client = BeeperDesktop()  # reads BEEPER_ACCESS_TOKEN from env

# List connected accounts
accounts = client.accounts.list()

# Search chats
chats = client.chats.search(query="John", limit=5)

# List messages in a chat (paginated)
messages = client.chats.messages.list(chat_id="!abc:beeper.local")

# Search messages globally
results = client.messages.search(query="contract", limit=20)

# Send a message
client.messages.send(chat_id="!abc:beeper.local", text="Hello!")

# Create a new chat
client.chats.create(account_id="whatsapp", participants=["user1"], text="Hey!")
```

### Key API endpoints (REST, v1)
- `GET /v1/accounts` — list connected messaging accounts
- `GET /v1/chats` — list chats (paginated)
- `GET /v1/chats/search` — search chats by query
- `GET /v1/chats/{chatID}` — get chat details
- `GET /v1/chats/{chatID}/messages` — list messages in a chat
- `POST /v1/chats/{chatID}/messages` — send a message
- `GET /v1/messages/search` — search messages globally
- `POST /v1/chats` — create a new chat
- `POST /v1/chats/{chatID}/archive` — archive/unarchive a chat

### Real-time messages (for the listening loop)
- **WebSocket (experimental)**: Check `https://developers.beeper.com/desktop-api/websocket-experimental` for real-time event streaming. This is ideal for M1.
- **Polling fallback**: If WebSocket is unavailable or unstable, poll `GET /v1/chats` sorted by recent activity every N seconds, then fetch new messages from chats with updated timestamps.

### MCP server (for Claude Code dev sessions)
You can add Beeper's MCP server to Claude Code for interactive development:
```bash
claude mcp add --transport stdio beeper_desktop_api_api \
  --env BEEPER_ACCESS_TOKEN="your-token" \
  -- npx -y @beeper/desktop-mcp --client=claude-code --tools=all
```

### Limitations
- Beeper Desktop must be running for the API to be accessible
- Message history may be limited initially (Beeper indexes in the background)
- Sending too many messages may trigger rate limits on underlying networks
- iMessage only works on macOS
- API is still evolving; check changelog at `https://developers.beeper.com/desktop-api/changelog`

### Full docs
- Main docs: https://developers.beeper.com/desktop-api
- Python SDK: https://developers.beeper.com/desktop-api-reference/python
- API reference: https://developers.beeper.com/desktop-api-reference
- Changelog: https://developers.beeper.com/desktop-api/changelog
- MCP server: https://developers.beeper.com/desktop-api/mcp
- Remote access (if needed later): https://developers.beeper.com/desktop-api/advanced/remote-access

## Lessons learned

### Beeper SDK gotchas
- **Messages resource**: Use `client.messages` for `list`, `send`, `search`, and `update` — NOT `client.chats.messages` (which only has `reactions`). Despite what the SDK docs suggest, all message operations are on the top-level `messages` resource.
- **`client.chats.list()` returns a paginated cursor**, not a plain list. You must iterate and break manually to limit results. `client.accounts.list()` returns a plain list — they're inconsistent.
- **Sort keys are strings but must be compared as integers.** WhatsApp uses small integers (~389K), iMessage uses timestamp-based ones (~1.7 trillion). String comparison breaks: `"389869" > "1772670457888"` is `True`. Always `int(sort_key)` before comparing.
- **Each chat has a `.preview` field** containing the latest message — useful for quick-checking new activity without fetching all messages.
- **Self-chats** (e.g., WhatsApp "Message Yourself") have `is_sender=True` on all messages — relevant if you're filtering by sender.
- **iMessage chat IDs break the SDK's URL construction.** Chat IDs like `imsg##thread:...` contain `#` characters that httpx interprets as URL fragment separators (per RFC 3986). The SDK interpolates chat_id directly into the path without encoding, so the server receives `/v1/chats/imsg` and returns 404. **Fix**: `BeeperPoller._needs_raw_http(chat_id)` detects `#` in chat IDs, and `_raw_list_messages()` / `_raw_retrieve_chat()` use `httpx.get()` with `urllib.parse.quote()` to properly encode the path. All message fetching and chat title resolution paths check this and fall back to raw HTTP when needed.
- **`messages.search()` does NOT index iMessage.** The search endpoint returns zero results for iMessage regardless of filters. **Fix**: `backfill_recent()` calls `_backfill_raw_http_chats()` separately to fetch iMessage messages directly from each chat via raw HTTP.
- **iMessage uses UUID-suffixed network IDs** like `imessage_df461d39ed5545ed025fcd30942f27e8` instead of the `imessagego` convention used by other networks. `normalize_network()` in `message_cache.py` strips both `go` suffixes and UUID suffixes to get the base name (e.g. `imessage`). `by_network()` tries all prefix variants (resolved alias + normalized forms) to match both conventions. `_display_network()` in `assistant.py` uses the same normalization to map to "iMessage".
- **`messages.search()` supports time-based filtering** via `date_after` and `date_before` (ISO 8601 strings). Returns a paginated iterator (`SyncCursorSearch`) with `Message` objects that have `chat_id` and `account_id` but NO `chat_title` — titles must be resolved separately via `chats.retrieve(chat_id)`. The iterator only exposes `items`, not the `chats` dict from the raw API response.
- **iMessage contacts often show as phone numbers.** Beeper's iMessage bridge frequently does not sync display names from macOS Contacts. Both `sender_name` on messages and `title` on chats come through as raw phone numbers (e.g., `+1 650-283-7070`). The chat participants API (`chat.participants.items`) also lacks `fullName` in these cases — only `phoneNumber` and `id` are populated. **Partial fix**: `_resolve_sender_name()` in `beeper_client.py` swaps phone-number sender names for the chat title in DM chats (`type == "single"`), which works when Beeper has resolved the title to a display name. When both are phone numbers, there's nothing to resolve against — this is a Beeper/iMessage bridge limitation. Ensuring contacts are named in macOS Contacts and restarting Beeper may trigger a re-sync.

### Timestamp gotchas
- **`str(datetime)` vs `.isoformat()`**: Python's `str()` on a datetime produces `2026-03-05 03:01:00+00:00` (space separator), while `.isoformat()` produces `2026-03-05T03:01:00+00:00` (T separator). SQLite string comparison breaks silently: space (0x20) < T (0x54), so a message at `03:07` with a space looks *older* than `02:43` with a T. Always use `.isoformat()` for timestamps that will be compared.

### Control channel feedback loop
- **The bot's own Telegram chat appears in Beeper.** If the bot sends an urgent notification to Telegram, Beeper picks it up as a new message. If that message contains urgency language ("urgent", "ASAP"), it gets classified as urgent → another notification → infinite loop. Fix: skip control channel messages entirely — don't triage them AND don't cache them (they're meta-conversation, not real messages). Matching is done on sender/chat name against `BOT_SENDER_NAMES` in config.

### State timing
- **`last_seen_at` must update AFTER the reply is sent**, not when the user's message is received or when the response is generated. If updated too early, the next "what's new?" query sees nothing because the timestamp already advanced past the current messages. This required threading an `on_reply_sent` callback through the channel abstraction so the Telegram handler calls `touch_last_seen()` only after `reply_text()` succeeds. On error, `last_seen_at` is not updated.
- **`last_seen_at` must NOT update on casual greetings.** When `handle_user_message` returns `no_query` (greetings, "thanks", etc.), the watermark should not advance. Otherwise: "Hey!" → watermark advances → "New messages?" → nothing found (because messages before the greeting are now "seen"). Fixed by having `handle_user_message` return `(response, queried_cache)` — the Telegram handler only calls `touch_last_seen()` when `queried_cache` is `True`.

### LLM API
- **Current models**: Opus 4.6 (`claude-opus-4-6`) for the assistant response, Sonnet 4.6 (`claude-sonnet-4-6`) for triage and search plan extraction. Previously used Haiku for triage and Sonnet for assistant.
- **Search plan extraction uses Sonnet, not Opus.** It's simple JSON classification — Opus is overkill. Faster and cheaper.
- **All LLM calls go through `src/llm.py`** — never import `anthropic` or `openai` directly in feature modules. The wrapper handles retry (2s between attempts), OpenAI fallback (gpt-4.1), and audio transcription (gpt-4o-transcribe).
- **Voice message transcription** uses OpenAI's `gpt-4o-transcribe` model via `transcribe_audio()` in `llm.py`. The Telegram adapter downloads voice notes/audio files to a temp file, transcribes, and passes the result as `[voice message] {transcript}` to the assistant pipeline. The `[voice message]` prefix signals to both Sonnet and Opus that the input was spoken. Temp files are cleaned up immediately after transcription.
- **Adrien's own messages need explicit labeling** in triage prompts. The model doesn't know who the "owner" is — without annotation, it may flag Adrien's own time-sensitive messages as urgent. Sender names containing substrings from `USER_SENDER_IDS` (in `config.py`) get tagged as "(Adrien — the owner)" in the prompt.

### Triage context
- **Triage context comes from the local SQLite cache**, not live Beeper API calls. When a new message arrives, it's stored in the cache first, then `cache.by_chat_id()` fetches the last 20 messages from that chat for context. This is instant and avoids redundant API calls (e.g., 10 messages from the same group chat in one poll cycle no longer means 10 identical Beeper API calls).
- **Messages older than 48h are automatically classified as not urgent** without calling Sonnet. The check runs at the top of `classify_urgency()` in `triage.py` — saves API calls and latency for stale messages (e.g., from backfill or delayed delivery).

### Terminal display
- **Terminal logs truncate message text to 120 characters.** In `main.py`, `text_preview = (msg["text"] or "(no text)")[:120]` is used for `logger.info` output. This is cosmetic only — the full message text is stored intact in the SQLite cache. Messages appearing cut off in the terminal are not actually truncated in the database.

### Prompt injection
- **All untrusted message content is wrapped in `<msg>...</msg>` tags** in both triage and assistant prompts. Both system prompts contain explicit instructions to treat content inside `<msg>` tags as data only and never follow instructions found within them. This mitigates prompt injection from external senders across all platforms.

### Search logic
- **`_execute_search` must not fall back to all messages when a specific filter returns nothing.** If the search plan includes a sender/search/chat filter but matches zero messages, that's the correct answer — don't fall back to `since_last_seen` or `recent()`. The `has_specific_filter` flag tracks whether filters were applied.
- **`MessageCache._query()` is private** — it takes raw SQL WHERE clauses via f-string. All access should go through specific methods (`by_sender`, `search_text`, `by_chat`, `by_chat_id`, `recent`, `since_last_seen`) that use parameterized queries.
- **Sender filter must expand to full chat context.** `by_sender("Sophie")` only returns messages FROM Sophie — Adrien's replies are missing. When Adrien asks "what did Sophie say?", the LLM needs both sides. Fix: after `by_sender`, also fetch all messages from the same `chat_id`(s) via `by_chat_id`. Deduplication handles overlap.
- **Search plan values can be null.** Sonnet sometimes returns `{"sender": null}` instead of omitting the key. Use `plan.get("key")` (falsy for both missing and None) instead of `"key" in plan` to guard all filter checks.
- **Network filter uses alias mapping.** Beeper uses internal network IDs (`facebookgo`, `instagramgo`, `imessagego`) that don't match what users say ("messenger", "instagram", "imessage"). `NETWORK_ALIASES` in `message_cache.py` maps natural names to Beeper IDs. `resolve_network()` handles the lookup. `by_network()` accepts either form. The search plan supports a `"network"` field that Sonnet extracts from queries like "show me my messenger convos". When combined with other filters (sender/search/chat), network acts as a post-filter to narrow results.

### Message format for LLM prompts
- **Unified format across all LLM prompts** (assistant response, compose context, triage context): `[timestamp] [network] chat_name=X | from=Y: <msg>text</msg>`. The explicit `chat_name=` and `from=` labels prevent the LLM from confusing chat titles with senders — previously, Opus misattributed messages in DM chats where the chat title is the other person's name but the sender was Adrien.
- **Adrien's sender_name is a raw Beeper ID** (`@alemercier:beeper.com`), not a human-readable name. The `_display_sender()` helper in `assistant.py` replaces it with `Adrien (you)` in all LLM prompts (assistant response + compose prompt) so the model knows which messages are from the owner. Triage uses a separate `(Adrien — the owner)` label.
- **Network names are mapped to human-friendly display names.** Beeper uses internal IDs (`twitter`, `facebookgo`, `instagramgo`, `imessagego`) that don't match what users expect. The `_display_network()` helper in `assistant.py` maps these to display names (`X`, `Messenger`, `Instagram`, `iMessage`). Applied everywhere network names are shown to Adrien or the LLM: summaries, compose prompts, sent confirmations, disambiguation lists, and urgent notifications.

### Channel separation (Telegram vs Beeper)
- **Diplo talks TO Adrien via Telegram only.** All responses, notifications, summaries, and automation results go through the Telegram control channel. Diplo must NEVER message Adrien via Beeper (WhatsApp, iMessage, etc.).
- **Diplo talks AS Adrien to other people via Beeper or Gmail.** When sending messages on Adrien's behalf (Beeper) or replying to emails (Gmail), the recipient should believe Adrien wrote it — Diplo impersonates, never reveals it's an AI.
- **Automation actions can trigger accidental Beeper sends to Adrien.** If a scheduled automation's action text says "notify me" or "tell me about...", Sonnet can misinterpret it as a reply intent with recipient "me", resolve Adrien in the contact registry, and send via Beeper to Adrien's WhatsApp. **Fix (prompt):** the search plan extraction prompt now explicitly states that "tell me", "notify me", "send me a summary" are NOT reply intents. **Fix (code):** `_is_adrien_recipient()` in `assistant.py` catches self-references ("me", "myself", "Adrien", "moi", and `USER_SENDER_IDS` matches) and blocks the Beeper send, returning the intent text as a normal response instead.

### Lookback search (insistence handling)
- **`last_seen_at` watermark can cause false "nothing new" responses.** If Adrien asks "what's new?" (advancing the watermark), then immediately asks "check again", the watermark is already past any messages that arrived in between. The `lookback_hours` intent bypasses `last_seen_at` entirely, querying `cache.recent(hours=N)` instead.
- **Code-level safety net for insistence.** If Diplo's last response contained "nothing new" / "all quiet" / etc. AND Sonnet returns `since_last_seen` again, the code overrides it to `{"lookback_hours": 1}`. This catches cases where Sonnet fails to detect insistence.
- **Lookback queries do NOT advance the watermark.** They return `queried_cache=False` so a follow-up "what's new?" still works correctly.
- **Diplo offers to go further back.** When responding to a lookback query, the prompt instructs Diplo to ask Adrien if he wants to look further back and by how much. Sonnet handles the follow-up naturally ("go back 3 hours" → `{"lookback_hours": 3}`).

### Google OAuth gotchas
- **Token refresh is automatic but can fail silently.** Both Calendar and Gmail tokens auto-refresh when expired. If the refresh token itself is revoked (e.g., user removed access in Google account settings), the refresh fails and the provider becomes non-functional until re-setup. Gmail logs the error; Calendar raises on next query.
- **Gmail historyId has limited retention (~7 days).** If the bot is down for more than a week, the incremental poll fails with a 404. The provider detects this and falls back to an initial fetch of the last 50 inbox emails. Some emails from the gap period may be missed.
- **Calendar OAuth consent flow is lazy.** The Google Calendar API service is built on first use. If no token exists, this triggers a browser popup — which blocks the current query until Adrien completes it. Subsequent queries use the cached token.
- **Gmail and Calendar use different OAuth scopes and different credential files.** Calendar uses `google_calendar_credentials.json` + `google_calendar_token.json` (single account). Gmail uses `email_client_secret.json` + per-mailbox tokens in `data/email_tokens/`. Don't confuse the two.
- **Google API libraries are synchronous.** Both `googleapiclient` calls are wrapped in `asyncio.run_in_executor()` to avoid blocking the event loop.

### Email integration gotchas
- **Email bodies can be huge.** The Gmail provider truncates to 3000 chars at parse time. The assistant further truncates to 500 chars in the LLM prompt. Without this, a few long emails can fill the entire context window.
- **HTML-only emails need conversion.** Many emails have no text/plain part. `_strip_html()` uses `html2text` (pip) if available, otherwise regex tag stripping. The regex fallback is lossy but adequate for triage.
- **Email `from` field parsing is messy.** Gmail returns `"Display Name <email@example.com>"` or just `"email@example.com"`. `_parse_from()` handles both. Some senders have quoted display names with escaped characters.
- **Email contact registry uses `email:{mailbox}` as network.** This distinguishes email senders from messaging senders with the same name. Display in the assistant uses `_display_network()` which maps `email:work` → `work email`.
