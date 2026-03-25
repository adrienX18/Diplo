# Diplo

An AI layer between you and the outside world. It watches all your messages, email, and calendar — across every platform — and gives you one unified interface to manage it all, via Telegram.

## What it does

- **Urgent alerts**: Diplo reads every incoming message and email, and pings you on Telegram when something actually needs your attention. Everything else stays quiet.
- **"What did I miss?"**: Ask in plain English and get a concise summary — filtered by person, platform, topic, or time range. Voice messages and photos work too.
- **Reply through it**: "Tell Sophie I'll be late" — Diplo composes a message matching the conversation's tone and sends it through the right platform.
- **Automations**: "Every morning at 9am, summarize my messages" or "Whenever Sophie messages, notify me" — set up recurring tasks and triggers in plain language.
- **Email & calendar**: Check your inbox, reply to threads, ask about your schedule. Gmail and Google Calendar are built in today — the architecture is provider-based, so adding Outlook, CalDAV, or any other service is just writing a small adapter.
- **Learns from you**: Tell Diplo "that wasn't urgent", "always prioritize Sophie", or "your summaries are too long" — it remembers your preferences and adjusts its behavior over time. Feedback is automatically distilled into rules that shape how it triages, summarizes, and responds.
- **Self-aware**: Ask Diplo what it can do, how it works, or why something went wrong — it knows its own capabilities and can investigate its own past decisions to explain exactly what happened.

Diplo does not browse the internet or access external websites. It only works with your messages, email, and calendar data.

## Prerequisites

You need these four things before starting. Set them up first, then the install is just copy-paste.

| What | Why | Get it |
|------|-----|--------|
| **Python 3.12+** | Diplo is written in Python | [python.org/downloads](https://www.python.org/downloads/) |
| **uv** | Package manager (fast, handles everything) | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| **Beeper Desktop** | Connects to all your messaging platforms via a local API | [beeper.com](https://www.beeper.com/) |
| **Telegram bot** | Your interface to talk to Diplo | Create one via [@BotFather](https://t.me/BotFather) — send `/newbot`, follow the prompts, copy the token |

You'll also need API keys:

| Key | What it's for | Get it |
|-----|---------------|--------|
| **Anthropic API key** | Powers the AI (Claude) | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| **Beeper access token** | Reads your messages | Beeper Desktop → Settings → Developers → API Access Token |
| **Telegram chat ID** | Restricts the bot to only you | Message [@userinfobot](https://t.me/userinfobot) on Telegram — it replies with your ID |
| OpenAI API key *(optional)* | Voice transcription + LLM fallback | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

## Quick start

Once you have the prerequisites, open a terminal and run these commands:

```bash
# 1. Clone the repo
git clone https://github.com/adrienX18/Diplo.git
cd Diplo

# 2. Install dependencies
uv sync

# 3. Set up your config
cp .env.example .env
```

Now open `.env` in any text editor and fill in your keys. The file has comments explaining each one — at minimum you need:

```
BEEPER_ACCESS_TOKEN=...
ANTHROPIC_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
USER_NAME=Your Name
USER_SENDER_IDS=your beeper username,your full name
```

Then start Diplo:

```bash
# 4. Run
python3 -m src.main
```

You should see:

```
Diplo — Listening + Triage + Telegram + Reply + Feedback + Gmail + Calendar
Telegram bot started
Listening for new messages. Press Ctrl+C to stop.
```

Send a message to your bot on Telegram. If it replies, you're good.

## Optional: Gmail & Google Calendar

Email monitoring and calendar queries are optional. See [GUIDE.md](GUIDE.md) for step-by-step setup instructions.

## Learn more

See [GUIDE.md](GUIDE.md) for the full picture — every capability explained, architecture decisions, and how things work under the hood.
