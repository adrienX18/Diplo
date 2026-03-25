import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BEEPER_ACCESS_TOKEN = os.environ["BEEPER_ACCESS_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = int(os.environ["TELEGRAM_CHAT_ID"]) if os.environ.get("TELEGRAM_CHAT_ID") else None

POLL_INTERVAL_SECONDS = 5
MESSAGE_CACHE_RETENTION_DAYS = 14
URGENT_BATCH_DELAY_SECONDS = 20
PRUNE_INTERVAL_SECONDS = 3600  # Prune + consolidation cycle
DEFAULT_TIMEZONE = "America/Los_Angeles"
SCHEDULER_INTERVAL_SECONDS = 30

# User identity
USER_NAME = os.environ.get("USER_NAME", "User")

# Sender IDs that identify the user across networks (lowercase substrings)
USER_SENDER_IDS = [
    s.strip().lower()
    for s in os.environ.get("USER_SENDER_IDS", "").split(",")
    if s.strip()
]

# Bot identity — used to detect and skip the control channel chat in Beeper
BOT_SENDER_NAMES = ["diplo", "pokebeeper"]

# Sender email addresses (for identifying the user's own emails)
USER_EMAIL_ADDRESSES = [
    a.strip().lower()
    for a in os.environ.get("USER_EMAIL_ADDRESSES", "").split(",")
    if a.strip()
]

# Email polling interval (seconds)
EMAIL_POLL_INTERVAL = 20
# How many emails to fetch on initial connect (backfills ~5 days)
EMAIL_INITIAL_FETCH_LIMIT = 1000

# Google Calendar OAuth2
GOOGLE_CALENDAR_CREDENTIALS = os.environ.get("GOOGLE_CALENDAR_CREDENTIALS", "")
GOOGLE_CALENDAR_TOKEN = os.environ.get("GOOGLE_CALENDAR_TOKEN", "")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "messages.db"
EMAIL_DB_PATH = DATA_DIR / "emails.db"
