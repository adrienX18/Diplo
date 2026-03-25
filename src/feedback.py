"""Feedback store — file-based feedback collection and rule consolidation.

Raw feedback lives in prompts/feedback.md (one entry per line, prefixed with --).
Consolidated rules live in prompts/learned_rules.md (max 30 lines).
Base prompts (triage_system.md, assistant_system.md) are never modified.
"""

import logging
import re
from pathlib import Path

from src.llm import complete
from src.llm_logger import new_context_id

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
FEEDBACK_FILE = PROMPTS_DIR / "feedback.md"
RULES_FILE = PROMPTS_DIR / "learned_rules.md"
BASE_TRIAGE_PROMPT = PROMPTS_DIR / "triage_system.md"
BASE_ASSISTANT_PROMPT = PROMPTS_DIR / "assistant_system.md"

CONSOLIDATION_MODEL = "claude-opus-4-6"
MAX_RULES_LINES = 30

# In-memory cache for learned rules. Populated on first load_rules() call,
# invalidated after consolidation writes new rules. Avoids reading from disk
# on every triage/assistant LLM call (rules change at most once per hour).
_rules_cache: str | None = None


def append_feedback(text: str):
    """Append a feedback entry to feedback.md."""
    FEEDBACK_FILE.touch(exist_ok=True)
    with open(FEEDBACK_FILE, "a") as f:
        f.write(f"-- {text}\n")
    logger.info("Stored feedback: %s", text[:100])


def load_rules() -> str:
    """Load the current learned rules, or empty string if none.

    Returns a cached copy after the first read. The cache is invalidated
    when run_consolidation() writes new rules.
    """
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    if RULES_FILE.exists():
        content = RULES_FILE.read_text().strip()
        lines = content.split("\n")
        if len(lines) > MAX_RULES_LINES:
            content = "\n".join(lines[:MAX_RULES_LINES])
        _rules_cache = content
        return content
    _rules_cache = ""
    return ""


def has_pending_feedback() -> bool:
    """Check if there's unprocessed feedback waiting for consolidation."""
    if not FEEDBACK_FILE.exists():
        return False
    content = FEEDBACK_FILE.read_text().strip()
    return bool(content)


async def run_consolidation():
    """Consolidate raw feedback into updated learned rules.

    Reads feedback.md + current learned_rules.md + base prompts,
    asks Opus to produce an updated ruleset, writes it, clears feedback.md.
    """
    global _rules_cache

    if not has_pending_feedback():
        return

    new_context_id()
    feedback_text = FEEDBACK_FILE.read_text().strip()
    # Clear immediately BEFORE the async Opus call to prevent data loss.
    # If new feedback arrives while Opus is thinking, it lands in the
    # now-empty file and is safe for the next consolidation cycle.
    FEEDBACK_FILE.write_text("")

    current_rules = load_rules()
    base_triage = BASE_TRIAGE_PROMPT.read_text().strip()
    base_assistant = BASE_ASSISTANT_PROMPT.read_text().strip()

    from src.config import USER_NAME
    system = f"""You are maintaining a set of learned rules for Diplo, {USER_NAME}'s AI messaging assistant.

{USER_NAME} gives freeform feedback over time ("that wasn't urgent", "always prioritize Sophie", "summaries are too long", etc.). Your job is to distill ALL feedback into a compact, updated set of rules.

Constraints:
- Output ONLY the rules, no preamble or explanation
- Maximum 30 lines
- Rules should be actionable and specific
- Stay true to the spirit of the base prompts (provided below) — your rules supplement them, never contradict them
- If new feedback contradicts an existing rule, update the rule
- If new feedback reinforces an existing rule, keep it (don't duplicate)
- Group related rules together
- Use bullet points (- prefix)"""

    user_content = f"""## Base triage prompt (DO NOT modify, for reference only)
{base_triage}

## Base assistant prompt (DO NOT modify, for reference only)
{base_assistant}

## Current learned rules
{current_rules if current_rules else "(none yet)"}

## New feedback to incorporate
{feedback_text}

Produce the updated learned rules (max 30 lines). Output ONLY the rules."""

    success = False
    try:
        new_rules = await complete(
            model=CONSOLIDATION_MODEL,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1500,
            call_type="consolidation",
        )

        # Validate output before writing
        if not _validate_rules(new_rules):
            logger.warning("Consolidation output failed validation — restoring feedback")
            return

        # Enforce line cap
        lines = new_rules.strip().split("\n")
        if len(lines) > MAX_RULES_LINES:
            new_rules = "\n".join(lines[:MAX_RULES_LINES])

        final = new_rules.strip()
        RULES_FILE.write_text(final + "\n")
        _rules_cache = final
        success = True
        logger.info("Consolidation complete: %d rules, cleared %d bytes of feedback",
                     len(new_rules.strip().split("\n")), len(feedback_text))
    except Exception:
        logger.exception("Feedback consolidation failed — restoring feedback for next cycle")
    finally:
        if not success:
            _restore_feedback(feedback_text)


_MAX_RULE_CHARS = 3000
_BULLET_RE = re.compile(r"^\s*-\s+\S")


def _validate_rules(text: str) -> bool:
    """Check that consolidation output looks like a valid ruleset.

    Rejects output that is:
    - Empty or whitespace-only
    - Too long (>3000 chars — likely an essay, not rules)
    - Missing bullet points (rules must use "- " prefix)
    """
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > _MAX_RULE_CHARS:
        return False
    lines = [l for l in stripped.split("\n") if l.strip()]
    if not lines:
        return False
    # At least a third of non-empty lines should be bullet points.
    # Rules may have section headers or continuation lines under bullets.
    bullet_count = sum(1 for l in lines if _BULLET_RE.match(l))
    if bullet_count < len(lines) / 3:
        return False
    return True


def _restore_feedback(feedback_text: str):
    """Prepend previously cleared feedback back to the file (new entries may exist)."""
    existing = ""
    if FEEDBACK_FILE.exists():
        existing = FEEDBACK_FILE.read_text()
    FEEDBACK_FILE.write_text(feedback_text + "\n" + existing if existing else feedback_text + "\n")
