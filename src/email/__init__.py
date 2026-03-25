"""Email integration — abstract provider, Gmail adapter, cache, and manager."""

from src.email.base import EmailProvider
from src.email.cache import EmailCache
from src.email.manager import EmailManager

__all__ = ["EmailProvider", "EmailCache", "EmailManager"]
