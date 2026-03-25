"""Calendar integration — abstract provider + aggregation manager."""

from src.calendar.base import CalendarEvent, CalendarProvider
from src.calendar.manager import CalendarManager

__all__ = ["CalendarEvent", "CalendarProvider", "CalendarManager"]
