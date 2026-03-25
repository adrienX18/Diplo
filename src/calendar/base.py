"""Abstract base class for calendar providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CalendarEvent:
    """A single calendar event, normalized across providers."""

    title: str
    start: datetime
    end: datetime
    location: str | None = None
    description: str | None = None
    calendar_name: str = ""
    all_day: bool = False

    def __str__(self) -> str:
        if self.all_day:
            date_str = self.start.strftime("%Y-%m-%d")
            time_part = "all day"
        else:
            date_str = self.start.strftime("%Y-%m-%d")
            time_part = f"{self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')}"
        parts = [f"[{date_str} {time_part}]"]
        if self.calendar_name:
            parts.append(f"({self.calendar_name})")
        parts.append(self.title)
        if self.location:
            parts.append(f"@ {self.location}")
        return " ".join(parts)


class CalendarProvider(ABC):
    """Base class for calendar providers (Google Calendar, Outlook, CalDAV, etc.)."""

    @abstractmethod
    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[CalendarEvent]:
        """Fetch all events within the given time range.

        Args:
            start: Start of range (timezone-aware).
            end: End of range (timezone-aware).

        Returns:
            List of events sorted by start time.
        """

    @abstractmethod
    async def search_events(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[CalendarEvent]:
        """Search events by text query within an optional time range.

        Args:
            query: Text to search for in event titles/descriptions.
            start: Optional start of range.
            end: Optional end of range.

        Returns:
            List of matching events sorted by start time.
        """
