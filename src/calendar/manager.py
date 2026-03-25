"""CalendarManager — aggregates multiple calendar providers into a single interface."""

import logging
from datetime import datetime

from src.calendar.base import CalendarEvent, CalendarProvider

logger = logging.getLogger(__name__)


class CalendarManager:
    """Aggregates multiple CalendarProviders and merges their results."""

    def __init__(self) -> None:
        self._providers: list[CalendarProvider] = []

    def add_provider(self, provider: CalendarProvider) -> None:
        """Register a calendar provider."""
        self._providers.append(provider)

    @property
    def has_providers(self) -> bool:
        """Whether any calendar providers are registered."""
        return len(self._providers) > 0

    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[CalendarEvent]:
        """Fetch events from all providers, merged and sorted by start time."""
        all_events: list[CalendarEvent] = []
        for provider in self._providers:
            try:
                events = await provider.get_events(start, end)
                all_events.extend(events)
            except Exception:
                logger.exception(
                    "Failed to fetch events from %s", type(provider).__name__
                )
        all_events.sort(key=lambda e: e.start)
        return all_events

    async def search_events(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[CalendarEvent]:
        """Search events across all providers, merged and sorted."""
        all_events: list[CalendarEvent] = []
        for provider in self._providers:
            try:
                events = await provider.search_events(query, start, end)
                all_events.extend(events)
            except Exception:
                logger.exception(
                    "Failed to search events from %s", type(provider).__name__
                )
        all_events.sort(key=lambda e: e.start)
        return all_events
