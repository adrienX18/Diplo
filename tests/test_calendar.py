"""Tests for the calendar integration — base, manager, Google provider, and assistant integration."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio

from src.calendar.base import CalendarEvent, CalendarProvider
from src.calendar.manager import CalendarManager
from src.calendar.google import GoogleCalendarProvider, _parse_event


# ---------------------------------------------------------------------------
# CalendarEvent
# ---------------------------------------------------------------------------

class TestCalendarEvent:
    def test_str_timed_event(self):
        e = CalendarEvent(
            title="Team standup",
            start=datetime(2026, 3, 16, 9, 0),
            end=datetime(2026, 3, 16, 9, 30),
            calendar_name="Work",
        )
        s = str(e)
        assert "2026-03-16" in s
        assert "09:00-09:30" in s
        assert "(Work)" in s
        assert "Team standup" in s

    def test_str_all_day_event(self):
        e = CalendarEvent(
            title="Birthday",
            start=datetime(2026, 3, 16),
            end=datetime(2026, 3, 17),
            all_day=True,
            calendar_name="Personal",
        )
        s = str(e)
        assert "all day" in s
        assert "Birthday" in s
        assert "(Personal)" in s

    def test_str_with_location(self):
        e = CalendarEvent(
            title="Lunch",
            start=datetime(2026, 3, 16, 12, 0),
            end=datetime(2026, 3, 16, 13, 0),
            location="Cafe Roma",
        )
        s = str(e)
        assert "@ Cafe Roma" in s

    def test_str_no_calendar_name(self):
        e = CalendarEvent(
            title="Quick call",
            start=datetime(2026, 3, 16, 14, 0),
            end=datetime(2026, 3, 16, 14, 15),
        )
        s = str(e)
        assert "Quick call" in s
        # No parenthesized calendar name
        assert "(" not in s

    def test_equality(self):
        kwargs = dict(
            title="A", start=datetime(2026, 1, 1, 10, 0),
            end=datetime(2026, 1, 1, 11, 0),
        )
        assert CalendarEvent(**kwargs) == CalendarEvent(**kwargs)

    def test_fields_optional(self):
        e = CalendarEvent(
            title="Minimal",
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 1),
        )
        assert e.location is None
        assert e.description is None
        assert e.calendar_name == ""
        assert e.all_day is False


# ---------------------------------------------------------------------------
# Dummy provider for manager tests
# ---------------------------------------------------------------------------

class DummyProvider(CalendarProvider):
    def __init__(self, events: list[CalendarEvent] | None = None, fail: bool = False):
        self._events = events or []
        self._fail = fail

    async def get_events(self, start, end):
        if self._fail:
            raise RuntimeError("provider failed")
        return [e for e in self._events if start <= e.start <= end]

    async def search_events(self, query, start=None, end=None):
        if self._fail:
            raise RuntimeError("provider failed")
        return [e for e in self._events if query.lower() in e.title.lower()]


# ---------------------------------------------------------------------------
# CalendarManager
# ---------------------------------------------------------------------------

class TestCalendarManager:
    def test_no_providers(self):
        mgr = CalendarManager()
        assert not mgr.has_providers

    def test_add_provider(self):
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider())
        assert mgr.has_providers

    @pytest.mark.asyncio
    async def test_get_events_single_provider(self):
        events = [
            CalendarEvent("A", datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)),
            CalendarEvent("B", datetime(2026, 3, 17, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 17, 10, 0, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events))

        result = await mgr.get_events(
            datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 18, 0, 0, tzinfo=timezone.utc),
        )
        assert len(result) == 2
        assert result[0].title == "A"
        assert result[1].title == "B"

    @pytest.mark.asyncio
    async def test_get_events_filters_by_range(self):
        events = [
            CalendarEvent("In range", datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)),
            CalendarEvent("Out of range", datetime(2026, 3, 20, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events))

        result = await mgr.get_events(
            datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        )
        assert len(result) == 1
        assert result[0].title == "In range"

    @pytest.mark.asyncio
    async def test_get_events_multiple_providers_merged_sorted(self):
        events_a = [
            CalendarEvent("Late", datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 15, 0, tzinfo=timezone.utc), calendar_name="Work"),
        ]
        events_b = [
            CalendarEvent("Early", datetime(2026, 3, 16, 8, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc), calendar_name="Personal"),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events_a))
        mgr.add_provider(DummyProvider(events_b))

        result = await mgr.get_events(
            datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        )
        assert len(result) == 2
        assert result[0].title == "Early"
        assert result[1].title == "Late"

    @pytest.mark.asyncio
    async def test_get_events_provider_failure_graceful(self):
        good_events = [
            CalendarEvent("OK", datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(fail=True))  # broken provider
        mgr.add_provider(DummyProvider(good_events))  # working provider

        result = await mgr.get_events(
            datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        )
        assert len(result) == 1
        assert result[0].title == "OK"

    @pytest.mark.asyncio
    async def test_search_events(self):
        events = [
            CalendarEvent("Team standup", datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 9, 30, tzinfo=timezone.utc)),
            CalendarEvent("Lunch with Sophie", datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 13, 0, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events))

        result = await mgr.search_events("sophie")
        assert len(result) == 1
        assert result[0].title == "Lunch with Sophie"

    @pytest.mark.asyncio
    async def test_search_events_provider_failure_graceful(self):
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(fail=True))

        result = await mgr.search_events("anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_providers_returns_empty(self):
        mgr = CalendarManager()
        result = await mgr.get_events(
            datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        )
        assert result == []


# ---------------------------------------------------------------------------
# Google Calendar — _parse_event
# ---------------------------------------------------------------------------

class TestParseEvent:
    def test_timed_event(self):
        raw = {
            "summary": "Meeting",
            "start": {"dateTime": "2026-03-16T09:00:00-07:00"},
            "end": {"dateTime": "2026-03-16T10:00:00-07:00"},
            "location": "Room A",
            "description": "Weekly sync",
        }
        event = _parse_event(raw, "Work")
        assert event is not None
        assert event.title == "Meeting"
        assert event.calendar_name == "Work"
        assert event.location == "Room A"
        assert event.description == "Weekly sync"
        assert not event.all_day
        assert event.start.hour == 9

    def test_all_day_event(self):
        raw = {
            "summary": "Holiday",
            "start": {"date": "2026-03-16"},
            "end": {"date": "2026-03-17"},
        }
        event = _parse_event(raw, "Personal")
        assert event is not None
        assert event.all_day is True
        assert event.title == "Holiday"
        assert event.start == datetime(2026, 3, 16)
        assert event.end == datetime(2026, 3, 17)

    def test_no_title(self):
        raw = {
            "start": {"dateTime": "2026-03-16T09:00:00Z"},
            "end": {"dateTime": "2026-03-16T10:00:00Z"},
        }
        event = _parse_event(raw, "Cal")
        assert event is not None
        assert event.title == "(No title)"

    def test_missing_start_returns_none(self):
        raw = {"summary": "Bad event", "start": {}, "end": {}}
        assert _parse_event(raw, "Cal") is None

    def test_no_location_no_description(self):
        raw = {
            "summary": "Quick",
            "start": {"dateTime": "2026-03-16T09:00:00Z"},
            "end": {"dateTime": "2026-03-16T09:15:00Z"},
        }
        event = _parse_event(raw, "Cal")
        assert event is not None
        assert event.location is None
        assert event.description is None

    def test_missing_end_same_as_start(self):
        raw = {
            "summary": "Point event",
            "start": {"dateTime": "2026-03-16T09:00:00Z"},
            "end": {},
        }
        event = _parse_event(raw, "Cal")
        assert event is not None
        assert event.start == event.end


# ---------------------------------------------------------------------------
# GoogleCalendarProvider (mocked service)
# ---------------------------------------------------------------------------

class TestGoogleCalendarProvider:
    def _make_provider(self, mock_service):
        """Create a provider with a pre-built mock service."""
        provider = GoogleCalendarProvider.__new__(GoogleCalendarProvider)
        provider._credentials_path = "fake.json"
        provider._token_path = "fake_token.json"
        provider._calendar_ids = None
        provider._service = mock_service
        return provider

    def _make_mock_service(self, calendars, events_by_cal):
        """Build a mock Google Calendar service.

        Args:
            calendars: list of (id, summary) tuples
            events_by_cal: dict of cal_id -> list of raw event dicts
        """
        service = MagicMock()

        # calendarList().list()
        cal_list_items = [{"id": cid, "summary": name} for cid, name in calendars]
        cal_list_response = {"items": cal_list_items}
        service.calendarList().list().execute.return_value = cal_list_response

        # events().list() — returns events based on calendarId
        def events_list_side_effect(**kwargs):
            cal_id = kwargs.get("calendarId", "")
            events = events_by_cal.get(cal_id, [])
            mock_exec = MagicMock()
            mock_exec.execute.return_value = {"items": events}
            return mock_exec

        service.events().list = events_list_side_effect
        return service

    @pytest.mark.asyncio
    async def test_get_events_basic(self):
        raw_events = [
            {
                "summary": "Meeting",
                "start": {"dateTime": "2026-03-16T09:00:00-07:00"},
                "end": {"dateTime": "2026-03-16T10:00:00-07:00"},
            },
        ]
        service = self._make_mock_service(
            [("primary", "My Calendar")],
            {"primary": raw_events},
        )
        provider = self._make_provider(service)

        events = await provider.get_events(
            datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        )
        assert len(events) == 1
        assert events[0].title == "Meeting"
        assert events[0].calendar_name == "My Calendar"

    @pytest.mark.asyncio
    async def test_get_events_multiple_calendars(self):
        service = self._make_mock_service(
            [("work", "Work"), ("personal", "Personal")],
            {
                "work": [{"summary": "Standup", "start": {"dateTime": "2026-03-16T09:00:00Z"}, "end": {"dateTime": "2026-03-16T09:30:00Z"}}],
                "personal": [{"summary": "Gym", "start": {"dateTime": "2026-03-16T07:00:00Z"}, "end": {"dateTime": "2026-03-16T08:00:00Z"}}],
            },
        )
        provider = self._make_provider(service)

        events = await provider.get_events(
            datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        )
        assert len(events) == 2
        # Should be sorted by start time
        assert events[0].title == "Gym"
        assert events[1].title == "Standup"

    @pytest.mark.asyncio
    async def test_search_events_with_query(self):
        raw_events = [
            {
                "summary": "Sophie lunch",
                "start": {"dateTime": "2026-03-16T12:00:00Z"},
                "end": {"dateTime": "2026-03-16T13:00:00Z"},
            },
        ]
        service = self._make_mock_service(
            [("primary", "My Calendar")],
            {"primary": raw_events},
        )
        provider = self._make_provider(service)

        events = await provider.search_events("Sophie")
        assert len(events) == 1
        assert events[0].title == "Sophie lunch"

    @pytest.mark.asyncio
    async def test_get_events_empty(self):
        service = self._make_mock_service(
            [("primary", "My Calendar")],
            {"primary": []},
        )
        provider = self._make_provider(service)

        events = await provider.get_events(
            datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        )
        assert events == []

    @pytest.mark.asyncio
    async def test_get_events_with_specific_calendar_ids(self):
        service = MagicMock()
        # When specific calendar_ids are given, provider calls calendars().get() per ID
        cal_response = MagicMock()
        cal_response.execute.return_value = {"summary": "My Work Cal"}
        service.calendars().get.return_value = cal_response

        events_response = MagicMock()
        events_response.execute.return_value = {
            "items": [
                {
                    "summary": "Review",
                    "start": {"dateTime": "2026-03-16T14:00:00Z"},
                    "end": {"dateTime": "2026-03-16T15:00:00Z"},
                },
            ]
        }
        service.events().list = lambda **kwargs: events_response

        provider = GoogleCalendarProvider.__new__(GoogleCalendarProvider)
        provider._credentials_path = "fake.json"
        provider._token_path = "fake_token.json"
        provider._calendar_ids = ["work@group.calendar.google.com"]
        provider._service = service

        events = await provider.get_events(
            datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        )
        assert len(events) == 1
        assert events[0].title == "Review"
        assert events[0].calendar_name == "My Work Cal"

    @pytest.mark.asyncio
    async def test_handles_all_day_events(self):
        raw_events = [
            {
                "summary": "Conference",
                "start": {"date": "2026-03-16"},
                "end": {"date": "2026-03-18"},
            },
        ]
        service = self._make_mock_service(
            [("primary", "Cal")],
            {"primary": raw_events},
        )
        provider = self._make_provider(service)

        events = await provider.get_events(
            datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 19, 0, 0, tzinfo=timezone.utc),
        )
        assert len(events) == 1
        assert events[0].all_day is True
        assert events[0].title == "Conference"


# ---------------------------------------------------------------------------
# Assistant integration — _fetch_calendar_events, _format_calendar_events
# ---------------------------------------------------------------------------

from src.assistant import _fetch_calendar_events, _format_calendar_events, handle_user_message


class TestFetchCalendarEvents:
    @pytest.mark.asyncio
    async def test_basic_date_range(self):
        events = [
            CalendarEvent("Meeting", datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events))

        result = await _fetch_calendar_events(
            {"start": "2026-03-16", "end": "2026-03-17"},
            mgr, "UTC",
        )
        assert len(result) == 1
        assert result[0].title == "Meeting"

    @pytest.mark.asyncio
    async def test_with_query(self):
        events = [
            CalendarEvent("Sophie lunch", datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 13, 0, tzinfo=timezone.utc)),
            CalendarEvent("Team standup", datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 9, 30, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events))

        result = await _fetch_calendar_events(
            {"start": "2026-03-16", "end": "2026-03-17", "query": "Sophie"},
            mgr, "UTC",
        )
        assert len(result) == 1
        assert result[0].title == "Sophie lunch"

    @pytest.mark.asyncio
    async def test_no_start_defaults_to_now(self):
        # Event in the far future
        events = [
            CalendarEvent("Future", datetime(2099, 1, 1, 9, 0, tzinfo=timezone.utc),
                          datetime(2099, 1, 1, 10, 0, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events))

        result = await _fetch_calendar_events({}, mgr, "UTC")
        # Default range is now to now+7 days, so far future event won't match
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_end_date_inclusive(self):
        """End date should include the full day (23:59:59)."""
        events = [
            CalendarEvent("Evening", datetime(2026, 3, 17, 20, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 17, 21, 0, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events))

        result = await _fetch_calendar_events(
            {"start": "2026-03-17", "end": "2026-03-17"},
            mgr, "UTC",
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_failure_returns_empty(self):
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(fail=True))

        result = await _fetch_calendar_events(
            {"start": "2026-03-16", "end": "2026-03-17"},
            mgr, "UTC",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_timezone_aware(self):
        tz = ZoneInfo("America/Los_Angeles")
        events = [
            CalendarEvent("LA event", datetime(2026, 3, 16, 9, 0, tzinfo=tz),
                          datetime(2026, 3, 16, 10, 0, tzinfo=tz)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(events))

        result = await _fetch_calendar_events(
            {"start": "2026-03-16", "end": "2026-03-17"},
            mgr, "America/Los_Angeles",
        )
        assert len(result) == 1


class TestFormatCalendarEvents:
    def test_timed_event(self):
        events = [
            CalendarEvent("Meeting", datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
                          calendar_name="Work"),
        ]
        text = _format_calendar_events(events, "UTC")
        assert "2026-03-16" in text
        assert "09:00-10:00" in text
        assert "(Work)" in text
        assert "Meeting" in text

    def test_all_day_event(self):
        events = [
            CalendarEvent("Holiday", datetime(2026, 3, 16),
                          datetime(2026, 3, 17), all_day=True),
        ]
        text = _format_calendar_events(events, "UTC")
        assert "(all day)" in text
        assert "Holiday" in text

    def test_with_location(self):
        events = [
            CalendarEvent("Lunch", datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 13, 0, tzinfo=timezone.utc),
                          location="Cafe Roma"),
        ]
        text = _format_calendar_events(events, "UTC")
        assert "@ Cafe Roma" in text

    def test_timezone_conversion(self):
        events = [
            CalendarEvent("Meeting", datetime(2026, 3, 16, 16, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 17, 0, tzinfo=timezone.utc)),
        ]
        text = _format_calendar_events(events, "America/Los_Angeles")
        # UTC 16:00 = PDT 09:00 (March is DST)
        assert "09:00-10:00" in text

    def test_multi_day_event(self):
        events = [
            CalendarEvent("Trip", datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc)),
        ]
        text = _format_calendar_events(events, "UTC")
        assert "2026-03-16 10:00" in text
        assert "2026-03-18 18:00" in text

    def test_empty_events(self):
        text = _format_calendar_events([], "UTC")
        assert text == ""

    def test_multiple_events_sorted_display(self):
        events = [
            CalendarEvent("Early", datetime(2026, 3, 16, 8, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)),
            CalendarEvent("Late", datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 15, 0, tzinfo=timezone.utc)),
        ]
        text = _format_calendar_events(events, "UTC")
        early_pos = text.index("Early")
        late_pos = text.index("Late")
        assert early_pos < late_pos


# ---------------------------------------------------------------------------
# Query plan extraction — calendar field
# ---------------------------------------------------------------------------

class TestQueryPlanCalendar:
    """Test that _extract_query_plan includes calendar field in the prompt when has_calendar=True."""

    @pytest.mark.asyncio
    async def test_calendar_section_included_when_has_calendar(self):
        """Verify the calendar instructions are in the prompt sent to Sonnet."""
        captured_system = {}

        async def mock_complete(model, system, messages, max_tokens, call_type):
            captured_system["prompt"] = system
            return '{"calendar": {"start": "2026-03-16", "end": "2026-03-22"}, "no_query": true}'

        with patch("src.assistant.complete", side_effect=mock_complete):
            from src.assistant import _extract_query_plan
            plan = await _extract_query_plan("when am I free next week?", None, "", has_calendar=True)

        assert "calendar" in captured_system["prompt"]
        assert "schedule" in captured_system["prompt"].lower() or "availability" in captured_system["prompt"].lower()

    @pytest.mark.asyncio
    async def test_calendar_section_not_included_without_calendar(self):
        captured_system = {}

        async def mock_complete(model, system, messages, max_tokens, call_type):
            captured_system["prompt"] = system
            return '{"since_last_seen": true}'

        with patch("src.assistant.complete", side_effect=mock_complete):
            from src.assistant import _extract_query_plan
            await _extract_query_plan("what's new?", None, "", has_calendar=False)

        # The calendar examples should NOT be in the prompt
        assert "when am I free" not in captured_system["prompt"]

    @pytest.mark.asyncio
    async def test_calendar_plan_parsed_correctly(self):
        async def mock_complete(model, system, messages, max_tokens, call_type):
            return '{"calendar": {"start": "2026-03-16", "end": "2026-03-22", "query": "Sophie"}, "no_query": true}'

        with patch("src.assistant.complete", side_effect=mock_complete):
            from src.assistant import _extract_query_plan
            plan = await _extract_query_plan("do I have a meeting with Sophie?", None, "", has_calendar=True)

        assert "calendar" in plan
        assert plan["calendar"]["start"] == "2026-03-16"
        assert plan["calendar"]["end"] == "2026-03-22"
        assert plan["calendar"]["query"] == "Sophie"
        assert plan.get("no_query") is True


# ---------------------------------------------------------------------------
# handle_user_message integration — calendar flows through to response
# ---------------------------------------------------------------------------

class TestHandleUserMessageCalendar:
    @pytest.mark.asyncio
    async def test_calendar_events_passed_to_response(self):
        """Calendar events should appear in the prompt sent to Opus for response generation."""
        captured_prompts = []

        async def mock_complete(model, system, messages, max_tokens, call_type):
            captured_prompts.append({"call_type": call_type, "user": messages[0]["content"] if messages else ""})
            if call_type == "route_intent":
                return "query"
            if call_type == "query_plan":
                return '{"calendar": {"start": "2026-03-16", "end": "2026-03-17"}, "no_query": true}'
            if call_type == "response":
                return "You have a meeting at 9am."
            return "{}"

        calendar_events = [
            CalendarEvent("Standup", datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 9, 30, tzinfo=timezone.utc), calendar_name="Work"),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(calendar_events))

        # Minimal cache mock
        cache = MagicMock()
        cache.get_timezone.return_value = "UTC"
        cache.get_last_seen.return_value = None

        convo = MagicMock()
        convo.format_for_prompt.return_value = ""
        convo.format_session_for_prompt.return_value = ""
        convo.recent.return_value = []

        with patch("src.assistant.complete", side_effect=mock_complete):
            response, queried = await handle_user_message(
                "what's on my calendar tomorrow?",
                cache, convo, calendar=mgr,
            )

        assert response == "You have a meeting at 9am."
        # Find the response call and check calendar events are in the prompt
        response_call = next(c for c in captured_prompts if c["call_type"] == "response")
        assert "Calendar events" in response_call["user"]
        assert "Standup" in response_call["user"]

    @pytest.mark.asyncio
    async def test_no_calendar_no_calendar_section(self):
        """Without a calendar manager, no calendar section should appear."""
        captured_prompts = []

        async def mock_complete(model, system, messages, max_tokens, call_type):
            captured_prompts.append({"call_type": call_type, "user": messages[0]["content"] if messages else ""})
            if call_type == "route_intent":
                return "query"
            if call_type == "query_plan":
                return '{"since_last_seen": true}'
            if call_type == "response":
                return "Nothing new boss."
            return "{}"

        cache = MagicMock()
        cache.get_timezone.return_value = "UTC"
        cache.get_last_seen.return_value = None
        cache.since_last_seen.return_value = []

        convo = MagicMock()
        convo.format_for_prompt.return_value = ""
        convo.format_session_for_prompt.return_value = ""
        convo.recent.return_value = []

        with patch("src.assistant.complete", side_effect=mock_complete):
            response, queried = await handle_user_message(
                "what's new?", cache, convo, calendar=None,
            )

        response_call = next(c for c in captured_prompts if c["call_type"] == "response")
        assert "Calendar events" not in response_call["user"]

    @pytest.mark.asyncio
    async def test_calendar_combined_with_messages(self):
        """Calendar can be fetched alongside message cache queries."""
        captured_prompts = []

        async def mock_complete(model, system, messages, max_tokens, call_type):
            captured_prompts.append({"call_type": call_type, "user": messages[0]["content"] if messages else ""})
            if call_type == "route_intent":
                return "query"
            if call_type == "query_plan":
                return '{"calendar": {"start": "2026-03-16", "end": "2026-03-17", "query": "Sophie"}, "sender": "Sophie"}'
            if call_type == "response":
                return "Sophie confirmed the meeting."
            return "{}"

        cal_events = [
            CalendarEvent("Lunch with Sophie", datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
                          datetime(2026, 3, 16, 13, 0, tzinfo=timezone.utc)),
        ]
        mgr = CalendarManager()
        mgr.add_provider(DummyProvider(cal_events))

        cache = MagicMock()
        cache.get_timezone.return_value = "UTC"
        cache.get_last_seen.return_value = None
        cache.by_sender.return_value = [
            {"message_id": "1", "chat_id": "c1", "chat_title": "Sophie", "network": "whatsapp",
             "sender_name": "Sophie", "text": "See you at noon!", "timestamp": "2026-03-16T10:00:00Z",
             "has_attachments": False},
        ]
        cache.by_chat_id.return_value = []
        cache.recent.return_value = []

        convo = MagicMock()
        convo.format_for_prompt.return_value = ""
        convo.format_session_for_prompt.return_value = ""
        convo.recent.return_value = []

        with patch("src.assistant.complete", side_effect=mock_complete):
            from src.assistant import handle_user_message
            response, _ = await handle_user_message(
                "did Sophie confirm our lunch meeting?",
                cache, convo, calendar=mgr,
            )

        response_call = next(c for c in captured_prompts if c["call_type"] == "response")
        # Both messages AND calendar events should be in the prompt
        assert "Sophie" in response_call["user"]
        assert "Calendar events" in response_call["user"]
        assert "Lunch with Sophie" in response_call["user"]

    @pytest.mark.asyncio
    async def test_empty_calendar_still_shows_section(self):
        """When calendar is queried but has no events, show a 'no events' section."""
        captured_prompts = []

        async def mock_complete(model, system, messages, max_tokens, call_type):
            captured_prompts.append({"call_type": call_type, "user": messages[0]["content"] if messages else ""})
            if call_type == "route_intent":
                return "query"
            if call_type == "query_plan":
                return '{"calendar": {"start": "2026-03-16", "end": "2026-03-17"}, "no_query": true}'
            if call_type == "response":
                return "Your calendar is clear!"
            return "{}"

        mgr = CalendarManager()
        mgr.add_provider(DummyProvider([]))  # empty calendar

        cache = MagicMock()
        cache.get_timezone.return_value = "UTC"
        cache.get_last_seen.return_value = None

        convo = MagicMock()
        convo.format_for_prompt.return_value = ""
        convo.format_session_for_prompt.return_value = ""
        convo.recent.return_value = []

        with patch("src.assistant.complete", side_effect=mock_complete):
            response, _ = await handle_user_message(
                "am I free tomorrow?", cache, convo, calendar=mgr,
            )

        response_call = next(c for c in captured_prompts if c["call_type"] == "response")
        assert "No calendar events found" in response_call["user"]


# ---------------------------------------------------------------------------
# Import / module structure
# ---------------------------------------------------------------------------

class TestImports:
    def test_calendar_package_imports(self):
        from src.calendar import CalendarEvent, CalendarProvider, CalendarManager
        assert CalendarEvent is not None
        assert CalendarProvider is not None
        assert CalendarManager is not None

    def test_google_provider_importable(self):
        from src.calendar.google import GoogleCalendarProvider
        assert GoogleCalendarProvider is not None

    def test_base_module_importable(self):
        from src.calendar.base import CalendarEvent, CalendarProvider
        assert CalendarEvent is not None

    def test_manager_module_importable(self):
        from src.calendar.manager import CalendarManager
        assert CalendarManager is not None
