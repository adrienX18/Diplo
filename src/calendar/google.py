"""Google Calendar provider — read-only access via Google Calendar API."""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.calendar.base import CalendarEvent, CalendarProvider

logger = logging.getLogger(__name__)

# Google API dependencies — imported lazily so the module can be imported
# even when google libs aren't installed (only fails when actually used).
_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def _build_service(credentials_path: str, token_path: str):
    """Build a Google Calendar API service object.

    Handles OAuth2 flow:
    - If a valid token exists at token_path, uses it (refreshing if expired).
    - Otherwise, runs the browser-based OAuth2 consent flow and saves the token.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    token = Path(token_path)

    if token.exists():
        creds = Credentials.from_authorized_user_file(str(token), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, _SCOPES
            )
            creds = flow.run_local_server(port=0)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def _parse_event(event: dict, calendar_name: str) -> CalendarEvent | None:
    """Parse a Google Calendar API event into a CalendarEvent."""
    title = event.get("summary", "(No title)")
    location = event.get("location")
    description = event.get("description")

    start_raw = event.get("start", {})
    end_raw = event.get("end", {})

    all_day = "date" in start_raw and "dateTime" not in start_raw

    if all_day:
        start = datetime.fromisoformat(start_raw["date"])
        end = datetime.fromisoformat(end_raw.get("date", start_raw["date"]))
    else:
        start_str = start_raw.get("dateTime")
        end_str = end_raw.get("dateTime")
        if not start_str:
            return None
        start = datetime.fromisoformat(start_str)
        end = datetime.fromisoformat(end_str) if end_str else start

    return CalendarEvent(
        title=title,
        start=start,
        end=end,
        location=location,
        description=description,
        calendar_name=calendar_name,
        all_day=all_day,
    )


class GoogleCalendarProvider(CalendarProvider):
    """Read-only Google Calendar provider.

    Args:
        credentials_path: Path to the OAuth2 client credentials JSON file
            (downloaded from Google Cloud Console).
        token_path: Path where the user's OAuth2 token will be stored/refreshed.
        calendar_ids: Optional list of calendar IDs to fetch. Defaults to
            all calendars visible to the authenticated user.
    """

    def __init__(
        self,
        credentials_path: str,
        token_path: str,
        calendar_ids: list[str] | None = None,
    ) -> None:
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._calendar_ids = calendar_ids
        self._service = None

    def _get_service(self):
        """Lazy-init the Google Calendar service."""
        if self._service is None:
            self._service = _build_service(
                self._credentials_path, self._token_path
            )
        return self._service

    def _get_calendar_ids_and_names(self) -> list[tuple[str, str]]:
        """Get (calendar_id, calendar_name) pairs to query.

        If calendar_ids was specified, uses those. Otherwise fetches all
        visible calendars from the API.
        """
        service = self._get_service()
        if self._calendar_ids:
            # Fetch names for the specified IDs
            result = []
            for cal_id in self._calendar_ids:
                try:
                    cal = service.calendars().get(calendarId=cal_id).execute()
                    result.append((cal_id, cal.get("summary", cal_id)))
                except Exception:
                    result.append((cal_id, cal_id))
            return result

        # Fetch all visible calendars
        calendars = []
        page_token = None
        while True:
            calendar_list = (
                service.calendarList()
                .list(pageToken=page_token)
                .execute()
            )
            for entry in calendar_list.get("items", []):
                cal_id = entry["id"]
                cal_name = entry.get("summary", cal_id)
                calendars.append((cal_id, cal_name))
            page_token = calendar_list.get("nextPageToken")
            if not page_token:
                break
        return calendars

    def _fetch_events(
        self,
        start: datetime,
        end: datetime,
        query: str | None = None,
    ) -> list[CalendarEvent]:
        """Synchronous event fetch across all calendars."""
        service = self._get_service()
        calendars = self._get_calendar_ids_and_names()

        # Ensure timezone-aware ISO format
        time_min = start.isoformat()
        time_max = end.isoformat()

        all_events: list[CalendarEvent] = []

        for cal_id, cal_name in calendars:
            try:
                page_token = None
                while True:
                    kwargs = {
                        "calendarId": cal_id,
                        "timeMin": time_min,
                        "timeMax": time_max,
                        "singleEvents": True,
                        "orderBy": "startTime",
                        "maxResults": 250,
                    }
                    if query:
                        kwargs["q"] = query
                    if page_token:
                        kwargs["pageToken"] = page_token

                    result = service.events().list(**kwargs).execute()

                    for event in result.get("items", []):
                        parsed = _parse_event(event, cal_name)
                        if parsed:
                            all_events.append(parsed)

                    page_token = result.get("nextPageToken")
                    if not page_token:
                        break
            except Exception:
                logger.exception("Failed to fetch events from calendar %s", cal_name)

        all_events.sort(key=lambda e: e.start)
        return all_events

    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[CalendarEvent]:
        """Fetch events within the time range from all calendars."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_events, start, end, None
        )

    async def search_events(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[CalendarEvent]:
        """Search events by text query."""
        # Default range: 30 days back to 90 days forward
        if start is None:
            from datetime import timedelta

            start = datetime.now(timezone.utc) - timedelta(days=30)
        if end is None:
            from datetime import timedelta

            end = datetime.now(timezone.utc) + timedelta(days=90)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_events, start, end, query
        )
