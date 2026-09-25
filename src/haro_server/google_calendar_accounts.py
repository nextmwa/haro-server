# src/haro_server/google_calendar_accounts.py
"""Loads the list of Google Calendar accounts to poll from a JSON file
(see tools/setup_google_calendar.py for how each account's token file is
produced, and .env.example for GOOGLE_CALENDAR_ACCOUNTS_PATH).

Kept out of config.py deliberately: every other Config field is a plain
env value with no I/O, but this one names a JSON file whose contents
(one entry per Google account, each with its own OAuth token and list of
calendar IDs) server.py needs to turn into real API client objects --
that construction happens in server.py's run_scheduler, matching how
GOOGLE_CALENDAR_CREDENTIALS_PATH's single Credentials object used to be
built there directly.
"""
import dataclasses
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GoogleCalendarAccount:
    label: str  # spoken in announcements, e.g. "tra poco hai [lavoro]: ..."
    credentials_path: str
    calendar_ids: list[str]  # usually just ["primary"], but a shared
    # calendar visible to this same account can be added here without any
    # new OAuth authorization -- see event_poller.py's poll_calendar().


def load_accounts(path: str) -> list[GoogleCalendarAccount]:
    with open(path) as f:
        raw = json.load(f)
    return [
        GoogleCalendarAccount(
            label=entry["label"],
            credentials_path=entry["credentials_path"],
            calendar_ids=entry["calendar_ids"],
        )
        for entry in raw
    ]


def build_calendar_services(path: str) -> list[tuple[str, Any, list[str]]]:
    """(label, Google Calendar API client, calendar ids) for every account in
    `path` whose token loads. One account's bad or expired token only drops
    that account (logged), never the others. Each call builds NEW clients:
    googleapiclient's HTTP layer isn't thread-safe, so the poller and the
    assistant's agenda tool (which reads calendars from worker threads)
    each keep their own."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    try:
        accounts = load_accounts(path)
    except Exception:
        logger.exception("failed to load %s -- Google Calendar disabled", path)
        return []
    services = []
    for account in accounts:
        try:
            credentials = Credentials.from_authorized_user_file(account.credentials_path)
            services.append((account.label, build("calendar", "v3", credentials=credentials), account.calendar_ids))
        except Exception:
            logger.exception("failed to initialize the Google Calendar client for account=%s", account.label)
    return services
