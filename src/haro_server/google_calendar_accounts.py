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
