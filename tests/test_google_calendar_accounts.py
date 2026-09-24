import json

from haro_server.google_calendar_accounts import GoogleCalendarAccount, load_accounts


def test_load_accounts_parses_every_entry(tmp_path):
    accounts_path = tmp_path / "accounts.json"
    accounts_path.write_text(
        json.dumps(
            [
                {
                    "label": "lavoro",
                    "credentials_path": "/app/google_calendar_token_lavoro.json",
                    "calendar_ids": ["primary"],
                },
                {
                    "label": "personale",
                    "credentials_path": "/app/google_calendar_token_personale.json",
                    "calendar_ids": ["primary", "famiglia@group.calendar.google.com"],
                },
            ]
        )
    )

    accounts = load_accounts(str(accounts_path))

    assert accounts == [
        GoogleCalendarAccount(
            label="lavoro",
            credentials_path="/app/google_calendar_token_lavoro.json",
            calendar_ids=["primary"],
        ),
        GoogleCalendarAccount(
            label="personale",
            credentials_path="/app/google_calendar_token_personale.json",
            calendar_ids=["primary", "famiglia@group.calendar.google.com"],
        ),
    ]
