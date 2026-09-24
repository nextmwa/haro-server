# tools/setup_google_calendar.py
"""One-time, interactive Google Calendar OAuth authorization for haro-
server's proactive event polling (event_poller.py's poll_calendar(),
wired up in server.py). Not imported by the server itself -- run this
manually, once, after creating an OAuth "Desktop app" client in Google
Cloud Console and downloading its client_secret_*.json.

Opens your default browser for the Google login/consent screen, then
writes a token file for that one Google account -- google-auth refreshes
it on its own from then on, using the refresh token inside, so this only
needs re-running if that token is deleted or access is revoked from that
account's own Google Account settings.

Run this ONCE PER GOOGLE ACCOUNT you want Haro to check (see
google_calendar_accounts.py and .env.example's GOOGLE_CALENDAR_ACCOUNTS_
PATH) -- give each account's run a distinct output_token.json name, add
an entry for it to google_calendar_accounts.json, and add a matching
bind mount in docker-compose.yml.

Usage:
    python3 tools/setup_google_calendar.py <client_secret.json> [output_token.json]

If output_token.json is omitted, defaults to google_calendar_token.json
in the current directory (already covered by this repo's .gitignore --
it holds a real, refreshable credential, same sensitivity as anything in
.env).
"""
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

# Read-only: poll_calendar() only ever lists upcoming events, never
# creates or modifies calendar entries, so the requested scope matches
# exactly what's needed -- not the broader read/write calendar scope.
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    client_secret_path = sys.argv[1]
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "google_calendar_token.json")

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
    # Starts a local HTTP server on a random free port and opens your
    # default browser there -- the standard, Google-documented flow for a
    # desktop/CLI app (no manual copy-pasting of an auth code needed).
    credentials = flow.run_local_server(port=0)

    output_path.write_text(credentials.to_json())
    print(f"Saved {output_path.resolve()}")
    print(
        "Add an entry for this account to google_calendar_accounts.json "
        "(label, credentials_path, calendar_ids), add a matching bind "
        "mount in docker-compose.yml, then restart haro-server."
    )


if __name__ == "__main__":
    main()
