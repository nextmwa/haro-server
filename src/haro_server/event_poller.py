import datetime
import logging

import httpx

from . import db
from .event_bus import Event

logger = logging.getLogger(__name__)


async def poll_github(
    client: httpx.AsyncClient, db_path: str, token: str, repos: list[str]
) -> list[Event]:
    """One poll cycle across every configured repo. Never raises: a
    failure on one repo (or all of them -- a network blip, an expired
    token) is logged and treated as "nothing new this cycle", not fatal
    to the scheduler calling this repeatedly (see the design spec's
    Error Handling section).
    """
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    events: list[Event] = []
    for repo in repos:
        try:
            events.extend(await _poll_one_repo(client, db_path, headers, repo))
        except Exception:
            logger.exception("GitHub poll failed for %s, skipping this cycle", repo)
    return events


async def _poll_one_repo(
    client: httpx.AsyncClient, db_path: str, headers: dict[str, str], repo: str
) -> list[Event]:
    events: list[Event] = []

    pulls_key = f"github:{repo}:pr"
    pulls_resp = await client.get(f"/repos/{repo}/pulls", headers=headers)
    pulls_resp.raise_for_status()
    last_pr_number = db.get_event_dedup_key(db_path, pulls_key)
    if last_pr_number is None:
        # Cold start (this repo has never been polled before): every
        # currently-open PR would otherwise look "new" against a watermark
        # of 0 and get announced all at once. Seed the watermark to the
        # current max PR number instead, without emitting any events --
        # only PRs that show up on a LATER cycle (once a real watermark
        # exists) are genuinely new and worth announcing.
        pr_numbers = [pr["number"] for pr in pulls_resp.json()]
        if pr_numbers:
            db.set_event_dedup_key(db_path, pulls_key, str(max(pr_numbers)))
    else:
        last_pr_number_int = int(last_pr_number)
        newest_pr_number = last_pr_number_int
        for pr in pulls_resp.json():
            if pr["number"] > last_pr_number_int:
                events.append(
                    Event(
                        source="github",
                        summary=f"nuova pull request su {repo}: {pr['title']}",
                        dedup_key=f"{pulls_key}:{pr['number']}",
                    )
                )
                newest_pr_number = max(newest_pr_number, pr["number"])
        if newest_pr_number != last_pr_number_int:
            db.set_event_dedup_key(db_path, pulls_key, str(newest_pr_number))

    repo_resp = await client.get(f"/repos/{repo}", headers=headers)
    repo_resp.raise_for_status()
    default_branch = repo_resp.json()["default_branch"]

    ci_key = f"github:{repo}:ci"
    checks_resp = await client.get(
        f"/repos/{repo}/commits/{default_branch}/check-runs", headers=headers
    )
    checks_resp.raise_for_status()
    check_runs = checks_resp.json().get("check_runs", [])
    conclusions = [c["conclusion"] for c in check_runs if c.get("conclusion")]
    if conclusions:
        # "failure" wins over any concurrent "success" for the same
        # commit -- one broken check is worth announcing even if others
        # on the same commit passed.
        overall = "failure" if "failure" in conclusions else "success"
        last_conclusion = db.get_event_dedup_key(db_path, ci_key)
        if overall != last_conclusion:
            if overall == "failure":
                events.append(
                    Event(
                        source="github",
                        summary=f"la build su {repo} e' fallita",
                        dedup_key=f"{ci_key}:{overall}",
                    )
                )
            db.set_event_dedup_key(db_path, ci_key, overall)

    return events


async def poll_calendar(
    service, db_path: str, label: str, calendar_ids: list[str], lookahead_minutes: int = 15
) -> list[Event]:
    """One Google account's worth of polling. `service` is a Google
    Calendar API v3 resource, as returned by
    `googleapiclient.discovery.build("calendar", "v3", credentials=...)`
    -- built once per account at server startup (one account's
    credentials can't list another account's calendars) and passed in
    here, not constructed by this function, so tests can inject a fake
    with the same `.events().list(**kwargs).execute()` shape. `label`
    identifies which account this is in both the dedup key (so the same
    calendar ID under two different accounts is never conflated) and the
    spoken announcement, e.g. "[lavoro] tra poco hai: ...". Each calendar
    in `calendar_ids` is polled independently -- same "log and skip"
    policy as poll_github's per-repo loop, so one bad calendar ID doesn't
    take the rest of this account's calendars down with it. Never raises.
    """
    now = datetime.datetime.now(datetime.UTC)
    time_max = now + datetime.timedelta(minutes=lookahead_minutes)

    events: list[Event] = []
    for calendar_id in calendar_ids:
        try:
            response = service.events().list(
                calendarId=calendar_id,
                timeMin=now.isoformat(),
                timeMax=time_max.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        except Exception:
            logger.exception(
                "Calendar poll failed for account=%s calendar=%s, skipping this cycle", label, calendar_id
            )
            continue

        for item in response.get("items", []):
            dedup_key = f"calendar:{label}:{calendar_id}:{item['id']}"
            if db.get_event_dedup_key(db_path, dedup_key) is not None:
                continue
            summary = item.get("summary", "un evento")
            events.append(
                Event(
                    source="calendar",
                    summary=f"[{label}] tra poco hai: {summary}",
                    dedup_key=dedup_key,
                )
            )
            db.set_event_dedup_key(db_path, dedup_key, "notified")

    return events
