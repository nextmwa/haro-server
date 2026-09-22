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
    last_pr_number_int = int(last_pr_number) if last_pr_number else 0
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
