import datetime
import httpx
import pytest

from haro_server import db
from haro_server.event_poller import poll_github, poll_calendar


class FakeGithubTransport(httpx.MockTransport):
    def __init__(self, pulls: list[dict], check_runs: list[dict]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/pulls"):
                return httpx.Response(200, json=pulls)
            if "/commits/" in request.url.path and request.url.path.endswith("/check-runs"):
                return httpx.Response(200, json={"check_runs": check_runs})
            if request.url.path.endswith("/acme/web"):
                return httpx.Response(200, json={"default_branch": "main"})
            return httpx.Response(404)

        super().__init__(handler)


async def test_poll_github_reports_a_new_open_pr(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    transport = FakeGithubTransport(pulls=[{"number": 42, "title": "Fix bug"}], check_runs=[])
    async with httpx.AsyncClient(transport=transport, base_url="https://api.github.com") as client:
        events = await poll_github(client, db_path, token="fake-token", repos=["acme/web"])

    assert len(events) == 1
    assert "42" in events[0].summary or "Fix bug" in events[0].summary


async def test_poll_github_does_not_repeat_an_already_seen_pr(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    transport = FakeGithubTransport(pulls=[{"number": 42, "title": "Fix bug"}], check_runs=[])
    async with httpx.AsyncClient(transport=transport, base_url="https://api.github.com") as client:
        await poll_github(client, db_path, token="fake-token", repos=["acme/web"])
        events = await poll_github(client, db_path, token="fake-token", repos=["acme/web"])

    assert events == []


async def test_poll_github_reports_a_failed_check_run(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    transport = FakeGithubTransport(
        pulls=[], check_runs=[{"conclusion": "failure", "name": "build"}]
    )
    async with httpx.AsyncClient(transport=transport, base_url="https://api.github.com") as client:
        events = await poll_github(client, db_path, token="fake-token", repos=["acme/web"])

    assert len(events) == 1
    assert "fallit" in events[0].summary.lower()


class FakeCalendarEventsList:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def execute(self):
        return {"items": self._items}


class FakeCalendarEvents:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def list(self, **kwargs):
        return FakeCalendarEventsList(self._items)


class FakeCalendarService:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def events(self):
        return FakeCalendarEvents(self._items)


def _soon_iso(minutes: int) -> str:
    return (datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=minutes)).isoformat()


async def test_poll_calendar_reports_an_upcoming_meeting(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    service = FakeCalendarService(
        items=[{"id": "evt1", "summary": "Standup", "start": {"dateTime": _soon_iso(10)}}]
    )
    events = await poll_calendar(service, db_path)

    assert len(events) == 1
    assert "Standup" in events[0].summary


async def test_poll_calendar_does_not_repeat_an_already_notified_meeting(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    service = FakeCalendarService(
        items=[{"id": "evt1", "summary": "Standup", "start": {"dateTime": _soon_iso(10)}}]
    )
    await poll_calendar(service, db_path)
    events = await poll_calendar(service, db_path)

    assert events == []
