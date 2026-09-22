# MCP Integrations and Proactive Event System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Haro answer spoken questions about GitHub and Google Calendar via MCP tool-calling, and proactively announce new events (a failed build, an upcoming meeting) it notices on a background poll, without the user asking.

**Architecture:** Two independent subsystems in `haro-server`, sharing nothing but the database and the audio-send path to the device. (1) A small event bus + per-source pollers that call GitHub's/Google's REST APIs directly on a timer, diff against last-seen state, and publish anything new; an announcer drains the bus and speaks through `Session` once it's idle. (2) An MCP client connected to GitHub's official MCP server, whose tools are handed to `LiteLlmClient` so the model can call them mid-conversation.

**Tech Stack:** Python 3.11+, FastAPI, LiteLLM, `httpx` (already a dependency, used for GitHub's REST API), `google-api-python-client` + `google-auth` (new dependencies, official Google libraries, used for Calendar's REST API), the `mcp` Python SDK (new dependency, for the MCP client session) + LiteLLM's `litellm.experimental_mcp_client` (tool-format glue).

**Spec:** `docs/superpowers/specs/2026-09-22-mcp-integrations-and-event-system-design.md`

## Global Constraints

- No inbound webhooks. All external-source checking is outbound polling only (default interval 180s, configurable via `EVENT_POLL_INTERVAL_SECONDS`).
- Every new `.env` variable is optional; unset means that integration is inactive, matching the existing `NAVIDROME_URL`/`NAVIDROME_USERNAME`/`NAVIDROME_PASSWORD` "all-or-nothing, unset = disabled" pattern in `config.py`.
- No firmware changes. A proactive announcement is delivered through the exact same `send_binary`/audio-chunk path a normal reply already uses.
- Secrets (`GITHUB_TOKEN`, Google OAuth credentials) live in `.env` only, never in code or logs, matching every other credential in this project.
- **Scope note on Google Calendar's MCP piece**: unlike GitHub (which has one official, well-documented MCP server: `github/github-mcp-server`), there is no single official Google Calendar MCP server — the ecosystem is several unofficial, community-maintained packages. This plan's on-demand MCP tool-calling (Tasks 8-10) therefore covers **GitHub only**. The proactive poller (Tasks 4-5) covers **both** GitHub and Calendar from the start, since polling talks to each service's own REST API directly and never needed an MCP server in the first place. Adding Calendar's on-demand MCP tools later is a follow-up task using the exact same `mcp_client.py` machinery Task 8 builds, once a specific package is chosen.

---

### Task 1: Event bus

**Files:**
- Create: `src/haro_server/event_bus.py`
- Test: `tests/test_event_bus.py`

**Interfaces:**
- Produces: `Event` dataclass (`source: str`, `summary: str`, `dedup_key: str`), `EventBus` class with `publish(event: Event) -> None` and `async def events(self) -> AsyncIterator[Event]` (an async generator that yields published events as they arrive, in publish order).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_bus.py
import asyncio

import pytest

from haro_server.event_bus import Event, EventBus


async def test_events_yields_published_events_in_order():
    bus = EventBus()
    bus.publish(Event(source="github", summary="build failed", dedup_key="gh:1"))
    bus.publish(Event(source="calendar", summary="meeting soon", dedup_key="cal:1"))

    received = []

    async def consume_two():
        async for event in bus.events():
            received.append(event)
            if len(received) == 2:
                break

    await asyncio.wait_for(consume_two(), timeout=1.0)

    assert [e.dedup_key for e in received] == ["gh:1", "cal:1"]


async def test_events_blocks_until_something_is_published():
    bus = EventBus()

    async def consume_one():
        async for event in bus.events():
            return event

    task = asyncio.ensure_future(consume_one())
    await asyncio.sleep(0.05)
    assert not task.done()  # nothing published yet, so the consumer is still waiting

    bus.publish(Event(source="github", summary="new PR", dedup_key="gh:2"))
    event = await asyncio.wait_for(task, timeout=1.0)
    assert event.dedup_key == "gh:2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_bus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.event_bus'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/haro_server/event_bus.py
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass(frozen=True)
class Event:
    source: str  # "github" or "calendar" -- which poller produced this
    summary: str  # what to say, e.g. "la build su main e' fallita"
    dedup_key: str  # unique per underlying thing (e.g. "github:owner/repo:pr:42"),
    # stored in db.py's event_state table so the same real-world event is
    # never announced twice


class EventBus:
    """In-process only -- no persistence, no cross-process delivery. This
    is deliberately the simplest thing that decouples "a poller noticed a
    change" from "something reacts to it", not a general message broker.
    See the design spec's Components section.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    def publish(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def events(self) -> AsyncIterator[Event]:
        while True:
            event = await self._queue.get()
            yield event
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_event_bus.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/haro_server/event_bus.py tests/test_event_bus.py
git commit -m "feat: add EventBus for the proactive-announcement pipeline"
```

---

### Task 2: Event dedup state in the database

**Files:**
- Modify: `src/haro_server/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing new (follows `db.py`'s existing per-call-connection pattern, see `get_config_value`/`set_config_value`).
- Produces: `db.get_event_dedup_key(db_path: str, source: str) -> str | None`, `db.set_event_dedup_key(db_path: str, source: str, dedup_key: str) -> None`. `source` is a poller name, e.g. `"github:owner/repo:pr"`, `"github:owner/repo:ci"`, `"calendar:primary"` -- each poller decides its own key granularity; the table just stores "last dedup_key seen for this string key".

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_db.py
def test_event_dedup_key_round_trips(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    assert db.get_event_dedup_key(db_path, "github:acme/web:pr") is None

    db.set_event_dedup_key(db_path, "github:acme/web:pr", "42")
    assert db.get_event_dedup_key(db_path, "github:acme/web:pr") == "42"

    db.set_event_dedup_key(db_path, "github:acme/web:pr", "43")
    assert db.get_event_dedup_key(db_path, "github:acme/web:pr") == "43"
```

(Check `tests/test_db.py`'s existing imports/fixtures first -- it already imports `db` and uses a `tmp_path`-based `db_path` for every other test in that file; match that pattern exactly, don't add a new fixture.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py::test_event_dedup_key_round_trips -v`
Expected: FAIL with `AttributeError: module 'haro_server.db' has no attribute 'get_event_dedup_key'`

- [ ] **Step 3: Write minimal implementation**

In `src/haro_server/db.py`, add to `_SCHEMA` (after the existing `config` table):

```python
CREATE TABLE IF NOT EXISTS event_state (
    source_key TEXT PRIMARY KEY,
    dedup_key TEXT NOT NULL
);
```

Add two functions at the end of the file, matching `get_config_value`/`set_config_value`'s exact shape immediately above them:

```python
def get_event_dedup_key(db_path: str, source_key: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT dedup_key FROM event_state WHERE source_key = ?", (source_key,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_event_dedup_key(db_path: str, source_key: str, dedup_key: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO event_state (source_key, dedup_key) VALUES (?, ?) "
            "ON CONFLICT(source_key) DO UPDATE SET dedup_key = excluded.dedup_key",
            (source_key, dedup_key),
        )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/haro_server/db.py tests/test_db.py
git commit -m "feat: add event_state table for poller dedup tracking"
```

---

### Task 3: `Session.is_busy()` and `Session.speak_announcement()`

**Files:**
- Modify: `src/haro_server/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `Session.is_busy() -> bool`, `async def Session.speak_announcement(self, text: str) -> None`. Later tasks (the announcer, Task 7) call both.

**Context:** `Session.handle_end_of_speech()` currently has no "a reply is in progress" flag at all -- read `src/haro_server/session.py` in full before starting. Its real signature is `Session(stt, llm, tts, send_text, send_binary, db_path=None, navidrome=None)`, and `handle_end_of_speech()` has **four** exit paths from one `async def`: an empty-transcript early return, a deterministic-action early return, a music-request early return (spawns `self._music_task` and returns without awaiting it), and the normal LLM/TTS streaming path at the end. Music plays via that background task (`self._music_task`, already set up by earlier work on this branch), which outlives `handle_end_of_speech()`'s own return -- so "busy" must check both: whether `handle_end_of_speech()`'s own synchronous work is in flight (covers all four exit paths), AND whether `self._music_task` is set and not done (covers a track still playing after the method already returned).

Reuse this file's real `_make_session(llm_chunks, sent_text=None, sent_binary=None)` helper (not a hypothetical `make_session`) -- it returns the 6-tuple `(session, stt, llm, tts, sent_text, sent_binary)` and already gives you the two capturing lists `speak_announcement()`'s test needs, so there is no need to reassign `session._send_binary`/`_send_text` by hand.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_session.py -- reuses this file's real _make_session(),
# FakeLlm classes.

async def test_is_busy_false_before_any_turn():
    session, *_ = _make_session(llm_chunks=["[emotion:neutral] ciao"])
    assert session.is_busy() is False


async def test_is_busy_true_during_a_normal_reply():
    # A slow-yielding fake LLM lets the test observe is_busy() while
    # handle_end_of_speech() is still awaiting the TTS stream.
    busy_during_reply = []

    session, stt, llm, tts, sent_text, sent_binary = _make_session(llm_chunks=[])

    class SlowLlm(FakeLlm):
        async def _chunk_generator(self):
            busy_during_reply.append(session.is_busy())
            yield "[emotion:neutral] ciao"

    session._llm = SlowLlm(chunks=[])
    await session.handle_end_of_speech()

    assert busy_during_reply == [True]
    assert session.is_busy() is False  # cleared once the turn finished


async def test_is_busy_true_while_a_music_track_is_still_playing():
    # handle_end_of_speech() returns immediately for a music request
    # (see session.py's own comment on why) -- is_busy() must still
    # report True while the background _music_task it spawned is
    # running, not just while handle_end_of_speech() itself is on the
    # stack.
    stt = FakeStt(transcript="metti musica")
    track = FakeTrack(id="song-1", title="Some Song", artist="Some Artist")
    llm = FakeLlm(chunks=[], music_query="musica", picked_track_index=0)
    navidrome = FakeNavidrome(search_results=[track], pcm_chunks=[b"pcm1", b"pcm2"])
    tts = FakeTts()

    async def send_text(text: str) -> None:
        pass

    async def send_binary(data: bytes) -> None:
        pass

    session = Session(stt, llm, tts, send_text, send_binary, navidrome=navidrome)

    await session.handle_end_of_speech()
    assert session.is_busy() is True  # music_task is running

    await session._music_task
    assert session.is_busy() is False


async def test_speak_announcement_sends_tts_audio_and_response_end():
    session, stt, llm, tts, sent_text, sent_binary = _make_session(llm_chunks=[])

    await session.speak_announcement("la build e' fallita")

    assert sent_binary  # at least one audio chunk was sent
    assert any("response_end" in t for t in sent_text)
```

(`test_is_busy_true_while_a_music_track_is_still_playing` needs `FakeStt`, `Session`, `FakeNavidrome`, `FakeTrack` -- all already defined at module level in this same file, so no extra import is needed.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session.py -k "is_busy or speak_announcement" -v`
Expected: FAIL with `AttributeError: 'Session' object has no attribute 'is_busy'`

- [ ] **Step 3: Write minimal implementation**

In `src/haro_server/session.py`, in `Session.__init__`, add after `self._music_task = None`:

```python
        # See is_busy()/speak_announcement() below -- true only while THIS
        # object's own synchronous turn-handling is in flight; music's
        # background task (self._music_task above) is checked separately
        # since it outlives handle_end_of_speech()'s return.
        self._reply_in_progress = False
```

`handle_end_of_speech()` has four exit points (empty-transcript return, action-match return, music-request return, and falling off the end of the normal-reply path), so the flag must be set before ANY of them can run and cleared in a `finally` wrapping the whole body -- not just the final streaming section. Indent the existing body one level under a new `try:` right after `self._reply_in_progress = True`, and add `finally: self._reply_in_progress = False` at the end, at the same indentation as the `try:`. The body's own content does not change, only its indentation:

```python
    async def handle_end_of_speech(self) -> None:
        self._reply_in_progress = True
        try:
            transcript = await self._stt.finalize()
            logger.debug("transcript: %s", transcript)

            if not transcript.strip():
                logger.info("empty transcript, skipping LLM/TTS turn")
                await self._send_text(protocol.encode_response_end())
                return

            action = actions.match_action(transcript)
            if action is not None:
                result = action.resolve()
                logger.info("action matched: %s -> %r", action.name, result)
                await self._send_text(protocol.encode_action(action.name, result))
                await self._send_text(protocol.encode_response_end())
                self._save_transcript(transcript, reply=f"[azione: {action.name} -> {result}]", emotion=None)
                return

            if music.is_music_request(transcript):
                if self._music_task is not None and not self._music_task.done():
                    self._music_task.cancel()
                self._music_task = asyncio.create_task(self._play_music(transcript))
                return

            raw_reply = self._llm.stream_reply(transcript)
            emotion, text_stream = await _split_emotion_prefix(raw_reply)
            await self._send_text(protocol.encode_emotion(emotion))

            reply_parts: list[str] = []
            teed_stream = _tee(text_stream, reply_parts)

            tts_stream = self._tts.synthesize(teed_stream)
            try:
                async for chunk in tts_stream:
                    await self._send_binary(chunk)
            finally:
                await tts_stream.aclose()
                await teed_stream.aclose()
                await text_stream.aclose()
                await raw_reply.aclose()

            await self._send_text(protocol.encode_response_end())

            full_reply = "".join(reply_parts)
            self._save_transcript(transcript, full_reply, emotion)
            self._spawn_fact_extraction(transcript, full_reply)
        finally:
            self._reply_in_progress = False
```

(The comments already on this method in the real file are unchanged -- they're omitted above only to keep this block focused on the indentation/wrapping change; keep them in place when editing the real file, don't delete them.)

Add two new methods after `handle_interrupt()`:

```python
    def is_busy(self) -> bool:
        return self._reply_in_progress or (self._music_task is not None and not self._music_task.done())

    async def speak_announcement(self, text: str) -> None:
        """Speaks `text` with no LLM/STT involvement -- used by the
        proactive-event announcer (event_bus.py's consumer, wired up in
        Task 7), never by a normal user turn. Callers are responsible for
        checking is_busy() first; this method does not check it itself,
        so it can also be used for other non-conversational speech later
        without re-deriving that policy here.
        """
        self._reply_in_progress = True
        try:
            await self._send_text(protocol.encode_emotion("neutral"))
            tts_stream = self._tts.synthesize(_single_chunk(text))
            try:
                async for chunk in tts_stream:
                    await self._send_binary(chunk)
            finally:
                await tts_stream.aclose()
            await self._send_text(protocol.encode_response_end())
        finally:
            self._reply_in_progress = False
```

Add the small helper near the other module-level async generators at the bottom of the file (next to `_prepend`/`_tee`):

```python
async def _single_chunk(text: str) -> AsyncIterator[str]:
    yield text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session.py -v`
Expected: PASS (all tests in the file, including the 4 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/haro_server/session.py tests/test_session.py
git commit -m "feat: add Session.is_busy()/speak_announcement() for proactive events"
```

---

### Task 4: GitHub poller

**Files:**
- Create: `src/haro_server/event_poller.py`
- Test: `tests/test_event_poller.py`

**Interfaces:**
- Consumes: `db.get_event_dedup_key`/`set_event_dedup_key` (Task 2), `Event`/`EventBus` (Task 1).
- Produces: `async def poll_github(client: httpx.AsyncClient, db_path: str, token: str, repos: list[str]) -> list[Event]`. `client` is injected so tests never hit the real network -- same reasoning `navidrome.py`'s own httpx usage already established in this codebase.

**Context:** Checks, per configured `owner/repo`: (a) whether the latest commit on the default branch has a new CI conclusion (`success`/`failure`) since last polled, and (b) whether there's a newer open PR number than last seen. Uses GitHub's REST API directly (`https://api.github.com`), not MCP -- this is a scheduled background check, not a conversation (see the design spec's Architecture section for why).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_poller.py
import httpx
import pytest

from haro_server import db
from haro_server.event_poller import poll_github


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_poller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.event_poller'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/haro_server/event_poller.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_event_poller.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/haro_server/event_poller.py tests/test_event_poller.py
git commit -m "feat: add GitHub poller for new PRs and CI status changes"
```

---

### Task 5: Calendar poller

**Files:**
- Modify: `src/haro_server/event_poller.py`
- Modify: `pyproject.toml`
- Test: `tests/test_event_poller.py`

**Interfaces:**
- Consumes: same `db`/`Event` as Task 4.
- Produces: `async def poll_calendar(service, db_path: str, lookahead_minutes: int = 15) -> list[Event]`. `service` is an injected Google Calendar API resource object (see Step 1 -- tests inject a fake with the same `.events().list(...).execute()` call shape the real `googleapiclient` resource has), so no real Google API call happens in tests.

**Context:** Google Calendar's REST API needs OAuth (not a static token like GitHub's PAT), so this uses the official `google-api-python-client` + `google-auth` libraries rather than hand-rolled OAuth refresh logic over raw `httpx` -- token refresh correctness matters here in a way it doesn't for GitHub's simple bearer token.

- [ ] **Step 1: Add the new dependency**

In `pyproject.toml`'s `dependencies` list, add (near `httpx`):

```python
    # event_poller.py's Calendar polling needs real OAuth token refresh
    # handling, which google-api-python-client/google-auth provide
    # correctly -- unlike GitHub's poller (a static PAT over plain httpx),
    # hand-rolling OAuth refresh here would be a real correctness risk for
    # little benefit.
    "google-api-python-client>=2.100.0",
    "google-auth>=2.23.0",
```

- [ ] **Step 2: Write the failing test**

```python
# Append to tests/test_event_poller.py
import datetime

from haro_server.event_poller import poll_calendar


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
```

(Add `from haro_server import db` to this file's imports if Task 4 didn't already add it.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_event_poller.py -k calendar -v`
Expected: FAIL with `ImportError: cannot import name 'poll_calendar' from 'haro_server.event_poller'`

- [ ] **Step 4: Write minimal implementation**

Append to `src/haro_server/event_poller.py`:

```python
import datetime


async def poll_calendar(service, db_path: str, lookahead_minutes: int = 15) -> list[Event]:
    """`service` is a Google Calendar API v3 resource, as returned by
    `googleapiclient.discovery.build("calendar", "v3", credentials=...)`
    -- built once at server startup (Task 10) and passed in here, not
    constructed by this function, so tests can inject a fake with the
    same `.events().list(**kwargs).execute()` shape. Never raises: same
    "log and skip this cycle" policy as poll_github above.
    """
    try:
        now = datetime.datetime.now(datetime.UTC)
        time_max = now + datetime.timedelta(minutes=lookahead_minutes)
        response = service.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=time_max.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
    except Exception:
        logger.exception("Calendar poll failed, skipping this cycle")
        return []

    events: list[Event] = []
    for item in response.get("items", []):
        dedup_key = f"calendar:primary:{item['id']}"
        if db.get_event_dedup_key(db_path, dedup_key) is not None:
            continue
        summary = item.get("summary", "un evento")
        events.append(
            Event(
                source="calendar",
                summary=f"tra poco hai: {summary}",
                dedup_key=dedup_key,
            )
        )
        db.set_event_dedup_key(db_path, dedup_key, "notified")

    return events
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_event_poller.py -v`
Expected: PASS (5 tests total: 3 from Task 4 + 2 new)

- [ ] **Step 6: Commit**

```bash
git add src/haro_server/event_poller.py tests/test_event_poller.py pyproject.toml
git commit -m "feat: add Calendar poller for upcoming meetings"
```

---

### Task 6: Configuration

**Files:**
- Modify: `src/haro_server/config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.github_token: str | None`, `Config.github_repos: list[str]`, `Config.google_calendar_credentials_path: str | None`, `Config.event_poll_interval_seconds: int`.

- [ ] **Step 1: Write the failing test**

Check `tests/test_config.py`'s existing structure first (it already has a pattern for asserting defaults and for asserting env-var overrides via `monkeypatch`/`os.environ` -- match that exact pattern, don't invent a new one). Add:

```python
def test_config_defaults_have_no_github_or_calendar_integration():
    config = Config()
    assert config.github_token is None
    assert config.github_repos == []
    assert config.google_calendar_credentials_path is None
    assert config.event_poll_interval_seconds == 180


def test_config_from_env_reads_github_repos_as_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
    monkeypatch.setenv("GITHUB_REPOS", "acme/web, acme/api")
    config = Config.from_env()
    assert config.github_token == "ghp_fake"
    assert config.github_repos == ["acme/web", "acme/api"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -k github -v`
Expected: FAIL with `TypeError: Config.__init__() got an unexpected keyword argument` or `AttributeError`

- [ ] **Step 3: Write minimal implementation**

In `src/haro_server/config.py`, add fields to the `Config` dataclass (after `navidrome_password`):

```python
    # GitHub/Calendar proactive polling (event_poller.py) and GitHub MCP
    # tool-calling (mcp_client.py) -- unset github_token means both are
    # inactive, matching navidrome_*'s "unset = disabled" pattern above.
    github_token: str | None = None
    github_repos: list[str] = dataclasses.field(default_factory=list)
    # Path to a Google OAuth credentials/token file (see event_poller.py's
    # poll_calendar() and the design spec's Configuration section) -- unset
    # means Calendar polling is inactive.
    google_calendar_credentials_path: str | None = None
    event_poll_interval_seconds: int = 180
```

In `Config.from_env()`, add:

```python
            github_token=os.environ.get("GITHUB_TOKEN") or None,
            github_repos=[r.strip() for r in os.environ.get("GITHUB_REPOS", "").split(",") if r.strip()],
            google_calendar_credentials_path=os.environ.get("GOOGLE_CALENDAR_CREDENTIALS_PATH") or None,
            event_poll_interval_seconds=int(os.environ.get("EVENT_POLL_INTERVAL_SECONDS", "180")),
```

In `.env.example`, add (after the `NAVIDROME_*` block):

```
# Proactive event announcements + GitHub MCP tool-calling (event_poller.py,
# mcp_client.py). All optional -- unset GITHUB_TOKEN disables both; unset
# GOOGLE_CALENDAR_CREDENTIALS_PATH disables Calendar polling only.
GITHUB_TOKEN=
GITHUB_REPOS=
GOOGLE_CALENDAR_CREDENTIALS_PATH=
EVENT_POLL_INTERVAL_SECONDS=180
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/haro_server/config.py .env.example tests/test_config.py
git commit -m "feat: add GitHub/Calendar configuration"
```

---

### Task 7: Scheduler and announcer wiring

**Files:**
- Create: `src/haro_server/announcer.py`
- Modify: `src/haro_server/server.py`
- Test: `tests/test_announcer.py`

**Interfaces:**
- Consumes: `EventBus`/`Event` (Task 1), `Session.is_busy()`/`speak_announcement()` (Task 3), `poll_github`/`poll_calendar` (Tasks 4-5), `Config` fields (Task 6).
- Produces: `run_announcer(bus: EventBus, get_active_session: Callable[[], Session | None], poll_interval_seconds: float = 1.0) -> None` in `announcer.py` -- a standalone, unit-tested function (also infinite, meant to be run as a background task). `get_active_session` returns the currently-connected session, or `None` if the device isn't connected -- `server.py` already only ever holds one live `Session` at a time (one robot, one connection), so a simple accessor is enough; no session registry is needed. The scheduler (the poll-and-publish loop) is **not** a standalone function -- Step 5 below defines it as a `run_scheduler()` closure inline inside `create_app()`, capturing `event_bus`/`config` from the enclosing scope, since it is pure wiring with no independent unit test of its own (Tasks 4-5 already cover `poll_github`/`poll_calendar` in isolation).

**Context:** `server.py`'s `create_app()` currently builds `stt`/`tts`/`llm`/`navidrome` once at startup and constructs a fresh `Session` per WebSocket connection inside `websocket_endpoint()`. Critically, `create_app()` itself is a plain (non-`async`) function, called from `main.py` as `app = create_app(config)` *before* `uvicorn.run(app, ...)` starts the event loop -- there is no running loop yet at that point, so `asyncio.create_task(...)` cannot be called directly inside `create_app()`'s body (it would raise `RuntimeError: no running event loop`). The background tasks this step adds must instead be started from a FastAPI startup event handler (`@app.on_event("startup")`), which FastAPI runs once uvicorn's loop is actually running -- registered as a nested function inside `create_app()`, the same closure style `websocket_endpoint()` already uses. Read `src/haro_server/server.py` in full before starting.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_announcer.py
import asyncio

import pytest

from haro_server.announcer import run_announcer
from haro_server.event_bus import Event, EventBus


class FakeSession:
    def __init__(self, busy: bool) -> None:
        self._busy = busy
        self.spoken: list[str] = []

    def is_busy(self) -> bool:
        return self._busy

    async def speak_announcement(self, text: str) -> None:
        self.spoken.append(text)


async def test_announcer_speaks_immediately_when_idle():
    bus = EventBus()
    session = FakeSession(busy=False)
    bus.publish(Event(source="github", summary="build failed", dedup_key="gh:1"))

    task = asyncio.ensure_future(run_announcer(bus, lambda: session))
    await asyncio.sleep(0.05)
    task.cancel()

    assert session.spoken == ["build failed"]


async def test_announcer_waits_for_idle_before_speaking():
    bus = EventBus()
    session = FakeSession(busy=True)
    bus.publish(Event(source="github", summary="build failed", dedup_key="gh:1"))

    task = asyncio.ensure_future(run_announcer(bus, lambda: session, poll_interval_seconds=0.01))
    await asyncio.sleep(0.03)
    assert session.spoken == []  # still busy, nothing spoken yet

    session._busy = False
    await asyncio.sleep(0.03)
    task.cancel()

    assert session.spoken == ["build failed"]


async def test_announcer_combines_multiple_queued_events_into_one_sentence():
    bus = EventBus()
    session = FakeSession(busy=False)
    bus.publish(Event(source="github", summary="build failed", dedup_key="gh:1"))
    bus.publish(Event(source="calendar", summary="meeting soon", dedup_key="cal:1"))
    await asyncio.sleep(0)  # let both publishes land in the queue before the announcer starts

    task = asyncio.ensure_future(run_announcer(bus, lambda: session))
    await asyncio.sleep(0.05)
    task.cancel()

    assert len(session.spoken) == 1
    assert "build failed" in session.spoken[0]
    assert "meeting soon" in session.spoken[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_announcer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.announcer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/haro_server/announcer.py
import asyncio
import logging
from typing import Callable

from .event_bus import Event, EventBus

logger = logging.getLogger(__name__)


class _SessionLike:
    def is_busy(self) -> bool: ...
    async def speak_announcement(self, text: str) -> None: ...


async def run_announcer(
    bus: EventBus,
    get_active_session: Callable[[], "_SessionLike | None"],
    poll_interval_seconds: float = 1.0,
) -> None:
    """Runs forever -- start as a background asyncio.Task at server
    startup (Task 7's server.py wiring), one instance for the whole
    process (there's only ever one connected robot). Drains one or more
    queued events, waits for an idle session, then speaks them as one
    combined sentence if more than one arrived -- see the design spec's
    Error Handling section for why multiple queued events are combined
    rather than announced back-to-back.
    """
    pending: list[Event] = []
    events_iter = bus.events()
    next_event_task: asyncio.Task | None = None

    while True:
        if next_event_task is None:
            next_event_task = asyncio.ensure_future(events_iter.__anext__())

        done, _ = await asyncio.wait({next_event_task}, timeout=poll_interval_seconds)
        if next_event_task in done:
            pending.append(next_event_task.result())
            next_event_task = None
            continue  # check for more queued events before deciding to speak

        if not pending:
            continue

        session = get_active_session()
        if session is None or session.is_busy():
            continue  # still busy (or nothing connected) -- keep waiting, pending stays queued

        text = _combine(pending)
        pending = []
        try:
            await session.speak_announcement(text)
        except Exception:
            # One delivery attempt only, per the design spec's Error
            # Handling section -- a stale build-status announcement
            # delivered late/retried is low-value, and the underlying
            # dedup_key is already marked seen by the poller that
            # produced this event, so it won't come back on the next cycle.
            logger.exception("failed to deliver a proactive announcement, dropping it")


def _combine(events: list[Event]) -> str:
    if len(events) == 1:
        return events[0].summary
    joined = "; ".join(e.summary for e in events)
    return f"Ci sono {len(events)} aggiornamenti: {joined}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_announcer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire the scheduler and announcer into `server.py`**

In `src/haro_server/server.py`, add near the top (with the other imports):

```python
from .announcer import run_announcer
from .event_bus import EventBus
from .event_poller import poll_calendar, poll_github
```

Inside `create_app()`, after the existing `navidrome` block and before the `@app.websocket("/")` decorator, add:

```python
    event_bus = EventBus()
    active_session: Session | None = None

    def get_active_session() -> Session | None:
        return active_session

    async def run_scheduler() -> None:
        calendar_service = None
        if config.google_calendar_credentials_path:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            credentials = Credentials.from_authorized_user_file(config.google_calendar_credentials_path)
            calendar_service = build("calendar", "v3", credentials=credentials)

        import httpx

        async with httpx.AsyncClient(base_url="https://api.github.com") as client:
            while True:
                if config.github_token and config.github_repos:
                    for event in await poll_github(
                        client, config.db_path, config.github_token, config.github_repos
                    ):
                        event_bus.publish(event)
                if calendar_service is not None:
                    for event in await poll_calendar(calendar_service, config.db_path):
                        event_bus.publish(event)
                await asyncio.sleep(config.event_poll_interval_seconds)

    @app.on_event("startup")
    async def _start_background_event_tasks() -> None:
        # Registered here (not called directly in create_app()'s body)
        # because create_app() runs before uvicorn's event loop exists --
        # see this task's Context note. FastAPI runs every "startup"
        # handler once that loop is actually up, which is the earliest
        # point asyncio.create_task() is legal here.
        if config.github_token or config.google_calendar_credentials_path:
            asyncio.create_task(run_scheduler())
            asyncio.create_task(run_announcer(event_bus, get_active_session))
            logger.info("proactive event polling started (interval=%ds)", config.event_poll_interval_seconds)
        else:
            logger.info("no GitHub token or Calendar credentials configured -- proactive events disabled")
```

Inside `websocket_endpoint()`, right after `session = Session(...)` is constructed, add:

```python
        nonlocal active_session
        active_session = session
```

`websocket_endpoint()` has two places that log `"session %s disconnected"` -- the normal-disconnect branch inside the receive loop (`if message["type"] == "websocket.disconnect": logger.info(...); break`) and the `except WebSocketDisconnect:` fallback handler below the loop. Add `active_session = None` right after the `logger.info(...)` call at **both** sites (each already runs inside `websocket_endpoint()`, so `nonlocal active_session` from the assignment above already covers them -- no second `nonlocal` declaration needed):

```python
                    logger.info("session %s disconnected", session.session_id)
                    active_session = None
                    break
```

and

```python
        except WebSocketDisconnect:
            logger.info("session %s disconnected", session.session_id)
            active_session = None
```

- [ ] **Step 6: Run the full test suite to check nothing broke**

Run: `pytest tests/ -v`
Expected: PASS (every existing test plus every test added in Tasks 1-7)

- [ ] **Step 7: Commit**

```bash
git add src/haro_server/announcer.py src/haro_server/server.py tests/test_announcer.py
git commit -m "feat: wire the event scheduler and announcer into server startup"
```

---

### Task 8: MCP client for GitHub

**Files:**
- Create: `src/haro_server/mcp_client.py`
- Modify: `pyproject.toml`
- Test: `tests/test_mcp_client.py`

**Interfaces:**
- Produces: `class McpToolClient` with `async def connect(self) -> None`, `async def close(self) -> None`, `tools: list[dict]` (OpenAI-format tool schemas, populated after `connect()`), `async def call_tool(self, tool_call) -> str` (`tool_call` is a LiteLLM/OpenAI-format tool call object; returns the tool's result as a string, ready to go into a `role: "tool"` message).

**Context -- read before writing any code:** This task connects to GitHub's official MCP server (`github/github-mcp-server`, a Go binary run as `github-mcp-server stdio` with a `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable -- verified against the server's own repo at plan-writing time, but MCP server distribution/install methods change; **re-verify the exact current install/run instructions at `https://github.com/github/github-mcp-server`'s README before writing Step 3 below**, and adjust the `command`/`args` in the code if they've changed).

This also needs the official `mcp` Python SDK (stdio client transport) and LiteLLM's own `litellm.experimental_mcp_client` module (tool-schema conversion + invocation glue) -- **both are real, but "experimental" APIs can shift between versions. Before writing Step 3, run this against the versions actually installed in this project's environment and read the real output**:

```bash
python -c "
import mcp
import litellm.experimental_mcp_client as emc
help(mcp.ClientSession)
help(emc.load_mcp_tools)
help(emc.call_openai_tool)
"
```

Write Step 3's implementation to match what that command actually prints, not from memory of MCP/LiteLLM documentation -- this project's established practice (see `tts_common.py`/`chatterbox_tts.py`/`pockettts_tts.py`'s own comments) is to verify a new library's real API via direct inspection before committing to it in code.

- [ ] **Step 1: Add the new dependencies**

In `pyproject.toml`'s `dependencies` list, add:

```python
    # mcp_client.py's connection to GitHub's MCP server (stdio transport).
    "mcp>=1.0.0",
```

(`litellm` is already a dependency and already provides `litellm.experimental_mcp_client` -- no separate package needed for that half.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_mcp_client.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from haro_server.mcp_client import McpToolClient


async def test_connect_populates_tools_from_the_mcp_session(monkeypatch):
    fake_tools = [{"type": "function", "function": {"name": "list_pull_requests", "parameters": {}}}]

    async def fake_load_mcp_tools(session, format):
        assert format == "openai"
        return fake_tools

    monkeypatch.setattr("haro_server.mcp_client._load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr("haro_server.mcp_client._open_stdio_session", AsyncMock(return_value=MagicMock()))

    client = McpToolClient(github_token="fake-token")
    await client.connect()

    assert client.tools == fake_tools
```

(This test intentionally mocks at the `_load_mcp_tools`/`_open_stdio_session` seam rather than spinning up a real MCP server process -- those two names are what Step 3 below must define as separate, mockable module-level functions, not inlined directly into `McpToolClient.connect()`.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_mcp_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.mcp_client'`

- [ ] **Step 4: Write the implementation**

First run the verification command from this task's Context section above and read its real output. Then write `src/haro_server/mcp_client.py` so that:
- `_open_stdio_session(command: str, args: list[str], env: dict[str, str])` opens an `mcp.ClientSession` over stdio to the given command (using whatever the real `mcp` package's stdio-client API turned out to be from the verification step) and returns the live session.
- `_load_mcp_tools(session, format)` is a thin wrapper around `litellm.experimental_mcp_client.load_mcp_tools` (or whatever the verification step showed its real name/signature to be).
- `McpToolClient.__init__(self, github_token: str)` stores the token, does no I/O.
- `McpToolClient.connect()` calls `_open_stdio_session("github-mcp-server", ["stdio"], {"GITHUB_PERSONAL_ACCESS_TOKEN": self._github_token})`, stores the session, then calls `_load_mcp_tools(session, "openai")` and stores the result on `self.tools`.
- `McpToolClient.call_tool(tool_call)` calls `litellm.experimental_mcp_client.call_openai_tool` (or the verified real name) with the stored session and the given tool call, and returns its result as a string.
- `McpToolClient.close()` closes the stdio session cleanly.

Do not guess field/method names beyond what the verification step actually printed -- if `help()`'s output differs from what's described above, follow the real output instead and note the difference in a code comment (matching how `chatterbox_tts.py`'s comments document what its own verification step found).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_mcp_client.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/haro_server/mcp_client.py pyproject.toml tests/test_mcp_client.py
git commit -m "feat: add MCP client for GitHub's official MCP server"
```

---

### Task 9: Tool-calling in `LiteLlmClient.stream_reply()`

**Files:**
- Modify: `src/haro_server/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `McpToolClient.tools`/`call_tool()` (Task 8).
- Produces: `LiteLlmClient.__init__` gains an optional `tools: list[dict] | None = None` and `call_tool: Callable[[Any], Awaitable[str]] | None = None` parameter pair, plus a new `LiteLlmClient.set_tools(self, tools: list[dict], call_tool) -> None` method (used by Task 10's server-startup wiring to attach tools *after* construction -- see that task's Context note for why). `stream_reply()`'s external contract (`AsyncIterator[str]`) is unchanged.

**Context:** Read this task's design-spec background carefully: LiteLLM's streaming responses have multiple currently-open upstream bugs around reconstructing `tool_calls` from chunked deltas (verified via real search results at plan-writing time -- e.g. BerriAI/litellm#39796, #17246, and crewAIInc/crewAI#7534, all about dropped/doubled/corrupted streamed tool-call data). To avoid that whole bug class, tool-calling here uses a **non-streamed** `acompletion()` call first (well-established, simple, no known issues) to let the model decide whether to call a tool; only the follow-up reply (once any tool results are in hand, or immediately if no tool was needed) is streamed, exactly like the existing no-tools code path. This means a turn that ends up calling a tool has a longer pause before the reply starts (no partial streaming during the decision step) -- an accepted tradeoff per the design spec, and only for deployments where `GITHUB_TOKEN` is configured at all.

`tests/test_llm.py` already has real fake-response scaffolding for this exact purpose -- `FakeChunk`/`_fake_stream` for a streamed response, `FakeMessage`/`FakeCompletionChoice`/`FakeCompletion` for a non-streamed one, and a `_client(tmp_path, model=...)` helper that builds a `LiteLlmClient` against a real (empty) temp db. Reuse all of them; mocking goes through `unittest.mock.patch("haro_server.llm.litellm.acompletion", new=AsyncMock())`, not `monkeypatch.setattr`, matching every existing test in that file.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_llm.py -- extends the existing FakeMessage to
# optionally carry tool_calls, and adds one fake tool-call object.

async def test_stream_reply_without_tools_is_unchanged(tmp_path):
    # Existing behavior: no tools configured -> the original single
    # streamed call, no tool-calling machinery involved at all.
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["[emotion:neutral] ciao"])
        chunks = [c async for c in client.stream_reply("ciao")]

    assert "".join(chunks) == "[emotion:neutral] ciao"
    assert mock_acompletion.call_count == 1
    assert "tools" not in mock_acompletion.call_args.kwargs


async def test_stream_reply_calls_a_tool_when_the_model_requests_one(tmp_path):
    fake_tool_call = FakeToolCall(id="call_1", name="list_pull_requests")
    decision_response = FakeCompletion("")
    decision_response.choices[0].message = FakeToolCallMessage([fake_tool_call])

    async def fake_call_tool(tool_call):
        assert tool_call is fake_tool_call
        return "2 open pull requests"

    client = _client(tmp_path)
    client.set_tools(
        tools=[{"type": "function", "function": {"name": "list_pull_requests"}}],
        call_tool=fake_call_tool,
    )

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.side_effect = [
            decision_response,
            _fake_stream(["[emotion:neutral] hai 2 PR aperte"]),
        ]
        chunks = [c async for c in client.stream_reply("ho PR aperte?")]

    assert "".join(chunks) == "[emotion:neutral] hai 2 PR aperte"
    assert mock_acompletion.call_count == 2  # one non-streamed decision call, one streamed follow-up
    first_call, second_call = mock_acompletion.call_args_list
    assert first_call.kwargs["tools"] == client._tools
    assert first_call.kwargs["stream"] is False
    assert second_call.kwargs["stream"] is True
    assert "tool_call_id" in str(second_call.kwargs["messages"])  # the tool result was folded in
```

Add these small test helpers near the top of `tests/test_llm.py`, alongside the existing `FakeDelta`/`FakeChoice`/`FakeChunk`/`FakeMessage`/`FakeCompletionChoice`/`FakeCompletion` classes:

```python
class FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str = "{}") -> None:
        self.id = id
        self.function = FakeToolCallFunction(name, arguments)


class FakeToolCallFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCallMessage:
    """Stands in for litellm's assistant message when the model decided
    to call a tool instead of replying directly -- .content is None (no
    text yet) and .tool_calls carries what to call. model_dump() only
    needs to round-trip through stream_reply()'s own messages.append()
    call, not match litellm's real serialization exactly.
    """

    def __init__(self, tool_calls: list[FakeToolCall]) -> None:
        self.content = None
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "tool_calls": self.tool_calls}
```

(The existing `FakeMessage` class has no `tool_calls` attribute -- `stream_reply()`'s new code checks `message.tool_calls` on whatever `decision.choices[0].message` is, so the no-tool-call path (an ordinary `FakeCompletion`/`FakeMessage`, unchanged) needs `tool_calls` to read as falsy. Add `self.tool_calls = None` to `FakeMessage.__init__` so both paths work through the same fake class family without a second one for "no tool call was made".)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py -k "stream_reply" -v`
Expected: FAIL with `AttributeError: 'LiteLlmClient' object has no attribute 'set_tools'`

- [ ] **Step 3: Write minimal implementation**

In `src/haro_server/llm.py`, modify `LiteLlmClient.__init__`, add `set_tools`, and modify `stream_reply`:

```python
    def __init__(
        self,
        model: str,
        db_path: str,
        tools: list[dict] | None = None,
        call_tool=None,
    ) -> None:
        self._model = model
        self._db_path = db_path
        self._tools = tools
        self._call_tool = call_tool

    def set_tools(self, tools: list[dict], call_tool) -> None:
        """Attaches MCP tools after construction -- server.py's startup
        event (Task 10) connects to the MCP server asynchronously, which
        can only happen once uvicorn's event loop is running, by which
        point this object already exists. The constructor's tools/
        call_tool parameters above stay for tests and any future case
        that has tools available up front.
        """
        self._tools = tools
        self._call_tool = call_tool

    async def stream_reply(self, transcript: str) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": _build_effective_system_prompt(self._db_path)},
            {"role": "user", "content": transcript},
        ]

        if not self._tools:
            response = await litellm.acompletion(model=self._model, messages=messages, stream=True)
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return

        # Tools configured: decide tool use via a non-streamed call first
        # -- see this task's Context note for why streaming isn't used
        # for this decision step.
        decision = await litellm.acompletion(
            model=self._model, messages=messages, tools=self._tools, tool_choice="auto", stream=False
        )
        message = decision.choices[0].message
        if not message.tool_calls:
            if message.content:
                yield message.content
            return

        messages.append(message.model_dump())
        for tool_call in message.tool_calls:
            result = await self._call_tool(tool_call)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        # Follow-up call, no tools offered this time -- a plain streamed
        # reply, identical in shape to the no-tools-configured path above.
        response = await litellm.acompletion(model=self._model, messages=messages, stream=True)
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (every existing test in the file plus the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/haro_server/llm.py tests/test_llm.py
git commit -m "feat: add MCP tool-calling to LiteLlmClient.stream_reply()"
```

---

### Task 10: Wire the MCP client into server startup

**Files:**
- Modify: `src/haro_server/server.py`

**Interfaces:**
- Consumes: `McpToolClient` (Task 8), `LiteLlmClient.set_tools()` (Task 9).

**Context:** Like Task 7's scheduler/announcer, `McpToolClient.connect()` is async and needs a running event loop, but `create_app()` runs before uvicorn's loop exists (see Task 7's Context note -- same constraint, same fix: a FastAPI startup event handler, not a direct call in `create_app()`'s body). `llm = LiteLlmClient(...)` is therefore still constructed synchronously, without tools, exactly as today; the MCP connection and `llm.set_tools(...)` call happen later, from a second `@app.on_event("startup")` handler alongside Task 7's.

- [ ] **Step 1: Update `create_app()`**

In `src/haro_server/server.py`, `llm = LiteLlmClient(model=config.default_model, db_path=config.db_path)`'s construction line does not change. Add, right after Task 7's `_start_background_event_tasks` startup handler (same indentation, same place inside `create_app()`, before the `@app.websocket("/")` decorator):

```python
    @app.on_event("startup")
    async def _connect_mcp_client() -> None:
        if config.github_token:
            from .mcp_client import McpToolClient

            mcp_client = McpToolClient(github_token=config.github_token)
            await mcp_client.connect()
            llm.set_tools(mcp_client.tools, mcp_client.call_tool)
            logger.info("GitHub MCP tools loaded (%d tool(s))", len(mcp_client.tools))
        else:
            logger.info("GITHUB_TOKEN not set -- GitHub MCP tool-calling disabled")
```

`mcp_client` is deliberately a local variable inside this handler, not stored on `create_app()`'s scope beyond it -- nothing else needs to reach it after `set_tools()` has run (this plan does not implement graceful shutdown/`close()` for it, matching how `stt`/`tts`/`llm`/`navidrome` are also just constructed once and left for the process's lifetime with no explicit teardown elsewhere in this file).

- [ ] **Step 2: Manual verification (no automated test for this task -- it only wires already-tested pieces together)**

Run the server locally with `GITHUB_TOKEN` and `GITHUB_REPOS` set in `.env` to a real (low-scope, read-only) personal access token and a repo you have access to. Confirm in the logs: `"GitHub MCP tools loaded (N tool(s))"` with `N > 0`. Ask Haro a GitHub-related question via a real voice turn (or `fake-robot-client`, per this repo's README) and confirm the reply reflects real tool output, not a hallucinated answer.

- [ ] **Step 3: Run the full test suite one more time**

Run: `pytest tests/ -v`
Expected: PASS (everything from Tasks 1-9, unaffected by this wiring-only task)

- [ ] **Step 4: Commit**

```bash
git add src/haro_server/server.py
git commit -m "feat: connect GitHub MCP tools at server startup"
```
