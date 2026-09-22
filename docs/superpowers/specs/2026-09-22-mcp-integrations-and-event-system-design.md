# MCP Integrations and Proactive Event System — Design

## Goal

Give Haro two new, related capabilities:

1. **On-demand external knowledge**: answer spoken questions about GitHub
   (pull requests, issues, CI status) and the user's Google Calendar by
   letting the LLM call out to MCP (Model Context Protocol) servers
   during a normal conversation.
2. **Proactive announcements**: notice when something new happens on
   those same sources (a build fails, a meeting starts in a few minutes)
   and speak up about it unprompted, without the user asking first.

These are two different mechanisms serving two different interaction
shapes, built to share the same underlying GitHub/Calendar access and the
same audio-delivery path to the device, so that a third source (e.g.
Sentry, Slack) can be added later by writing only the piece specific to
that source.

## Context

Haro is a personal desk voice assistant: an ESP32-S3 firmware talking
over WebSocket to `haro-server` (this repo, FastAPI + LiteLLM for the
LLM call + a TTS engine). Today, every spoken reply is triggered by the
user: wake word → mic audio streamed to the server → STT → LLM →
TTS → audio streamed back. `session.py`'s `Session` owns one
conversation's state; `db.py` already has a SQLite database used for
transcripts, long-term memory, and the admin-editable system prompt.

The user is a web developer who uses Haro at a work desk. Confirmed
during design: GitHub (not GitLab) and a work Google Calendar are the
two sources to start with. No public endpoint is available for GitHub
to push webhooks to, so all external-source checking is done by
`haro-server` polling outward — nothing needs to be exposed to the
internet.

## Non-Goals

- No inbound webhooks (GitHub, Google, or otherwise) in this iteration.
  Polling only, per the user's explicit choice.
- No new firmware behavior. The ESP32 side is unchanged: it still just
  streams mic audio up and plays whatever PCM audio the server sends it.
  A proactive announcement uses the exact same "server sends audio,
  device plays it" path a normal reply already uses.
- No UI for configuring which repos/calendars are watched in this
  iteration — configuration is `.env`-based, matching every other piece
  of configuration in this project (`NAVIDROME_URL`, `DEFAULT_MODEL`,
  etc.).
- No summarization/prioritization intelligence beyond the simple
  "more than one event queued → one combined sentence" rule below.
  Sentry/Slack/other sources are explicitly out of scope for this spec;
  the design just needs to not preclude adding them later.

## Architecture

Both new capabilities live entirely in `haro-server`. The firmware needs
no changes: it already receives audio to play over the same WebSocket
connection used for normal replies, regardless of what caused the server
to send it.

Two independent subsystems:

**1. MCP client (on-demand path).** `haro-server` connects to MCP
servers (GitHub, Google Calendar) as an MCP client, and their tools are
exposed to the LLM via LiteLLM's tool/function-calling. When the user
asks something the LLM judges relevant ("ho PR da revieware?", "quando
ho la prossima riunione?"), the LLM decides to call the matching tool,
gets a result, and folds it into its normal reply — the same
turn-taking flow `session.py` already implements, just with the LLM
occasionally pausing mid-turn to call a tool before finishing its
answer.

**2. Event poller + bus (proactive path).** A background asyncio task
polls GitHub and Calendar directly (their REST/API clients, not through
MCP or the LLM — this is a scheduled check, not a conversation) on a
fixed interval, diffs the result against state persisted in the
existing SQLite database, and publishes anything genuinely new onto an
in-process event queue. A single listener drains that queue, holds each
event until Haro is idle, then delivers it as speech over the same
audio-send path a normal reply uses.

This split exists because MCP tool-calling only runs when something
(the user's turn) asks the LLM to run — it has no way to notice a change
on its own between conversations. The poller is the piece that notices;
the MCP tools are the piece that answers when asked.

## Components

All new files in `src/haro_server/`:

- **`mcp_client.py`** — Connects to the MCP servers configured via
  `.env` (see Configuration below), fetches their tool schemas, and
  hands them to `llm.py`'s `LiteLlmClient` so tool-calling is available
  during a normal turn. One client instance, shared across sessions
  (mirrors how `stt`/`tts`/`llm` are already loaded once at server
  startup in `server.py`).

- **`event_poller.py`** — One async function per source
  (`poll_github(...)`, `poll_calendar(...)`), each returning a list of
  new `Event` objects (a small dataclass: `source`, `summary` text,
  `dedup_key`). A scheduler task calls each poller every
  `EVENT_POLL_INTERVAL_SECONDS` (default 180s = 3 minutes — comfortably
  under both APIs' rate limits for a single user) and publishes
  whatever each poller returns onto the event bus. Adding a third
  source later means writing one more `poll_x()` function with this
  same shape; nothing else changes.

- **`event_bus.py`** — A minimal in-process async queue
  (`asyncio.Queue`) with `publish(event)` and an async iterator for
  consumption. No persistence, no cross-process delivery — this is
  intentionally the simplest thing that decouples "something noticed a
  change" from "something reacts to it", not a general message broker.

- **Announcement delivery**, inside `server.py` or a small
  `announcer.py`: consumes the event bus, and for each event (or batch,
  see Error Handling), waits until the connected session reports itself
  idle, synthesizes the announcement text through the same
  `tts.py`/`tts_common.py` pipeline `Session` already uses, and streams
  it to the device the same way a normal reply's audio is streamed.
  "Idle" here is new, server-side state: `Session` doesn't currently
  track whether a turn is in progress at all (there's no firmware-
  orchestrator-state visibility on the server -- HARO_STATE_LISTENING/
  THINKING/SPEAKING live only in the ESP32's orchestrator.c). This adds
  an `is_busy()` check on `Session`, true while `handle_end_of_speech()`
  is running its reply OR while `self._music_task` is set and not
  `.done()` -- music streams as a background task that outlives
  `handle_end_of_speech()`'s own return (see that method's existing
  `_music_task` handling, already used by `handle_interrupt()`), so
  checking only "is a reply in flight" would wrongly report idle while
  a track is still playing and let an announcement barge in over music.
  The announcer polls this check, not the firmware's state machine, and
  reuses `Session`'s existing `send_binary`/audio-chunk-streaming logic
  rather than duplicating it.

- **`db.py` additions** — one new table, `event_state`, storing the last
  `dedup_key` seen per source (e.g. last-seen PR IDs for GitHub, last
  CI status per branch, last-notified calendar event IDs), so a poller
  run that finds nothing new compared to the last run correctly returns
  an empty list.

## Data Flow

**On-demand (MCP):**
1. Wake word → mic audio streamed to server (unchanged).
2. STT transcribes, `LiteLlmClient` gets the turn.
3. If the LLM decides a GitHub/Calendar question is being asked, it
   calls the matching MCP tool, gets a result, and uses it while
   composing its reply.
4. Reply streamed back as TTS audio (unchanged).

**Proactive (poller):**
1. Every `EVENT_POLL_INTERVAL_SECONDS`, `event_poller.py` calls each
   source's poll function directly against GitHub's/Google's APIs.
2. Each poller diffs its result against `event_state` in the database.
3. Anything new is published to `event_bus.py`.
4. The announcer drains the bus. For each event: if `Session.is_busy()`
   is false, synthesize and speak it now; if a reply or music track is
   in progress, hold it and re-check once the session goes idle.
5. `event_state` is updated once an event is either announced or
   deliberately dropped (see Error Handling) — either way, it is never
   announced a second time.

## Error Handling

- **Multiple events queued while busy**: announced one at a time, in
  order, once idle — except when more than one is waiting at the moment
  Haro becomes idle, in which case they're combined into one sentence
  ("Ci sono 3 aggiornamenti: ...") rather than three separate
  back-to-back announcements.
- **Device unreachable when an event is ready to announce**: one
  delivery attempt; if it fails (no connected session, or the send
  itself fails), the event is dropped, not retried or queued
  indefinitely — a build-failed notice from 20 minutes ago delivered
  late is low-value, and the same information stays available on-demand
  via the MCP tools in the meantime. `event_state` is still updated so
  a dropped event isn't re-announced once connectivity returns.
- **API rate limits**: `EVENT_POLL_INTERVAL_SECONDS` default (180s) is
  chosen to stay well under GitHub's and Google Calendar's per-user rate
  limits; not a real risk for a single-user deployment.
- **Poller failures** (network error, API error, auth expired): logged
  and skipped for that cycle, not fatal to the poller task — the next
  scheduled poll tries again. Mirrors how the rest of this project
  already treats a degraded network as an expected, recoverable
  condition rather than a crash.
- **Credentials**: GitHub token and Google Calendar OAuth
  credentials/refresh token are read from `.env`, matching every other
  secret in this project (`ANTHROPIC_API_KEY`, `NAVIDROME_PASSWORD`,
  etc.) — never committed, never logged.

## Configuration (`.env`)

New variables, all optional (unset = that integration is inactive,
matching the existing `NAVIDROME_*` "unset = disabled" convention):

```
GITHUB_TOKEN=
GITHUB_REPOS=            # comma-separated "owner/repo" list to watch
GOOGLE_CALENDAR_CREDENTIALS_PATH=   # path to an OAuth client/token file
EVENT_POLL_INTERVAL_SECONDS=180
```

## Testing

- `event_poller.py`'s poll functions take an injected API client, so
  they're unit-testable against a fake client returning canned
  GitHub/Calendar responses — verifying dedup against `event_state`
  works without hitting a real API.
- `event_bus.py` is tested directly: publish N events, assert an
  iterating consumer sees exactly those N in order.
- The announcer's idle/busy branching is tested against a fake
  `Session`-like object exposing just the state check and a recorded
  "spoke X" call, the same `FakePipeline`-style fake-object pattern
  `test_tts.py` already uses elsewhere in this repo.
- MCP tool-calling itself is exercised against LiteLLM's existing
  function-calling test coverage patterns (a fake tool call in, a fake
  tool result out) rather than a real MCP server round-trip in CI.
