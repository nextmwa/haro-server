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

        # Block indefinitely for the first event (nothing to speak yet
        # anyway). Once something is pending, only give an already-queued
        # sibling event a zero-timeout chance to land -- so several
        # events published back-to-back get combined into one
        # announcement -- rather than waiting a full poll interval for
        # more to show up before ever checking whether we can speak.
        timeout = None if not pending else 0
        done, _ = await asyncio.wait({next_event_task}, timeout=timeout)
        if next_event_task in done:
            pending.append(next_event_task.result())
            next_event_task = None
            continue  # check for more queued events before deciding to speak

        if not pending:
            continue

        session = get_active_session()
        if session is None:
            # No connected session at all: per the design spec's Error
            # Handling section, this is a single delivery attempt, not a
            # queue -- there's nothing to wait for here (no robot could
            # reconnect and retroactively receive this), so drop `pending`
            # instead of holding it forever. Each event's dedup_key is
            # already marked seen by the poller that produced it, so
            # dropping it here doesn't cause it to reappear on the next
            # poll cycle.
            logger.info(
                "dropping %d queued announcement(s): no connected session", len(pending)
            )
            pending = []
            continue
        if session.is_busy():
            # Busy (mid-reply or mid-track), but a session IS connected --
            # unlike the no-session case above, this is worth waiting out:
            # keep pending queued and retry at poll_interval_seconds,
            # without spinning the loop or re-blocking on the next event
            # indefinitely.
            await asyncio.sleep(poll_interval_seconds)
            continue

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


_MAX_COMBINED_SUMMARIES = 3


def _combine(events: list[Event]) -> str:
    if len(events) == 1:
        return events[0].summary
    shown = events[:_MAX_COMBINED_SUMMARIES]
    joined = "; ".join(e.summary for e in shown)
    remaining = len(events) - len(shown)
    if remaining > 0:
        # Even after the cold-start fix (I6), a burst of many events in
        # one cycle shouldn't turn into an unbounded monologue -- list the
        # first few and summarize the rest by count.
        return f"Ci sono {len(events)} aggiornamenti: {joined}; e altri {remaining}"
    return f"Ci sono {len(events)} aggiornamenti: {joined}"
