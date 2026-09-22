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
