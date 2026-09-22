# tests/test_announcer.py
import asyncio

from haro_server.announcer import _combine, run_announcer
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


async def test_announcer_speaks_to_a_session_that_connects_after_an_earlier_drop():
    # Companion to the drop test above: once an event has been dropped
    # because no session was connected, a session connecting afterward
    # must NOT receive a stale announcement for it (there is nothing left
    # in `pending` to speak) -- but the announcer must still be alive and
    # able to handle a brand new event normally.
    bus = EventBus()
    bus.publish(Event(source="github", summary="build failed", dedup_key="gh:1"))

    session_holder: dict[str, FakeSession | None] = {"session": None}
    task = asyncio.ensure_future(
        run_announcer(bus, lambda: session_holder["session"], poll_interval_seconds=0.01)
    )
    await asyncio.sleep(0.05)  # the first event is dropped: no session yet

    session_holder["session"] = FakeSession(busy=False)
    bus.publish(Event(source="calendar", summary="meeting soon", dedup_key="cal:1"))
    await asyncio.sleep(0.05)
    task.cancel()

    assert session_holder["session"].spoken == ["meeting soon"]


def test_combine_caps_at_three_summaries_and_counts_the_rest():
    events = [
        Event(source="github", summary=f"evento {i}", dedup_key=f"gh:{i}")
        for i in range(5)
    ]

    text = _combine(events)

    for i in range(3):
        assert f"evento {i}" in text
    assert "evento 3" not in text
    assert "evento 4" not in text
    assert "e altri 2" in text


def test_combine_with_three_or_fewer_events_lists_all_of_them():
    events = [
        Event(source="github", summary="a", dedup_key="1"),
        Event(source="github", summary="b", dedup_key="2"),
        Event(source="github", summary="c", dedup_key="3"),
    ]

    text = _combine(events)

    assert "a" in text and "b" in text and "c" in text
    assert "e altri" not in text
