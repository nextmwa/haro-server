import asyncio

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
