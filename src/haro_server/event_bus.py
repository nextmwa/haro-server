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
