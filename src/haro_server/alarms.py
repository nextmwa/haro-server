"""Rings alarms and timers (set by the LLM through assistant_tools.py) when
they come due: every few seconds, due alarms are handed to the connected
robot's session (Session.ring_alarm()), which alternates gentle chimes
(alarm_sound.py) with a spoken message until the user stops it with the
wake word or ROUNDS run out.
"""
import asyncio
import datetime
import logging
from typing import Callable

from . import db
from .assistant_tools import TIMEZONE

logger = logging.getLogger(__name__)

# One round = CHIME_SECONDS of chimes, then the spoken message. Volume rises
# across rounds (see Session.ring_alarm()): ~6 x 25s = about 2.5 minutes.
ROUNDS = 6
CHIME_SECONDS = 20
POLL_SECONDS = 5
# An alarm due while no robot is connected keeps waiting for it this long
# (Wi-Fi hiccup, server restart), then is marked missed instead of ringing
# at a surprising time later on.
MISSED_AFTER = datetime.timedelta(minutes=10)


def _spoken_time(when: datetime.datetime) -> str:
    local = when.astimezone(TIMEZONE)
    return f"{local.hour}" if local.minute == 0 else f"{local.hour} e {local.minute}"


def message_for(alarm: db.Alarm) -> str:
    if alarm.kind == "timer":
        return f"Il timer è scaduto: {alarm.label}." if alarm.label else "Il timer è scaduto."
    time_part = f"Sono le {_spoken_time(alarm.fire_at)}."
    if alarm.label:
        return f"{time_part} Promemoria: {alarm.label}."
    return f"Buongiorno! {time_part} È ora di alzarsi."


async def check_once(db_path: str, get_active_session: Callable, now: Callable[[], datetime.datetime]) -> None:
    for alarm in db.get_due_alarms(db_path, now()):
        session = get_active_session()
        if session is None:
            if now() - alarm.fire_at > MISSED_AFTER:
                logger.warning("alarm %d missed: no robot connected", alarm.id)
                db.set_alarm_status(db_path, alarm.id, "missed")
            continue
        logger.info("ringing %s %d", alarm.kind, alarm.id)
        db.set_alarm_status(db_path, alarm.id, "ringing")
        try:
            await session.ring_alarm(message_for(alarm))
        except Exception:
            logger.exception("alarm %d failed while ringing", alarm.id)
        finally:
            db.set_alarm_status(db_path, alarm.id, "done")


async def run_alarms(db_path: str, get_active_session: Callable) -> None:
    def utc_now() -> datetime.datetime:
        return datetime.datetime.now(datetime.UTC)

    while True:
        try:
            await check_once(db_path, get_active_session, utc_now)
        except Exception:
            logger.exception("alarm check failed")
        await asyncio.sleep(POLL_SECONDS)
