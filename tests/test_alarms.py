import asyncio
import datetime

import numpy as np
from test_session import FakeLlm, FakeStt, FakeTts

from haro_server import alarms, db, protocol
from haro_server.alarm_sound import gentle_chimes
from haro_server.assistant_tools import TIMEZONE
from haro_server.session import Session
from haro_server.tts_common import DEVICE_SAMPLE_RATE


def test_chimes_have_the_requested_length_and_rise_in_volume():
    pcm = gentle_chimes(10, start_gain=0.05, end_gain=0.8, seed=1)
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32767

    assert len(samples) == 10 * DEVICE_SAMPLE_RATE
    first, last = samples[: 2 * DEVICE_SAMPLE_RATE], samples[-2 * DEVICE_SAMPLE_RATE:]
    assert np.max(np.abs(first)) < 0.2
    assert np.max(np.abs(last)) > 0.4
    assert np.max(np.abs(samples)) <= 0.8 + 1e-3


def test_alarm_messages_are_speakable():
    at = datetime.datetime(2026, 9, 25, 7, 30, tzinfo=TIMEZONE)
    assert alarms.message_for(db.Alarm(1, at, "", "alarm", "pending")) == (
        "Buongiorno! Sono le 7 e 30. È ora di alzarsi."
    )
    assert alarms.message_for(db.Alarm(2, at.replace(minute=0), "palestra", "alarm", "pending")) == (
        "Sono le 7. Promemoria: palestra."
    )
    assert alarms.message_for(db.Alarm(3, at, "pasta", "timer", "pending")) == "Il timer è scaduto: pasta."


def _session(sent_text, sent_audio):
    async def send_text(text):
        sent_text.append(text)

    async def send_binary(data):
        sent_audio.append(data)

    return Session(FakeStt(""), FakeLlm(chunks=[]), FakeTts(), send_text, send_binary)


async def test_ring_alarm_plays_chimes_and_speech_then_ends_the_response(monkeypatch):
    monkeypatch.setattr(alarms, "ROUNDS", 2)
    monkeypatch.setattr(alarms, "CHIME_SECONDS", 1)
    sent_text, sent_audio = [], []
    session = _session(sent_text, sent_audio)

    await session.ring_alarm("Sono le 7.")

    assert sent_text[0] == protocol.encode_emotion("happy")
    assert sent_text[-1] == protocol.encode_response_end()
    chime_bytes = sum(len(a) for a in sent_audio)
    assert chime_bytes >= 2 * 1 * DEVICE_SAMPLE_RATE * 2  # two rounds of 1s chimes


async def test_an_interrupt_stops_a_ringing_alarm(monkeypatch):
    monkeypatch.setattr(alarms, "ROUNDS", 50)
    monkeypatch.setattr(alarms, "CHIME_SECONDS", 1)
    sent_text, sent_audio = [], []
    session = _session(sent_text, sent_audio)

    ringing = asyncio.create_task(session.ring_alarm("Sono le 7."))
    await asyncio.sleep(0.05)
    await session.handle_interrupt()
    await asyncio.wait_for(ringing, timeout=2)

    assert protocol.encode_response_end() not in sent_text


async def test_the_runner_rings_due_alarms_and_marks_them_done(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    now = datetime.datetime(2026, 9, 25, 5, 0, tzinfo=datetime.UTC)
    due = db.add_alarm(db_path, now - datetime.timedelta(seconds=3), "", "alarm")
    later = db.add_alarm(db_path, now + datetime.timedelta(hours=1), "", "alarm")
    rung = []

    class FakeSession:
        async def ring_alarm(self, message):
            rung.append(message)

    await alarms.check_once(db_path, lambda: FakeSession(), lambda: now)

    assert len(rung) == 1
    statuses = {a.id for a in db.get_pending_alarms(db_path)}
    assert statuses == {later}
    assert due not in statuses


async def test_an_alarm_with_no_robot_connected_waits_then_is_marked_missed(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    fire_at = datetime.datetime(2026, 9, 25, 5, 0, tzinfo=datetime.UTC)
    alarm_id = db.add_alarm(db_path, fire_at, "", "alarm")

    await alarms.check_once(db_path, lambda: None, lambda: fire_at + datetime.timedelta(minutes=1))
    assert [a.id for a in db.get_pending_alarms(db_path)] == [alarm_id]  # still waiting for the robot

    await alarms.check_once(db_path, lambda: None, lambda: fire_at + alarms.MISSED_AFTER + datetime.timedelta(seconds=1))
    assert db.get_pending_alarms(db_path) == []
