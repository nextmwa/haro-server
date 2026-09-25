import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient

from haro_server import db, turn_audio
from haro_server.admin import create_admin_router
from haro_server.session import Session

from test_session import FakeLlm, FakeStt, FakeTts


def test_save_writes_a_16khz_mono_pcm16_wav(tmp_path):
    db_path = str(tmp_path / "haro.db")
    pcm = b"\x01\x00" * 16000  # 1s

    turn_audio.save(db_path, 7, pcm)

    path = turn_audio.path_for(db_path, 7)
    assert path is not None
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() == 16000


def test_path_for_is_none_when_no_audio_was_saved(tmp_path):
    assert turn_audio.path_for(str(tmp_path / "haro.db"), 42) is None


def test_save_keeps_only_the_newest_files(tmp_path, monkeypatch):
    monkeypatch.setattr(turn_audio, "MAX_FILES", 3)
    db_path = str(tmp_path / "haro.db")
    for turn_id in range(1, 6):
        turn_audio.save(db_path, turn_id, b"\x00\x00")

    kept = [i for i in range(1, 6) if turn_audio.path_for(db_path, i) is not None]
    assert kept == [3, 4, 5]


def test_save_transcript_returns_the_new_row_id(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)

    first = db.save_transcript(db_path, "s", "uno", None, None)
    second = db.save_transcript(db_path, "s", "due", None, None)

    assert second == first + 1


async def _noop(_):
    pass


async def test_a_turn_saves_the_audio_the_robot_sent(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    session = Session(
        FakeStt(transcript="che ore sono"), FakeLlm(chunks=["[emotion:neutral] Le dieci."]),
        FakeTts(), _noop, _noop, db_path=db_path,
    )
    await session.handle_audio_frame(b"\x01\x00" * 100)
    await session.handle_audio_frame(b"\x02\x00" * 100)

    await session.handle_end_of_speech()

    [row] = db.get_transcripts(db_path)
    path = turn_audio.path_for(db_path, row.id)
    assert path is not None
    with wave.open(str(path)) as w:
        assert w.readframes(w.getnframes()) == b"\x01\x00" * 100 + b"\x02\x00" * 100


async def test_an_empty_transcript_is_still_logged_with_its_audio(tmp_path):
    # Exactly the turns worth listening to: the robot heard *something*
    # but STT made nothing of it.
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    session = Session(FakeStt(transcript=""), FakeLlm(chunks=[]), FakeTts(), _noop, _noop, db_path=db_path)
    await session.handle_audio_frame(b"\x05\x00" * 50)

    await session.handle_end_of_speech()

    [row] = db.get_transcripts(db_path)
    assert row.transcript == ""
    assert row.reply == "[trascrizione vuota, nessuna risposta]"
    assert turn_audio.path_for(db_path, row.id) is not None


async def test_the_next_turn_does_not_include_the_previous_turns_audio(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    session = Session(
        FakeStt(transcript="ciao"), FakeLlm(chunks=["[emotion:neutral] Ciao."]),
        FakeTts(), _noop, _noop, db_path=db_path,
    )
    await session.handle_audio_frame(b"\x01\x00" * 10)
    await session.handle_end_of_speech()
    await session.handle_audio_frame(b"\x02\x00" * 10)
    await session.handle_end_of_speech()

    newest = db.get_transcripts(db_path)[0]
    with wave.open(str(turn_audio.path_for(db_path, newest.id))) as w:
        assert w.readframes(w.getnframes()) == b"\x02\x00" * 10


def test_admin_serves_turn_audio_behind_the_password(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    turn_audio.save(db_path, 3, b"\x01\x00" * 10)
    app = FastAPI()
    app.include_router(create_admin_router(db_path, "secret"))
    client = TestClient(app)

    assert client.get("/admin/audio/3.wav").status_code == 401
    r = client.get("/admin/audio/3.wav", auth=("any", "secret"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert client.get("/admin/audio/4.wav", auth=("any", "secret")).status_code == 404


async def test_a_reply_cut_off_by_a_disconnect_is_still_logged(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)

    async def send_binary_disconnected(_):
        raise ConnectionError("robot went away")

    session = Session(
        FakeStt(transcript="raccontami una storia"),
        FakeLlm(chunks=["[emotion:happy] C'era una volta un robot."]),
        FakeTts(), _noop, send_binary_disconnected, db_path=db_path,
    )
    await session.handle_audio_frame(b"\x01\x00" * 10)

    try:
        await session.handle_end_of_speech()
    except ConnectionError:
        pass

    [row] = db.get_transcripts(db_path)
    assert row.transcript == "raccontami una storia"
    assert row.reply.endswith("[risposta interrotta]")
    assert turn_audio.path_for(db_path, row.id) is not None
