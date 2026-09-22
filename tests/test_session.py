import asyncio

import pytest

from haro_server import db
from haro_server.session import Session
from haro_server.session import _background_tasks as background_tasks


class FakeStt:
    def __init__(self, transcript: str) -> None:
        self.fed_frames: list[bytes] = []
        self._transcript = transcript

    def feed(self, frame: bytes) -> None:
        self.fed_frames.append(frame)

    async def finalize(self) -> str:
        return self._transcript


class FakeLlm:
    def __init__(
        self,
        chunks: list[str],
        facts: list[str] | None = None,
        music_query: str = "",
        picked_track_index: int | None = 0,
    ) -> None:
        self._chunks = chunks
        self._facts = facts if facts is not None else []
        self._music_query = music_query
        self._picked_track_index = picked_track_index
        self.received_transcript: str | None = None
        self.stream_reply_calls = 0
        self.extract_facts_calls: list[tuple[str, str]] = []
        self.extract_music_query_calls: list[str] = []
        self.pick_best_track_calls: list[tuple[str, list]] = []

    def stream_reply(self, transcript: str):
        self.stream_reply_calls += 1
        self.received_transcript = transcript
        return self._chunk_generator()

    async def _chunk_generator(self):
        for chunk in self._chunks:
            yield chunk

    async def extract_facts(self, transcript: str, reply: str) -> list[str]:
        self.extract_facts_calls.append((transcript, reply))
        return self._facts

    async def extract_music_query(self, transcript: str) -> str:
        self.extract_music_query_calls.append(transcript)
        return self._music_query

    async def pick_best_track(self, transcript: str, candidates: list):
        self.pick_best_track_calls.append((transcript, candidates))
        if self._picked_track_index is None or not candidates:
            return None
        return candidates[self._picked_track_index]


class FakeTts:
    def __init__(self) -> None:
        self.received_text: list[str] = []
        # Counted in the plain (non-generator) wrapper, so a call that is
        # never iterated is still recorded -- an `async def ... yield` body
        # would not run at all until the first __anext__.
        self.synthesize_calls = 0

    def synthesize(self, text_stream):
        self.synthesize_calls += 1
        return self._synthesize(text_stream)

    async def _synthesize(self, text_stream):
        async for text in text_stream:
            self.received_text.append(text)
            yield f"audio:{text}".encode()


class FakeTrack:
    def __init__(self, id: str, title: str, artist: str, album: str = "") -> None:
        self.id = id
        self.title = title
        self.artist = artist
        self.album = album


class FakeNavidrome:
    def __init__(self, search_results: list[FakeTrack] | None = None, pcm_chunks: list[bytes] | None = None) -> None:
        self._search_results = search_results if search_results is not None else []
        self._pcm_chunks = pcm_chunks if pcm_chunks is not None else [b"pcm1", b"pcm2"]
        self.search_calls: list[str] = []
        self.stream_calls: list[str] = []
        # Set when stream_track_as_pcm16's generator is cancelled mid-
        # iteration -- lets a test prove handle_interrupt() actually tore
        # down the stream rather than just cancelling the outer task.
        self.stream_cancelled = False

    async def search(self, query: str, count: int = 10) -> list[FakeTrack]:
        self.search_calls.append(query)
        return self._search_results

    async def stream_track_as_pcm16(self, track_id: str):
        self.stream_calls.append(track_id)
        try:
            for chunk in self._pcm_chunks:
                yield chunk
        except GeneratorExit:
            # session.py's finally block calls stream.aclose() on every
            # exit path (see its comment) -- aclose() throws GeneratorExit
            # into the generator at its suspended `yield`, NOT
            # CancelledError (that's what the *task* running _play_music()
            # received; this generator is a separate object). Mirrors
            # navidrome.py's real generator, which cleans up its ffmpeg
            # subprocess via a plain `finally` regardless of exception type
            # for the same reason.
            self.stream_cancelled = True
            raise


def _make_session(llm_chunks, sent_text=None, sent_binary=None):
    sent_text = sent_text if sent_text is not None else []
    sent_binary = sent_binary if sent_binary is not None else []

    async def send_text(text: str) -> None:
        sent_text.append(text)

    async def send_binary(data: bytes) -> None:
        sent_binary.append(data)

    stt = FakeStt(transcript="ciao come stai")
    llm = FakeLlm(chunks=llm_chunks)
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary)
    return session, stt, llm, tts, sent_text, sent_binary


async def test_handle_hello_stores_session_id():
    session, *_ = _make_session(llm_chunks=[])

    await session.handle_hello("session-1")

    assert session.session_id == "session-1"


async def test_handle_audio_frame_feeds_stt():
    session, stt, *_ = _make_session(llm_chunks=[])

    await session.handle_audio_frame(b"\x01\x02")

    assert stt.fed_frames == [b"\x01\x02"]


async def test_end_of_speech_sends_emotion_then_audio_then_response_end():
    # Record every send onto ONE shared, ordered list so the test can
    # actually tell if audio was sent before the emotion message (two
    # separate lists, one per channel, cannot prove cross-channel order).
    events: list[tuple[str, object]] = []

    async def send_text(text: str) -> None:
        events.append(("text", text))

    async def send_binary(data: bytes) -> None:
        events.append(("binary", data))

    stt = FakeStt(transcript="ciao come stai")
    # Emotion prefix split across two LLM chunks, on purpose.
    llm = FakeLlm(chunks=["[emo", "tion:happy] Ciao! ", "Come posso aiutarti?"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary)

    await session.handle_end_of_speech()

    assert llm.received_transcript == "ciao come stai"
    assert events == [
        ("text", '{"type": "emotion", "value": "happy"}'),
        ("binary", b"audio:Ciao! "),
        ("binary", b"audio:Come posso aiutarti?"),
        ("text", '{"type": "response_end"}'),
    ]
    assert tts.received_text == ["Ciao! ", "Come posso aiutarti?"]


async def test_end_of_speech_falls_back_to_neutral_when_no_emotion_tag():
    session, stt, llm, tts, sent_text, sent_binary = _make_session(
        llm_chunks=["Ciao!"]
    )

    await session.handle_end_of_speech()

    assert sent_text[0] == '{"type": "emotion", "value": "neutral"}'
    assert tts.received_text == ["Ciao!"]


async def test_end_of_speech_falls_back_to_neutral_for_long_reply_without_tag():
    # No bracket anywhere; longer than the internal scan window. All text
    # must still reach TTS -- none of it should be silently dropped.
    long_reply = "Questa e' una risposta piuttosto lunga senza alcun tag di emozione all'inizio, per verificare che il fallback non perda testo."
    session, stt, llm, tts, sent_text, sent_binary = _make_session(
        llm_chunks=[long_reply]
    )

    await session.handle_end_of_speech()

    assert sent_text[0] == '{"type": "emotion", "value": "neutral"}'
    assert "".join(tts.received_text) == long_reply


async def test_end_of_speech_with_empty_transcript_skips_llm_and_tts():
    events: list[tuple[str, object]] = []

    async def send_text(text: str) -> None:
        events.append(("text", text))

    async def send_binary(data: bytes) -> None:
        events.append(("binary", data))

    # Whitespace-only: silence or a VAD false trigger, not speech.
    stt = FakeStt(transcript="   \n ")
    llm = FakeLlm(chunks=["[emotion:happy] non dovrebbe accadere"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary)

    await session.handle_end_of_speech()

    assert llm.stream_reply_calls == 0
    assert tts.synthesize_calls == 0
    assert tts.received_text == []
    # Only response_end: no emotion message, no audio.
    assert events == [("text", '{"type": "response_end"}')]


async def test_end_of_speech_with_action_transcript_skips_llm_and_tts():
    import json

    events: list[tuple[str, object]] = []

    async def send_text(text: str) -> None:
        events.append(("text", text))

    async def send_binary(data: bytes) -> None:
        events.append(("binary", data))

    stt = FakeStt(transcript="lancia un dado")
    llm = FakeLlm(chunks=["[emotion:happy] non dovrebbe accadere"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary)

    await session.handle_end_of_speech()

    assert llm.stream_reply_calls == 0
    assert tts.synthesize_calls == 0
    assert len(events) == 2
    action_event = json.loads(events[0][1])
    assert action_event["type"] == "action"
    assert action_event["name"] == "dice_roll"
    assert action_event["result"] in range(1, 7)
    assert events[1] == ("text", '{"type": "response_end"}')


class _TrackingAsyncGen:
    """Wraps a plain async generator, counting explicit aclose() calls
    separately from natural exhaustion (which also runs a generator's
    `finally` block) -- used to prove the Session promptly closes a
    stream on both success and abort, instead of relying on GC.
    """

    def __init__(self, gen) -> None:
        self._gen = gen
        self.aclose_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        self.aclose_calls += 1
        await self._gen.aclose()


class _TrackingStream(_TrackingAsyncGen):
    def __init__(self, chunks: list[str]) -> None:
        super().__init__(self._make(chunks))

    async def _make(self, chunks: list[str]):
        for chunk in chunks:
            yield chunk


class TrackingLlm:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.stream: _TrackingStream | None = None

    def stream_reply(self, transcript: str):
        self.stream = _TrackingStream(self._chunks)
        return self.stream


class TrackingTts:
    """Fake TTS engine whose synthesize() return value tracks aclose()
    calls the same way TrackingLlm does for the LLM stream -- used to
    prove the Session closes the TTS generator too, on both success and
    abort, instead of only relying on GC.
    """

    def __init__(self) -> None:
        self.stream: _TrackingAsyncGen | None = None

    def synthesize(self, text_stream):
        self.stream = _TrackingAsyncGen(self._synthesize(text_stream))
        return self.stream

    async def _synthesize(self, text_stream):
        async for text in text_stream:
            yield f"audio:{text}".encode()


async def test_end_of_speech_closes_llm_stream_after_normal_completion():
    sent_text: list[str] = []
    sent_binary: list[bytes] = []

    async def send_text(text: str) -> None:
        sent_text.append(text)

    async def send_binary(data: bytes) -> None:
        sent_binary.append(data)

    stt = FakeStt(transcript="ciao come stai")
    llm = TrackingLlm(chunks=["[emotion:happy] Ciao!"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary)

    await session.handle_end_of_speech()

    # The fix must not break the successful path: it completes normally,
    # response_end is still sent, and the LLM stream is closed exactly once.
    assert sent_text[-1] == '{"type": "response_end"}'
    assert llm.stream is not None
    assert llm.stream.aclose_calls == 1


async def test_end_of_speech_closes_llm_stream_when_turn_is_aborted():
    calls = {"n": 0}

    async def send_text(text: str) -> None:
        pass

    async def send_binary(data: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("client disconnected")

    stt = FakeStt(transcript="ciao come stai")
    llm = TrackingLlm(chunks=["[emotion:happy] Ciao! ", "Come posso aiutarti?"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary)

    with pytest.raises(RuntimeError):
        await session.handle_end_of_speech()

    assert llm.stream is not None
    assert llm.stream.aclose_calls == 1


async def test_end_of_speech_closes_tts_stream_after_normal_completion():
    sent_text: list[str] = []
    sent_binary: list[bytes] = []

    async def send_text(text: str) -> None:
        sent_text.append(text)

    async def send_binary(data: bytes) -> None:
        sent_binary.append(data)

    stt = FakeStt(transcript="ciao come stai")
    llm = FakeLlm(chunks=["[emotion:happy] Ciao!"])
    tts = TrackingTts()
    session = Session(stt, llm, tts, send_text, send_binary)

    await session.handle_end_of_speech()

    # The fix must not break the successful path: it completes normally,
    # response_end is still sent, and the TTS stream is closed exactly once.
    assert sent_text[-1] == '{"type": "response_end"}'
    assert tts.stream is not None
    assert tts.stream.aclose_calls == 1


async def test_end_of_speech_closes_tts_stream_when_turn_is_aborted():
    calls = {"n": 0}

    async def send_text(text: str) -> None:
        pass

    async def send_binary(data: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("client disconnected")

    stt = FakeStt(transcript="ciao come stai")
    llm = FakeLlm(chunks=["[emotion:happy] Ciao! ", "Come posso aiutarti?"])
    tts = TrackingTts()
    session = Session(stt, llm, tts, send_text, send_binary)

    with pytest.raises(RuntimeError):
        await session.handle_end_of_speech()

    assert tts.stream is not None
    assert tts.stream.aclose_calls == 1


async def _drain_background_tasks() -> None:
    """Awaits every fact-extraction task Session._spawn_fact_extraction has
    scheduled so far, so a test can assert on its effects (a fire-and-
    forget asyncio.create_task() would otherwise still be pending, or not
    even started, when the test function itself returns).
    """
    for task in list(background_tasks):
        await task


async def test_end_of_speech_saves_the_transcript_when_db_path_is_set(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)

    async def send_text(text: str) -> None:
        pass

    async def send_binary(data: bytes) -> None:
        pass

    stt = FakeStt(transcript="che tempo fa oggi")
    llm = FakeLlm(chunks=["[emotion:neutral] Fa bello oggi."])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary, db_path=db_path)
    await session.handle_hello("session-xyz")

    await session.handle_end_of_speech()
    await _drain_background_tasks()

    transcripts = db.get_transcripts(db_path)
    assert len(transcripts) == 1
    assert transcripts[0].session_id == "session-xyz"
    assert transcripts[0].transcript == "che tempo fa oggi"
    assert transcripts[0].reply == "Fa bello oggi."
    assert transcripts[0].emotion == "neutral"


async def test_end_of_speech_saves_action_turns_to_the_transcript_log(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)

    async def send_text(text: str) -> None:
        pass

    async def send_binary(data: bytes) -> None:
        pass

    stt = FakeStt(transcript="lancia un dado")
    llm = FakeLlm(chunks=["non dovrebbe essere chiamato"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary, db_path=db_path)

    await session.handle_end_of_speech()

    transcripts = db.get_transcripts(db_path)
    assert len(transcripts) == 1
    assert transcripts[0].transcript == "lancia un dado"
    assert transcripts[0].reply is not None
    assert transcripts[0].reply.startswith("[azione: dice_roll")


async def test_end_of_speech_stores_extracted_facts_as_memories(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)

    async def send_text(text: str) -> None:
        pass

    async def send_binary(data: bytes) -> None:
        pass

    stt = FakeStt(transcript="il mio compleanno e' il 5 maggio")
    llm = FakeLlm(
        chunks=["[emotion:happy] Bello saperlo!"],
        facts=["il compleanno dell'utente e' il 5 maggio"],
    )
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary, db_path=db_path)
    await session.handle_hello("session-abc")

    await session.handle_end_of_speech()
    await _drain_background_tasks()

    assert llm.extract_facts_calls == [("il mio compleanno e' il 5 maggio", "Bello saperlo!")]
    memories = db.get_memories(db_path)
    assert len(memories) == 1
    assert memories[0].fact == "il compleanno dell'utente e' il 5 maggio"
    assert memories[0].source_session_id == "session-abc"


async def test_end_of_speech_does_not_touch_the_db_when_db_path_is_none():
    # Every other test in this file constructs Session without db_path
    # (defaults to None) and must keep working with zero DB side effects --
    # this test only makes that "no persistence configured" contract
    # explicit rather than relying on it being implied by every other test
    # simply not checking for a db file.
    session, *_ = _make_session(llm_chunks=["[emotion:happy] ok"])
    assert session._db_path is None

    await session.handle_end_of_speech()
    await _drain_background_tasks()  # must be a no-op: nothing was spawned


async def test_music_request_without_navidrome_configured_sends_an_error():
    events: list[tuple[str, object]] = []

    async def send_text(text: str) -> None:
        events.append(("text", text))

    async def send_binary(data: bytes) -> None:
        events.append(("binary", data))

    stt = FakeStt(transcript="metti della musica di Vasco Rossi")
    llm = FakeLlm(chunks=["non dovrebbe essere chiamato"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary)  # navidrome=None (default)

    await session.handle_end_of_speech()
    await asyncio.sleep(0)  # let the spawned task run

    assert any(e[0] == "text" and '"type": "error"' in e[1] for e in events)
    assert any(e[1] == '{"type": "response_end"}' for e in events)
    assert events[-1] == ("text", '{"type": "response_end"}')


async def test_music_request_plays_the_picked_track():
    events: list[tuple[str, object]] = []

    async def send_text(text: str) -> None:
        events.append(("text", text))

    async def send_binary(data: bytes) -> None:
        events.append(("binary", data))

    stt = FakeStt(transcript="metti Vita Spericolata di Vasco Rossi")
    track = FakeTrack(id="song-1", title="Vita Spericolata", artist="Vasco Rossi")
    llm = FakeLlm(chunks=[], music_query="Vita Spericolata Vasco Rossi", picked_track_index=0)
    navidrome = FakeNavidrome(search_results=[track], pcm_chunks=[b"pcm1", b"pcm2"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary, navidrome=navidrome)

    await session.handle_end_of_speech()
    await session._music_task

    assert llm.extract_music_query_calls == ["metti Vita Spericolata di Vasco Rossi"]
    assert navidrome.search_calls == ["Vita Spericolata Vasco Rossi"]
    assert navidrome.stream_calls == ["song-1"]
    assert events[0] == ("text", '{"type": "action", "name": "music_playing", "result": "Vita Spericolata - Vasco Rossi"}')
    assert events[1] == ("binary", b"pcm1")
    assert events[2] == ("binary", b"pcm2")
    assert events[-1] == ("text", '{"type": "response_end"}')


async def test_music_request_with_no_search_results_sends_an_error():
    events: list[tuple[str, object]] = []

    async def send_text(text: str) -> None:
        events.append(("text", text))

    async def send_binary(data: bytes) -> None:
        events.append(("binary", data))

    stt = FakeStt(transcript="metti qualcosa che non esiste")
    llm = FakeLlm(chunks=[], music_query="qualcosa che non esiste")
    navidrome = FakeNavidrome(search_results=[])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary, navidrome=navidrome)

    await session.handle_end_of_speech()
    await session._music_task

    assert navidrome.stream_calls == []
    assert any(e[0] == "text" and '"type": "error"' in e[1] for e in events)
    assert events[-1] == ("text", '{"type": "response_end"}')


async def test_handle_interrupt_cancels_an_in_flight_music_task():
    events: list[tuple[str, object]] = []

    async def send_text(text: str) -> None:
        events.append(("text", text))

    async def send_binary(data: bytes) -> None:
        events.append(("binary", data))
        await asyncio.sleep(0.05)  # simulate real send pacing so there's time to interrupt

    stt = FakeStt(transcript="metti musica")
    track = FakeTrack(id="song-1", title="Some Song", artist="Some Artist")
    llm = FakeLlm(chunks=[], music_query="musica", picked_track_index=0)
    navidrome = FakeNavidrome(search_results=[track], pcm_chunks=[b"pcm1", b"pcm2", b"pcm3", b"pcm4"])
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary, navidrome=navidrome)

    await session.handle_end_of_speech()
    await asyncio.sleep(0.01)  # let the first chunk or two send
    await session.handle_interrupt()

    with pytest.raises(asyncio.CancelledError):
        await session._music_task

    assert navidrome.stream_cancelled is True
    # Interrupted mid-track: response_end must NOT have been sent (the
    # robot is about to start a new LISTENING turn, not finish this one).
    assert not any(e == ("text", '{"type": "response_end"}') for e in events)


async def test_handle_interrupt_with_no_active_music_task_is_a_no_op():
    session, *_ = _make_session(llm_chunks=[])

    await session.handle_interrupt()  # must not raise


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


class GatedTts:
    """TTS fake whose synthesize() blocks mid-stream on an asyncio.Event
    until the test releases it -- used to hold Session._lock open long
    enough that a concurrent caller can be proven to actually block on
    it, not just win a lucky race. `entered` fires the moment the first
    chunk is about to be produced (i.e. once the caller is definitely
    inside the locked section), so a test can wait on that instead of an
    arbitrary sleep.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.entered = asyncio.Event()

    def synthesize(self, text_stream):
        return self._gen(text_stream)

    async def _gen(self, text_stream):
        async for text in text_stream:
            self.entered.set()
            await self.release.wait()
            yield f"audio:{text}".encode()


async def test_speak_announcement_and_a_user_turn_cannot_run_concurrently():
    # I3: proves the *lock* closes the race, not just is_busy() -- a real
    # user turn started while speak_announcement() is mid-flight must
    # block until the announcement's lock is released, rather than
    # interleaving TTS/WebSocket calls with it.
    tts = GatedTts()

    async def send_text(text: str) -> None:
        pass

    async def send_binary(data: bytes) -> None:
        pass

    stt = FakeStt(transcript="ciao come stai")
    llm = FakeLlm(chunks=["[emotion:neutral] ciao"])
    session = Session(stt, llm, tts, send_text, send_binary)

    announce_task = asyncio.ensure_future(session.speak_announcement("aggiornamento"))
    await asyncio.wait_for(tts.entered.wait(), timeout=1.0)  # now inside the locked section

    turn_task = asyncio.ensure_future(session.handle_end_of_speech())
    await asyncio.sleep(0.02)

    # The user turn must still be stuck waiting on the lock -- it hasn't
    # even reached its own body (stt.finalize()/llm.stream_reply()) yet.
    assert not turn_task.done()
    assert llm.stream_reply_calls == 0

    tts.release.set()  # let the announcement finish and release the lock
    await announce_task
    await turn_task

    assert llm.stream_reply_calls == 1


async def test_handle_end_of_speech_and_speak_announcement_cannot_run_concurrently_either_order():
    # Same race, opposite ordering: a user turn already holding the lock
    # must block a proactive announcement that arrives mid-turn.
    tts = GatedTts()

    async def send_text(text: str) -> None:
        pass

    async def send_binary(data: bytes) -> None:
        pass

    stt = FakeStt(transcript="ciao come stai")
    llm = FakeLlm(chunks=["[emotion:neutral] ciao"])
    session = Session(stt, llm, tts, send_text, send_binary)

    turn_task = asyncio.ensure_future(session.handle_end_of_speech())
    await asyncio.wait_for(tts.entered.wait(), timeout=1.0)  # now inside the locked section

    announce_task = asyncio.ensure_future(session.speak_announcement("aggiornamento"))
    await asyncio.sleep(0.02)

    assert not announce_task.done()

    tts.release.set()
    await turn_task
    await announce_task  # completes once the turn releases the lock


async def test_handle_interrupt_cancels_an_in_flight_announcement():
    events: list[tuple[str, object]] = []

    async def send_text(text: str) -> None:
        events.append(("text", text))

    async def send_binary(data: bytes) -> None:
        events.append(("binary", data))
        await asyncio.sleep(0.05)  # simulate real send pacing so there's time to interrupt

    class SlowMultiChunkTts:
        """Unlike the real TTS engines, yields several binary chunks per
        single text chunk -- announcements are always a single
        `_single_chunk(text)` input, so this is needed to have more than
        one binary frame in flight to interrupt mid-stream."""

        def synthesize(self, text_stream):
            return self._gen(text_stream)

        async def _gen(self, text_stream):
            async for _ in text_stream:
                for chunk in [b"a1", b"a2", b"a3", b"a4"]:
                    yield chunk

    stt = FakeStt(transcript="")
    llm = FakeLlm(chunks=[])
    session = Session(stt, llm, SlowMultiChunkTts(), send_text, send_binary)

    announce_task = asyncio.ensure_future(session.speak_announcement("la build e' fallita"))
    await asyncio.sleep(0.01)  # let the first chunk or two send
    assert session._announcement_task is not None

    await session.handle_interrupt()
    await announce_task  # must NOT raise -- speak_announcement() swallows the cancellation itself

    assert session._announcement_task is None
    # Interrupted mid-announcement: response_end must NOT have been sent.
    assert not any(e == ("text", '{"type": "response_end"}') for e in events)
    # At least one chunk got out before the interrupt landed, proving this
    # was cancelled mid-stream rather than never starting.
    assert any(e[0] == "binary" for e in events)


async def test_handle_interrupt_with_no_active_announcement_is_a_no_op():
    session, *_ = _make_session(llm_chunks=[])

    await session.handle_interrupt()  # must not raise (covers both music and announcement branches)
