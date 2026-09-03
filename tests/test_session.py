import pytest

from haro_server.session import Session


class FakeStt:
    def __init__(self, transcript: str) -> None:
        self.fed_frames: list[bytes] = []
        self._transcript = transcript

    def feed(self, frame: bytes) -> None:
        self.fed_frames.append(frame)

    async def finalize(self) -> str:
        return self._transcript


class FakeLlm:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.received_transcript: str | None = None

    def stream_reply(self, transcript: str):
        self.received_transcript = transcript
        return self._chunk_generator()

    async def _chunk_generator(self):
        for chunk in self._chunks:
            yield chunk


class FakeTts:
    def __init__(self) -> None:
        self.received_text: list[str] = []

    async def synthesize(self, text_stream):
        async for text in text_stream:
            self.received_text.append(text)
            yield f"audio:{text}".encode()


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
