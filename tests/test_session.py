from haro_server.session import Session


class FakeStt:
    def __init__(self, transcript: str) -> None:
        self.fed_frames: list[bytes] = []
        self._transcript = transcript

    def feed(self, frame: bytes) -> None:
        self.fed_frames.append(frame)

    def finalize(self) -> str:
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
    # Emotion prefix split across two LLM chunks, on purpose.
    session, stt, llm, tts, sent_text, sent_binary = _make_session(
        llm_chunks=["[emo", "tion:happy] Ciao! ", "Come posso aiutarti?"]
    )

    await session.handle_end_of_speech()

    assert llm.received_transcript == "ciao come stai"
    assert sent_text[0] == '{"type": "emotion", "value": "happy"}'
    assert sent_text[-1] == '{"type": "response_end"}'
    assert sent_binary == [b"audio:Ciao! ", b"audio:Come posso aiutarti?"]
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
