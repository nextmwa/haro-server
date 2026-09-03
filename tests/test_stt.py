import numpy as np

from haro_server.stt import ParakeetSttEngine, _bytes_to_float32


class FakeModel:
    def __init__(self, transcript: str) -> None:
        self.transcribe_calls: list[list[np.ndarray]] = []
        self._transcript = transcript

    def transcribe(self, audio_batch: list[np.ndarray]):
        self.transcribe_calls.append(audio_batch)

        class _Result:
            def __init__(self, text: str) -> None:
                self.text = text

        return [_Result(self._transcript)]


def test_bytes_to_float32_normalizes_int16_range():
    pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes()
    audio = _bytes_to_float32(pcm)
    assert audio.dtype == np.float32
    assert audio[0] == 0.0
    assert 0.49 < audio[1] < 0.51
    assert -0.51 < audio[2] < -0.49


def test_finalize_transcribes_all_fed_audio_and_resets():
    engine = ParakeetSttEngine.__new__(ParakeetSttEngine)  # bypass real __init__
    engine._model = FakeModel(transcript="ciao mondo")
    engine._buffer = bytearray()

    engine.feed(b"\x00\x00\x01\x00")
    engine.feed(b"\x02\x00\x03\x00")
    text = engine.finalize()

    assert text == "ciao mondo"
    assert len(engine._model.transcribe_calls) == 1
    fed_audio = engine._model.transcribe_calls[0][0]
    assert len(fed_audio) == 4  # 4 samples fed across the two frames

    # finalize() must reset state so the next turn starts clean.
    assert bytes(engine._buffer) == b""


def test_finalize_with_no_audio_returns_empty_string_without_calling_model():
    engine = ParakeetSttEngine.__new__(ParakeetSttEngine)
    engine._model = FakeModel(transcript="should not be used")
    engine._buffer = bytearray()

    text = engine.finalize()

    assert text == ""
    assert engine._model.transcribe_calls == []
