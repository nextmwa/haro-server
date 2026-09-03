import numpy as np

from haro_server.tts import KokoroTtsEngine, _find_sentence_boundary, _to_pcm16


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str, voice: str):
        self.calls.append(text)
        # One "sentence" -> one fake audio segment of quiet, valid samples.
        audio = np.full(10, 0.1, dtype=np.float32)
        yield ("graphemes", "phonemes", audio)


async def _text_stream(chunks: list[str]):
    for chunk in chunks:
        yield chunk


def test_find_sentence_boundary_finds_end_of_first_sentence():
    assert _find_sentence_boundary("Ciao! Come stai?") == len("Ciao! ")


def test_find_sentence_boundary_returns_none_without_terminator():
    assert _find_sentence_boundary("nessun punto qui") is None


def test_to_pcm16_converts_float_audio_to_16_bit_bytes():
    audio = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    pcm = _to_pcm16(audio)
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples[0] == 0
    assert samples[1] == int(0.5 * 32767)
    # Out-of-range values are clipped, not wrapped.
    assert samples[3] == 32767
    assert samples[4] == -32767


async def test_synthesize_flushes_on_sentence_boundaries():
    engine = KokoroTtsEngine.__new__(KokoroTtsEngine)  # bypass real __init__
    engine._pipeline = FakePipeline()
    engine._voice = "if_sara"

    chunks = [c async for c in engine.synthesize(_text_stream(["Ciao! ", "Come stai?"]))]

    assert engine._pipeline.calls == ["Ciao! ", "Come stai?"]
    assert len(chunks) == 2
    assert all(isinstance(c, bytes) for c in chunks)
