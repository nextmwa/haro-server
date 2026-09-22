import numpy as np

from haro_server.chatterbox_tts import ChatterboxTtsEngine


class FakeChatterboxModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.sr = 24000

    def generate(self, text: str, language_id: str):
        self.calls.append((text, language_id))
        # Real generate() returns shape (1, num_samples), not a flat 1-D
        # array -- see _synthesize_sentence()'s squeeze() comment.
        return np.full((1, 10), 0.1, dtype=np.float32)


async def _text_stream(chunks: list[str]):
    for chunk in chunks:
        yield chunk


def _fake_engine(language: str = "it") -> ChatterboxTtsEngine:
    engine = ChatterboxTtsEngine.__new__(ChatterboxTtsEngine)  # bypass real __init__
    engine._model = FakeChatterboxModel()
    engine._language = language
    return engine


async def test_synthesize_flushes_on_sentence_boundaries():
    engine = _fake_engine()

    chunks = [c async for c in engine.synthesize(_text_stream(["Ciao! ", "Come stai?"]))]

    assert engine._model.calls == [("Ciao! ", "it"), ("Come stai?", "it")]
    assert len(chunks) == 2
    assert all(isinstance(c, bytes) for c in chunks)


async def test_synthesize_passes_the_configured_language_id():
    engine = _fake_engine(language="fr")

    [_ async for _ in engine.synthesize(_text_stream(["Bonjour! "]))]

    assert engine._model.calls == [("Bonjour! ", "fr")]


async def test_synthesize_flushes_trailing_text_without_a_terminator():
    engine = _fake_engine()

    chunks = [c async for c in engine.synthesize(_text_stream(["nessun punto finale"]))]

    assert engine._model.calls == [("nessun punto finale", "it")]
    assert len(chunks) == 1


def test_synthesize_sentence_squeezes_the_batch_dimension_before_converting():
    engine = _fake_engine()

    pcm = engine._synthesize_sentence("Ciao")

    # FakeChatterboxModel yields shape (1, 10) at sr=24000 -- resampled
    # down to DEVICE_SAMPLE_RATE (16000), so not exactly 10 samples/20
    # bytes, but must still be a flat, even-length PCM16 buffer (not
    # corrupted by an un-squeezed leading dimension).
    assert isinstance(pcm, bytes)
    assert len(pcm) % 2 == 0
