import numpy as np

from haro_server.tts import KokoroTtsEngine


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


async def test_synthesize_flushes_on_sentence_boundaries():
    engine = KokoroTtsEngine.__new__(KokoroTtsEngine)  # bypass real __init__
    engine._pipeline = FakePipeline()
    engine._voice = "if_sara"

    chunks = [c async for c in engine.synthesize(_text_stream(["Ciao! ", "Come stai?"]))]

    assert engine._pipeline.calls == ["Ciao!", "Come stai?"]
    assert len(chunks) == 2
    assert all(isinstance(c, bytes) for c in chunks)
