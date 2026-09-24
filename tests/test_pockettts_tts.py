import numpy as np

from haro_server.pockettts_tts import PocketTtsEngine, resolve_voice
from haro_server.tts_common import DEVICE_SAMPLE_RATE


class FakePocketModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []
        self.sample_rate = 24000

    def get_state_for_audio_prompt(self, voice: str):
        return f"voice-state:{voice}"

    def generate_audio(self, voice_state, text: str):
        self.calls.append((voice_state, text))
        # Real generate_audio() returns a flat 1-D tensor, unlike
        # Chatterbox's (1, num_samples) -- no squeeze() needed downstream.
        return np.full(10, 0.1, dtype=np.float32)

    def generate_audio_stream(self, voice_state, text: str):
        self.calls.append((voice_state, text))
        # Real Pocket TTS yields one Mimi frame per chunk: 80ms at 24kHz.
        for _ in range(self.stream_chunks_per_sentence):
            yield np.full(1920, 0.1, dtype=np.float32)

    stream_chunks_per_sentence = 2


async def _text_stream(chunks: list[str]):
    for chunk in chunks:
        yield chunk


def _fake_engine(voice: str = "giovanni") -> PocketTtsEngine:
    engine = PocketTtsEngine.__new__(PocketTtsEngine)  # bypass real __init__
    engine._model = FakePocketModel()
    engine._voice_state = engine._model.get_state_for_audio_prompt(voice)
    return engine


async def test_synthesize_flushes_on_sentence_boundaries():
    engine = _fake_engine()

    chunks = [c async for c in engine.synthesize(_text_stream(["Ciao! ", "Come stai?"]))]

    assert engine._model.calls == [
        ("voice-state:giovanni", "Ciao! "),
        ("voice-state:giovanni", "Come stai?"),
    ]
    assert len(chunks) >= 2
    assert all(isinstance(c, bytes) for c in chunks)


async def test_synthesize_sends_frames_immediately_until_ahead_then_coalesces():
    engine = _fake_engine()
    # 150 x 80ms = 12s of audio, generated instantly (far above real time).
    engine._model.stream_chunks_per_sentence = 150

    chunks = [c async for c in engine.synthesize(_text_stream(["Una frase lunga."]))]

    seconds = [len(c) / 2 / DEVICE_SAMPLE_RATE for c in chunks]
    # First send: one frame, not a whole sentence -- the robot starts
    # talking as soon as possible.
    assert seconds[0] < 0.2
    # Once the robot has 1s+ queued, sends are coalesced to >=1s, so the
    # firmware's 32-slot event queue can't be flooded with 80ms frames.
    assert all(s >= 1.0 for s in seconds[-5:-1])
    assert len(chunks) < 32
    # No audio lost or duplicated by the streaming resampler.
    assert abs(sum(seconds) - 12.0) < 0.05


async def test_synthesize_keeps_sending_frames_immediately_while_not_ahead(monkeypatch):
    engine = _fake_engine()
    engine._model.stream_chunks_per_sentence = 10
    # An unreachable "ahead" threshold: the robot is never far enough
    # ahead, as when generation only just keeps up with playback -- every
    # frame must go out right away, never held back to build a bigger send.
    import haro_server.pockettts_tts as mod
    monkeypatch.setattr(mod, "STREAM_COALESCE_SECONDS", 1e6)

    chunks = [c async for c in engine.synthesize(_text_stream(["Frase."]))]

    assert all(len(c) / 2 / DEVICE_SAMPLE_RATE < 0.2 for c in chunks)


async def test_synthesize_uses_the_configured_voice_state():
    engine = _fake_engine(voice="lola")

    [_ async for _ in engine.synthesize(_text_stream(["Hola! "]))]

    assert engine._model.calls == [("voice-state:lola", "Hola! ")]


async def test_synthesize_flushes_trailing_text_without_a_terminator():
    engine = _fake_engine()

    chunks = [c async for c in engine.synthesize(_text_stream(["nessun punto finale"]))]

    assert engine._model.calls == [("voice-state:giovanni", "nessun punto finale")]
    assert len(chunks) >= 1


def test_synthesize_sentence_converts_flat_tensor_to_pcm16():
    engine = _fake_engine()

    pcm = engine._synthesize_sentence("Ciao")

    # FakePocketModel yields a flat 1-D array of 10 samples at sr=24000 --
    # resampled down to DEVICE_SAMPLE_RATE (16000), so not exactly 10
    # samples/20 bytes, but must still be a valid, even-length PCM16
    # buffer.
    assert isinstance(pcm, bytes)
    assert len(pcm) % 2 == 0


def test_resolve_voice_maps_a_name_to_its_wav_in_the_voices_dir(tmp_path):
    (tmp_path / "fujiko.wav").write_bytes(b"RIFF")
    assert resolve_voice("fujiko", voices_dir=str(tmp_path)) == str(tmp_path / "fujiko.wav")


def test_resolve_voice_passes_presets_and_paths_through(tmp_path):
    assert resolve_voice("giovanni", voices_dir=str(tmp_path)) == "giovanni"
    assert resolve_voice("/some/where/else.wav", voices_dir=str(tmp_path)) == "/some/where/else.wav"

