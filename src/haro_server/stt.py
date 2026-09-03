import asyncio

import numpy as np


def _bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    # np.frombuffer raises ValueError if the buffer length isn't a whole
    # number of int16 samples, which a truncated/odd-sized frame from the
    # robot can produce. Drop the dangling byte instead of failing the turn.
    truncated = pcm_bytes[: len(pcm_bytes) & ~1]
    return np.frombuffer(truncated, dtype=np.int16).astype(np.float32) / 32768.0


def load_parakeet_model(device: str = "cpu"):
    """Loads the (large, slow-to-load) Parakeet model once. Call this at
    server startup and share the returned model across every connection's
    ParakeetSttEngine -- constructing it per-connection would reload
    hundreds of MB of weights on every single robot (re)connection.
    """
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    return model.to(device)


class ParakeetSttEngine:
    """Cheap, per-connection wrapper around a shared, pre-loaded model.

    Each connection must get its OWN ParakeetSttEngine instance (its
    `_buffer` is per-turn state that must not be shared across concurrent
    robot connections), but all instances should share the SAME `model`
    (loaded once via `load_parakeet_model`) so the expensive weights are
    only ever loaded a single time for the whole server process.

    This is a batch (not streaming) implementation: `feed()` just buffers
    raw PCM16 audio bytes in memory, and `finalize()` runs one batch
    `transcribe()` call over the whole buffered utterance. NeMo's true
    incremental/cache-aware streaming API is significantly more complex and
    was deliberately deferred -- see the task-7 brief's confidence note.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._buffer = bytearray()

    def feed(self, frame: bytes) -> None:
        self._buffer.extend(frame)

    async def finalize(self) -> str:
        if not self._buffer:
            return ""
        audio = _bytes_to_float32(bytes(self._buffer))
        # Reset BEFORE transcribing: if transcribe() raises, this turn's
        # audio must not leak into the next turn's buffer.
        self._buffer = bytearray()
        # transcribe() is blocking, CPU-bound model inference; running it
        # directly on the event loop would freeze every connection on the
        # server for its whole duration.
        result = await asyncio.to_thread(self._model.transcribe, [audio])
        return result[0].text if result else ""
