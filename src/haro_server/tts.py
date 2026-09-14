import asyncio
import re
from typing import AsyncIterator

import librosa
import numpy as np

_SENTENCE_END_RE = re.compile(r"[.!?]\s")

# Kokoro-82M's KPipeline always synthesizes at 24kHz (its fixed native
# output rate -- not configurable). The ESP32 firmware's speaker output is
# fixed at 16kHz: found on real hardware that its mic (RX) and speaker (TX)
# share one full-duplex I2S controller, and esp_codec_dev's own driver
# refuses to open them at two different sample rates ("conflict
# sample_rate" -- see managed_components/espressif__esp_codec_dev's
# audio_codec_data_i2s.c in the firmware repo), so the firmware side can't
# just match Kokoro's rate instead. Playing 24kHz audio through a
# 16kHz-configured DAC came out slowed down and garbled -- unintelligible,
# not just lower quality. Resample here instead.
_KOKORO_SAMPLE_RATE = 24000
_DEVICE_SAMPLE_RATE = 16000


def _resample_to_device_rate(audio: np.ndarray) -> np.ndarray:
    if _KOKORO_SAMPLE_RATE == _DEVICE_SAMPLE_RATE:
        return audio
    return librosa.resample(audio, orig_sr=_KOKORO_SAMPLE_RATE, target_sr=_DEVICE_SAMPLE_RATE)


def _find_sentence_boundary(text: str) -> int | None:
    match = _SENTENCE_END_RE.search(text)
    if match is None:
        return None
    return match.end()


def _to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _as_numpy(audio) -> np.ndarray:
    # The real KPipeline yields `KPipeline.Result.audio` as a
    # torch.FloatTensor (see this module's KokoroTtsEngine docstring for
    # the version this was verified against); FakePipeline in the tests
    # yields plain numpy arrays. Normalize either to a numpy array here
    # rather than in `_to_pcm16`, so that function's tested behavior
    # (clip + scale + convert) stays exactly as specified against the fake.
    if isinstance(audio, np.ndarray):
        return audio
    return audio.detach().cpu().numpy()


class KokoroTtsEngine:
    def __init__(self, voice: str = "if_sara", device: str = "cpu") -> None:
        # Verified against `kokoro==0.9.4`, pinned in pyproject.toml (see
        # its comment for why that version, not the plan's original
        # `kokoro>=1.0` -- which does not exist on PyPI -- see this task's
        # confidence note). `KPipeline.__init__(lang_code, repo_id=None,
        # model=True, trf=False, en_callable=None, device=None)` still
        # accepts `lang_code` and `device` as keywords exactly as below;
        # 0.9.4 added the optional `repo_id` param, passed explicitly here
        # to pin the model repo and suppress its "defaulting repo_id"
        # warning. "if_sara" is a real Italian ("i") female voice for this
        # model. Italian G2P goes through espeak-ng (`EspeakG2P`), so the
        # `espeak-ng` system package must be installed wherever this runs
        # (see Task 9's Dockerfile).
        from kokoro import KPipeline

        self._pipeline = KPipeline(
            lang_code="i", repo_id="hexgrad/Kokoro-82M", device=device
        )
        self._voice = voice

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        buffer = ""
        async for chunk in text_stream:
            buffer += chunk
            boundary = _find_sentence_boundary(buffer)
            while boundary is not None:
                sentence, buffer = buffer[:boundary], buffer[boundary:]
                for pcm_chunk in await asyncio.to_thread(
                    self._synthesize_sentence, sentence
                ):
                    yield pcm_chunk
                boundary = _find_sentence_boundary(buffer)
        if buffer.strip():
            for pcm_chunk in await asyncio.to_thread(self._synthesize_sentence, buffer):
                yield pcm_chunk

    def _synthesize_sentence(self, sentence: str) -> list[bytes]:
        # Blocking, CPU-bound model inference (2-3s per sentence on CPU per
        # the spec). It MUST run off the event loop -- calling it inline
        # would freeze every other connection on the server, not just this
        # turn -- hence `asyncio.to_thread` at both call sites above. The
        # whole sentence is synthesized into a list inside the worker
        # thread, rather than yielding lazily, because the real KPipeline
        # returns a synchronous generator that cannot be iterated
        # incrementally from the event loop without blocking it again.
        return [
            _to_pcm16(_resample_to_device_rate(_as_numpy(audio)))
            for _, _, audio in self._pipeline(sentence, voice=self._voice)
        ]
