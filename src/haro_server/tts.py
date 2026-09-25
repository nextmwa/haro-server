import asyncio
from typing import AsyncIterator

from .tts_common import (
    as_numpy,
    speakable_sentences,
    resample_to_device_rate,
    to_pcm16,
)

# Kokoro-82M's KPipeline always synthesizes at 24kHz (its fixed native
# output rate -- not configurable). See tts_common.DEVICE_SAMPLE_RATE's
# comment for why this needs resampling down rather than the firmware
# matching Kokoro's rate instead.
_KOKORO_SAMPLE_RATE = 24000


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
        async for sentence in speakable_sentences(text_stream):
            for pcm_chunk in await asyncio.to_thread(self._synthesize_sentence, sentence):
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
            to_pcm16(resample_to_device_rate(as_numpy(audio), _KOKORO_SAMPLE_RATE))
            for _, _, audio in self._pipeline(sentence, voice=self._voice)
        ]
