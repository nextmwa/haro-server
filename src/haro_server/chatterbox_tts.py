import asyncio
from typing import AsyncIterator

from .tts_common import as_numpy, find_sentence_boundary, resample_to_device_rate, to_pcm16


class ChatterboxTtsEngine:
    """Resemble AI's Chatterbox Multilingual (github.com/resemble-ai/chatterbox,
    PyPI `chatterbox-tts`) -- tried as a more natural-sounding alternative
    to Kokoro (tts.py) for Italian specifically: Kokoro is Apache-2.0/very
    lightweight but trained mostly on English, and came across as accented
    on Italian text in real use. Chatterbox Multilingual covers 23
    languages including Italian natively.

    Deliberately its own file/class, implementing the exact same
    `synthesize(text_stream) -> AsyncIterator[bytes]` shape as
    KokoroTtsEngine (session.py's TtsEngineLike protocol) -- server.py
    picks between the two by config (Config.tts_engine), so switching back
    to Kokoro, or adding a third engine later, is a one-line config change,
    never a rewrite of session.py or this file.
    """

    def __init__(self, language: str = "it", device: str = "cpu") -> None:
        # Import deferred to construction time, same reasoning as
        # KokoroTtsEngine's own `from kokoro import KPipeline` -- keeps
        # this module importable (e.g. for tests, via __new__) even where
        # the real (large) chatterbox-tts package isn't installed.
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self._language = language

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        buffer = ""
        async for chunk in text_stream:
            buffer += chunk
            boundary = find_sentence_boundary(buffer)
            while boundary is not None:
                sentence, buffer = buffer[:boundary], buffer[boundary:]
                yield await asyncio.to_thread(self._synthesize_sentence, sentence)
                boundary = find_sentence_boundary(buffer)
        if buffer.strip():
            yield await asyncio.to_thread(self._synthesize_sentence, buffer)

    def _synthesize_sentence(self, sentence: str) -> bytes:
        # Blocking, CPU-bound model inference -- run off the event loop the
        # same way and for the same reason as KokoroTtsEngine._synthesize_sentence
        # (an inline call here would freeze every other connection on the
        # server, not just this turn). Unlike Kokoro's KPipeline, which
        # yields one (graphemes, phonemes, audio) tuple per internally-split
        # clause, generate() returns the whole sentence as a single tensor
        # -- one PCM chunk per call, not a list.
        audio = self._model.generate(text=sentence, language_id=self._language)
        # generate() returns shape (1, num_samples) (a batch/channel
        # dimension of 1, not a flat 1-D waveform like Kokoro's tuples) --
        # squeeze() drops that dimension rather than letting it flow into
        # to_pcm16()'s tobytes() and silently produce the wrong byte
        # layout. Harmless no-op if a future version already returns 1-D.
        samples = as_numpy(audio).squeeze()
        return to_pcm16(resample_to_device_rate(samples, self._model.sr))
