import asyncio
import os
from typing import AsyncIterator

from .tts_common import as_numpy, find_sentence_boundary, resample_to_device_rate, to_pcm16


# Where custom cloned voices live inside the container (bind-mounted from
# server/voices/ by docker-compose.yml). A voice named "fujiko" is the file
# voices/fujiko.wav -- see resolve_voice().
VOICES_DIR = "/app/voices"


def resolve_voice(voice: str, voices_dir: str = VOICES_DIR) -> str:
    """Map POCKETTTS_VOICE to what get_state_for_audio_prompt() accepts.

    A bare name with a matching <voices_dir>/<name>.wav is a custom cloned
    voice and resolves to that file's path; anything else (a built-in preset
    like "giovanni", an explicit path, a URL) passes through unchanged.
    """
    if "/" not in voice:
        candidate = os.path.join(voices_dir, f"{voice}.wav")
        if os.path.isfile(candidate):
            return candidate
    return voice


class PocketTtsEngine:
    """Kyutai's Pocket TTS (kyutai-labs/pocket-tts, PyPI `pocket-tts`) --
    tried as a faster alternative to Chatterbox Multilingual
    (chatterbox_tts.py) for Italian: Chatterbox's Italian output is
    accurate but slow on CPU (~16s to synthesize one short sentence,
    measured against the real installed package). Pocket TTS is built
    CPU-first (its own docs: "a TTS that fits in your CPU", ~6x real-time
    on 2 CPU cores) and ships a built-in Italian voice/language config, so
    it's the latency-focused option from the same shortlist.

    Deliberately its own file/class, implementing the exact same
    `synthesize(text_stream) -> AsyncIterator[bytes]` shape as
    KokoroTtsEngine and ChatterboxTtsEngine (session.py's TtsEngineLike
    protocol) -- server.py picks between engines by config
    (Config.tts_engine), so switching, or adding a fourth engine later, is
    a one-line config change, never a rewrite of session.py or this file.

    language defaults to "italian_24l" (the higher-quality 24-layer
    variant -- Pocket TTS also ships a lighter/faster 12-layer "italian")
    and quantize defaults to True: measured against the real installed
    package, italian_24l unquantized took ~11s to synthesize one short
    sentence (worse than Chatterbox), but int8 quantization cut that to
    ~2-3s for the first call and then ~1x realtime (e.g. 1.8s to
    synthesize a 1.9s sentence) for every call after -- still the
    higher-quality voice, just far faster. See __init__'s warm-up call for
    why "every call after" excludes the very first one.
    """

    def __init__(self, language: str = "italian_24l", voice: str = "giovanni", quantize: bool = True) -> None:
        # Import deferred to construction time, same reasoning as
        # ChatterboxTtsEngine's own deferred import -- keeps this module
        # importable (e.g. for tests, via __new__) even where the real
        # (large) pocket-tts package isn't installed.
        from pocket_tts import TTSModel

        self._model = TTSModel.load_model(language=language, quantize=quantize)
        # Unlike Chatterbox's generate(text, language_id=...), Pocket TTS
        # separates *language* (picked at model-load time above, since
        # it's a whole model variant) from *voice* (a cloned-from-audio
        # timbre, prepared once here and reused for every sentence rather
        # than re-derived per call). "giovanni" is Pocket TTS's own
        # built-in Italian voice preset -- see its docs' voice list.
        self._voice_state = self._model.get_state_for_audio_prompt(resolve_voice(voice))
        # Measured on the real installed package: the first
        # generate_audio() call after loading a quantized model pays a
        # one-off ~8s cost (kernel/cache warm-up), then every later call
        # settles to ~1x realtime. Paying that cost here, once, at server
        # startup (server.py builds this engine before logging "models
        # loaded, ready to accept connections") means Haro's very first
        # real reply of the day isn't the slow one.
        self._synthesize_sentence("Pronto.")

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
        # same way and for the same reason as the other engines'
        # _synthesize_sentence. generate_audio() returns a flat 1-D
        # tensor already (unlike Chatterbox's (1, num_samples)), so no
        # squeeze() is needed here.
        audio = self._model.generate_audio(self._voice_state, sentence)
        samples = as_numpy(audio)
        return to_pcm16(resample_to_device_rate(samples, self._model.sample_rate))
