import asyncio
import os
from typing import AsyncIterator

import numpy as np
import soxr

from .tts_common import DEVICE_SAMPLE_RATE, as_numpy, find_sentence_boundary, resample_to_device_rate, to_pcm16

# Streaming output sizing (see PocketTtsEngine.synthesize()). While the
# robot has less than this much audio queued ahead of playback, every
# ~80ms frame is sent as soon as it's ready -- measured on the real model
# (2026-09-24, ~1.1x real time in Docker): holding frames back to build 1s
# sends left the robot silent for ~0.9s right after its first word, and
# again at every sentence start. Once it's this far ahead, frames are
# coalesced into sends of at least this long instead: the firmware queues
# at most 32 server events and drops audio beyond that (main.c's
# s_server_queue), so 80ms sends from a generator well above real time
# (e.g. native macOS, ~3.3x) would overflow it; 1s sends give it 30s+.
STREAM_COALESCE_SECONDS = 1.0


# Where custom cloned voices live, relative to the server's working
# directory: server/voices/ when run natively (haroctl), and /app/voices in
# Docker (WORKDIR /app, bind-mounted by docker-compose.yml). A voice named
# "fujiko" is the file voices/fujiko.wav -- see resolve_voice().
VOICES_DIR = "voices"


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

    That measurement only holds inside Docker's Linux VM. Natively on
    macOS (Apple Silicon, 2026-09-24 benchmark, same sentences and voice)
    it's the other way round: fp32 ran at 3.3x real time vs 1.3x for int8
    (Docker: int8 1.1x, fp32 0.5x) -- hence POCKETTTS_QUANTIZE, false for
    the native server and forced true in docker-compose.yml.
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
        # Pacing across the whole reply, not per sentence: how much audio
        # has been sent vs. how much the robot has had time to play since
        # the first send (see STREAM_COALESCE_SECONDS).
        loop = asyncio.get_running_loop()
        first_send_at: float | None = None
        sent_seconds = 0.0
        pending: list[np.ndarray] = []
        pending_samples = 0
        coalesce_samples = int(STREAM_COALESCE_SECONDS * DEVICE_SAMPLE_RATE)

        def take_pending() -> bytes:
            nonlocal first_send_at, sent_seconds, pending, pending_samples
            if first_send_at is None:
                first_send_at = loop.time()
            sent_seconds += pending_samples / DEVICE_SAMPLE_RATE
            audio = to_pcm16(np.concatenate(pending))
            pending, pending_samples = [], 0
            return audio

        async def frames_for_text() -> AsyncIterator[np.ndarray]:
            buffer = ""
            async for chunk in text_stream:
                buffer += chunk
                boundary = find_sentence_boundary(buffer)
                while boundary is not None:
                    sentence, buffer = buffer[:boundary], buffer[boundary:]
                    async for frame in self._stream_sentence(sentence):
                        yield frame
                    boundary = find_sentence_boundary(buffer)
            if buffer.strip():
                async for frame in self._stream_sentence(buffer):
                    yield frame

        async for frame in frames_for_text():
            pending.append(frame)
            pending_samples += frame.size
            played = 0.0 if first_send_at is None else loop.time() - first_send_at
            ahead = sent_seconds - played
            if ahead < STREAM_COALESCE_SECONDS or pending_samples >= coalesce_samples:
                yield take_pending()
        if pending:
            yield take_pending()

    async def _stream_sentence(self, sentence: str) -> AsyncIterator[np.ndarray]:
        # Measured on the real model (2026-09-24): whole-sentence
        # generate_audio() made the robot wait 2-5s per sentence before
        # hearing anything, while generate_audio_stream()'s first frame is
        # ready in ~0.2-0.4s. Frames are ~80ms each at the model's 24kHz;
        # a stateful soxr stream resamples them to DEVICE_SAMPLE_RATE, since
        # resampling each frame on its own would click at every boundary.
        # The generator is advanced one frame per to_thread() call -- same
        # off-the-event-loop reasoning as _synthesize_sentence -- never
        # concurrently.
        frames = self._model.generate_audio_stream(self._voice_state, sentence)
        resampler = soxr.ResampleStream(self._model.sample_rate, DEVICE_SAMPLE_RATE, 1, dtype="float32")
        while True:
            frame = await asyncio.to_thread(next, frames, None)
            last = frame is None
            samples = np.zeros(0, dtype=np.float32) if last else as_numpy(frame).astype(np.float32)
            out = resampler.resample_chunk(samples, last=last)
            if out.size:
                yield out
            if last:
                return

    def _synthesize_sentence(self, sentence: str) -> bytes:
        # Blocking, CPU-bound model inference -- run off the event loop the
        # same way and for the same reason as the other engines'
        # _synthesize_sentence. generate_audio() returns a flat 1-D
        # tensor already (unlike Chatterbox's (1, num_samples)), so no
        # squeeze() is needed here.
        audio = self._model.generate_audio(self._voice_state, sentence)
        samples = as_numpy(audio)
        return to_pcm16(resample_to_device_rate(samples, self._model.sample_rate))
