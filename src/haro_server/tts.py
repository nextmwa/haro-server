import re
from typing import AsyncIterator

import numpy as np

_SENTENCE_END_RE = re.compile(r"[.!?]\s")


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
        # Verified against the real, installed `kokoro==0.7.4` package (see
        # pyproject.toml for why it's pinned to exactly that version --
        # there is no 1.0 release; the plan's original `kokoro>=1.0` pin
        # does not exist on PyPI, see this task's confidence note).
        # `KPipeline.__init__(lang_code, model=True, trf=False, device=None)`
        # accepts `device` as a keyword exactly as below. "if_sara" is a
        # real Italian ("i") female voice for this model. Italian G2P goes
        # through espeak-ng (`EspeakG2P`), so the `espeak-ng` system package
        # must be installed wherever this runs (see Task 9's Dockerfile).
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code="i", device=device)
        self._voice = voice

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        buffer = ""
        async for chunk in text_stream:
            buffer += chunk
            boundary = _find_sentence_boundary(buffer)
            while boundary is not None:
                sentence, buffer = buffer[:boundary], buffer[boundary:]
                for _, _, audio in self._pipeline(sentence, voice=self._voice):
                    yield _to_pcm16(_as_numpy(audio))
                boundary = _find_sentence_boundary(buffer)
        if buffer.strip():
            for _, _, audio in self._pipeline(buffer, voice=self._voice):
                yield _to_pcm16(_as_numpy(audio))
