"""Audio-format helpers shared by every TTS engine adapter (tts.py's
KokoroTtsEngine, chatterbox_tts.py's ChatterboxTtsEngine, and whatever
comes next) -- kept separate from any one engine so swapping which model
actually does the synthesis never touches this file. Every engine's output
has to end up in the exact same shape regardless of what it started as:
16-bit signed PCM, mono, at DEVICE_SAMPLE_RATE -- the ESP32 firmware's one
fixed speaker format (see DEVICE_SAMPLE_RATE's own comment).
"""
import re
from typing import AsyncIterator

import librosa
import numpy as np

# A sentence ends at . ! ? (or …) followed by whitespace, or at a line
# break. Line breaks matter for LLM replies formatted as lists despite the
# system prompt: found on real hardware (2026-09-24) that bullet items with
# no final punctuation all reached Pocket TTS as ONE long "sentence" (74
# tokens vs its 50-token chunk limit -- "generation may skip words"),
# spoken with no pauses between items and flat intonation.
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”»)]*\s|\n")

# Markdown and typography a voice can't speak: list markers, headings,
# emphasis, inline code, quotes, emoji and other pictographs.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•–—+]|\d{1,2}[.)])\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*")
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|`+)")
_QUOTES_RE = re.compile(r"[\"“”«»„]")
_PICTOGRAPH_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
_SPACES_RE = re.compile(r"\s+")

# The ESP32 firmware's speaker output is fixed at 16kHz: found on real
# hardware that its mic (RX) and speaker (TX) share one full-duplex I2S
# controller, and esp_codec_dev's own driver refuses to open them at two
# different sample rates ("conflict sample_rate" -- see
# managed_components/espressif__esp_codec_dev's audio_codec_data_i2s.c in
# the firmware repo), so the firmware side can't just match a given
# engine's native rate instead. Playing audio at the wrong rate through a
# 16kHz-configured DAC came out slowed down and garbled -- unintelligible,
# not just lower quality -- confirmed on real hardware with Kokoro's 24kHz
# output before resample_to_device_rate() existed.
DEVICE_SAMPLE_RATE = 16000


def find_sentence_boundary(text: str) -> int | None:
    match = _SENTENCE_END_RE.search(text)
    if match is None:
        return None
    return match.end()


async def speakable_sentences(text_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Splits a streamed LLM reply into sentences as soon as each one is
    complete, cleaned for speech (see clean_for_speech()); sentences with
    nothing speakable left are skipped. Shared by every TTS engine."""
    buffer = ""
    async for chunk in text_stream:
        buffer += chunk
        boundary = find_sentence_boundary(buffer)
        while boundary is not None:
            sentence, buffer = buffer[:boundary], buffer[boundary:]
            cleaned = clean_for_speech(sentence)
            if cleaned:
                yield cleaned
            boundary = find_sentence_boundary(buffer)
    cleaned = clean_for_speech(buffer)
    if cleaned:
        yield cleaned


def clean_for_speech(sentence: str) -> str:
    """Turns one sentence as split by find_sentence_boundary() into text a
    TTS engine can read naturally: formatting stripped, and a final "."
    added where the sentence has no closing punctuation (a list item, a
    heading, a line ending in ":"), so the voice pauses there instead of
    running straight into the next one. ? and ! are kept as-is -- they
    carry the intonation. Returns "" when nothing speakable is left.
    """
    text = _LIST_MARKER_RE.sub("", sentence)
    text = _HEADING_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    text = _QUOTES_RE.sub("", text)
    text = _PICTOGRAPH_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text).strip()
    text = text.rstrip(" :;,")
    if not text or not any(c.isalnum() for c in text):
        return ""
    if text[-1] not in ".!?…":
        text += "."
    return text


def to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def resample_to_device_rate(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == DEVICE_SAMPLE_RATE:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=DEVICE_SAMPLE_RATE)


def as_numpy(audio) -> np.ndarray:
    # Every engine used here yields audio as a torch.FloatTensor (Kokoro's
    # KPipeline.Result.audio, Chatterbox's generate() return value); each
    # engine's own tests instead use a fake that yields a plain numpy
    # array directly. Normalize either to a numpy array here rather than
    # in `to_pcm16`, so that function's tested behavior (clip + scale +
    # convert) stays exactly as specified against a fake.
    if isinstance(audio, np.ndarray):
        return audio
    return audio.detach().cpu().numpy()
