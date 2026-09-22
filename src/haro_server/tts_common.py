"""Audio-format helpers shared by every TTS engine adapter (tts.py's
KokoroTtsEngine, chatterbox_tts.py's ChatterboxTtsEngine, and whatever
comes next) -- kept separate from any one engine so swapping which model
actually does the synthesis never touches this file. Every engine's output
has to end up in the exact same shape regardless of what it started as:
16-bit signed PCM, mono, at DEVICE_SAMPLE_RATE -- the ESP32 firmware's one
fixed speaker format (see DEVICE_SAMPLE_RATE's own comment).
"""
import re

import librosa
import numpy as np

_SENTENCE_END_RE = re.compile(r"[.!?]\s")

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
