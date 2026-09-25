"""A gentle wake-up sound, generated rather than shipped as a file: soft
bell/marimba-like tones on a pentatonic scale (no two notes ever clash),
each with a slow attack and a long natural decay, at an unhurried random
pace -- closer to wind chimes than to a siren. Volume starts almost
inaudible and rises gradually, so it wakes rather than startles.

Output is the robot's playback format: 16kHz mono PCM16.
"""
import numpy as np

from .tts_common import DEVICE_SAMPLE_RATE, to_pcm16

# C major pentatonic, C5..E6 -- bright but soft on a small speaker.
_NOTES_HZ = [523.25, 587.33, 659.25, 783.99, 880.00, 1046.50, 1174.66, 1318.51]


def _tone(freq: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * DEVICE_SAMPLE_RATE)) / DEVICE_SAMPLE_RATE
    # Fundamental plus a quiet, fast-decaying upper partial for a bell-like
    # shimmer; 20ms attack (no click), then a long exponential decay.
    body = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.9)
    shimmer = 0.25 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-t / 0.25)
    attack = np.minimum(1.0, t / 0.02)
    return (body + shimmer) * attack


def gentle_chimes(seconds: float, start_gain: float, end_gain: float, seed: int = 0) -> bytes:
    """`seconds` of chimes whose volume ramps from start_gain to end_gain
    (0..1 of full scale). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    total = int(seconds * DEVICE_SAMPLE_RATE)
    out = np.zeros(total + 3 * DEVICE_SAMPLE_RATE, dtype=np.float64)
    position = 0.0
    while position < seconds:
        tone = _tone(float(rng.choice(_NOTES_HZ)), 3.0) * rng.uniform(0.6, 1.0)
        start = int(position * DEVICE_SAMPLE_RATE)
        out[start:start + len(tone)] += tone
        position += rng.uniform(0.45, 1.1)
    out = out[:total]
    peak = np.max(np.abs(out)) or 1.0
    ramp = np.linspace(start_gain, end_gain, total)
    return to_pcm16(out / peak * ramp)
