import numpy as np

from haro_server.tts_common import (
    DEVICE_SAMPLE_RATE,
    find_sentence_boundary,
    resample_to_device_rate,
    to_pcm16,
)


def test_find_sentence_boundary_finds_end_of_first_sentence():
    assert find_sentence_boundary("Ciao! Come stai?") == len("Ciao! ")


def test_find_sentence_boundary_returns_none_without_terminator():
    assert find_sentence_boundary("nessun punto qui") is None


def test_to_pcm16_converts_float_audio_to_16_bit_bytes():
    audio = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    pcm = to_pcm16(audio)
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples[0] == 0
    assert samples[1] == int(0.5 * 32767)
    # Out-of-range values are clipped, not wrapped.
    assert samples[3] == 32767
    assert samples[4] == -32767


def test_resample_to_device_rate_shortens_audio_to_the_device_rate():
    # 1 second of audio at some engine's native rate must come out as ~1
    # second at the device's rate (a couple of samples of slack for the
    # resampling filter's edge behavior, not an exact count). 24000 here is
    # arbitrary -- any rate other than DEVICE_SAMPLE_RATE exercises the
    # actual resample path, this isn't specific to any one engine.
    one_second = np.zeros(24000, dtype=np.float32)
    resampled = resample_to_device_rate(one_second, orig_sr=24000)
    assert abs(resampled.shape[0] - DEVICE_SAMPLE_RATE) < 10


def test_resample_to_device_rate_is_a_no_op_when_already_at_device_rate():
    audio = np.zeros(DEVICE_SAMPLE_RATE, dtype=np.float32)
    resampled = resample_to_device_rate(audio, orig_sr=DEVICE_SAMPLE_RATE)
    assert resampled is audio
