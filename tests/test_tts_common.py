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


from haro_server.tts_common import clean_for_speech  # noqa: E402


def test_a_newline_ends_a_sentence_so_list_items_are_spoken_separately():
    text = "Ti butto qualche idea:\n- Serata film\n- Uscita tranquilla\n"
    first = find_sentence_boundary(text)
    assert text[:first].strip() == "Ti butto qualche idea:"


def test_clean_for_speech_strips_list_markers_and_adds_a_closing_pause():
    assert clean_for_speech("- Serata film o serie con snack e copertina\n") == "Serata film o serie con snack e copertina."
    assert clean_for_speech("* Hobby: libro, videogiochi\n") == "Hobby: libro, videogiochi."
    assert clean_for_speech("• Relax totale") == "Relax totale."
    assert clean_for_speech("2. Uscita tranquilla") == "Uscita tranquilla."


def test_clean_for_speech_keeps_question_and_exclamation_marks():
    assert clean_for_speech("Sei più da casa o da uscire? ") == "Sei più da casa o da uscire?"
    assert clean_for_speech("Dipende da che umore hai stasera! ") == "Dipende da che umore hai stasera!"


def test_clean_for_speech_turns_a_trailing_colon_into_a_pause():
    assert clean_for_speech("Ti butto qualche idea veloce, poi mi dici cosa ti ispira:\n") == (
        "Ti butto qualche idea veloce, poi mi dici cosa ti ispira."
    )


def test_clean_for_speech_removes_markdown_quotes_and_emoji():
    assert clean_for_speech("**Serata “cucina”**: provi una ricetta 🍕\n") == "Serata cucina: provi una ricetta."
    assert clean_for_speech("## Consigli\n") == "Consigli."
    assert clean_for_speech("usa `ls` nel terminale.") == "usa ls nel terminale."


def test_clean_for_speech_of_a_bare_marker_is_empty():
    assert clean_for_speech("- \n") == ""
    assert clean_for_speech("   ") == ""
