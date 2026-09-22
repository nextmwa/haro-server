from haro_server import music


def test_is_music_request_matches_metti():
    assert music.is_music_request("metti della musica di Vasco Rossi")


def test_is_music_request_matches_riproduci():
    assert music.is_music_request("riproduci qualcosa di allegro")


def test_is_music_request_matches_suona():
    assert music.is_music_request("suona una canzone")


def test_is_music_request_matches_fai_sentire():
    assert music.is_music_request("fammi sentire qualcosa di Vasco")


def test_is_music_request_returns_false_for_unrelated_speech():
    assert not music.is_music_request("che tempo fa oggi")


def test_is_music_request_returns_false_for_dice_action():
    assert not music.is_music_request("lancia un dado")
