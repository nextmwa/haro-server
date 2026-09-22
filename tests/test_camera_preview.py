from haro_server import camera_preview


def test_get_latest_frame_returns_none_before_anything_set():
    camera_preview._latest_jpeg = None
    camera_preview._latest_face_box = None

    assert camera_preview.get_latest_frame() is None
    assert camera_preview.get_latest_face_box() is None


def test_set_and_get_latest_frame_roundtrip():
    camera_preview.set_latest_frame(b"\xff\xd8\xff fake jpeg")

    assert camera_preview.get_latest_frame() == b"\xff\xd8\xff fake jpeg"
    assert camera_preview.get_latest_face_box() is None


def test_set_latest_frame_overwrites_the_previous_one():
    camera_preview.set_latest_frame(b"first")
    camera_preview.set_latest_frame(b"second")

    assert camera_preview.get_latest_frame() == b"second"


def test_set_latest_frame_with_face_box():
    camera_preview.set_latest_frame(b"jpeg bytes", (10, 20, 110, 140))

    assert camera_preview.get_latest_frame() == b"jpeg bytes"
    assert camera_preview.get_latest_face_box() == (10, 20, 110, 140)


def test_set_latest_frame_without_face_box_clears_a_previous_one():
    camera_preview.set_latest_frame(b"frame with face", (10, 20, 110, 140))
    camera_preview.set_latest_frame(b"frame without face")

    assert camera_preview.get_latest_face_box() is None
