import cv2
import numpy as np

from haro_server import face_tracking


def _blank_jpeg(width: int = 320, height: int = 240) -> bytes:
    image = np.zeros((height, width), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_detect_face_returns_none_for_a_blank_frame():
    # A blank frame has no face by construction -- this asserts the
    # "nothing found" path works, not detection accuracy on a real face
    # (no appropriately-licensed test photo is checked into this repo --
    # see this module's own test-scope note below).
    assert face_tracking.detect_face(_blank_jpeg()) is None


def test_detect_face_returns_none_for_corrupt_data_instead_of_raising():
    assert face_tracking.detect_face(b"not a jpeg at all") is None


def test_detect_face_returns_none_for_empty_bytes():
    assert face_tracking.detect_face(b"") is None


# Deliberately not tested here: actually detecting a real face. Haar
# cascades need a real photographic face to exercise meaningfully (not a
# synthetic shape), and no appropriately-licensed test photo was available
# to check into this repo -- this was instead verified against real
# frames from the robot's own camera during development (see the
# /admin/camera live preview). If a suitable public-domain face photo
# becomes available, add it under tests/fixtures/ and a
# test_detect_face_finds_a_real_face() case here.
