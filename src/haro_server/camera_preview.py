"""Holds the most recent camera snapshot the robot has sent (see
protocol.py's CameraFrameMessage), for admin.py's /admin/camera live-
preview page. A single in-process value, not per-session: this is a
one-robot device, and a debug/monitoring feature has no reason to carry
the complexity of a per-session store.

Plain module-level state, no lock: the WebSocket handler (server.py) is the
only writer, admin.py's HTTP route is the only reader, and a `bytes`
assignment/read is atomic enough under the GIL for "occasionally show a
slightly stale preview frame" to be a non-issue -- this is a monitoring aid,
not something correctness depends on.
"""

_latest_jpeg: bytes | None = None
# (x1, y1, x2, y2) pixel coords of the detected face in _latest_jpeg's own
# frame, or None if no face was found in that particular frame -- always
# set together with _latest_jpeg (see set_latest_frame()) so the two never
# drift out of sync with each other.
_latest_face_box: tuple[int, int, int, int] | None = None


def set_latest_frame(jpeg: bytes, face_box: tuple[int, int, int, int] | None = None) -> None:
    global _latest_jpeg, _latest_face_box
    _latest_jpeg = jpeg
    _latest_face_box = face_box


def get_latest_frame() -> bytes | None:
    return _latest_jpeg


def get_latest_face_box() -> tuple[int, int, int, int] | None:
    return _latest_face_box
