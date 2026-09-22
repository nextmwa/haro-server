"""Server-side face detection on JPEG frames the robot's camera sends
(protocol.py's CameraFrameMessage). Moved here from the ESP32 firmware:
real-hardware testing found the on-device esp-dl models either missed a
clearly visible face for many consecutive seconds (the default 2-stage
MSRMNP cascade -- its first, region-proposal stage is a recall bottleneck)
or crashed the board with ESP_ERR_NO_MEM alongside the rest of its workload
(the more accurate single-stage ESPDet model). The server has neither
problem: no memory ceiling, and CPU to spare between voice turns.

Haar cascade (OpenCV's classic, bundled-with-the-wheel detector), not a
DNN-based one -- simplest option that needs no separate model download,
and fast enough on CPU for a frame every ~500ms. If its accuracy ever
proves insufficient, OpenCV's DNN face detector (res10_300x300 SSD) is the
natural upgrade, at the cost of bundling ~10MB of model files.
"""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@dataclass(frozen=True)
class FaceResult:
    # Normalized to [-1, 1], 0 = frame center -- same convention the
    # removed on-device camera_face_track.cpp used, so main.c's existing
    # gaze/servo consumer needs no changes to how it interprets these.
    dx: float
    dy: float
    # Pixel coords in the source frame (top-left/bottom-right), for
    # admin.py's /admin/camera detection-box overlay.
    box: tuple[int, int, int, int]


def detect_face(jpeg: bytes) -> FaceResult | None:
    """Never raises: a corrupt/undecodable frame (e.g. one truncated by a
    dropped WebSocket fragment) just yields "no face" rather than crashing
    the turn that happens to arrive around the same time.
    """
    try:
        array = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        height, width = image.shape[:2]
        faces = _cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            return None

        # Largest face wins (closest/most prominent), matching the removed
        # on-device logic's own "largest box" tie-breaker.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + w / 2.0, y + h / 2.0
        # Horizontal sign flipped (mirror effect): the camera faces the user
        # head-on, so an un-flipped dx (face right of frame center -> +dx)
        # makes the robot's eyes/servo turn the opposite way from how the
        # person actually moved -- confirmed backwards on real hardware.
        # Flipping dx makes the robot track like a mirror instead (step to
        # your right, the robot's gaze follows to that same side, the way
        # your reflection would), which is what people intuitively expect
        # from anything that looks back at them. No flip needed vertically:
        # a mirror only reverses the left-right axis.
        dx = -(cx - width / 2.0) / (width / 2.0)
        dy = (cy - height / 2.0) / (height / 2.0)
        return FaceResult(dx=dx, dy=dy, box=(int(x), int(y), int(x + w), int(y + h)))
    except Exception:
        logger.exception("face detection failed, treating as no face")
        return None
