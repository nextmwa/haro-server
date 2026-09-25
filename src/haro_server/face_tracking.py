"""Server-side face detection on JPEG frames the robot's camera sends
(protocol.py's CameraFrameMessage). Moved here from the ESP32 firmware:
real-hardware testing found the on-device esp-dl models either missed a
clearly visible face for many consecutive seconds (the default 2-stage
MSRMNP cascade -- its first, region-proposal stage is a recall bottleneck)
or crashed the board with ESP_ERR_NO_MEM alongside the rest of its workload
(the more accurate single-stage ESPDet model). The server has neither
problem: no memory ceiling, and CPU to spare between voice turns.

Detector: YuNet (OpenCV Zoo's face_detection_yunet_2023mar, its current
release), a small CNN run through cv2.FaceDetectorYN, with the ~230KB
.onnx model shipped in models/. Replaced the original Haar cascade on
2026-09-24: measured on 9 real frames from the robot, Haar found the face
3/3 but also produced 13 false positives on a whiteboard's handwriting --
bigger than the (distant) real face, so "largest face wins" put the box
on the whiteboard. YuNet on the same frames: 3/3, zero false positives,
nothing when the face was covered by a hand.
"""
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
# YuNet's own defaults from OpenCV Zoo's demo: a candidate needs 0.8+
# confidence; 0.3 NMS IoU merges overlapping boxes for the same face.
_SCORE_THRESHOLD = 0.8
_NMS_THRESHOLD = 0.3
# Faces smaller than this (px) are ignored -- the same minimum size the
# Haar detector used, so someone far across the room doesn't steal focus.
_MIN_FACE_SIZE = 40

# One detector instance, reused (building it loads the model); its input
# size is per-frame state, so detection is serialized with a lock --
# detect_face() runs in asyncio.to_thread() worker threads.
_detector = cv2.FaceDetectorYN.create(str(_MODEL_PATH), "", (320, 240), _SCORE_THRESHOLD, _NMS_THRESHOLD, 5000)
_detector_lock = threading.Lock()


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
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            return None

        height, width = image.shape[:2]
        with _detector_lock:
            _detector.setInputSize((width, height))
            _, detections = _detector.detect(image)
        if detections is None:
            return None
        # Each row: x, y, w, h, 5 landmark (x, y) pairs, score.
        faces = [
            (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
            for d in detections
            if d[2] >= _MIN_FACE_SIZE and d[3] >= _MIN_FACE_SIZE
        ]
        if not faces:
            return None

        # Largest face wins (closest/most prominent), matching the removed
        # on-device logic's own "largest box" tie-breaker.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # YuNet boxes can extend slightly past the frame edge -- clip them.
        x2, y2 = min(x + w, float(width)), min(y + h, float(height))
        x, y = max(0.0, x), max(0.0, y)
        w, h = x2 - x, y2 - y
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
