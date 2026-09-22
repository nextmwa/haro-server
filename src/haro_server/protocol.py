import base64
import dataclasses
import json
from typing import Union


class ProtocolError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class HelloMessage:
    session_id: str


@dataclasses.dataclass(frozen=True)
class EndOfSpeechMessage:
    pass


@dataclasses.dataclass(frozen=True)
class InterruptMessage:
    # Sent by the firmware when the wake word fires during a long-running,
    # server-driven audio stream (currently: music playback -- see
    # session.py's handle_interrupt()) that isn't a normal short TTS reply
    # and needs to be told to stop rather than just having its audio
    # dropped client-side.
    pass


@dataclasses.dataclass(frozen=True)
class CameraFrameMessage:
    # A JPEG snapshot from the robot's camera, base64-decoded already. Two
    # consumers: admin.py's /admin/camera live-preview route, and
    # face_tracking.py's server-side detection (session.py runs it on
    # every frame and replies with encode_face_position() below -- face
    # detection itself is NOT done on the firmware, see face_tracking.py's
    # module docstring for why). Sent over the text channel (base64-encoded
    # JSON), not the binary channel, because binary messages from the
    # firmware already have a fixed meaning (raw mic audio --
    # Session.handle_audio_frame()) that this must not collide with.
    jpeg: bytes


ClientMessage = Union[HelloMessage, EndOfSpeechMessage, InterruptMessage, CameraFrameMessage]


def parse_client_message(text: str) -> ClientMessage:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {text!r}") from exc

    if not isinstance(data, dict):
        raise ProtocolError(f"expected a JSON object, got: {text!r}")

    msg_type = data.get("type")
    if msg_type == "hello":
        session_id = data.get("session_id")
        if not isinstance(session_id, str):
            raise ProtocolError(f"hello message missing session_id: {data!r}")
        return HelloMessage(session_id=session_id)
    if msg_type == "end_of_speech":
        return EndOfSpeechMessage()
    if msg_type == "interrupt":
        return InterruptMessage()
    if msg_type == "camera_frame":
        b64_data = data.get("data")
        if not isinstance(b64_data, str):
            raise ProtocolError(f"camera_frame message missing data: {data!r}")
        try:
            jpeg = base64.b64decode(b64_data, validate=True)
        except Exception as exc:
            raise ProtocolError(f"camera_frame message has invalid base64 data: {exc}") from exc
        return CameraFrameMessage(jpeg=jpeg)
    raise ProtocolError(f"unknown message type: {msg_type!r}")


def encode_emotion(value: str) -> str:
    return json.dumps({"type": "emotion", "value": value})


def encode_action(name: str, result: object) -> str:
    return json.dumps({"type": "action", "name": name, "result": result})


def encode_response_end() -> str:
    return json.dumps({"type": "response_end"})


def encode_error(message: str) -> str:
    return json.dumps({"type": "error", "message": message})


def encode_face_position(found: bool, dx: float = 0.0, dy: float = 0.0) -> str:
    # Reply to a CameraFrameMessage once face_tracking.py has run on it --
    # see the firmware's protocol.c PROTOCOL_EVENT_FACE_POSITION, which
    # feeds this straight into the same gaze/servo system the removed
    # on-device detection used to drive. dx/dy are ignored (and omitted
    # from the JSON) when found is False -- there is nothing meaningful to
    # send when no face was detected in that frame.
    if not found:
        return json.dumps({"type": "face_position", "found": False})
    return json.dumps({"type": "face_position", "found": True, "dx": dx, "dy": dy})
