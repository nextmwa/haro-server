import json

import pytest

from haro_server import protocol


def test_parse_hello_message():
    msg = protocol.parse_client_message('{"type": "hello", "session_id": "abc123"}')
    assert msg == protocol.HelloMessage(session_id="abc123")


def test_parse_end_of_speech_message():
    msg = protocol.parse_client_message('{"type": "end_of_speech"}')
    assert msg == protocol.EndOfSpeechMessage()


def test_parse_interrupt_message():
    msg = protocol.parse_client_message('{"type": "interrupt"}')
    assert msg == protocol.InterruptMessage()


def test_parse_camera_frame_message_decodes_base64():
    import base64

    jpeg_bytes = b"\xff\xd8\xff\xe0fake jpeg data"
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")

    msg = protocol.parse_client_message(json.dumps({"type": "camera_frame", "data": encoded}))

    assert msg == protocol.CameraFrameMessage(jpeg=jpeg_bytes)


def test_parse_camera_frame_message_rejects_missing_data():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_client_message('{"type": "camera_frame"}')


def test_parse_camera_frame_message_rejects_invalid_base64():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_client_message('{"type": "camera_frame", "data": "not valid base64!!"}')


def test_parse_invalid_json_raises_protocol_error():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_client_message("not json")


def test_parse_non_object_json_raises_protocol_error():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_client_message("42")


def test_parse_unknown_type_raises_protocol_error():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_client_message('{"type": "mystery"}')


def test_parse_hello_missing_session_id_raises_protocol_error():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_client_message('{"type": "hello"}')


def test_encode_emotion():
    text = protocol.encode_emotion("happy")
    assert json.loads(text) == {"type": "emotion", "value": "happy"}


def test_encode_response_end():
    text = protocol.encode_response_end()
    assert json.loads(text) == {"type": "response_end"}


def test_encode_error():
    text = protocol.encode_error("boom")
    assert json.loads(text) == {"type": "error", "message": "boom"}


def test_encode_action_with_int_result():
    text = protocol.encode_action("dice_roll", 4)
    assert json.loads(text) == {"type": "action", "name": "dice_roll", "result": 4}


def test_encode_action_with_string_result():
    text = protocol.encode_action("coin_flip", "testa")
    assert json.loads(text) == {"type": "action", "name": "coin_flip", "result": "testa"}


def test_encode_face_position_when_found():
    text = protocol.encode_face_position(True, dx=0.21, dy=-0.06)
    assert json.loads(text) == {"type": "face_position", "found": True, "dx": 0.21, "dy": -0.06}


def test_encode_face_position_when_not_found_omits_dx_dy():
    text = protocol.encode_face_position(False)
    assert json.loads(text) == {"type": "face_position", "found": False}
