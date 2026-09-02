import json

import pytest

from haro_server import protocol


def test_parse_hello_message():
    msg = protocol.parse_client_message('{"type": "hello", "session_id": "abc123"}')
    assert msg == protocol.HelloMessage(session_id="abc123")


def test_parse_end_of_speech_message():
    msg = protocol.parse_client_message('{"type": "end_of_speech"}')
    assert msg == protocol.EndOfSpeechMessage()


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
