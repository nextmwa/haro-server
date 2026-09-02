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


ClientMessage = Union[HelloMessage, EndOfSpeechMessage]


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
    raise ProtocolError(f"unknown message type: {msg_type!r}")


def encode_emotion(value: str) -> str:
    return json.dumps({"type": "emotion", "value": value})


def encode_response_end() -> str:
    return json.dumps({"type": "response_end"})


def encode_error(message: str) -> str:
    return json.dumps({"type": "error", "message": message})
