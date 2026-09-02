# Haro AI Backend Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working, containerized conversational AI backend for the Haro robot: it accepts the robot's WebSocket connection, transcribes streamed speech, gets a reply from a configurable LLM provider (Claude/OpenAI/Gemini via LiteLLM), and streams back synthesized speech with an emotion tag — a complete, testable conversation loop with no persistent memory yet (memory is a separate follow-on plan).

**Architecture:** A Python 3.11+ asyncio service (FastAPI + Uvicorn) exposing one WebSocket endpoint implementing the robot's existing protocol contract exactly. A `Session` class owns the per-connection turn state machine and depends only on duck-typed Protocol interfaces for STT/LLM/TTS (never their concrete classes), mirroring the pattern already used in the robot project's `Orchestrator` — this keeps the turn logic fully unit-testable with fakes, independent of any real model or API key. Three adapter modules (`stt.py`, `llm.py`, `tts.py`) implement those Protocols against real libraries and are wired together only in `server.py`.

**Tech Stack:** Python 3.11+, `asyncio`, FastAPI + Uvicorn (WebSocket server), `litellm` (multi-provider LLM routing), NVIDIA Parakeet TDT v3 via NeMo (streaming-oriented STT), Kokoro-82M (streaming TTS), `pytest` + `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-09-02-haro-ai-backend-design.md`

## Global Constraints

- Python 3.11+.
- This server implements the robot's existing, already-deployed WebSocket protocol exactly — see the spec's "The protocol contract" section, sourced from `../haro/src/haro/protocol.py`. This project does not get to change that contract.
- Mic audio arrives as 16-bit PCM mono at **16kHz**. Synthesized reply audio must be sent as 16-bit PCM mono at **24kHz** (pinned by the spec to match the robot's `AudioOutput` default).
- Every external boundary (STT engine, LLM client, TTS engine) is a duck-typed Protocol the `Session` class depends on — `session.py` must never import a concrete adapter class, only the Protocols it defines itself. This is what makes the turn state machine unit-testable without any real model or API key.
- **Honesty about library API certainty:** the LiteLLM adapter (Task 5) is written with high confidence — LiteLLM's `acompletion(..., stream=True)` shape has been stable and OpenAI-API-compatible for a long time. The Parakeet STT adapter (Task 7) and Kokoro TTS adapter (Task 6) involve libraries whose exact Python API may have evolved since this plan was written (both are recent, fast-moving projects). Those two tasks are explicitly flagged with verification steps — implementers must check the actually-installed package's real API (via its docs, `--help`, or inspecting the package) rather than trusting the example code as gospel, and must fall back to the documented simpler alternative if the exact calls don't match. The Protocol interface each adapter must satisfy, and the tests proving `Session` works correctly against that interface, are NOT uncertain — those are this plan's fixed, load-bearing contract.
- Hardware/network-dependent pieces (the real STT/TTS/LLM adapters wired to real models and API keys, the FastAPI server wiring, Docker) have no pytest suite for their real-integration behavior by design — verified manually, since real model weights and API keys aren't available in an automated test run. Their *interface-shape* correctness (do they implement the Protocol correctly) is still checked via type-consistent construction in `server.py`.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/haro_server/__init__.py`
- Create: `tests/test_scaffolding.py`
- Create: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Produces: an installable `haro_server` package under `src/haro_server/`, a `tests/` directory collected by `pytest`, `pytest-asyncio` in auto mode.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "haro-server"
version = "0.1.0"
description = "AI backend server for the Haro personal desk robot"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "websockets>=12.0",
    "litellm>=1.50.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

(STT/TTS-specific dependencies are added in Tasks 6 and 7, since their exact package names/extras need verification at that point.)

- [ ] **Step 2: Create the package init**

Create `src/haro_server/__init__.py` (empty file).

- [ ] **Step 3: Create `.env.example`**

```bash
# Copy to .env and fill in the key(s) matching your DEFAULT_MODEL.
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
DEFAULT_MODEL=claude-sonnet-5

STT_DEVICE=cpu
TTS_DEVICE=cpu

HOST=0.0.0.0
PORT=8765
```

- [ ] **Step 4: Create `.gitignore`**

```
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.venv/
venv/
.env
```

- [ ] **Step 5: Write a trivial scaffolding test**

```python
# tests/test_scaffolding.py
import haro_server


def test_package_imports():
    assert haro_server is not None
```

- [ ] **Step 6: Install the package in editable mode with dev dependencies**

Run: `pip install -e ".[dev]"`

- [ ] **Step 7: Run the test suite to verify the harness works**

Run: `pytest -v`
Expected: 1 test passes.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/haro_server/__init__.py tests/test_scaffolding.py .env.example .gitignore
git commit -m "chore: scaffold haro_server Python package and test harness"
```

---

### Task 2: Config module

**Files:**
- Create: `src/haro_server/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass with fields `anthropic_api_key: str | None`, `openai_api_key: str | None`, `gemini_api_key: str | None`, `default_model: str`, `stt_device: str`, `tts_device: str`, `host: str`, `port: int`; `Config.from_env() -> Config` (reads `os.environ`, with sensible defaults for everything except the API keys). Used by `main.py` (Task 8) and by the LLM adapter (Task 5, via `litellm` reading the same env var names directly — see that task).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import os

from haro_server.config import Config


def test_from_env_uses_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("STT_DEVICE", raising=False)
    monkeypatch.delenv("TTS_DEVICE", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    config = Config.from_env()

    assert config.anthropic_api_key is None
    assert config.openai_api_key is None
    assert config.gemini_api_key is None
    assert config.default_model == "claude-sonnet-5"
    assert config.stt_device == "cpu"
    assert config.tts_device == "cpu"
    assert config.host == "0.0.0.0"
    assert config.port == 8765


def test_from_env_reads_set_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.1")
    monkeypatch.setenv("STT_DEVICE", "cuda")
    monkeypatch.setenv("PORT", "9000")

    config = Config.from_env()

    assert config.anthropic_api_key == "sk-ant-test"
    assert config.default_model == "gpt-5.1"
    assert config.stt_device == "cuda"
    assert config.port == 9000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.config'`

- [ ] **Step 3: Implement `Config`**

```python
# src/haro_server/config.py
import dataclasses
import os


@dataclasses.dataclass
class Config:
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    default_model: str = "claude-sonnet-5"
    stt_device: str = "cpu"
    tts_device: str = "cpu"
    host: str = "0.0.0.0"
    port: int = 8765

    @staticmethod
    def from_env() -> "Config":
        return Config(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
            default_model=os.environ.get("DEFAULT_MODEL", "claude-sonnet-5"),
            stt_device=os.environ.get("STT_DEVICE", "cpu"),
            tts_device=os.environ.get("TTS_DEVICE", "cpu"),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8765")),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/haro_server/config.py tests/test_config.py
git commit -m "feat: add Config reading server settings from environment variables"
```

---

### Task 3: Protocol module

**Files:**
- Create: `src/haro_server/protocol.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Produces: `ProtocolError`; `HelloMessage(session_id: str)`, `EndOfSpeechMessage()` (frozen dataclasses); `ClientMessage` type alias (union of the two); `parse_client_message(text: str) -> ClientMessage` (raises `ProtocolError` on invalid JSON, a non-object, or an unrecognized/malformed message); `encode_emotion(value: str) -> str`; `encode_response_end() -> str`; `encode_error(message: str) -> str`. Used by `session.py` (Task 4) and `server.py` (Task 8).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_protocol.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.protocol'`

- [ ] **Step 3: Implement `protocol.py`**

```python
# src/haro_server/protocol.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_protocol.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/haro_server/protocol.py tests/test_protocol.py
git commit -m "feat: add WebSocket protocol parsing/encoding matching the robot's contract"
```

---

### Task 4: Session turn orchestrator

**Files:**
- Create: `src/haro_server/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `haro_server.protocol` (`encode_emotion`, `encode_response_end`).
- Produces: `SttEngineLike` Protocol (`feed(frame: bytes) -> None`, `finalize() -> str`); `LlmClientLike` Protocol (`stream_reply(transcript: str) -> AsyncIterator[str]`); `TtsEngineLike` Protocol (`synthesize(text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]`); `Session(stt, llm, tts, send_text, send_binary)` with `async def handle_hello(session_id: str) -> None`, `async def handle_audio_frame(frame: bytes) -> None`, `async def handle_end_of_speech() -> None`, and a public `.session_id: str | None` attribute. This is the module `server.py` (Task 8) wires to a real WebSocket, and the module the STT/LLM/TTS adapters (Tasks 5-7) are built to satisfy — but `session.py` itself never imports any of those concrete adapter classes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session.py
from haro_server.session import Session


class FakeStt:
    def __init__(self, transcript: str) -> None:
        self.fed_frames: list[bytes] = []
        self._transcript = transcript

    def feed(self, frame: bytes) -> None:
        self.fed_frames.append(frame)

    def finalize(self) -> str:
        return self._transcript


class FakeLlm:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.received_transcript: str | None = None

    def stream_reply(self, transcript: str):
        self.received_transcript = transcript
        return self._chunk_generator()

    async def _chunk_generator(self):
        for chunk in self._chunks:
            yield chunk


class FakeTts:
    def __init__(self) -> None:
        self.received_text: list[str] = []

    async def synthesize(self, text_stream):
        async for text in text_stream:
            self.received_text.append(text)
            yield f"audio:{text}".encode()


def _make_session(llm_chunks, sent_text=None, sent_binary=None):
    sent_text = sent_text if sent_text is not None else []
    sent_binary = sent_binary if sent_binary is not None else []

    async def send_text(text: str) -> None:
        sent_text.append(text)

    async def send_binary(data: bytes) -> None:
        sent_binary.append(data)

    stt = FakeStt(transcript="ciao come stai")
    llm = FakeLlm(chunks=llm_chunks)
    tts = FakeTts()
    session = Session(stt, llm, tts, send_text, send_binary)
    return session, stt, llm, tts, sent_text, sent_binary


async def test_handle_hello_stores_session_id():
    session, *_ = _make_session(llm_chunks=[])

    await session.handle_hello("session-1")

    assert session.session_id == "session-1"


async def test_handle_audio_frame_feeds_stt():
    session, stt, *_ = _make_session(llm_chunks=[])

    await session.handle_audio_frame(b"\x01\x02")

    assert stt.fed_frames == [b"\x01\x02"]


async def test_end_of_speech_sends_emotion_then_audio_then_response_end():
    # Emotion prefix split across two LLM chunks, on purpose.
    session, stt, llm, tts, sent_text, sent_binary = _make_session(
        llm_chunks=["[emo", "tion:happy] Ciao! ", "Come posso aiutarti?"]
    )

    await session.handle_end_of_speech()

    assert llm.received_transcript == "ciao come stai"
    assert sent_text[0] == '{"type": "emotion", "value": "happy"}'
    assert sent_text[-1] == '{"type": "response_end"}'
    assert sent_binary == [b"audio:Ciao! ", b"audio:Come posso aiutarti?"]
    assert tts.received_text == ["Ciao! ", "Come posso aiutarti?"]


async def test_end_of_speech_falls_back_to_neutral_when_no_emotion_tag():
    session, stt, llm, tts, sent_text, sent_binary = _make_session(
        llm_chunks=["Ciao!"]
    )

    await session.handle_end_of_speech()

    assert sent_text[0] == '{"type": "emotion", "value": "neutral"}'
    assert tts.received_text == ["Ciao!"]


async def test_end_of_speech_falls_back_to_neutral_for_long_reply_without_tag():
    # No bracket anywhere; longer than the internal scan window. All text
    # must still reach TTS -- none of it should be silently dropped.
    long_reply = "Questa e' una risposta piuttosto lunga senza alcun tag di emozione all'inizio, per verificare che il fallback non perda testo."
    session, stt, llm, tts, sent_text, sent_binary = _make_session(
        llm_chunks=[long_reply]
    )

    await session.handle_end_of_speech()

    assert sent_text[0] == '{"type": "emotion", "value": "neutral"}'
    assert "".join(tts.received_text) == long_reply
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.session'`

- [ ] **Step 3: Implement `session.py`**

```python
# src/haro_server/session.py
import logging
import re
from typing import AsyncIterator, Awaitable, Callable, Protocol

from . import protocol

logger = logging.getLogger(__name__)


class SttEngineLike(Protocol):
    def feed(self, frame: bytes) -> None: ...
    def finalize(self) -> str: ...


class LlmClientLike(Protocol):
    def stream_reply(self, transcript: str) -> AsyncIterator[str]: ...


class TtsEngineLike(Protocol):
    def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]: ...


SendText = Callable[[str], Awaitable[None]]
SendBinary = Callable[[bytes], Awaitable[None]]

_EMOTION_RE = re.compile(r"^\[emotion:(happy|sad|confused|neutral)\]\s*")
_MAX_PREFIX_SCAN = 40  # generous upper bound for "[emotion:confused] "


class Session:
    def __init__(
        self,
        stt: SttEngineLike,
        llm: LlmClientLike,
        tts: TtsEngineLike,
        send_text: SendText,
        send_binary: SendBinary,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._send_text = send_text
        self._send_binary = send_binary
        self.session_id: str | None = None

    async def handle_hello(self, session_id: str) -> None:
        self.session_id = session_id
        logger.info("session started: %s", session_id)

    async def handle_audio_frame(self, frame: bytes) -> None:
        self._stt.feed(frame)

    async def handle_end_of_speech(self) -> None:
        transcript = self._stt.finalize()
        logger.info("transcript: %s", transcript)

        raw_reply = self._llm.stream_reply(transcript)
        emotion, text_stream = await _split_emotion_prefix(raw_reply)
        await self._send_text(protocol.encode_emotion(emotion))

        async for chunk in self._tts.synthesize(text_stream):
            await self._send_binary(chunk)

        await self._send_text(protocol.encode_response_end())


async def _split_emotion_prefix(
    chunks: AsyncIterator[str],
) -> tuple[str, AsyncIterator[str]]:
    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        match = _EMOTION_RE.match(buffer)
        if match:
            emotion = match.group(1)
            remainder = buffer[match.end():]
            return emotion, _prepend(remainder, chunks)
        if len(buffer) >= _MAX_PREFIX_SCAN:
            break
    # No recognizable emotion prefix appeared within the scan window (or the
    # stream ended first); treat whatever was accumulated as real reply text
    # and keep streaming normally rather than waiting indefinitely.
    return "neutral", _prepend(buffer, chunks)


async def _prepend(text: str, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    if text:
        yield text
    async for chunk in rest:
        yield chunk
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_session.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add src/haro_server/session.py tests/test_session.py
git commit -m "feat: add Session turn orchestrator with emotion-prefix streaming"
```

---

### Task 5: LLM adapter (LiteLLM)

**Files:**
- Create: `src/haro_server/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `litellm.acompletion`.
- Produces: `SYSTEM_PROMPT: str`; `LiteLlmClient(model: str)` implementing `LlmClientLike` (`stream_reply(transcript: str) -> AsyncIterator[str]`). Used by `server.py` (Task 8).

**Confidence note:** this task's shape is high-confidence — LiteLLM's `acompletion(model=..., messages=[...], stream=True)` interface, returning chunks with `.choices[0].delta.content`, mirrors the OpenAI SDK's streaming shape and has been stable for a long time. LiteLLM reads `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`GEMINI_API_KEY` from the environment automatically based on the `model` string's prefix — this adapter does not need to pass a key explicitly, it only needs those env vars to already be set (which `Config`/`.env` already handle).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm.py
from unittest.mock import AsyncMock, patch

from haro_server.llm import LiteLlmClient


class FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = FakeDelta(content)


class FakeChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [FakeChoice(content)]


async def _fake_stream(chunks):
    for content in chunks:
        yield FakeChunk(content)


async def test_stream_reply_yields_non_empty_deltas_in_order():
    client = LiteLlmClient(model="claude-sonnet-5")

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["Ciao", None, "!", ""])

        collected = [chunk async for chunk in client.stream_reply("ciao")]

    assert collected == ["Ciao", "!"]


async def test_stream_reply_passes_model_and_transcript():
    client = LiteLlmClient(model="gpt-5.1")

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["ok"])

        [_ async for _ in client.stream_reply("che ore sono")]

    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.1"
    assert call_kwargs["stream"] is True
    messages = call_kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "che ore sono"}
    assert messages[0]["role"] == "system"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.llm'`

- [ ] **Step 3: Install `litellm` and implement `llm.py`**

Run: `pip install -e ".[dev]"` (picks up `litellm` from `pyproject.toml`, already added in Task 1)

```python
# src/haro_server/llm.py
from typing import AsyncIterator

import litellm

SYSTEM_PROMPT = """Sei Haro, un assistente vocale amichevole su un piccolo robot da scrivania. Rispondi in modo naturale e conciso, adatto a una conversazione vocale (frasi brevi, colloquiali, senza formattazione markdown).

Inizia SEMPRE la tua risposta con un tag di emozione tra parentesi quadre, scegliendo esattamente uno tra: happy, sad, confused, neutral. Formato esatto, senza eccezioni: [emotion:happy] seguito dal testo della risposta.

Esempio: [emotion:happy] Certo, posso aiutarti con quello!"""


class LiteLlmClient:
    def __init__(self, model: str) -> None:
        self._model = model

    async def stream_reply(self, transcript: str) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add src/haro_server/llm.py tests/test_llm.py
git commit -m "feat: add LiteLLM-based multi-provider streaming chat client"
```

---

### Task 6: TTS adapter (Kokoro)

**Files:**
- Create: `src/haro_server/tts.py`
- Test: `tests/test_tts.py`
- Modify: `pyproject.toml` (add `kokoro` dependency)

**Interfaces:**
- Produces: `KokoroTtsEngine(voice: str = "if_sara", device: str = "cpu")` implementing `TtsEngineLike` (`synthesize(text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]`, sentence-chunked, yielding 16-bit PCM mono at 24kHz). Used by `server.py` (Task 8).

**Confidence note:** the sentence-chunking logic and the `_to_pcm16` audio conversion in this task are fully specified and testable without the real `kokoro` package (Step 1's tests use a fake pipeline). The real `KPipeline` construction/call shape in Step 3 reflects Kokoro's API at its original release and may not exactly match the currently-installed version — **before writing the real adapter, run a quick check of the installed package** (e.g. `python -c "import kokoro; help(kokoro.KPipeline)"` or read its README) and adjust the constructor/call arguments to match what you actually find, keeping the class's outward shape (`synthesize(text_stream)` yielding PCM16 bytes) exactly as specified — that outward shape is what `server.py` and the tests depend on, not the internal `KPipeline` call details.

- [ ] **Step 1: Write the failing tests (against a fake pipeline, not the real `kokoro` package)**

```python
# tests/test_tts.py
import numpy as np

from haro_server.tts import KokoroTtsEngine, _find_sentence_boundary, _to_pcm16


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str, voice: str):
        self.calls.append(text)
        # One "sentence" -> one fake audio segment of quiet, valid samples.
        audio = np.full(10, 0.1, dtype=np.float32)
        yield ("graphemes", "phonemes", audio)


async def _text_stream(chunks: list[str]):
    for chunk in chunks:
        yield chunk


def test_find_sentence_boundary_finds_end_of_first_sentence():
    assert _find_sentence_boundary("Ciao! Come stai?") == len("Ciao! ")


def test_find_sentence_boundary_returns_none_without_terminator():
    assert _find_sentence_boundary("nessun punto qui") is None


def test_to_pcm16_converts_float_audio_to_16_bit_bytes():
    audio = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    pcm = _to_pcm16(audio)
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples[0] == 0
    assert samples[1] == int(0.5 * 32767)
    # Out-of-range values are clipped, not wrapped.
    assert samples[3] == 32767
    assert samples[4] == -32767


async def test_synthesize_flushes_on_sentence_boundaries():
    engine = KokoroTtsEngine.__new__(KokoroTtsEngine)  # bypass real __init__
    engine._pipeline = FakePipeline()
    engine._voice = "if_sara"

    chunks = [c async for c in engine.synthesize(_text_stream(["Ciao! ", "Come stai?"]))]

    assert engine._pipeline.calls == ["Ciao! ", "Come stai?"]
    assert len(chunks) == 2
    assert all(isinstance(c, bytes) for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.tts'`

- [ ] **Step 3: Add the `kokoro` dependency and implement `tts.py`**

In `pyproject.toml`, add `"kokoro>=1.0",` and `"numpy>=1.24.0",` to `dependencies`. Then `pip install -e ".[dev]"`.

If the installed `kokoro` package requires a system-level phonemizer backend (commonly `espeak-ng` on Debian/Ubuntu — check the package's own install instructions), note that as a Dockerfile requirement for Task 9, but it is not needed to run this task's own tests (which use `FakePipeline`, no real `kokoro` import at test time beyond what Step 1 already needs — `numpy`).

```python
# src/haro_server/tts.py
import re
from typing import AsyncIterator

import numpy as np

_SENTENCE_END_RE = re.compile(r"[.!?]\s")


def _find_sentence_boundary(text: str) -> int | None:
    match = _SENTENCE_END_RE.search(text)
    if match is None:
        return None
    return match.end()


def _to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


class KokoroTtsEngine:
    def __init__(self, voice: str = "if_sara", device: str = "cpu") -> None:
        # NOTE: verify this constructor and the call signature below against
        # the actually-installed `kokoro` package -- see this task's
        # confidence note in the plan. "if_sara" is Kokoro's Italian voice
        # naming convention (language prefix "i") at the library's original
        # release; confirm the exact voice name available in your version.
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code="i", device=device)
        self._voice = voice

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        buffer = ""
        async for chunk in text_stream:
            buffer += chunk
            boundary = _find_sentence_boundary(buffer)
            while boundary is not None:
                sentence, buffer = buffer[:boundary], buffer[boundary:]
                for _, _, audio in self._pipeline(sentence, voice=self._voice):
                    yield _to_pcm16(audio)
                boundary = _find_sentence_boundary(buffer)
        if buffer.strip():
            for _, _, audio in self._pipeline(buffer, voice=self._voice):
                yield _to_pcm16(audio)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tts.py -v`
Expected: PASS (4 tests) — these tests exercise the sentence-boundary logic and PCM conversion against `FakePipeline`, not the real `kokoro` package, so they should pass even before the real package is confirmed working.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Manually verify against the real Kokoro package**

Run a short script constructing a real `KokoroTtsEngine()` and feeding it a short async text stream, confirming it produces non-empty PCM bytes without raising. Adjust the constructor/call arguments in `tts.py` if the installed package's real API differs from Step 3's code, per this task's confidence note. Record in your report whether you needed to adjust anything.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/haro_server/tts.py tests/test_tts.py
git commit -m "feat: add Kokoro-based sentence-chunked streaming TTS engine"
```

---

### Task 7: STT adapter (Parakeet)

**Files:**
- Create: `src/haro_server/stt.py`
- Test: `tests/test_stt.py`
- Modify: `pyproject.toml` (add NeMo ASR dependency)

**Interfaces:**
- Produces: `load_parakeet_model(device: str = "cpu")` (loads the model once, expensive, call at server startup); `ParakeetSttEngine(model)` implementing `SttEngineLike` (`feed(frame: bytes) -> None`, `finalize() -> str`) — a cheap per-connection wrapper around a shared, pre-loaded model. Used by `server.py` (Task 8), which must call `load_parakeet_model` exactly once at startup and construct one `ParakeetSttEngine` per connection, all sharing that one model.

**Confidence note — read before starting:** NeMo's streaming ASR API (cache-aware chunked inference, carrying encoder state across calls) is genuinely complex and this plan cannot give you verified-exact code for it — NeMo's exact streaming method names/parameters for this specific model may differ from what's sketched below, and this is a newer model than this plan's author has hands-on experience with. **Do this instead:**

1. First, get the SIMPLE version working and tested: `feed()` just buffers raw audio bytes in memory; `finalize()` converts the full buffer to a normalized float32 numpy array and runs one batch `transcribe()` call, returning the text. This satisfies the `SttEngineLike` interface completely and is correct, even though it isn't true incremental streaming — Parakeet TDT is fast enough (per this plan's research, faster than Whisper Large v3 even on CPU) that a single-pass transcription of a short utterance should still feel responsive. Write and pass this version's tests first.
2. Only if you have time/interest after that's working: look at whether the installed NeMo version exposes a documented streaming/chunked inference API (check `nemo.collections.asr` model methods, NeMo's own streaming ASR examples/tutorials shipped with the package or in its docs) and, if it's straightforward, upgrade `feed()` to do incremental processing that reduces `finalize()`'s latency. This is optional polish, not required for this task to be DONE — **report in your final report which version you shipped**, so the controller and a future task both know the actual latency profile.

- [ ] **Step 1: Write the failing tests (against a fake model, not real NeMo)**

```python
# tests/test_stt.py
import numpy as np

from haro_server.stt import ParakeetSttEngine, _bytes_to_float32


class FakeModel:
    def __init__(self, transcript: str) -> None:
        self.transcribe_calls: list[list[np.ndarray]] = []
        self._transcript = transcript

    def transcribe(self, audio_batch: list[np.ndarray]):
        self.transcribe_calls.append(audio_batch)

        class _Result:
            def __init__(self, text: str) -> None:
                self.text = text

        return [_Result(self._transcript)]


def test_bytes_to_float32_normalizes_int16_range():
    pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes()
    audio = _bytes_to_float32(pcm)
    assert audio.dtype == np.float32
    assert audio[0] == 0.0
    assert 0.49 < audio[1] < 0.51
    assert -0.51 < audio[2] < -0.49


def test_finalize_transcribes_all_fed_audio_and_resets():
    engine = ParakeetSttEngine.__new__(ParakeetSttEngine)  # bypass real __init__
    engine._model = FakeModel(transcript="ciao mondo")
    engine._buffer = bytearray()

    engine.feed(b"\x00\x00\x01\x00")
    engine.feed(b"\x02\x00\x03\x00")
    text = engine.finalize()

    assert text == "ciao mondo"
    assert len(engine._model.transcribe_calls) == 1
    fed_audio = engine._model.transcribe_calls[0][0]
    assert len(fed_audio) == 4  # 4 samples fed across the two frames

    # finalize() must reset state so the next turn starts clean.
    assert bytes(engine._buffer) == b""


def test_finalize_with_no_audio_returns_empty_string_without_calling_model():
    engine = ParakeetSttEngine.__new__(ParakeetSttEngine)
    engine._model = FakeModel(transcript="should not be used")
    engine._buffer = bytearray()

    text = engine.finalize()

    assert text == ""
    assert engine._model.transcribe_calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'haro_server.stt'`

- [ ] **Step 3: Add the NeMo ASR dependency and implement `stt.py`**

In `pyproject.toml`, add `"nemo_toolkit[asr]>=2.0.0",` to `dependencies` (verify this is still the correct install extra for the installed/available version — check the package's current install instructions if this fails). Then `pip install -e ".[dev]"` (this is a heavy dependency with a real download; expect it to take a while).

```python
# src/haro_server/stt.py
import numpy as np


def _bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def load_parakeet_model(device: str = "cpu"):
    """Loads the (large, slow-to-load) Parakeet model once. Call this at
    server startup and share the returned model across every connection's
    ParakeetSttEngine -- constructing it per-connection would reload
    hundreds of MB of weights on every single robot (re)connection.

    NOTE: verify against the installed NeMo version -- see this task's
    confidence note in the plan.
    """
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    return model.to(device)


class ParakeetSttEngine:
    """Cheap, per-connection wrapper around a shared, pre-loaded model.

    Each connection must get its OWN ParakeetSttEngine instance (its
    `_buffer` is per-turn state that must not be shared across concurrent
    robot connections), but all instances should share the SAME `model`
    (loaded once via `load_parakeet_model`) so the expensive weights are
    only ever loaded a single time for the whole server process.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._buffer = bytearray()

    def feed(self, frame: bytes) -> None:
        self._buffer.extend(frame)

    def finalize(self) -> str:
        if not self._buffer:
            return ""
        audio = _bytes_to_float32(bytes(self._buffer))
        result = self._model.transcribe([audio])
        self._buffer = bytearray()
        return result[0].text if result else ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stt.py -v`
Expected: PASS (3 tests) — these run against `FakeModel`, not the real NeMo model, so they should pass regardless of whether the real model loads correctly.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Manually verify against the real Parakeet model**

Call `load_parakeet_model()` for a real model, construct `ParakeetSttEngine(model)`, feed it a short real audio sample (16-bit PCM mono 16kHz — e.g. record a short WAV and read its raw frames, or synthesize a test tone and confirm it at least doesn't crash even if the transcript is nonsense for a tone), and confirm `finalize()` returns a string without raising. Record in your report whether the loading/call needed adjustment from Step 3's code, and whether you attempted the optional streaming upgrade from this task's confidence note.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/haro_server/stt.py tests/test_stt.py
git commit -m "feat: add Parakeet-based STT engine (batch transcription on finalize)"
```

---

### Task 8: FastAPI WebSocket server

**Files:**
- Create: `src/haro_server/server.py`
- Create: `src/haro_server/main.py`

**Interfaces:**
- Consumes: `Config` (Task 2), `protocol` (Task 3), `Session` (Task 4), `LiteLlmClient` (Task 5), `KokoroTtsEngine` (Task 6), `load_parakeet_model`/`ParakeetSttEngine` (Task 7).
- Produces: `create_app(config: Config) -> fastapi.FastAPI`; `main() -> None` (console entry point). No pytest suite — wires real hardware-dependent adapters (real models, real API keys), verified manually. `create_app`'s own routing logic (translating WebSocket frames to `Session` calls) is a thin, mechanical layer over already-tested `Session`/`protocol` code.

**Design note carried over from this plan's self-review:** `load_parakeet_model` and `KokoroTtsEngine` load real model weights (hundreds of MB) — `create_app` must construct them exactly **once**, at app-creation time (server startup), not inside the per-connection handler. `LiteLlmClient` is cheap but is also shared for consistency. Only `ParakeetSttEngine` (which holds per-turn buffer state) is constructed fresh per connection, wrapping the one shared model. This also gives the spec's "fail fast at startup" behavior for free: if a model fails to load, `create_app()` raises immediately rather than lazily on the first robot connection.

- [ ] **Step 1: Implement `server.py`**

```python
# src/haro_server/server.py
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import protocol
from .config import Config
from .llm import LiteLlmClient
from .session import Session
from .stt import ParakeetSttEngine, load_parakeet_model
from .tts import KokoroTtsEngine

logger = logging.getLogger(__name__)


def create_app(config: Config) -> FastAPI:
    app = FastAPI()

    # Loaded once, at server startup, and shared across every connection --
    # see this task's design note. A failure here fails server startup
    # immediately rather than lazily on the first robot connection.
    logger.info("loading STT model (device=%s)...", config.stt_device)
    stt_model = load_parakeet_model(device=config.stt_device)
    logger.info("loading TTS model (device=%s)...", config.tts_device)
    tts = KokoroTtsEngine(device=config.tts_device)
    llm = LiteLlmClient(model=config.default_model)
    logger.info("models loaded, ready to accept connections")

    @app.websocket("/")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()

        # Cheap: wraps the shared stt_model with a fresh per-connection
        # buffer. tts/llm have no per-turn state, so the shared instances
        # from above are reused directly across all connections.
        stt = ParakeetSttEngine(model=stt_model)

        async def send_text(text: str) -> None:
            await websocket.send_text(text)

        async def send_binary(data: bytes) -> None:
            await websocket.send_bytes(data)

        session = Session(stt, llm, tts, send_text, send_binary)

        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] is not None:
                    try:
                        parsed = protocol.parse_client_message(message["text"])
                    except protocol.ProtocolError as exc:
                        logger.warning("protocol error: %s", exc)
                        await send_text(protocol.encode_error(str(exc)))
                        continue
                    if isinstance(parsed, protocol.HelloMessage):
                        await session.handle_hello(parsed.session_id)
                    elif isinstance(parsed, protocol.EndOfSpeechMessage):
                        try:
                            await session.handle_end_of_speech()
                        except Exception as exc:
                            logger.exception("turn failed")
                            await send_text(protocol.encode_error(str(exc)))
                elif "bytes" in message and message["bytes"] is not None:
                    await session.handle_audio_frame(message["bytes"])
        except WebSocketDisconnect:
            logger.info("session %s disconnected", session.session_id)

    return app
```

- [ ] **Step 2: Implement `main.py`**

```python
# src/haro_server/main.py
import logging

import uvicorn

from .config import Config
from .server import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = Config.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, loop="uvloop")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the modules import cleanly**

Run: `python -c "from haro_server import server, main; print('ok')"`

Note this deliberately does NOT call `create_app(...)` — since `create_app` now eagerly loads the real STT/TTS models (per this task's design note), calling it requires real model downloads/weights that may not be available in this environment. This step only confirms the modules themselves import without a syntax/reference error (i.e. every name `server.py`/`main.py` reference actually exists in Tasks 2-7's modules). Full startup — actually calling `create_app` and loading real models — is verified in Task 9's manual smoke test and Task 10's Docker check.

- [ ] **Step 3b: Self-review the wiring against the real signatures**

Before moving on, re-read `config.py` (Task 2), `session.py` (Task 4), `llm.py` (Task 5), `tts.py` (Task 6), and `stt.py` (Task 7) as they actually exist on disk right now, and confirm every constructor call and attribute access in `server.py`/`main.py` matches exactly — field names on `Config`, `Session`'s constructor parameter order, `LiteLlmClient`/`KokoroTtsEngine`/`load_parakeet_model`/`ParakeetSttEngine`'s exact signatures. This wiring task is exactly the kind of place a small name mismatch would silently break the whole server at runtime without any test catching it.

- [ ] **Step 4: Commit**

```bash
git add src/haro_server/server.py src/haro_server/main.py
git commit -m "feat: add FastAPI WebSocket endpoint wiring the full turn pipeline"
```

---

### Task 9: Fake robot client (manual testing tool)

**Files:**
- Create: `tools/fake_robot_client.py`

**Interfaces:**
- Standalone script simulating the robot side of the protocol (the mirror image of the robot repo's own `tools/fake_server.py`, which simulates this server for testing the robot). Not imported by any `haro_server` module. Lets you manually exercise this server end-to-end without physical robot hardware.

- [ ] **Step 1: Implement the fake client**

```python
# tools/fake_robot_client.py
"""Simulates the Haro robot's side of the WebSocket protocol, for manually
testing this server without physical robot hardware. Sends a short burst of
silent PCM audio (not real speech -- useful for confirming the pipeline runs
end-to-end and produces *some* transcript/reply/audio, not for judging STT
quality) then end_of_speech, and prints what comes back.
"""
import asyncio
import json
import sys

import websockets

SAMPLE_RATE = 16000
FRAME_BYTES = 2560  # matches the robot's own outgoing frame size


async def main(url: str) -> None:
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "hello", "session_id": "fake-robot-1"}))

        silence_frame = b"\x00" * FRAME_BYTES
        for _ in range(20):  # ~3.2s of silence
            await ws.send(silence_frame)

        await ws.send(json.dumps({"type": "end_of_speech"}))
        print("sent end_of_speech, waiting for response...")

        audio_bytes_received = 0
        async for message in ws:
            if isinstance(message, bytes):
                audio_bytes_received += len(message)
            else:
                print("received:", message)
                if json.loads(message).get("type") == "response_end":
                    break

        print(f"total audio bytes received: {audio_bytes_received}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765/"
    asyncio.run(main(url))
```

- [ ] **Step 2: Manually verify against a running server**

Start the server (`python -m haro_server.main`, with at least one real API key set in the environment), then in another terminal run `python tools/fake_robot_client.py`. Expected: prints the `emotion` message, then reports a non-zero total audio bytes received, then the loop exits after `response_end`. This is a smoke test — silence as input won't produce a meaningful transcript, but it proves the pipeline (STT → LLM → TTS → protocol) runs to completion without crashing.

- [ ] **Step 3: Commit**

```bash
git add tools/fake_robot_client.py
git commit -m "chore: add fake robot client for manual end-to-end server testing"
```

---

### Task 10: Docker

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Interfaces:**
- Standalone deployment files; not consumed by any Python module.

- [ ] **Step 1: Create the `Dockerfile`**

```dockerfile
FROM python:3.11-slim

# espeak-ng: required by Kokoro's phonemizer backend. ffmpeg: commonly
# needed by audio-processing libraries in this stack. Verify against
# Tasks 6/7's actual installed packages whether anything else is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

EXPOSE 8765

CMD ["python", "-m", "haro_server.main"]
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
services:
  app:
    build: .
    ports:
      - "8765:8765"
    env_file:
      - .env
    restart: unless-stopped
```

(No GPU override or Neo4j service in this plan — those belong to the follow-on memory plan and a later GPU-acceleration pass. This compose file runs the whole core pipeline on CPU with a single container.)

- [ ] **Step 3: Manually verify the container builds and runs**

`docker compose build && docker compose up`, with a real `.env` file (copied from `.env.example`, at least one API key filled in). Confirm the container starts without crashing and logs show Uvicorn listening on port 8765. Run `tools/fake_robot_client.py` from Task 9 against `ws://localhost:8765/` to confirm the containerized server responds end-to-end.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "chore: add Dockerfile and docker-compose for the AI backend"
```
