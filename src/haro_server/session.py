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
        logger.debug("transcript: %s", transcript)

        raw_reply = self._llm.stream_reply(transcript)
        emotion, text_stream = await _split_emotion_prefix(raw_reply)
        await self._send_text(protocol.encode_emotion(emotion))

        try:
            async for chunk in self._tts.synthesize(text_stream):
                await self._send_binary(chunk)
        finally:
            # text_stream (the _prepend wrapper) only delegates to raw_reply
            # via `async for`, which does NOT cascade .aclose() to it -- so
            # both must be closed explicitly, or an aborted turn (e.g. the
            # robot disconnects mid-reply) leaves the LLM's generator/
            # connection suspended until garbage collection eventually gets
            # to it, instead of closing promptly.
            await text_stream.aclose()
            await raw_reply.aclose()

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
