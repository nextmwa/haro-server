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
                if message["type"] == "websocket.disconnect":
                    logger.info("session %s disconnected", session.session_id)
                    break
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
                        except Exception:
                            # Deliberately generic on the wire: str(exc) here
                            # leaked provider tracebacks and absolute
                            # filesystem paths to an unauthenticated client.
                            # logger.exception already records the full
                            # traceback server-side, where it belongs.
                            logger.exception("turn failed")
                            await send_text(
                                protocol.encode_error("internal error during turn")
                            )
                elif "bytes" in message and message["bytes"] is not None:
                    await session.handle_audio_frame(message["bytes"])
        except WebSocketDisconnect:
            # Defensive fallback in case some other code path raises this
            # (e.g. a future refactor using receive_text()/receive_bytes()),
            # but the explicit "websocket.disconnect" check above is what
            # actually handles the normal disconnect case with the raw
            # receive() loop used here.
            logger.info("session %s disconnected", session.session_id)

    return app
