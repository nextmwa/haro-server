# src/haro_server/server.py
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import camera_preview, db, face_tracking, protocol
from .admin import create_admin_router
from .chatterbox_tts import ChatterboxTtsEngine
from .config import Config
from .llm import LiteLlmClient
from .navidrome import NavidromeClient
from .pockettts_tts import PocketTtsEngine
from .session import Session
from .stt import ParakeetSttEngine, load_parakeet_model
from .tts import KokoroTtsEngine

logger = logging.getLogger(__name__)


def _build_tts_engine(config: Config):
    # The only place that knows about a specific TTS engine's class --
    # session.py depends only on the TtsEngineLike protocol (just
    # `synthesize()`), so adding a third engine later, or switching which
    # one is active, never touches session.py or the other engine's file.
    # See tts_common.py's module docstring for the full reasoning.
    if config.tts_engine == "chatterbox":
        return ChatterboxTtsEngine(device=config.tts_device)
    if config.tts_engine == "pockettts":
        # No device= here: Pocket TTS is CPU-first by design (its docs:
        # "a TTS that fits in your CPU") and TTSModel.load_model() takes
        # no device kwarg -- see pockettts_tts.py.
        return PocketTtsEngine()
    if config.tts_engine == "kokoro":
        return KokoroTtsEngine(device=config.tts_device)
    raise ValueError(
        f"unknown TTS_ENGINE {config.tts_engine!r} -- expected \"kokoro\", \"chatterbox\", or \"pockettts\""
    )


def create_app(config: Config) -> FastAPI:
    app = FastAPI()

    db.init_db(config.db_path)

    if config.admin_password:
        app.include_router(create_admin_router(config.db_path, config.admin_password))
        logger.info("admin UI mounted at /admin")
    else:
        # No unauthenticated fallback: an admin UI that edits the system
        # prompt and shows every transcript is not something to expose
        # without a password just because none was configured -- it stays
        # off entirely (ADMIN_PASSWORD unset in .env is "no admin UI", not
        # "open admin UI").
        logger.warning("ADMIN_PASSWORD not set -- /admin UI disabled")

    # Loaded once, at server startup, and shared across every connection --
    # see this task's design note. A failure here fails server startup
    # immediately rather than lazily on the first robot connection.
    logger.info("loading STT model (device=%s)...", config.stt_device)
    stt_model = load_parakeet_model(device=config.stt_device)
    logger.info("loading TTS model (engine=%s, device=%s)...", config.tts_engine, config.tts_device)
    tts = _build_tts_engine(config)
    llm = LiteLlmClient(model=config.default_model, db_path=config.db_path)
    logger.info("models loaded, ready to accept connections")

    navidrome = None
    if config.navidrome_url and config.navidrome_username and config.navidrome_password:
        navidrome = NavidromeClient(
            base_url=config.navidrome_url,
            username=config.navidrome_username,
            password=config.navidrome_password,
        )
        logger.info("Navidrome music playback configured (%s)", config.navidrome_url)
    else:
        logger.info("Navidrome not configured -- music requests will get an error reply")

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

        session = Session(stt, llm, tts, send_text, send_binary, db_path=config.db_path, navidrome=navidrome)

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
                    elif isinstance(parsed, protocol.InterruptMessage):
                        # Reachable precisely because handle_end_of_speech()'s
                        # music branch does NOT await playback inline (see
                        # its own comment) -- this receive loop is free to
                        # keep pulling messages, including this one, while a
                        # track streams in the background.
                        await session.handle_interrupt()
                    elif isinstance(parsed, protocol.CameraFrameMessage):
                        # asyncio.to_thread: Haar cascade detection is CPU-
                        # bound (a handful of ms at QVGA, but still a
                        # blocking call) -- offloaded the same way STT/TTS
                        # inference already is elsewhere in this codebase,
                        # so it never stalls the event loop for other
                        # connections.
                        face_result = await asyncio.to_thread(face_tracking.detect_face, parsed.jpeg)
                        camera_preview.set_latest_frame(
                            parsed.jpeg, face_result.box if face_result is not None else None
                        )
                        if face_result is not None:
                            await send_text(protocol.encode_face_position(True, face_result.dx, face_result.dy))
                        else:
                            await send_text(protocol.encode_face_position(False))
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
