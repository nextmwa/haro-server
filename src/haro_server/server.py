# src/haro_server/server.py
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import camera_preview, db, face_tracking, protocol
from .admin import create_admin_router
from .announcer import run_announcer
from .chatterbox_tts import ChatterboxTtsEngine
from .config import Config
from .event_bus import EventBus
from .event_poller import poll_calendar, poll_github
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

    event_bus = EventBus()
    active_session: Session | None = None

    def get_active_session() -> Session | None:
        return active_session

    async def run_scheduler() -> None:
        calendar_service = None
        if config.google_calendar_credentials_path:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            credentials = Credentials.from_authorized_user_file(config.google_calendar_credentials_path)
            calendar_service = build("calendar", "v3", credentials=credentials)

        import httpx

        async with httpx.AsyncClient(base_url="https://api.github.com") as client:
            while True:
                if config.github_token and config.github_repos:
                    for event in await poll_github(
                        client, config.db_path, config.github_token, config.github_repos
                    ):
                        event_bus.publish(event)
                if calendar_service is not None:
                    for event in await poll_calendar(calendar_service, config.db_path):
                        event_bus.publish(event)
                await asyncio.sleep(config.event_poll_interval_seconds)

    @app.on_event("startup")
    async def _start_background_event_tasks() -> None:
        # Registered here (not called directly in create_app()'s body)
        # because create_app() runs before uvicorn's event loop exists --
        # see this task's Context note. FastAPI runs every "startup"
        # handler once that loop is actually up, which is the earliest
        # point asyncio.create_task() is legal here.
        if config.github_token or config.google_calendar_credentials_path:
            asyncio.create_task(run_scheduler())
            asyncio.create_task(run_announcer(event_bus, get_active_session))
            logger.info("proactive event polling started (interval=%ds)", config.event_poll_interval_seconds)
        else:
            logger.info("no GitHub token or Calendar credentials configured -- proactive events disabled")

    @app.on_event("startup")
    async def _connect_mcp_client() -> None:
        if config.github_token:
            from .mcp_client import McpToolClient

            mcp_client = McpToolClient(github_token=config.github_token)
            await mcp_client.connect()
            llm.set_tools(mcp_client.tools, mcp_client.call_tool)
            logger.info("GitHub MCP tools loaded (%d tool(s))", len(mcp_client.tools))
        else:
            logger.info("GITHUB_TOKEN not set -- GitHub MCP tool-calling disabled")

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
        nonlocal active_session
        active_session = session

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    logger.info("session %s disconnected", session.session_id)
                    active_session = None
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
            active_session = None

    return app
