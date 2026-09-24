# src/haro_server/server.py
import asyncio
import contextlib
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import camera_preview, db, face_tracking, protocol
from .admin import create_admin_router
from .announcer import run_announcer
from .chatterbox_tts import ChatterboxTtsEngine
from .config import Config
from .event_bus import EventBus
from .event_poller import poll_calendar, poll_github
from .google_calendar_accounts import load_accounts
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
        return PocketTtsEngine(voice=config.pockettts_voice)
    if config.tts_engine == "kokoro":
        return KokoroTtsEngine(device=config.tts_device)
    raise ValueError(
        f"unknown TTS_ENGINE {config.tts_engine!r} -- expected \"kokoro\", \"chatterbox\", or \"pockettts\""
    )


def create_app(config: Config) -> FastAPI:
    db.init_db(config.db_path)

    # Loaded once, at server startup, and shared across every connection --
    # see this task's design note. A failure here fails server startup
    # immediately rather than lazily on the first robot connection.
    logger.info("loading STT model (device=%s)...", config.stt_device)
    stt_model = load_parakeet_model(device=config.stt_device)
    logger.info("loading TTS model (engine=%s, device=%s, pockettts_voice=%s)...", config.tts_engine, config.tts_device, config.pockettts_voice)
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
        # (label, service, calendar_ids) for every account whose token
        # loaded and whose client built successfully. One account's bad
        # or expired token must degrade only that account's polling --
        # not the other accounts, and not GitHub polling either (this
        # used to run before the loop with no guard at all, so any
        # failure here killed run_scheduler() entirely -- see C2 in the
        # final review, which this per-account loop preserves).
        calendar_accounts = []
        if config.google_calendar_accounts_path:
            try:
                from google.oauth2.credentials import Credentials
                from googleapiclient.discovery import build

                accounts = load_accounts(config.google_calendar_accounts_path)
            except Exception:
                logger.exception(
                    "failed to load %s -- Calendar polling disabled",
                    config.google_calendar_accounts_path,
                )
                accounts = []
            for account in accounts:
                try:
                    credentials = Credentials.from_authorized_user_file(account.credentials_path)
                    service = build("calendar", "v3", credentials=credentials)
                    calendar_accounts.append((account.label, service, account.calendar_ids))
                except Exception:
                    logger.exception(
                        "failed to initialize the Google Calendar client for account=%s -- "
                        "this account's polling disabled",
                        account.label,
                    )

        import httpx

        async with httpx.AsyncClient(base_url="https://api.github.com") as client:
            while True:
                try:
                    if config.github_token and config.github_repos:
                        for event in await poll_github(
                            client, config.db_path, config.github_token, config.github_repos
                        ):
                            event_bus.publish(event)
                    for label, service, calendar_ids in calendar_accounts:
                        for event in await poll_calendar(service, config.db_path, label, calendar_ids):
                            event_bus.publish(event)
                except Exception:
                    # Never let one bad poll cycle kill this task
                    # permanently and silently -- log it and try again
                    # next interval (see C2 in the final review).
                    logger.exception("proactive event poll cycle failed, will retry next interval")
                await asyncio.sleep(config.event_poll_interval_seconds)

    def _log_if_task_exited_unexpectedly(name: str):
        def _callback(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.error("%s background task exited unexpectedly", name, exc_info=exc)

        return _callback

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup and shutdown for both the background poller/announcer
        # tasks (Task 7) and the GitHub MCP client connection (Task 10)
        # used to be two separate, order-coupled `@app.on_event("startup")`
        # handlers with no shutdown counterpart at all -- deprecated in
        # the installed FastAPI/Starlette, and structurally unable to
        # either retain task references or close the MCP subprocess (see
        # C1/C2/I8 in the final review). A single lifespan context
        # manager fixes all three: startup runs before `yield`, shutdown
        # after it, and both halves run in this same coroutine -- which
        # matters because mcp_client.py's close() MUST be awaited from
        # the same asyncio task that awaited connect() (see its own
        # docstring).
        background_tasks: list[asyncio.Task] = []
        if config.github_token or config.google_calendar_accounts_path:
            scheduler_task = asyncio.create_task(run_scheduler())
            announcer_task = asyncio.create_task(run_announcer(event_bus, get_active_session))
            scheduler_task.add_done_callback(_log_if_task_exited_unexpectedly("run_scheduler"))
            announcer_task.add_done_callback(_log_if_task_exited_unexpectedly("run_announcer"))
            # Retained here for the lifetime of the app -- an
            # asyncio.Task with no strong reference held anywhere can be
            # garbage-collected mid-flight with no warning (the same
            # footgun session.py's own _background_tasks set guards
            # against for fact-extraction tasks).
            background_tasks.extend([scheduler_task, announcer_task])
            logger.info("proactive event polling started (interval=%ds)", config.event_poll_interval_seconds)
        else:
            logger.info("no GitHub token or Calendar credentials configured -- proactive events disabled")

        mcp_client = None
        if config.github_token:
            from .mcp_client import McpToolClient

            mcp_client = McpToolClient(github_token=config.github_token)
            try:
                await mcp_client.connect()
            except Exception:
                # An optional integration (github-mcp-server missing from
                # PATH, an invalid/expired token, a handshake timeout)
                # must never take the whole robot down over it -- STT,
                # TTS, conversation, and music all work fine without
                # GitHub tool-calling. Leave llm tool-less instead of
                # propagating and crashing server startup (see C1 in the
                # final review).
                logger.exception(
                    "failed to connect to the GitHub MCP server -- GitHub tool-calling disabled"
                )
                mcp_client = None
            else:
                llm.set_tools(mcp_client.tools, mcp_client.call_tool)
                logger.info("GitHub MCP tools loaded (%d tool(s))", len(mcp_client.tools))
        else:
            logger.info("GITHUB_TOKEN not set -- GitHub MCP tool-calling disabled")

        try:
            yield
        finally:
            for task in background_tasks:
                task.cancel()
            for task in background_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if mcp_client is not None:
                # Same coroutine that awaited connect() above -- required
                # by mcp_client.py's close() docstring (anyio cancel
                # scopes are bound to the task that entered them).
                # Without this, every server restart leaked an orphan
                # github-mcp-server subprocess holding a GitHub token in
                # its environment (see I8 in the final review).
                await mcp_client.close()

    app = FastAPI(lifespan=lifespan)

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
        finally:
            # Reliably clears active_session no matter how the loop above
            # exited -- a normal "websocket.disconnect" message, a
            # WebSocketDisconnect exception, or anything else raised out
            # of the loop (e.g. send_text()/send_binary() raising on a
            # half-closed socket) used to leave a dead session registered
            # forever (see I4 in the final review).
            #
            # The identity check matters on its own: on a flaky-WiFi
            # reconnect, the OLD connection's disconnect handler can run
            # AFTER a new connection has already registered its own
            # (live) session -- without this check, the old handler would
            # null out the new, still-live session's registration.
            if active_session is session:
                active_session = None

    return app
