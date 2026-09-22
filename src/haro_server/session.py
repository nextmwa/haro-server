import asyncio
import logging
import re
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

from . import actions, db, music, protocol

logger = logging.getLogger(__name__)

# Fire-and-forget background fact-extraction tasks (see Session._spawn_fact_extraction)
# need a strong reference kept somewhere until they finish, or asyncio may
# garbage-collect a task mid-flight with no warning (a well-known asyncio
# footgun -- see the "Save a reference to the result" note in
# asyncio.create_task's own docs). Module-level since these tasks outlive
# any single Session/turn.
_background_tasks: set[asyncio.Task] = set()


class SttEngineLike(Protocol):
    def feed(self, frame: bytes) -> None: ...
    async def finalize(self) -> str: ...


class LlmClientLike(Protocol):
    def stream_reply(self, transcript: str) -> AsyncIterator[str]: ...
    async def extract_facts(self, transcript: str, reply: str) -> list[str]: ...
    async def extract_music_query(self, transcript: str) -> str: ...
    async def pick_best_track(self, transcript: str, candidates: list[Any]) -> Any | None: ...


class TtsEngineLike(Protocol):
    def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]: ...


class NavidromeLike(Protocol):
    async def search(self, query: str, count: int = 10) -> list[Any]: ...
    def stream_track_as_pcm16(self, track_id: str) -> AsyncIterator[bytes]: ...


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
        db_path: str | None = None,
        navidrome: NavidromeLike | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._send_text = send_text
        self._send_binary = send_binary
        # None means "no persistence configured" (also what every existing
        # test that doesn't care about it passes implicitly) -- every save/
        # extract call below is a no-op in that case.
        self._db_path = db_path
        # None means "Navidrome not configured" -- a music request then
        # gets a plain error reply instead of attempting playback. See
        # _play_music()/handle_interrupt().
        self._navidrome = navidrome
        self._music_task: asyncio.Task | None = None
        # Set only while speak_announcement()'s own task is in flight --
        # handle_interrupt() cancels this the same way it cancels
        # _music_task, so a user turn can interrupt a proactive
        # announcement, not just a music track.
        self._announcement_task: asyncio.Task | None = None
        # See is_busy()/speak_announcement() below -- true only while THIS
        # object's own synchronous turn-handling is in flight; music's
        # background task (self._music_task above) is checked separately
        # since it outlives handle_end_of_speech()'s return.
        self._reply_in_progress = False
        # Serializes handle_end_of_speech() and speak_announcement() so
        # only one can be mid-flight at a time: both may call
        # self._tts.synthesize() (not documented as safe for concurrent
        # calls) and both push binary audio frames to the same WebSocket
        # via self._send_binary -- two coroutines doing that concurrently
        # would interleave frames on the wire. is_busy() below is a cheap
        # pre-filter the announcer checks before ever trying to acquire
        # this (avoids pointless lock contention from a background
        # poller), but this lock is what actually prevents the race, not
        # the boolean.
        self._lock = asyncio.Lock()
        self.session_id: str | None = None

    async def handle_hello(self, session_id: str) -> None:
        self.session_id = session_id
        logger.info("session started: %s", session_id)

    async def handle_audio_frame(self, frame: bytes) -> None:
        self._stt.feed(frame)

    async def handle_end_of_speech(self) -> None:
        # Held for the full turn -- see self._lock's docstring in
        # __init__ -- so this can never run concurrently with
        # speak_announcement(). A real user turn simply waits for an
        # in-flight announcement to finish rather than racing it.
        async with self._lock:
            await self._handle_end_of_speech_locked()

    async def _handle_end_of_speech_locked(self) -> None:
        self._reply_in_progress = True
        try:
            transcript = await self._stt.finalize()
            logger.debug("transcript: %s", transcript)

            if not transcript.strip():
                # Silence, or a VAD false trigger: nothing was said, so there is
                # nothing to reply to. Providers reject empty user content, so
                # driving a full LLM/TTS turn here would just error out. End the
                # turn cleanly instead so the robot stops waiting.
                logger.info("empty transcript, skipping LLM/TTS turn")
                await self._send_text(protocol.encode_response_end())
                return

            action = actions.match_action(transcript)
            if action is not None:
                # Deterministic utility command (dice, coin flip, ...) -- see
                # actions.py's module docstring for why this bypasses the LLM
                # entirely rather than asking it to "roll a die" itself.
                result = action.resolve()
                logger.info("action matched: %s -> %r", action.name, result)
                await self._send_text(protocol.encode_action(action.name, result))
                await self._send_text(protocol.encode_response_end())
                self._save_transcript(transcript, reply=f"[azione: {action.name} -> {result}]", emotion=None)
                return

            if music.is_music_request(transcript):
                # Spawned as a background task, NOT awaited inline: a track can
                # play for minutes, and the outer server.py receive loop must
                # stay free to catch an `interrupt` message (wake word firing
                # mid-song) the whole time -- see handle_interrupt() and this
                # method's own docstring-length comment on _play_music() for
                # why this is the one turn type that can't just be a straight
                # await like every other branch in this method.
                if self._music_task is not None and not self._music_task.done():
                    self._music_task.cancel()
                self._music_task = asyncio.create_task(self._play_music(transcript))
                return

            raw_reply = self._llm.stream_reply(transcript)
            emotion, text_stream = await _split_emotion_prefix(raw_reply)
            await self._send_text(protocol.encode_emotion(emotion))

            # Tees the reply text past TTS into `reply_parts` too, so the full
            # reply is available afterward for the transcript log and fact
            # extraction below without a second LLM pass.
            reply_parts: list[str] = []
            teed_stream = _tee(text_stream, reply_parts)

            tts_stream = self._tts.synthesize(teed_stream)
            try:
                async for chunk in tts_stream:
                    await self._send_binary(chunk)
            finally:
                # tts_stream, teed_stream, text_stream (the _prepend wrapper),
                # and raw_reply each only delegate to the next via `async for`,
                # which does NOT cascade .aclose() through the chain -- so all
                # four must be closed explicitly, or an aborted turn (e.g. the
                # robot disconnects mid-reply) leaves generators/connections
                # suspended until garbage collection eventually gets to them,
                # instead of closing promptly.
                await tts_stream.aclose()
                await teed_stream.aclose()
                await text_stream.aclose()
                await raw_reply.aclose()

            await self._send_text(protocol.encode_response_end())

            full_reply = "".join(reply_parts)
            self._save_transcript(transcript, full_reply, emotion)
            self._spawn_fact_extraction(transcript, full_reply)
        finally:
            self._reply_in_progress = False

    def _save_transcript(self, transcript: str, reply: str | None, emotion: str | None) -> None:
        if self._db_path is None:
            return
        db.save_transcript(self._db_path, self.session_id or "unknown", transcript, reply, emotion)

    def _spawn_fact_extraction(self, transcript: str, reply: str) -> None:
        if self._db_path is None:
            return
        db_path = self._db_path
        session_id = self.session_id

        async def _run() -> None:
            facts = await self._llm.extract_facts(transcript, reply)
            for fact in facts:
                db.add_memory(db_path, fact, session_id)
            if facts:
                logger.info("stored %d new memory fact(s)", len(facts))

        task = asyncio.create_task(_run())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    async def handle_interrupt(self) -> None:
        """The wake word fired while a background music task -- or a
        proactive announcement (see speak_announcement()) -- is playing.
        Cancel whichever is in flight. Cancelling a task raises
        CancelledError inside whatever it's awaiting (most likely
        navidrome.py's stream_track_as_pcm16(), mid `async for`, for
        music; the TTS stream for an announcement), which the relevant
        try/finally tears down promptly rather than leaving it running
        unattended. A no-op if nothing is currently playing (e.g. the
        interrupt lost a race with the track/announcement finishing
        naturally).
        """
        if self._music_task is not None and not self._music_task.done():
            self._music_task.cancel()
        if self._announcement_task is not None and not self._announcement_task.done():
            self._announcement_task.cancel()

    def is_busy(self) -> bool:
        return self._reply_in_progress or (self._music_task is not None and not self._music_task.done())

    async def speak_announcement(self, text: str) -> None:
        """Speaks `text` with no LLM/STT involvement -- used by the
        proactive-event announcer (event_bus.py's consumer, wired up in
        Task 7), never by a normal user turn. Callers are responsible for
        checking is_busy() first as a cheap pre-filter; the actual mutual
        exclusion against a concurrent handle_end_of_speech() call is
        enforced by self._lock (see its docstring in __init__), acquired
        here too.

        Runs as its own cancellable task (self._announcement_task) so
        handle_interrupt() can cut it short if a real user turn (wake
        word) fires while this is still speaking -- awaited here so
        callers still see this as one coroutine that completes when the
        announcement is delivered (or dropped, if interrupted).
        """
        task = asyncio.ensure_future(self._speak_announcement_locked(text))
        self._announcement_task = task
        try:
            await task
        except asyncio.CancelledError:
            # handle_interrupt() cancelled self._announcement_task (not
            # this coroutine directly) -- swallow it here rather than
            # letting it propagate to the caller (announcer.py's own
            # loop), which must keep running afterward.
            logger.info("proactive announcement interrupted by a user turn")
        finally:
            self._announcement_task = None

    async def _speak_announcement_locked(self, text: str) -> None:
        async with self._lock:
            self._reply_in_progress = True
            try:
                await self._send_text(protocol.encode_emotion("neutral"))
                tts_stream = self._tts.synthesize(_single_chunk(text))
                try:
                    async for chunk in tts_stream:
                        await self._send_binary(chunk)
                finally:
                    await tts_stream.aclose()
                await self._send_text(protocol.encode_response_end())
            finally:
                self._reply_in_progress = False

    async def _play_music(self, transcript: str) -> None:
        if self._navidrome is None:
            logger.info("music request received but Navidrome is not configured")
            await self._send_text(protocol.encode_error("la riproduzione musicale non e' configurata"))
            await self._send_text(protocol.encode_response_end())
            return

        # Captured here (not iterated inline) so the finally block below
        # can explicitly .aclose() it on every exit path -- an `async for`
        # exiting via a propagating exception does NOT implicitly close the
        # iterator, the same reason handle_end_of_speech()'s normal-reply
        # path above explicitly closes tts_stream/teed_stream/text_stream/
        # raw_reply instead of relying on GC. Without this, an interrupt
        # cancelling this task never reached navidrome.py's own cleanup
        # (confirmed missing by test_handle_interrupt_cancels_an_in_flight_music_task
        # before this fix -- the ffmpeg subprocess would leak until GC).
        stream: AsyncIterator[bytes] | None = None
        try:
            query = await self._llm.extract_music_query(transcript)
            candidates = await self._navidrome.search(query)
            track = await self._llm.pick_best_track(transcript, candidates) if candidates else None

            if track is None:
                logger.info("no matching track found for query: %r", query)
                await self._send_text(protocol.encode_error("non ho trovato nessun brano corrispondente"))
                await self._send_text(protocol.encode_response_end())
                return

            logger.info("playing track: %s - %s", track.title, track.artist)
            await self._send_text(protocol.encode_action("music_playing", f"{track.title} - {track.artist}"))
            self._save_transcript(transcript, reply=f"[musica: {track.title} - {track.artist}]", emotion=None)

            stream = self._navidrome.stream_track_as_pcm16(track.id)
            async for chunk in stream:
                await self._send_binary(chunk)

            await self._send_text(protocol.encode_response_end())
        except asyncio.CancelledError:
            logger.info("music playback interrupted")
            raise
        except Exception:
            logger.exception("music playback failed")
            try:
                await self._send_text(protocol.encode_error("errore durante la riproduzione musicale"))
                await self._send_text(protocol.encode_response_end())
            except Exception:
                pass  # connection is likely already gone
        finally:
            if stream is not None:
                await stream.aclose()
            self._music_task = None


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


async def _tee(chunks: AsyncIterator[str], collected: list[str]) -> AsyncIterator[str]:
    async for chunk in chunks:
        collected.append(chunk)
        yield chunk


async def _single_chunk(text: str) -> AsyncIterator[str]:
    yield text
