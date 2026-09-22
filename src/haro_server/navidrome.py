import asyncio
import contextlib
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import httpx

logger = logging.getLogger(__name__)


# Matches tts_common.DEVICE_SAMPLE_RATE (16-bit mono PCM at 16kHz) -- the
# firmware's one fixed playback format, see stream_track_as_pcm16()'s
# ffmpeg invocation and docstring.
_PCM_BYTES_PER_SECOND = 16000 * 2


@dataclass(frozen=True)
class Track:
    id: str
    title: str
    artist: str
    album: str


async def _align_to_sample_boundary(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Wraps a raw byte-chunk stream (e.g. successive proc.stdout.read(N)
    calls) so every chunk this yields has an even length -- 16-bit PCM is 2
    bytes/sample, and a raw pipe read has no obligation to land on a
    sample boundary. An odd-length chunk passed straight to the firmware
    desyncs every sample after it once played, audible as harsh static
    instead of the track (confirmed on real hardware -- see
    stream_track_as_pcm16()'s call site comment). Any trailing odd byte is
    carried over and prepended to the next chunk instead of being dropped;
    a genuinely odd-length total is flushed as a final short chunk once
    `chunks` is exhausted, rather than silently losing that last byte.
    """
    leftover = b""
    async for raw in chunks:
        if not raw:
            continue
        chunk = leftover + raw
        if len(chunk) % 2 != 0:
            leftover = chunk[-1:]
            chunk = chunk[:-1]
        else:
            leftover = b""
        if chunk:
            yield chunk
    if leftover:
        yield leftover


# How far ahead of real-time playback this is allowed to run before
# _pace_to_realtime() below starts throttling -- enough slack to smooth
# over brief network/scheduling hiccups without either device-side buffers
# or the sender's own read-ahead growing unbounded.
_MAX_LEAD_SECONDS = 0.5


async def _pace_to_realtime(
    chunks: AsyncIterator[bytes],
    bytes_per_second: float,
    *,
    sleep: Callable[[float], "asyncio.Future[None]"] = asyncio.sleep,
    now: Callable[[], float] = time.monotonic,
    max_lead_seconds: float = _MAX_LEAD_SECONDS,
) -> AsyncIterator[bytes]:
    """Throttles `chunks` so they're handed to the caller no faster than
    they'll actually be played, plus up to `max_lead_seconds` of read-ahead.

    Why this exists: ffmpeg decodes an mp3 far faster than real-time (a
    whole track in a couple of seconds), and nothing about
    stream_track_as_pcm16()'s httpx-fetch + ffmpeg-pipe path waits on
    anything playback-speed-related -- unlike tts.py's chunks, which are
    naturally paced by how long Kokoro actually takes to synthesize each
    sentence (comparable to or slower than that sentence's own playback
    time). Without this, the whole track gets pushed at network speed,
    confirmed on real hardware to arrive faster than the firmware's
    fixed-size playback queue (main.c's 8-deep server_client event queue)
    and I2S write pacing can absorb -- audible as overlapping fragments of
    the track rather than one continuous stream. Pacing sends to
    real-time (with a small buffer for smoothness) keeps the device never
    more than `max_lead_seconds` of audio ahead of what it's actually
    playing.

    `sleep`/`now` are injectable so tests can exercise the throttling
    decision without a real clock or real waiting.
    """
    start = now()
    audio_seconds_sent = 0.0
    async for chunk in chunks:
        lead = audio_seconds_sent - (now() - start)
        if lead > max_lead_seconds:
            await sleep(lead - max_lead_seconds)
        audio_seconds_sent += len(chunk) / bytes_per_second
        yield chunk


class NavidromeClient:
    """Subsonic-API client for a self-hosted Navidrome server.

    Auth follows the classic Subsonic salted-token scheme (still Navidrome's
    documented standard): token = md5(password + salt), sent with a fresh
    random salt on every request (per-request salts are what the scheme is
    designed around -- reusing one would defeat the point of salting).
    Verified against Navidrome's own docs (navidrome.org/docs/developers/
    subsonic-api, navidrome.org/docs/usage/integration/authentication) as of
    2026-09, NOT against a live server in this session (none was reachable
    here) -- the exact JSON response shape (parsed in search() below) needs
    confirming against a real instance before trusting it blindly.
    """

    API_VERSION = "1.16.1"
    CLIENT_NAME = "haro"

    def __init__(self, base_url: str, username: str, password: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password

    def _auth_params(self) -> dict[str, str]:
        salt = secrets.token_hex(6)
        token = hashlib.md5((self._password + salt).encode("utf-8")).hexdigest()
        return {
            "u": self._username,
            "t": token,
            "s": salt,
            "v": self.API_VERSION,
            "c": self.CLIENT_NAME,
            "f": "json",
        }

    async def search(self, query: str, count: int = 10) -> list[Track]:
        params = {
            **self._auth_params(),
            "query": query,
            "songCount": str(count),
            "artistCount": "0",
            "albumCount": "0",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self._base_url}/rest/search3", params=params)
            response.raise_for_status()
            data = response.json()

        songs = data.get("subsonic-response", {}).get("searchResult3", {}).get("song", [])
        return [
            Track(
                id=song["id"],
                title=song.get("title", "?"),
                artist=song.get("artist", "?"),
                album=song.get("album", "?"),
            )
            for song in songs
        ]

    def _stream_url(self, track_id: str) -> str:
        # format=mp3: a container ffmpeg can reliably auto-detect while
        # reading from a non-seekable pipe (stream_track_as_pcm16() below),
        # unlike some other containers Navidrome could otherwise return.
        # maxBitRate=64: this plays through a small single desk-robot
        # speaker, not hifi equipment -- no reason to pull (and decode)
        # more bitrate than that.
        params = {**self._auth_params(), "id": track_id, "format": "mp3", "maxBitRate": "64"}
        query = "&".join(f"{key}={value}" for key, value in params.items())
        return f"{self._base_url}/rest/stream?{query}"

    async def stream_track_as_pcm16(self, track_id: str) -> AsyncIterator[bytes]:
        """Fetches the track from Navidrome (server-transcoded to a low-
        bitrate mp3, see _stream_url()) and decodes it to raw 16-bit mono
        PCM at 16kHz via a piped ffmpeg subprocess -- the same target
        format tts_common.resample_to_device_rate() already produces for
        the firmware's fixed-rate speaker (every TTS engine adapter uses
        it), so the firmware side needs no changes to play this the same
        way it plays TTS audio.

        An async generator, closed the same explicit way every other stream
        in this codebase is (see session.py's existing aclose() discipline)
        -- closing it (or cancelling whatever awaits it) tears down the
        ffmpeg subprocess and the feed task promptly instead of leaking
        them, which matters here specifically because session.py cancels
        this mid-track when the robot's wake word interrupts playback.

        Every yielded chunk is guaranteed sample-aligned (even byte count)
        via _align_to_sample_boundary() -- see that function's docstring
        for why a raw pipe read needs this and tts.py's chunks never did.
        Chunks are also paced to roughly real-time playback speed via
        _pace_to_realtime() -- see that function's docstring for why an
        ffmpeg decode (much faster than real-time) needs this too.
        """
        url = self._stream_url(track_id)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
                    "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                )
                assert proc.stdin is not None and proc.stdout is not None
                feed_task = asyncio.create_task(self._feed_ffmpeg(response, proc))

                async def _raw_reads() -> AsyncIterator[bytes]:
                    while True:
                        raw = await proc.stdout.read(4096)
                        if not raw:
                            return
                        yield raw

                aligned = _align_to_sample_boundary(_raw_reads())
                try:
                    async for chunk in _pace_to_realtime(aligned, _PCM_BYTES_PER_SECOND):
                        yield chunk
                finally:
                    feed_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await feed_task
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()

    @staticmethod
    async def _feed_ffmpeg(response: httpx.Response, proc: "asyncio.subprocess.Process") -> None:
        # Runs concurrently with the caller reading proc.stdout above --
        # both directions must be pumped at once, or ffmpeg (and/or the
        # HTTP response's own internal buffer) stalls once either pipe's
        # OS buffer fills, deadlocking the whole transfer.
        try:
            async for chunk in response.aiter_bytes():
                proc.stdin.write(chunk)
                await proc.stdin.drain()
        finally:
            proc.stdin.close()
