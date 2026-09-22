import hashlib
import http.server
import shutil
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from haro_server.navidrome import NavidromeClient, Track, _align_to_sample_boundary, _pace_to_realtime

FIXTURES_DIR = Path(__file__).parent / "fixtures"
requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_auth_params_token_matches_the_documented_subsonic_scheme():
    client = NavidromeClient(base_url="http://navidrome.local", username="leo", password="s3cret")

    params = client._auth_params()

    assert params["u"] == "leo"
    assert params["f"] == "json"
    assert params["c"] == "haro"
    # token must equal md5(password + salt) -- the documented Subsonic
    # scheme (navidrome.org/docs/usage/integration/authentication).
    expected_token = hashlib.md5(("s3cret" + params["s"]).encode("utf-8")).hexdigest()
    assert params["t"] == expected_token


def test_auth_params_uses_a_fresh_salt_each_call():
    client = NavidromeClient(base_url="http://navidrome.local", username="leo", password="s3cret")

    salts = {client._auth_params()["s"] for _ in range(20)}

    assert len(salts) == 20  # no repeats across 20 calls


def test_stream_url_requests_a_low_bitrate_mp3():
    client = NavidromeClient(base_url="http://navidrome.local", username="leo", password="s3cret")

    url = client._stream_url("song-42")

    assert url.startswith("http://navidrome.local/rest/stream?")
    assert "id=song-42" in url
    assert "format=mp3" in url
    assert "maxBitRate=64" in url


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


async def test_search_parses_songs_from_subsonic_response(monkeypatch):
    client = NavidromeClient(base_url="http://navidrome.local", username="leo", password="s3cret")

    fake_payload = {
        "subsonic-response": {
            "status": "ok",
            "searchResult3": {
                "song": [
                    {"id": "1", "title": "Vita Spericolata", "artist": "Vasco Rossi", "album": "Bollicine"},
                    {"id": "2", "title": "Albachiara", "artist": "Vasco Rossi", "album": "Non siamo mica..."},
                ]
            },
        }
    }

    mock_get = AsyncMock(return_value=_FakeResponse(fake_payload))
    with patch("haro_server.navidrome.httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value.get = mock_get
        tracks = await client.search("Vasco Rossi")

    assert tracks == [
        Track(id="1", title="Vita Spericolata", artist="Vasco Rossi", album="Bollicine"),
        Track(id="2", title="Albachiara", artist="Vasco Rossi", album="Non siamo mica..."),
    ]


async def test_search_returns_empty_list_when_no_songs_found():
    client = NavidromeClient(base_url="http://navidrome.local", username="leo", password="s3cret")
    fake_payload = {"subsonic-response": {"status": "ok", "searchResult3": {}}}

    mock_get = AsyncMock(return_value=_FakeResponse(fake_payload))
    with patch("haro_server.navidrome.httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value.get = mock_get
        tracks = await client.search("nonexistent")

    assert tracks == []


async def _fake_chunks(*parts: bytes):
    for part in parts:
        yield part


async def test_align_to_sample_boundary_passes_through_already_even_chunks():
    out = [chunk async for chunk in _align_to_sample_boundary(_fake_chunks(b"abcd", b"efgh"))]
    assert out == [b"abcd", b"efgh"]


async def test_align_to_sample_boundary_carries_an_odd_trailing_byte_to_the_next_chunk():
    # b"abcde" (5 bytes) then b"fgh" (3 bytes) -- naive 4096-at-a-time
    # reads would yield an odd 5-byte chunk here, desyncing every 16-bit
    # sample after it (see this function's docstring). The trailing "e"
    # must instead be held and prepended to the next chunk. Total length
    # (8 bytes) is even, so no final flush is involved here -- that case
    # is covered separately below.
    out = [chunk async for chunk in _align_to_sample_boundary(_fake_chunks(b"abcde", b"fgh"))]
    assert out == [b"abcd", b"efgh"]
    assert all(len(chunk) % 2 == 0 for chunk in out)
    assert b"".join(out) == b"abcdefgh"


async def test_align_to_sample_boundary_flushes_a_trailing_odd_byte_at_end_of_stream():
    # Total input length (5 bytes) is itself odd -- there's no partner
    # byte coming, ever. The last byte must still be flushed once the
    # source is exhausted, not silently dropped.
    out = [chunk async for chunk in _align_to_sample_boundary(_fake_chunks(b"abcde"))]
    assert out == [b"abcd", b"e"]
    assert b"".join(out) == b"abcde"


async def test_align_to_sample_boundary_never_yields_an_odd_length_chunk_across_many_small_reads():
    # Simulates a pipe delivering awkward, arbitrary-sized fragments (1-3
    # bytes at a time) of a message whose total length is even -- every
    # yielded chunk must still be even-length, and the concatenation must
    # reproduce the original bytes exactly.
    source = bytes(range(37)) + bytes(range(37, 74))  # 74 bytes, even total
    parts = [source[i:i + 3] for i in range(0, len(source), 3)]
    out = [chunk async for chunk in _align_to_sample_boundary(_fake_chunks(*parts))]
    assert all(len(chunk) % 2 == 0 for chunk in out)
    assert b"".join(out) == source


class _FakeClock:
    """Deterministic now()/sleep() pair for testing _pace_to_realtime()
    without a real clock or real waiting: sleep() advances the clock by
    exactly the requested duration and records the call.
    """

    def __init__(self) -> None:
        self.t = 0.0
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, duration: float) -> None:
        self.sleep_calls.append(duration)
        self.t += duration


async def test_pace_to_realtime_sleeps_once_it_gets_too_far_ahead():
    clock = _FakeClock()
    # 10 chunks of 3200 bytes = 0.1s of audio each at 32000 bytes/sec, with
    # zero real time elapsing between them (no ffmpeg/network delay in this
    # fake) -- exactly the "decoded far faster than real-time" scenario
    # this function exists for. With max_lead_seconds=0.5, the first 6
    # chunks (0.6s of audio accumulated) fit under the lead budget; each
    # chunk after that should trigger a 0.1s sleep to hold the lead at 0.5s.
    chunks = _fake_chunks(*[bytes(3200) for _ in range(10)])
    out = [
        chunk
        async for chunk in _pace_to_realtime(chunks, bytes_per_second=32000, sleep=clock.sleep, now=clock.now, max_lead_seconds=0.5)
    ]
    assert len(out) == 10
    assert clock.sleep_calls == pytest.approx([0.1, 0.1, 0.1, 0.1])


async def test_pace_to_realtime_does_not_sleep_when_already_at_or_behind_playback_speed():
    clock = _FakeClock()

    async def slow_chunks():
        for _ in range(5):
            clock.t += 1.0  # simulate real elapsed time (network/decode) before each chunk
            yield bytes(3200)  # 0.1s of audio -- far less than the 1s that just passed

    out = [chunk async for chunk in _pace_to_realtime(slow_chunks(), bytes_per_second=32000, sleep=clock.sleep, now=clock.now)]
    assert len(out) == 5
    assert clock.sleep_calls == []


async def test_pace_to_realtime_yields_chunks_unchanged_and_in_order():
    clock = _FakeClock()
    parts = [b"first", b"second", b"third"]
    out = [chunk async for chunk in _pace_to_realtime(_fake_chunks(*parts), bytes_per_second=32000, sleep=clock.sleep, now=clock.now)]
    assert out == parts


@pytest.fixture
def local_file_server():
    """Serves tests/fixtures/ over HTTP on localhost, for exercising
    stream_track_as_pcm16()'s real httpx-fetch + ffmpeg-pipe mechanism
    against a real (tiny, checked-in) MP3 without needing a live Navidrome
    server anywhere.
    """
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(FIXTURES_DIR), **kwargs
    )
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


@requires_ffmpeg
async def test_stream_track_as_pcm16_decodes_to_16bit_mono_16khz_pcm(local_file_server):
    client = NavidromeClient(base_url="http://navidrome.local", username="leo", password="s3cret")
    client._stream_url = lambda track_id: f"{local_file_server}/test_tone.mp3"

    total_bytes = 0
    async for chunk in client.stream_track_as_pcm16("fake-id"):
        assert isinstance(chunk, bytes)
        total_bytes += len(chunk)

    # 0.5s @ 16kHz, 16-bit mono = 0.5 * 16000 * 2 = 16000 bytes, +/- a small
    # amount of encoder/decoder frame padding.
    assert abs(total_bytes - 16000) < 4000


@requires_ffmpeg
async def test_stream_track_as_pcm16_cleans_up_promptly_on_cancellation(local_file_server):
    import asyncio
    import time

    client = NavidromeClient(base_url="http://navidrome.local", username="leo", password="s3cret")
    client._stream_url = lambda track_id: f"{local_file_server}/test_tone.mp3"

    received: list[int] = []

    async def _play():
        async for chunk in client.stream_track_as_pcm16("fake-id"):
            received.append(len(chunk))
            await asyncio.sleep(0.01)

    task = asyncio.create_task(_play())
    await asyncio.sleep(0.02)
    start = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.monotonic() - start

    # A hung ffmpeg/feed-task cleanup would make this take much longer than
    # a clean cancellation should.
    assert elapsed < 2.0
