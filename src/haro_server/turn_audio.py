"""Keeps the raw audio the robot sent for each voice turn, as a WAV next to
the database (<db dir>/audio/<transcript id>.wav), so the admin page can
play it back beside its transcript -- to judge whether a bad transcription
was the STT model's fault or the audio's (noise, clipped start, ...).

Only the newest MAX_FILES turns are kept: this is a monitoring aid, not an
archive of everything said in the room.
"""
import logging
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# Same format the robot streams and stt.py decodes: 16kHz mono PCM16.
_SAMPLE_RATE = 16000
MAX_FILES = 100


def _audio_dir(db_path: str) -> Path:
    return Path(db_path).resolve().parent / "audio"


def save(db_path: str, transcript_id: int, pcm16: bytes) -> None:
    """Never raises: failing to keep a debug recording must not fail the
    voice turn it belongs to."""
    try:
        directory = _audio_dir(db_path)
        directory.mkdir(parents=True, exist_ok=True)
        with wave.open(str(directory / f"{transcript_id}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SAMPLE_RATE)
            w.writeframes(pcm16)
        files = sorted(directory.glob("*.wav"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
        for old in files[:-MAX_FILES]:
            old.unlink(missing_ok=True)
    except Exception:
        logger.exception("failed to save audio for transcript %s", transcript_id)


def path_for(db_path: str, transcript_id: int) -> Path | None:
    path = _audio_dir(db_path) / f"{transcript_id}.wav"
    return path if path.is_file() else None
