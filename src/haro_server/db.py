import datetime
import sqlite3
from dataclasses import dataclass

# SQLite, not a client/server database: this runs as a single self-hosted
# instance on one machine (see docker-compose.yml's model_cache volume --
# same "one process, one box" deployment shape), so a file-based DB needs
# no extra infrastructure and is trivial to back up (it's just a file).
#
# sqlite3's default connection is not safe to share across threads/asyncio
# tasks that might run concurrently, so every function here opens (and
# closes) its own short-lived connection rather than holding one open for
# the server's lifetime -- these are small, infrequent reads/writes (a
# couple of times per voice turn, plus occasional admin-page hits), not a
# hot path where connection setup cost matters.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    transcript TEXT NOT NULL,
    reply TEXT,
    emotion TEXT
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    fact TEXT NOT NULL,
    source_session_id TEXT
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fire_at TEXT NOT NULL,
    label TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS event_state (
    source_key TEXT PRIMARY KEY,
    dedup_key TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Transcript:
    id: int
    session_id: str
    created_at: str
    transcript: str
    reply: str | None
    emotion: str | None


@dataclass(frozen=True)
class Alarm:
    id: int
    fire_at: datetime.datetime  # timezone-aware, UTC
    label: str
    kind: str  # "alarm" (sveglia) or "timer"
    status: str  # pending, ringing, done, cancelled, missed


@dataclass(frozen=True)
class Memory:
    id: int
    created_at: str
    fact: str
    source_session_id: str | None


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def save_transcript(
    db_path: str, session_id: str, transcript: str, reply: str | None, emotion: str | None
) -> int:
    """Returns the new row's id (turn_audio.py names the turn's WAV by it)."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO transcripts (session_id, created_at, transcript, reply, emotion) VALUES (?, ?, ?, ?, ?)",
            (session_id, _now(), transcript, reply, emotion),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def get_transcripts(db_path: str, limit: int = 50, offset: int = 0) -> list[Transcript]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, session_id, created_at, transcript, reply, emotion "
            "FROM transcripts ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [Transcript(*row) for row in rows]
    finally:
        conn.close()


def add_memory(db_path: str, fact: str, source_session_id: str | None) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO memories (created_at, fact, source_session_id) VALUES (?, ?, ?)",
            (_now(), fact, source_session_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_memories(db_path: str) -> list[Memory]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, created_at, fact, source_session_id FROM memories ORDER BY id DESC"
        ).fetchall()
        return [Memory(*row) for row in rows]
    finally:
        conn.close()


def delete_memory(db_path: str, memory_id: int) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
    finally:
        conn.close()


def get_config_value(db_path: str, key: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_config_value(db_path: str, key: str, value: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_event_dedup_key(db_path: str, source_key: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT dedup_key FROM event_state WHERE source_key = ?", (source_key,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_event_dedup_key(db_path: str, source_key: str, dedup_key: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO event_state (source_key, dedup_key) VALUES (?, ?) "
            "ON CONFLICT(source_key) DO UPDATE SET dedup_key = excluded.dedup_key",
            (source_key, dedup_key),
        )
        conn.commit()
    finally:
        conn.close()


def add_alarm(db_path: str, fire_at: datetime.datetime, label: str, kind: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO alarms (fire_at, label, kind, created_at, status) VALUES (?, ?, ?, ?, 'pending')",
            (fire_at.astimezone(datetime.UTC).isoformat(), label, kind, _now()),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def _alarm(row) -> Alarm:
    return Alarm(row[0], datetime.datetime.fromisoformat(row[1]), row[2], row[3], row[4])


def get_pending_alarms(db_path: str) -> list[Alarm]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, fire_at, label, kind, status FROM alarms WHERE status = 'pending' ORDER BY fire_at"
        ).fetchall()
        return [_alarm(r) for r in rows]
    finally:
        conn.close()


def get_due_alarms(db_path: str, now: datetime.datetime) -> list[Alarm]:
    return [a for a in get_pending_alarms(db_path) if a.fire_at <= now]


def set_alarm_status(db_path: str, alarm_id: int, status: str) -> bool:
    """Returns whether an alarm with that id existed."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("UPDATE alarms SET status = ? WHERE id = ?", (status, alarm_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
