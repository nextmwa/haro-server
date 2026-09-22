from haro_server import db


def test_init_db_is_idempotent(tmp_path):
    path = str(tmp_path / "haro.db")
    db.init_db(path)
    db.init_db(path)  # must not raise on a second call


def test_save_and_get_transcripts_orders_newest_first(tmp_path):
    path = str(tmp_path / "haro.db")
    db.init_db(path)

    db.save_transcript(path, "session-1", "ciao", "ciao a te", "happy")
    db.save_transcript(path, "session-1", "che ore sono", "sono le tre", "neutral")

    transcripts = db.get_transcripts(path)
    assert [t.transcript for t in transcripts] == ["che ore sono", "ciao"]
    assert transcripts[0].reply == "sono le tre"
    assert transcripts[0].emotion == "neutral"
    assert transcripts[0].session_id == "session-1"


def test_get_transcripts_respects_limit(tmp_path):
    path = str(tmp_path / "haro.db")
    db.init_db(path)
    for i in range(5):
        db.save_transcript(path, "s", f"turno {i}", None, None)

    assert len(db.get_transcripts(path, limit=2)) == 2


def test_add_and_get_memories_orders_newest_first(tmp_path):
    path = str(tmp_path / "haro.db")
    db.init_db(path)

    db.add_memory(path, "il compleanno e' il 5 maggio", "session-1")
    db.add_memory(path, "preferisce il caffe' senza zucchero", "session-2")

    memories = db.get_memories(path)
    assert [m.fact for m in memories] == [
        "preferisce il caffe' senza zucchero",
        "il compleanno e' il 5 maggio",
    ]
    assert memories[1].source_session_id == "session-1"


def test_delete_memory_removes_it(tmp_path):
    path = str(tmp_path / "haro.db")
    db.init_db(path)
    db.add_memory(path, "fatto da eliminare", None)
    memory_id = db.get_memories(path)[0].id

    db.delete_memory(path, memory_id)

    assert db.get_memories(path) == []


def test_config_value_roundtrip(tmp_path):
    path = str(tmp_path / "haro.db")
    db.init_db(path)

    assert db.get_config_value(path, "system_prompt") is None

    db.set_config_value(path, "system_prompt", "Sei un assistente.")
    assert db.get_config_value(path, "system_prompt") == "Sei un assistente."

    db.set_config_value(path, "system_prompt", "Prompt aggiornato.")
    assert db.get_config_value(path, "system_prompt") == "Prompt aggiornato."


def test_event_dedup_key_round_trips(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    assert db.get_event_dedup_key(db_path, "github:acme/web:pr") is None

    db.set_event_dedup_key(db_path, "github:acme/web:pr", "42")
    assert db.get_event_dedup_key(db_path, "github:acme/web:pr") == "42"

    db.set_event_dedup_key(db_path, "github:acme/web:pr", "43")
    assert db.get_event_dedup_key(db_path, "github:acme/web:pr") == "43"
