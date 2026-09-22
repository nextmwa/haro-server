import os

from haro_server.config import Config


def test_from_env_uses_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("STT_DEVICE", raising=False)
    monkeypatch.delenv("TTS_DEVICE", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    config = Config.from_env()

    assert config.anthropic_api_key is None
    assert config.openai_api_key is None
    assert config.gemini_api_key is None
    assert config.default_model == "claude-sonnet-5"
    assert config.stt_device == "cpu"
    assert config.tts_device == "cpu"
    assert config.host == "0.0.0.0"
    assert config.port == 8765


def test_from_env_reads_set_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.1")
    monkeypatch.setenv("STT_DEVICE", "cuda")
    monkeypatch.setenv("PORT", "9000")

    config = Config.from_env()

    assert config.anthropic_api_key == "sk-ant-test"
    assert config.default_model == "gpt-5.1"
    assert config.stt_device == "cuda"
    assert config.port == 9000


def test_config_defaults_have_no_github_or_calendar_integration():
    config = Config()
    assert config.github_token is None
    assert config.github_repos == []
    assert config.google_calendar_credentials_path is None
    assert config.event_poll_interval_seconds == 180


def test_config_from_env_reads_github_repos_as_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
    monkeypatch.setenv("GITHUB_REPOS", "acme/web, acme/api")
    config = Config.from_env()
    assert config.github_token == "ghp_fake"
    assert config.github_repos == ["acme/web", "acme/api"]


def test_repr_redacts_every_secret_shaped_field():
    config = Config(
        anthropic_api_key="sk-ant-secret",
        openai_api_key="sk-openai-secret",
        gemini_api_key="gemini-secret",
        admin_password="admin-secret",
        navidrome_password="navidrome-secret",
        github_token="ghp_secret",
    )

    text = repr(config)

    for secret in [
        "sk-ant-secret",
        "sk-openai-secret",
        "gemini-secret",
        "admin-secret",
        "navidrome-secret",
        "ghp_secret",
    ]:
        assert secret not in text
    assert text.count("<redacted>") == 6
    # Non-secret fields still show through -- redaction shouldn't turn
    # this into an opaque blob that's useless for debugging.
    assert "default_model='claude-sonnet-5'" in text


def test_repr_leaves_unset_secret_fields_as_none_not_redacted():
    config = Config()
    assert "github_token=None" in repr(config)
    assert "<redacted>" not in repr(config)
