import dataclasses
import os

_SECRET_FIELD_SUFFIXES = ("_token", "_key", "_password")


def _looks_like_a_secret_field(field_name: str) -> bool:
    return field_name.endswith(_SECRET_FIELD_SUFFIXES)


@dataclasses.dataclass
class Config:
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    default_model: str = "claude-sonnet-5"
    stt_device: str = "cpu"
    tts_device: str = "cpu"
    # "kokoro", "chatterbox", or "pockettts" -- see server.py's
    # construction site and tts_common.py's module docstring for why
    # swapping this is a one-line config change, not a code change.
    # Defaults to "kokoro" (the original, lighter engine) so nothing
    # changes for a deployment that doesn't set this explicitly.
    tts_engine: str = "kokoro"
    host: str = "0.0.0.0"
    port: int = 8765
    db_path: str = "haro.db"
    # Protects the /admin web UI (system prompt editor, transcript log,
    # long-term memory browser) with HTTP Basic Auth -- see admin.py. None
    # means "no admin UI credentials configured"; server.py refuses to
    # mount /admin in that case rather than serving it unprotected.
    admin_password: str | None = None
    # Navidrome music playback (navidrome.py, session.py's _play_music()).
    # All three must be set for it to activate -- server.py treats any
    # missing piece as "not configured" and music requests get a plain
    # error reply instead.
    navidrome_url: str | None = None
    navidrome_username: str | None = None
    navidrome_password: str | None = None
    # GitHub/Calendar proactive polling (event_poller.py) and GitHub MCP
    # tool-calling (mcp_client.py) -- unset github_token means both are
    # inactive, matching navidrome_*'s "unset = disabled" pattern above.
    github_token: str | None = None
    github_repos: list[str] = dataclasses.field(default_factory=list)
    # Path to a Google OAuth credentials/token file (see event_poller.py's
    # poll_calendar() and the design spec's Configuration section) -- unset
    # means Calendar polling is inactive.
    google_calendar_credentials_path: str | None = None
    event_poll_interval_seconds: int = 180

    def __repr__(self) -> str:
        # The default dataclass __repr__ would print every secret field
        # (github_token, anthropic_api_key, navidrome_password, ...) in
        # full if this object is ever logged (e.g. an unhandled exception
        # that includes local variables, or a stray `logger.debug(config)`
        # somewhere). Redact anything field-name-shaped like a secret
        # instead of trusting every call site to remember not to log it.
        parts = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if _looks_like_a_secret_field(field.name) and value:
                value = "<redacted>"
            parts.append(f"{field.name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    __str__ = __repr__

    @staticmethod
    def from_env() -> "Config":
        return Config(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
            default_model=os.environ.get("DEFAULT_MODEL", "claude-sonnet-5"),
            stt_device=os.environ.get("STT_DEVICE", "cpu"),
            tts_device=os.environ.get("TTS_DEVICE", "cpu"),
            tts_engine=os.environ.get("TTS_ENGINE", "kokoro"),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8765")),
            db_path=os.environ.get("DB_PATH", "haro.db"),
            admin_password=os.environ.get("ADMIN_PASSWORD") or None,
            navidrome_url=os.environ.get("NAVIDROME_URL") or None,
            navidrome_username=os.environ.get("NAVIDROME_USERNAME") or None,
            navidrome_password=os.environ.get("NAVIDROME_PASSWORD") or None,
            github_token=os.environ.get("GITHUB_TOKEN") or None,
            github_repos=[r.strip() for r in os.environ.get("GITHUB_REPOS", "").split(",") if r.strip()],
            google_calendar_credentials_path=os.environ.get("GOOGLE_CALENDAR_CREDENTIALS_PATH") or None,
            event_poll_interval_seconds=int(os.environ.get("EVENT_POLL_INTERVAL_SECONDS", "180")),
        )
