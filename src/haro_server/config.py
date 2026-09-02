import dataclasses
import os


@dataclasses.dataclass
class Config:
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    default_model: str = "claude-sonnet-5"
    stt_device: str = "cpu"
    tts_device: str = "cpu"
    host: str = "0.0.0.0"
    port: int = 8765

    @staticmethod
    def from_env() -> "Config":
        return Config(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
            default_model=os.environ.get("DEFAULT_MODEL", "claude-sonnet-5"),
            stt_device=os.environ.get("STT_DEVICE", "cpu"),
            tts_device=os.environ.get("TTS_DEVICE", "cpu"),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8765")),
        )
