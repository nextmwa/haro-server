# haro-server

AI backend server for the Haro personal desk robot: a FastAPI WebSocket
service that transcribes speech (Parakeet), gets a reply from
Claude/OpenAI/Gemini (LiteLLM), and synthesizes speech (Kokoro).

## Local development

```
pip install -e ".[dev]"
pytest
python -m haro_server.main
```

Requires **Python 3.11 or 3.12**. Python 3.13+ is not supported, because
`kokoro`'s and `nemo_toolkit`'s dependencies don't yet support it — hence
the `requires-python = ">=3.11,<3.13"` bound in `pyproject.toml`, which
turns that into an explicit resolution error rather than a silent pip
backtrack to an ancient `kokoro`.

Copy `.env.example` to `.env` and fill in the API key matching your
`DEFAULT_MODEL` (see that file's note on Gemini model strings).
