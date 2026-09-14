# haro-server

AI backend server for the Haro personal desk robot: a FastAPI WebSocket
service that transcribes speech (Parakeet), gets a reply from
Claude/OpenAI/Gemini (LiteLLM), and synthesizes speech (Kokoro). Talks to
the robot's firmware, [haro-firmware](https://github.com/nextmwa/haro-firmware),
over the WebSocket protocol implemented in `src/haro_server/protocol.py`.

## Setup

Copy `.env.example` to `.env` and fill in the API key matching your
`DEFAULT_MODEL` (see that file's note on Gemini model strings).

```
cp .env.example .env
# edit .env: set DEFAULT_MODEL and the matching *_API_KEY
```

## Run with Docker (recommended)

The dependencies here are large (PyTorch, NeMo, Kokoro -- multi-GB, slow to
install), so Docker is the easiest way to get a consistent environment:

```
docker-compose up --build
```

This exposes the server on `0.0.0.0:8765` and persists downloaded model
weights (Kokoro + Parakeet, several GB) in a named volume so they aren't
re-downloaded on every container recreate. Point `haro-firmware`'s server
URL at `ws://<this-machine's-LAN-IP>:8765`.

After editing `.env`, `docker-compose restart` does **not** reload it --
use `docker-compose up -d --force-recreate` (or `--build` if source also
changed) instead.

## Local development (without Docker)

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

You'll also need the `espeak-ng` and `ffmpeg` system packages installed
(see `Dockerfile` for why).

## Manual testing without a robot

`tools/fake_robot_client.py` speaks the robot's side of the protocol (sends
silence + `end_of_speech`, prints what comes back) for testing the server
in isolation:

```
python tools/fake_robot_client.py ws://localhost:8765/
```

## License

[MIT](LICENSE)
