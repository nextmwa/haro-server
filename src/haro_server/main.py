# src/haro_server/main.py
import logging
import os

import uvicorn
from dotenv import load_dotenv

from .config import Config
from .server import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Running natively (haroctl / launchd) there's no docker-compose
    # env_file to inject .env, so load it here. Never overrides variables
    # already set, so Docker's env_file/environment still win there.
    env_file = os.environ.get("HARO_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
    config = Config.from_env()
    app = create_app(config)
    # Bounded graceful shutdown: by default uvicorn waits indefinitely for
    # in-flight work (a streaming reply, an LLM call) while keeping the port
    # bound, so a restart's new process couldn't bind for ~60s.
    uvicorn.run(app, host=config.host, port=config.port, loop="uvloop", timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
