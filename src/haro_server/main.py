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
    uvicorn.run(app, host=config.host, port=config.port, loop="uvloop")


if __name__ == "__main__":
    main()
