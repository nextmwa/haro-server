# src/haro_server/main.py
import logging

import uvicorn

from .config import Config
from .server import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = Config.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, loop="uvloop")


if __name__ == "__main__":
    main()
