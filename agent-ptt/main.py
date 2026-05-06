"""agent-ptt entry point.

Starts three concurrent async tasks:
  1. FastAPI REST server (uvicorn) — receives text from PA via POST /speak
  2. Zello WebSocket client — maintains PTT channel connection
  3. Bridge speak task — TTS + Zello transmission for queued text
"""
import asyncio
import logging

import uvicorn

from api import create_app
from bridge import Bridge
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)


async def main() -> None:
    bridge = Bridge()
    app = create_app(bridge)

    server = uvicorn.Server(
        uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=cfg.api_port,
            loop="asyncio",
            log_level="info",
        )
    )

    await asyncio.gather(
        server.serve(),
        bridge.zello.run(),
        bridge.speak_task(),
    )


if __name__ == "__main__":
    asyncio.run(main())
