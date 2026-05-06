"""FastAPI REST server.

Endpoints:
  POST /speak   — PA sends text to be spoken on Zello
  GET  /health  — liveness check
"""
import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from config import cfg

logger = logging.getLogger(__name__)


class SpeakRequest(BaseModel):
    text: str


def create_app(bridge) -> FastAPI:
    app = FastAPI(title="agent-ptt")

    @app.post("/speak", status_code=202)
    async def speak(
        request: SpeakRequest,
        x_api_key: str = Header(default=""),
    ):
        if x_api_key != cfg.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text must not be empty")
        await bridge.speak_queue.put(text)
        logger.info("Queued speak request: %r", text[:80])
        return {"queued": True}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "zello_connected": bridge.zello.connected,
            "speak_queue_size": bridge.speak_queue.qsize(),
        }

    return app
