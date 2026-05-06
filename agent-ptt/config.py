import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Required environment variable {key!r} is not set")
    return val


@dataclass(frozen=True)
class Config:
    speaches_base_url: str
    speaches_stt_model: str
    speaches_tts_model: str
    speaches_tts_voice: str
    zello_server_url: str
    zello_auth_token: str
    zello_channel: str
    zello_username: str
    zello_password: str
    pa_webhook_url: str
    api_port: int
    api_key: str


cfg = Config(
    speaches_base_url=_require("SPEACHES_BASE_URL"),
    speaches_stt_model=os.getenv("SPEACHES_STT_MODEL", "Systran/faster-distil-whisper-small.en"),
    speaches_tts_model=os.getenv("SPEACHES_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX"),
    speaches_tts_voice=os.getenv("SPEACHES_TTS_VOICE", "af_heart"),
    zello_server_url=os.getenv("ZELLO_SERVER_URL", "wss://zello.io/ws"),
    zello_auth_token=os.getenv("ZELLO_AUTH_TOKEN", ""),
    zello_channel=_require("ZELLO_CHANNEL"),
    zello_username=_require("ZELLO_USERNAME"),
    zello_password=_require("ZELLO_PASSWORD"),
    pa_webhook_url=_require("PA_WEBHOOK_URL"),
    api_port=int(os.getenv("API_PORT", "8080")),
    api_key=_require("API_KEY"),
)
