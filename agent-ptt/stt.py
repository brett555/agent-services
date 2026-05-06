import io
import logging

from openai import AsyncOpenAI

from config import cfg

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key="speaches", base_url=cfg.speaches_base_url)


async def transcribe(wav_bytes: bytes) -> str:
    """Transcribe WAV audio bytes to text via speaches STT."""
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"
    try:
        result = await _client.audio.transcriptions.create(
            model=cfg.speaches_stt_model,
            file=audio_file,
            response_format="text",
        )
        text = result if isinstance(result, str) else result.text
        text = text.strip()
        logger.info("STT result: %r", text)
        return text
    except Exception:
        logger.exception("STT transcription failed")
        return ""
