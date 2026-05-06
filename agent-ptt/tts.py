import logging

from openai import AsyncOpenAI

from config import cfg

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key="speaches", base_url=cfg.speaches_base_url)


async def synthesize(text: str) -> bytes:
    """Convert text to WAV audio bytes via speaches TTS."""
    logger.info("TTS synthesizing: %r", text)
    try:
        async with _client.audio.speech.with_streaming_response.create(
            model=cfg.speaches_tts_model,
            voice=cfg.speaches_tts_voice,
            input=text,
            response_format="wav",
        ) as response:
            return await response.read()
    except Exception:
        logger.exception("TTS synthesis failed")
        return b""
