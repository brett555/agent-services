"""Bridge: orchestrates the PA ↔ Zello message flow.

PA → User:  speak_queue receives text → TTS → Opus → Zello
User → PA:  Zello audio → Opus decode → WAV → STT → PA webhook
"""
import asyncio
import logging

import httpx

from config import cfg
from stt import transcribe
from tts import synthesize
from zello.audio import opus_frames_to_wav, wav_to_opus_frames
from zello.client import ZelloClient

logger = logging.getLogger(__name__)


class Bridge:
    def __init__(self) -> None:
        self.speak_queue: asyncio.Queue[str] = asyncio.Queue()
        self.zello = ZelloClient(on_audio_received=self._on_user_spoke)

    async def _on_user_spoke(self, frames: list[bytes], sample_rate: int, frame_ms: int) -> None:
        """Decode PTT audio, transcribe it, and forward the text to the PA webhook."""
        wav_bytes = opus_frames_to_wav(frames, sample_rate, frame_ms)
        if not wav_bytes:
            logger.warning("Empty audio received from Zello")
            return

        text = await transcribe(wav_bytes)
        if not text:
            logger.warning("STT returned empty result")
            return

        logger.info("User said: %r → forwarding to PA", text)
        await self._post_to_pa(text)

    async def _post_to_pa(self, text: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(cfg.pa_webhook_url, json={"text": text})
                resp.raise_for_status()
                logger.debug("PA webhook responded: %d", resp.status_code)
        except Exception:
            logger.exception("Failed to POST transcription to PA webhook at %s", cfg.pa_webhook_url)

    async def speak_task(self) -> None:
        """Dequeue text from the API, synthesize it, and transmit via Zello."""
        while True:
            text = await self.speak_queue.get()
            try:
                wav_bytes = await synthesize(text)
                if not wav_bytes:
                    logger.warning("TTS produced no audio for: %r", text)
                    continue
                codec_header, frames = wav_to_opus_frames(wav_bytes)
                if frames:
                    await self.zello.send_audio(codec_header, frames)
                else:
                    logger.warning("Audio encoding produced no frames")
            except Exception:
                logger.exception("Error processing speak request: %r", text)
            finally:
                self.speak_queue.task_done()
