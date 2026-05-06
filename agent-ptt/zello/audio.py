"""Opus <-> WAV conversion utilities for Zello audio streams.

Zello uses Opus codec inside its binary WebSocket frames. The codec header
is a 4-byte little-endian structure: [sample_rate(u16), frames_per_packet(u8),
frame_duration_ms(u8)].
"""
import base64
import io
import logging
import struct
from math import gcd

import numpy as np
import opuslib
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

ZELLO_SAMPLE_RATE = 16000
ZELLO_CHANNELS = 1
ZELLO_FRAME_DURATION_MS = 20
ZELLO_FRAME_SIZE = ZELLO_SAMPLE_RATE * ZELLO_FRAME_DURATION_MS // 1000  # 320 samples


def make_codec_header(
    sample_rate: int = ZELLO_SAMPLE_RATE,
    frames_per_packet: int = 1,
    frame_duration_ms: int = ZELLO_FRAME_DURATION_MS,
) -> str:
    """Return base64-encoded Zello Opus codec header."""
    header = struct.pack("<HBB", sample_rate, frames_per_packet, frame_duration_ms)
    return base64.b64encode(header).decode()


def parse_codec_header(header_b64: str) -> tuple[int, int, int]:
    """Parse base64 Zello codec header → (sample_rate, frames_per_packet, frame_duration_ms)."""
    data = base64.b64decode(header_b64)
    sample_rate, frames_per_packet, frame_duration_ms = struct.unpack("<HBB", data[:4])
    return sample_rate, frames_per_packet, frame_duration_ms


def wav_to_opus_frames(wav_bytes: bytes) -> tuple[str, list[bytes]]:
    """Convert WAV bytes to (codec_header_b64, [opus_frame_bytes, ...]).

    Resamples to ZELLO_SAMPLE_RATE Hz mono and encodes with Opus at
    ZELLO_FRAME_DURATION_MS ms per frame.
    """
    buf = io.BytesIO(wav_bytes)
    data, src_rate = sf.read(buf, dtype="int16", always_2d=False)

    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.int16)

    if src_rate != ZELLO_SAMPLE_RATE:
        g = gcd(ZELLO_SAMPLE_RATE, src_rate)
        up, down = ZELLO_SAMPLE_RATE // g, src_rate // g
        resampled = resample_poly(data.astype(np.float32), up, down)
        data = np.clip(resampled, -32768, 32767).astype(np.int16)

    encoder = opuslib.Encoder(ZELLO_SAMPLE_RATE, ZELLO_CHANNELS, opuslib.APPLICATION_VOIP)
    frames: list[bytes] = []
    pcm = data.tobytes()
    frame_bytes = ZELLO_FRAME_SIZE * 2  # 2 bytes per int16 sample

    for i in range(0, len(pcm), frame_bytes):
        chunk = pcm[i : i + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        frames.append(encoder.encode(chunk, ZELLO_FRAME_SIZE))

    logger.debug("WAV→Opus: %d frames (%d Hz → %d Hz)", len(frames), src_rate, ZELLO_SAMPLE_RATE)
    return make_codec_header(), frames


def opus_frames_to_wav(frames: list[bytes], sample_rate: int, frame_duration_ms: int) -> bytes:
    """Decode a list of Opus frames to WAV bytes."""
    frame_size = sample_rate * frame_duration_ms // 1000
    decoder = opuslib.Decoder(sample_rate, ZELLO_CHANNELS)
    pcm_chunks: list[bytes] = []

    for frame in frames:
        try:
            pcm_chunks.append(decoder.decode(frame, frame_size))
        except opuslib.OpusError as e:
            logger.warning("Opus decode error (skipping frame): %s", e)

    if not pcm_chunks:
        return b""

    pcm = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    logger.debug("Opus→WAV: %d frames decoded", len(frames))
    return buf.getvalue()
