"""Zello Channel API WebSocket client.

Protocol: https://github.com/zelloptt/zello-channel-api/blob/main/API.md

Binary audio frame layout (network byte order):
  [type: u8=0x01][stream_id: u32][packet_id: u32][opus_data: bytes]
"""
import asyncio
import json
import logging
import struct
from collections.abc import Callable, Coroutine
from typing import Any

import aiohttp

from config import cfg
from zello.audio import parse_codec_header, ZELLO_FRAME_DURATION_MS

logger = logging.getLogger(__name__)

_AUDIO_TYPE = 0x01
_FRAME_HDR_FMT = ">BII"
_FRAME_HDR_SIZE = struct.calcsize(_FRAME_HDR_FMT)

AudioCallback = Callable[[list[bytes], int, int], Coroutine[Any, Any, None]]


class ZelloClient:
    def __init__(self, on_audio_received: AudioCallback) -> None:
        """
        on_audio_received: async callback(frames, sample_rate, frame_duration_ms)
        Called once per complete PTT transmission.
        """
        self._on_audio_received = on_audio_received
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._seq = 0
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._streams: dict[int, list[bytes]] = {}
        self._stream_meta: dict[int, tuple[int, int]] = {}
        self._send_lock = asyncio.Lock()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def _send_json(self, msg: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Not connected to Zello")
        await self._ws.send_str(json.dumps(msg))

    async def _send_command(self, cmd: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        """Send a command and wait for its seq-matched response."""
        seq = self._next_seq()
        cmd["seq"] = seq
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending[seq] = fut
        await self._send_json(cmd)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(seq, None)
            raise

    def _dispatch_json(self, data: dict[str, Any]) -> None:
        seq = data.get("seq")
        if seq and seq in self._pending:
            fut = self._pending.pop(seq)
            if not fut.done():
                fut.set_result(data)
            return

        command = data.get("command", "")

        if command == "on_channel_status":
            status = data.get("status", "")
            logger.info("Zello channel status: %s (channel=%s)", status, data.get("channel", ""))
            self._connected = status == "online"

        elif command == "on_stream_start":
            stream_id: int = data["stream_id"]
            codec_header = data.get("codec_header", "")
            try:
                sample_rate, _, frame_ms = parse_codec_header(codec_header)
            except Exception:
                sample_rate, frame_ms = 16000, ZELLO_FRAME_DURATION_MS
            self._streams[stream_id] = []
            self._stream_meta[stream_id] = (sample_rate, frame_ms)
            logger.debug("Stream start: id=%d, rate=%d, frame_ms=%d", stream_id, sample_rate, frame_ms)

        elif command == "on_stream_stop":
            stream_id = data["stream_id"]
            frames = self._streams.pop(stream_id, [])
            meta = self._stream_meta.pop(stream_id, (16000, ZELLO_FRAME_DURATION_MS))
            logger.debug("Stream stop: id=%d, frames=%d", stream_id, len(frames))
            if frames:
                task = asyncio.create_task(self._on_audio_received(frames, meta[0], meta[1]))
                task.add_done_callback(self._log_task_error)

        elif command == "on_error":
            logger.error("Zello server error: %s", data.get("error", "unknown"))

        else:
            logger.debug("Unhandled Zello command: %s", command)

    @staticmethod
    def _log_task_error(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            logger.error("Audio callback raised: %s", task.exception())

    def _handle_binary(self, data: bytes) -> None:
        if len(data) < _FRAME_HDR_SIZE:
            return
        frame_type, stream_id, _packet_id = struct.unpack_from(_FRAME_HDR_FMT, data)
        if frame_type != _AUDIO_TYPE:
            return
        if stream_id in self._streams:
            self._streams[stream_id].append(data[_FRAME_HDR_SIZE:])

    async def _read_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    self._dispatch_json(json.loads(msg.data))
                except json.JSONDecodeError:
                    logger.warning("Non-JSON text from Zello: %s", msg.data[:100])
            elif msg.type == aiohttp.WSMsgType.BINARY:
                self._handle_binary(msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                logger.info("Zello WebSocket closing")
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("Zello WebSocket error: %s", msg.data)
                break

    async def send_audio(self, codec_header: str, frames: list[bytes]) -> None:
        """Transmit Opus frames as a PTT stream on the Zello channel."""
        if not self._ws or self._ws.closed:
            logger.warning("Cannot send audio: Zello not connected")
            return

        async with self._send_lock:
            response = await self._send_command({
                "command": "start_stream",
                "type": "audio",
                "codec": "opus",
                "codec_header": codec_header,
                "packet_duration": ZELLO_FRAME_DURATION_MS,
            })
            if not response.get("success"):
                logger.error("start_stream rejected: %s", response)
                return

            stream_id: int = response["stream_id"]
            logger.debug("Sending %d frames on stream %d", len(frames), stream_id)

            for packet_id, frame in enumerate(frames):
                header = struct.pack(_FRAME_HDR_FMT, _AUDIO_TYPE, stream_id, packet_id)
                await self._ws.send_bytes(header + frame)
                await asyncio.sleep(ZELLO_FRAME_DURATION_MS / 1000)

            seq = self._next_seq()
            await self._send_json({"seq": seq, "command": "stop_stream", "stream_id": stream_id})
            logger.debug("Stream %d complete", stream_id)

    async def run(self) -> None:
        """Connect to Zello and reconnect automatically on failure."""
        while True:
            try:
                await self._connect_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Zello connection lost: %s — reconnecting in 5s", e)
            finally:
                self._connected = False
                self._ws = None
                for fut in self._pending.values():
                    if not fut.done():
                        fut.cancel()
                self._pending.clear()
                self._streams.clear()
                self._stream_meta.clear()
            await asyncio.sleep(5)

    async def _connect_once(self) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(cfg.zello_server_url, heartbeat=30.0) as ws:
                self._ws = ws
                self._seq = 0
                logger.info("Connected to Zello at %s", cfg.zello_server_url)

                reader = asyncio.create_task(self._read_loop(ws))
                try:
                    logon_cmd: dict[str, Any] = {
                        "command": "logon",
                        "channel": cfg.zello_channel,
                        "username": cfg.zello_username,
                        "password": cfg.zello_password,
                    }
                    # auth_token is only required for consumer Zello, not Zello Work
                    if cfg.zello_auth_token:
                        logon_cmd["auth_token"] = cfg.zello_auth_token
                    response = await self._send_command(logon_cmd)
                    if not response.get("success"):
                        logger.error("Zello logon failed: %s", response)
                        reader.cancel()
                        return

                    logger.info("Logged in to Zello channel %r", cfg.zello_channel)
                    await reader
                except asyncio.CancelledError:
                    reader.cancel()
                    raise
                except Exception:
                    reader.cancel()
                    raise
