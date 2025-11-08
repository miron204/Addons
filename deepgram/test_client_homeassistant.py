#!/usr/bin/env python3
"""
Minimal Wyoming client that mirrors Home Assistant's STT flow.

Sequence:
 1. Connect to tcp://host:port
 2. Send `transcribe` request event
 3. Send `audio-start`
 4. Stream `audio-chunk` payloads (raw 16‑bit PCM)
 5. Send `audio-stop`
 6. Print every event returned by the server

Usage:
    python3 test_client_homeassistant.py --file path/to/audio.wav

Notes:
  * Audio file must be mono, 16 kHz, 16-bit little-endian PCM. Resample beforehand.
  * Chunks default to 80 ms (matching Home Assistant and Faster Whisper add-on).
"""

import argparse
import asyncio
import contextlib
import logging
import sys
import wave
from pathlib import Path

from wyoming.client import AsyncClient
from wyoming.event import Event

logger = logging.getLogger("wyoming-test-client")


def load_wav_pcm(path: Path, expected_rate: int) -> bytes:
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()

        if nchannels != 1 or sampwidth != 2 or framerate != expected_rate:
            raise ValueError(
                f"Expected mono, 16-bit PCM @ {expected_rate} Hz. "
                f"Got channels={nchannels}, width={sampwidth}, rate={framerate}"
            )

        frames = wf.readframes(wf.getnframes())
    return frames


async def read_events(client: AsyncClient):
    """Continuously read events from the server until EOF."""
    try:
        while True:
            event = await client.read_event()
            if event is None:
                break
            logger.info("⬅️  %s", event)
    except asyncio.CancelledError:
        logger.debug("Event reader cancelled")
        raise
    except Exception as exc:
        logger.error("Error reading event: %s", exc, exc_info=True)


async def send_audio(
    client: AsyncClient,
    pcm: bytes,
    sample_rate: int,
    chunk_ms: float,
):
    chunk_size = int(sample_rate * (chunk_ms / 1000.0) * 2)  # 16-bit

    transcribe_event = Event(
        type="transcribe",
        data={"language": "en"},
    )
    await client.write_event(transcribe_event)
    logger.info("➡️  Sent transcribe request")

    await client.write_event(
        Event(
            type="audio-start",
            data={"rate": sample_rate, "width": 2, "channels": 1},
        )
    )
    logger.info("➡️  Sent audio-start")

    for idx in range(0, len(pcm), chunk_size):
        chunk = pcm[idx : idx + chunk_size]
        if not chunk:
            continue
        await client.write_event(Event(type="audio-chunk", payload=chunk))
        await asyncio.sleep(chunk_ms / 1000.0)  # mimic live streaming timing

    await client.write_event(Event(type="audio-stop"))
    logger.info("➡️  Sent audio-stop")


async def run(uri: str, wav_path: Path, sample_rate: int, chunk_ms: float):
    pcm = load_wav_pcm(wav_path, sample_rate)

    async with AsyncClient.from_uri(uri) as client:
        reader_task = asyncio.create_task(read_events(client))
        await send_audio(client, pcm, sample_rate, chunk_ms)

        try:
            await asyncio.wait_for(reader_task, timeout=6.0)
        except asyncio.TimeoutError:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task


def main():
    parser = argparse.ArgumentParser(
        description="Home Assistant style Wyoming streaming client"
    )
    parser.add_argument(
        "--uri",
        default="tcp://127.0.0.1:10301",
        help="Wyoming server URI (default: %(default)s)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to mono, 16 kHz, 16-bit PCM WAV file",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the audio (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=80.0,
        help="Chunk duration in milliseconds (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: %(default)s)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.file.exists():
        logger.error("Audio file not found: %s", args.file)
        return 1

    try:
        asyncio.run(run(args.uri, args.file, args.sample_rate, args.chunk_ms))
    except KeyboardInterrupt:
        logger.info("Cancelled by user")
    except Exception as exc:
        logger.exception("Error running client: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

