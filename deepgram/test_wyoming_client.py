#!/usr/bin/env python3
"""
Wyoming protocol test client - mimics Home Assistant's behavior exactly.

This client connects to the Deepgram Wyoming server, sends audio,
and displays all received transcript events to verify the server is working correctly.
"""

import argparse
import asyncio
import logging
import sys
import wave
from pathlib import Path
from typing import Optional

from wyoming.client import AsyncClient
from wyoming.event import Event

logger = logging.getLogger("wyoming-test")


def load_wav_pcm(path: Path, expected_rate: int = 16000) -> bytes:
    """Load WAV file and return raw PCM audio data."""
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


async def read_events(client: AsyncClient, timeout: float = 10.0):
    """Read and display all events from the Wyoming server."""
    transcripts = []
    start_time = asyncio.get_event_loop().time()
    final_received = False
    final_received_time = None
    grace_period = 0.5  # Wait 0.5s after final transcript to ensure no more events
    
    try:
        while True:
            try:
                event = await asyncio.wait_for(client.read_event(), timeout=1.0)
                elapsed = asyncio.get_event_loop().time() - start_time
                
                if event.type == "transcript":
                    text = event.data.get("text", "") if event.data else ""
                    is_final = event.data.get("final", False) if event.data else False
                    transcripts.append((elapsed, text, is_final))
                    logger.info(
                        f"[{elapsed:.2f}s] 📝 Transcript ({'FINAL' if is_final else 'interim'}): {text}"
                    )
                    
                    # Track when final transcript is received
                    if is_final:
                        final_received = True
                        final_received_time = asyncio.get_event_loop().time()
                else:
                    logger.info(f"[{elapsed:.2f}s] 📨 Event: {event.type} - {event.data}")
                    
            except asyncio.TimeoutError:
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                # If we received a final transcript, check grace period
                if final_received and final_received_time:
                    if current_time - final_received_time >= grace_period:
                        logger.info(f"Final transcript received, stopping after {grace_period}s grace period")
                        break
                
                # Check if we've exceeded total timeout
                if elapsed > timeout:
                    logger.info(f"Timeout after {timeout}s, stopping event reader")
                    break
                continue
                
    except Exception as e:
        logger.error(f"Error reading events: {e}", exc_info=True)
    
    return transcripts


async def send_audio_stream(
    client: AsyncClient,
    pcm: bytes,
    sample_rate: int,
    chunk_ms: float = 80.0,
):
    """Send audio stream to Wyoming server, mimicking Home Assistant's behavior."""
    chunk_size = int(sample_rate * (chunk_ms / 1000.0) * 2)  # 16-bit = 2 bytes per sample

    # Step 1: Send transcribe request
    transcribe_event = Event(
        type="transcribe",
        data={"language": "en"},
    )
    await client.write_event(transcribe_event)
    logger.info("➡️  Sent transcribe request")

    # Step 2: Send audio-start
    await client.write_event(
        Event(
            type="audio-start",
            data={"rate": sample_rate, "width": 2, "channels": 1},
        )
    )
    logger.info("➡️  Sent audio-start")

    # Step 3: Send audio chunks (mimicking real-time streaming)
    chunk_count = 0
    for idx in range(0, len(pcm), chunk_size):
        chunk = pcm[idx : idx + chunk_size]
        if not chunk:
            continue
        await client.write_event(Event(type="audio-chunk", payload=chunk))
        chunk_count += 1
        await asyncio.sleep(chunk_ms / 1000.0)  # Mimic live streaming timing

    logger.info(f"➡️  Sent {chunk_count} audio chunks")

    # Step 4: Send audio-stop
    await client.write_event(Event(type="audio-stop"))
    logger.info("➡️  Sent audio-stop")


async def test_wyoming_server(
    uri: str,
    wav_path: Optional[Path] = None,
    sample_rate: int = 16000,
    chunk_ms: float = 80.0,
    read_timeout: float = 10.0,
):
    """Test Wyoming server with audio file or wait for manual input."""
    async with AsyncClient.from_uri(uri) as client:
        # Start reading events in background
        reader_task = asyncio.create_task(read_events(client, timeout=read_timeout))

        if wav_path:
            # Send audio file
            logger.info(f"Loading audio from: {wav_path}")
            pcm = load_wav_pcm(wav_path, sample_rate)
            logger.info(f"Loaded {len(pcm)} bytes of PCM audio")

            sender_task = asyncio.create_task(
                send_audio_stream(client, pcm, sample_rate, chunk_ms)
            )
            await sender_task
        else:
            # Interactive mode - wait for user to send audio
            logger.info("Interactive mode - send audio through Home Assistant or another client")
            logger.info("Press Ctrl+C to stop")

        # Wait for all events to be received
        try:
            transcripts = await asyncio.wait_for(reader_task, timeout=read_timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for events after {read_timeout}s")
            transcripts = []

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRANSCRIPT SUMMARY:")
        logger.info("=" * 60)
        if transcripts:
            for elapsed, text, is_final in transcripts:
                status = "FINAL" if is_final else "interim"
                logger.info(f"  [{elapsed:.2f}s] [{status}] {text}")
            
            # Show final transcript
            final_transcripts = [t for t in transcripts if t[2]]
            if final_transcripts:
                final_text = final_transcripts[-1][1]
                logger.info(f"\n✅ Final transcript: {final_text}")
            else:
                logger.warning("\n⚠️  No final transcript received!")
                if transcripts:
                    logger.info(f"   Last interim: {transcripts[-1][1]}")
        else:
            logger.error("❌ No transcripts received!")


def main():
    parser = argparse.ArgumentParser(
        description="Test Wyoming Deepgram STT server"
    )
    parser.add_argument(
        "--uri",
        default="tcp://127.0.0.1:10301",
        help="Wyoming server URI (default: %(default)s)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to mono, 16 kHz, 16-bit PCM WAV file (optional - if not provided, waits for external input)",
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
        "--read-timeout",
        type=float,
        default=10.0,
        help="Timeout for reading events in seconds (default: %(default)s)",
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
        datefmt="%H:%M:%S",
    )

    if args.file and not args.file.exists():
        logger.error(f"Audio file not found: {args.file}")
        return 1

    try:
        asyncio.run(
            test_wyoming_server(
                args.uri,
                args.file,
                args.sample_rate,
                args.chunk_ms,
                args.read_timeout,
            )
        )
    except KeyboardInterrupt:
        logger.info("\nCancelled by user")
    except Exception as exc:
        logger.exception(f"Error running test: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

