#!/usr/bin/env python3
"""
Test client for Deepgram Wyoming Server

This script connects to the Deepgram server and tests transcription.
It can either:
1. Send a describe event to get server info
2. Send audio data from a WAV file for transcription
3. Generate test audio (silence) to verify connection
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path

try:
    from wyoming.client import AsyncClient
    from wyoming.event import Event
except ImportError:
    print("Error: wyoming package not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

async def test_describe(uri: str):
    """Test the describe event to get server information."""
    logger.info(f"Connecting to {uri}...")
    
    async with AsyncClient.from_uri(uri) as client:
        logger.info("✅ Connected to server")
        
        # Send describe event
        describe_event = Event(type="describe")
        await client.write_event(describe_event)
        logger.info("📤 Sent describe event")
        
        # Wait for response
        info_event = await client.read_event()
        if info_event:
            logger.info(f"📥 Received: {info_event.type}")
            if info_event.data:
                import json
                logger.info(f"Server info: {json.dumps(info_event.data, indent=2)}")
        else:
            logger.warning("No response received")

async def test_transcription(uri: str, audio_file: str = None, sample_rate: int = 16000):
    """Test transcription with audio data."""
    logger.info(f"Connecting to {uri}...")
    
    async with AsyncClient.from_uri(uri) as client:
        logger.info("✅ Connected to server")
        
        # Read audio file if provided
        audio_data = b""
        if audio_file:
            audio_path = Path(audio_file)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return
            
            logger.info(f"Reading audio file: {audio_file}")
            audio_data = audio_path.read_bytes()
            logger.info(f"Read {len(audio_data)} bytes of audio")
        else:
            # Generate 1 second of silence (16-bit PCM, mono, 16kHz)
            logger.info("Generating 1 second of silence (16kHz, 16-bit, mono)")
            silence_samples = sample_rate * 2  # 2 bytes per sample (16-bit)
            audio_data = b"\x00\x00" * silence_samples
        
        # Send audio data in chunks
        chunk_size = 4096
        total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
        logger.info(f"Sending audio data in {total_chunks} chunk(s)...")
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            audio_event = Event(
                type="audio-chunk",
                payload=chunk
            )
            await client.write_event(audio_event)
        
        # Send audio-stop event to trigger transcription
        logger.info("Sending audio-stop event...")
        stop_event = Event(type="audio-stop")
        
        # Start timing
        start_time = time.time()
        await client.write_event(stop_event)
        
        # Wait for transcript
        logger.info("Waiting for transcript...")
        transcript_received = False
        response_time = None
        
        try:
            # Wait for transcript event with timeout (60s to allow for Flux processing which can take 25-30s)
            transcript_event = await asyncio.wait_for(client.read_event(), timeout=60.0)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            if transcript_event and transcript_event.type == "transcript":
                text = transcript_event.data.get("text", "") if transcript_event.data else ""
                logger.info(f"✅ Transcript received: {text}")
                print(f"\n{'='*60}")
                print(f"TRANSCRIPT: {text}")
                if response_time:
                    print(f"RESPONSE TIME: {response_time:.3f} seconds ({response_time*1000:.1f} ms)")
                print(f"{'='*60}\n")
                transcript_received = True
            else:
                logger.warning(f"Unexpected event type: {transcript_event.type if transcript_event else 'None'}")
        
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            logger.error(f"❌ Timeout waiting for transcript (waited {response_time:.3f}s)")
        
        if not transcript_received:
            logger.warning("No transcript received")

async def interactive_test(uri: str):
    """Interactive test mode."""
    logger.info(f"Connecting to {uri}...")
    
    async with AsyncClient.from_uri(uri) as client:
        logger.info("✅ Connected to server")
        logger.info("\nInteractive mode - Enter commands:")
        logger.info("  'describe' - Get server info")
        logger.info("  'test' - Send test audio for transcription")
        logger.info("  'quit' - Exit")
        logger.info("")
        
        while True:
            try:
                command = input("> ").strip().lower()
                
                if command == "quit" or command == "q":
                    break
                elif command == "describe":
                    describe_event = Event(type="describe")
                    await client.write_event(describe_event)
                    logger.info("📤 Sent describe event")
                    
                    info_event = await client.read_event()
                    if info_event:
                        logger.info(f"📥 Received: {info_event.type}")
                elif command == "test":
                    # Generate 1 second of silence
                    sample_rate = 16000
                    silence_samples = sample_rate * 2
                    audio_data = b"\x00\x00" * silence_samples
                    
                    logger.info("Sending test audio...")
                    chunk_size = 4096
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        audio_event = Event(type="audio-chunk", payload=chunk)
                        await client.write_event(audio_event)
                    
                    stop_event = Event(type="audio-stop")
                    start_time = time.time()
                    await client.write_event(stop_event)
                    logger.info("Waiting for transcript...")
                    
                    try:
                        transcript_event = await asyncio.wait_for(client.read_event(), timeout=60.0)
                        response_time = time.time() - start_time
                        if transcript_event and transcript_event.type == "transcript":
                            text = transcript_event.data.get("text", "") if transcript_event.data else ""
                            logger.info(f"✅ Transcript: {text}")
                            logger.info(f"⏱️  Response time: {response_time:.3f}s ({response_time*1000:.1f}ms)")
                    except asyncio.TimeoutError:
                        response_time = time.time() - start_time
                        logger.error(f"❌ Timeout waiting for transcript (waited {response_time:.3f}s)")
                else:
                    logger.warning(f"Unknown command: {command}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Test client for Deepgram Wyoming Server")
    parser.add_argument(
        "--uri",
        default="tcp://localhost:10301",
        help="Server URI (default: tcp://localhost:10301)"
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Test describe event to get server info"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to WAV audio file to transcribe"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            await interactive_test(args.uri)
        elif args.describe:
            await test_describe(args.uri)
        elif args.audio:
            await test_transcription(args.uri, args.audio, args.sample_rate)
        else:
            # Default: test transcription with silence
            logger.info("No specific test specified. Running default transcription test with silence...")
            await test_transcription(args.uri, None, args.sample_rate)
    
    except ConnectionRefusedError:
        logger.error(f"❌ Connection refused. Is the server running on {args.uri}?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

