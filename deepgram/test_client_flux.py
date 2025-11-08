#!/usr/bin/env python3
"""
Real-time Flux streaming test client for Deepgram Wyoming Server

This script connects to the Deepgram server and tests Flux model with:
- Real-time microphone input
- Streaming audio chunks as they're captured
- Live transcript updates as they arrive
- Perfect for testing Flux's streaming capabilities
"""

import asyncio
import argparse
import logging
import sys
import time
import signal
from typing import Optional

try:
    from wyoming.client import AsyncClient
    from wyoming.event import Event
except ImportError:
    print("Error: wyoming package not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    try:
        import sounddevice as sd
        import numpy as np
        SOUNDDEVICE_AVAILABLE = True
        PYAUDIO_AVAILABLE = False
    except ImportError:
        print("Error: Need either pyaudio or sounddevice for microphone input.")
        print("Install with: pip install pyaudio")
        print("  OR: pip install sounddevice")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    logger.info("\n🛑 Shutting down...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)

class MicrophoneCapture:
    """Capture audio from microphone"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = None
        self.stream = None
        
    def __enter__(self):
        if PYAUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        else:
            # sounddevice will be used in read_chunk
            pass
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
    
    def read_chunk(self) -> Optional[bytes]:
        """Read a chunk of audio data"""
        global shutdown_flag
        if shutdown_flag:
            return None
            
        if PYAUDIO_AVAILABLE:
            if self.stream:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    return data
                except OSError as e:
                    if "Input overflowed" in str(e):
                        # This is usually fine, just skip this chunk
                        return b'\x00' * (self.chunk_size * 2)  # Return silence
                    logger.error(f"Error reading from microphone: {e}")
                    logger.error("💡 On macOS, check System Settings > Privacy & Security > Microphone")
                    return None
                except Exception as e:
                    logger.error(f"Error reading from microphone: {e}")
                    if "Permission denied" in str(e) or "access denied" in str(e).lower():
                        logger.error("💡 Microphone permission denied!")
                        logger.error("   On macOS: System Settings > Privacy & Security > Microphone > Terminal (or Python)")
                    return None
        else:
            # Use sounddevice
            try:
                samples = sd.rec(
                    self.chunk_size,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='int16'
                )
                sd.wait()
                return samples.tobytes()
            except Exception as e:
                logger.error(f"Error reading from microphone: {e}")
                if "Permission denied" in str(e) or "access denied" in str(e).lower():
                    logger.error("💡 Microphone permission denied!")
                    logger.error("   On macOS: System Settings > Privacy & Security > Microphone > Terminal (or Python)")
                return None
    
    def get_audio_level(self, chunk: bytes) -> float:
        """Calculate audio level (RMS) from chunk"""
        if not chunk:
            return 0.0
        
        if PYAUDIO_AVAILABLE:
            import struct
            samples = struct.unpack(f'{len(chunk)//2}h', chunk)
            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            return rms / 32768.0  # Normalize to 0-1
        else:
            import numpy as np
            samples = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(samples**2))
            return rms / 32768.0  # Normalize to 0-1

async def test_flux_streaming(uri: str, sample_rate: int = 16000, chunk_size: int = 1024):
    """Test Flux streaming with real-time microphone input"""
    global shutdown_flag
    
    logger.info(f"Connecting to {uri}...")
    logger.info("🎤 Starting real-time Flux streaming test")
    logger.info("📝 Speak into your microphone. Transcripts will appear in real-time.")
    logger.info("⏹️  Press Ctrl+C to stop\n")
    
    async with AsyncClient.from_uri(uri) as client:
        logger.info("✅ Connected to server")
        
        # Send audio-start event (optional, but good practice)
        start_event = Event(type="audio-start", data={"rate": sample_rate, "width": 2, "channels": 1})
        await client.write_event(start_event)
        logger.info("📤 Sent audio-start event")
        
        # Start capturing audio
        try:
            with MicrophoneCapture(sample_rate=sample_rate, chunk_size=chunk_size) as mic:
                logger.info("🎤 Microphone ready. Start speaking...\n")
                
                # Test microphone first
                logger.info("🔍 Testing microphone...")
                test_chunk = mic.read_chunk()
                if test_chunk:
                    test_level = mic.get_audio_level(test_chunk)
                    if test_level < 0.001:
                        logger.warning("⚠️  Microphone appears silent. Check:")
                        logger.warning("   - Microphone permissions (macOS: System Settings > Privacy & Security > Microphone)")
                        logger.warning("   - Microphone is connected and not muted")
                        logger.warning("   - Try speaking louder")
                    else:
                        logger.info(f"✅ Microphone working (level: {test_level*100:.1f}%)")
                else:
                    logger.error("❌ Failed to read from microphone!")
                    logger.error("💡 On macOS, grant microphone permission:")
                    logger.error("   System Settings > Privacy & Security > Microphone > Terminal (or Python)")
                    return
                
                logger.info("")
                
                # Task to read transcript events
                transcript_task = asyncio.create_task(read_transcripts(client))
                
                # Task to send audio chunks
                audio_task = asyncio.create_task(send_audio_chunks(client, mic, show_levels=True))
                
                try:
                    # Wait for either task to complete or shutdown
                    done, pending = await asyncio.wait(
                        [transcript_task, audio_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # If audio task completed, cancel it (we're done sending)
                    if audio_task in done:
                        logger.info("📤 Finished sending audio")
                    else:
                        # Audio task is still pending, cancel it
                        audio_task.cancel()
                        try:
                            await audio_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Don't cancel transcript task - we need it to receive the response
                            
                except KeyboardInterrupt:
                    logger.info("\n🛑 Interrupted by user")
                finally:
                    # Cancel audio task if still running
                    if not audio_task.done():
                        audio_task.cancel()
                        try:
                            await audio_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Send audio-stop event to trigger transcription
                    stop_event = Event(type="audio-stop")
                    await client.write_event(stop_event)
                    logger.info("📤 Sent audio-stop event")
                    logger.info("⏳ Waiting for transcription (this may take 20-30 seconds for Flux)...")
                    
                    # Wait longer for transcript - Flux can take 20-30 seconds
                    # The transcript reader task should still be running
                    if not transcript_task.done():
                        try:
                            # Give transcript reader more time to get the response
                            await asyncio.wait_for(transcript_task, timeout=60.0)
                        except asyncio.TimeoutError:
                            logger.warning("⏱️  Timeout waiting for transcript")
                        except asyncio.CancelledError:
                            # Task was cancelled, that's OK
                            pass
                    else:
                        logger.info("📥 Transcript task already completed")
                    
                    logger.info("\n✅ Streaming test complete")
        except Exception as e:
            logger.error(f"❌ Failed to initialize microphone: {e}")
            if "Permission denied" in str(e) or "access denied" in str(e).lower():
                logger.error("\n💡 MICROPHONE PERMISSION REQUIRED!")
                logger.error("   On macOS:")
                logger.error("   1. Open System Settings")
                logger.error("   2. Go to Privacy & Security")
                logger.error("   3. Click Microphone")
                logger.error("   4. Enable Terminal (or Python)")
                logger.error("   5. Restart this script")
            return

async def send_audio_chunks(client: AsyncClient, mic: MicrophoneCapture, show_levels: bool = True):
    """Send audio chunks as they're captured"""
    global shutdown_flag
    chunk_count = 0
    silent_chunks = 0
    max_level = 0.0
    
    try:
        while not shutdown_flag:
            chunk = mic.read_chunk()
            if chunk is None:
                if shutdown_flag:
                    break
                await asyncio.sleep(0.01)
                continue
            
            # Check audio level
            level = mic.get_audio_level(chunk)
            max_level = max(max_level, level)
            
            # Warn if audio seems too quiet
            if level < 0.001:  # Very quiet
                silent_chunks += 1
                if silent_chunks == 100:  # ~6 seconds of silence
                    logger.warning("⚠️  No audio detected! Check:")
                    logger.warning("   1. Microphone is connected and working")
                    logger.warning("   2. Microphone permissions are granted")
                    logger.warning("   3. Speak louder or move closer to mic")
                    silent_chunks = 0  # Reset warning
            else:
                silent_chunks = 0
            
            # Show audio level indicator
            if show_levels and chunk_count % 10 == 0:  # Every ~0.6 seconds
                bar_length = int(level * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                level_pct = level * 100
                print(f"\r🎤 Audio Level: [{bar}] {level_pct:5.1f}%  ", end="", flush=True)
            
            # Send audio chunk
            audio_event = Event(
                type="audio-chunk",
                payload=chunk
            )
            await client.write_event(audio_event)
            chunk_count += 1
            
            # Small delay to prevent overwhelming the server
            await asyncio.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Error sending audio chunks: {e}")
    finally:
        print()  # New line after audio level indicator
        logger.info(f"📤 Finished sending {chunk_count} audio chunks (max level: {max_level*100:.1f}%)")

async def read_transcripts(client: AsyncClient):
    """Read and display transcript events as they arrive"""
    global shutdown_flag
    transcript_count = 0
    last_transcript = ""
    shutdown_time = None
    max_wait_after_shutdown = 60.0  # Wait up to 60 seconds after shutdown
    
    try:
        # Keep reading until we get a transcript or timeout
        while True:
            try:
                # Wait for transcript event
                timeout = 1.0  # Short timeout for active reading
                event = await asyncio.wait_for(client.read_event(), timeout=timeout)
                
                if event and event.type == "transcript":
                    text = event.data.get("text", "") if event.data else ""
                    if text and text != last_transcript:
                        transcript_count += 1
                        last_transcript = text
                        
                        # Display transcript
                        print(f"\n{'='*60}")
                        print(f"📝 TRANSCRIPT #{transcript_count}:")
                        print(f"{text}")
                        print(f"{'='*60}\n")
                        
                        logger.info(f"✅ Received transcript #{transcript_count} ({len(text)} chars)")
                        
                        # Continue reading for more transcripts (server may send updates)
                    elif text == last_transcript:
                        # Duplicate transcript, ignore
                        pass
                elif event:
                    logger.debug(f"Received event: {event.type}")
                    # Continue waiting for transcript
                    
            except asyncio.TimeoutError:
                # No event received in this iteration
                if shutdown_flag:
                    # Track when shutdown happened
                    if shutdown_time is None:
                        shutdown_time = time.time()
                        if transcript_count == 0:
                            logger.info("⏳ Waiting for transcript...")
                        else:
                            logger.info("⏳ Waiting for additional transcripts...")
                    
                    # Check how long we've been waiting since shutdown
                    elapsed = time.time() - shutdown_time
                    
                    # If we've received at least one transcript, wait a bit more for final one
                    # But don't wait as long if we already have a transcript
                    max_wait = max_wait_after_shutdown if transcript_count == 0 else 30.0
                    
                    if elapsed >= max_wait:
                        if transcript_count == 0:
                            logger.warning(f"⏱️  Timeout: No transcript received after {elapsed:.1f}s")
                        else:
                            logger.info(f"⏱️  No more transcripts after {elapsed:.1f}s (received {transcript_count})")
                        break
                    
                    # Log progress every 10 seconds (but not on every iteration)
                    elapsed_int = int(elapsed)
                    if elapsed_int > 0 and elapsed_int % 10 == 0 and elapsed - elapsed_int < 1.0:
                        remaining = max_wait - elapsed
                        if transcript_count == 0:
                            logger.info(f"⏳ Still waiting for transcript... ({elapsed_int}s elapsed, ~{remaining:.0f}s remaining)")
                        else:
                            logger.info(f"⏳ Still waiting for final transcript... ({elapsed_int}s elapsed, ~{remaining:.0f}s remaining)")
                    
                    # Continue waiting - we haven't hit the max wait time yet
                    continue
                else:
                    # Not shutdown yet, continue waiting
                    continue
            except Exception as e:
                logger.error(f"Error reading transcript: {e}")
                if shutdown_flag:
                    if shutdown_time is None:
                        shutdown_time = time.time()
                    elapsed = time.time() - shutdown_time if shutdown_time else 0
                    if elapsed >= max_wait_after_shutdown:
                        break
                await asyncio.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error in transcript reader: {e}")
    finally:
        if transcript_count == 0:
            if shutdown_flag and shutdown_time:
                elapsed = time.time() - shutdown_time
                logger.warning(f"⚠️  No transcripts received after {elapsed:.1f}s. The server may still be processing...")
            else:
                logger.warning("⚠️  No transcripts received.")
        else:
            logger.info(f"📥 Finished reading {transcript_count} transcript(s)")

async def test_flux_with_silence_detection(uri: str, sample_rate: int = 16000, chunk_size: int = 1024, 
                                          silence_threshold: float = 0.01, silence_duration: float = 2.0):
    """Test Flux with automatic silence detection - sends audio-stop after silence"""
    global shutdown_flag
    
    logger.info(f"Connecting to {uri}...")
    logger.info("🎤 Starting Flux streaming with silence detection")
    logger.info(f"📝 Speak into your microphone. Will auto-stop after {silence_duration}s of silence.")
    logger.info("⏹️  Press Ctrl+C to stop manually\n")
    
    async with AsyncClient.from_uri(uri) as client:
        logger.info("✅ Connected to server")
        
        # Send audio-start event
        start_event = Event(type="audio-start", data={"rate": sample_rate, "width": 2, "channels": 1})
        await client.write_event(start_event)
        logger.info("📤 Sent audio-start event")
        
        # Start capturing audio
        with MicrophoneCapture(sample_rate=sample_rate, chunk_size=chunk_size) as mic:
            logger.info("🎤 Microphone ready. Start speaking...\n")
            
            # Task to read transcript events
            transcript_task = asyncio.create_task(read_transcripts(client))
            
            # Track silence
            last_sound_time = time.time()
            chunk_count = 0
            
            try:
                while not shutdown_flag:
                    chunk = mic.read_chunk()
                    if chunk is None:
                        if shutdown_flag:
                            break
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Calculate audio level using mic's method
                    level = mic.get_audio_level(chunk)
                    
                    # Check if sound detected
                    if level > silence_threshold:
                        last_sound_time = time.time()
                    
                    # Send audio chunk
                    audio_event = Event(
                        type="audio-chunk",
                        payload=chunk
                    )
                    await client.write_event(audio_event)
                    chunk_count += 1
                    
                    # Check for silence timeout
                    silence_elapsed = time.time() - last_sound_time
                    if silence_elapsed >= silence_duration and chunk_count > 100:  # At least some audio sent
                        logger.info(f"\n🔇 Silence detected ({silence_elapsed:.1f}s). Sending audio-stop...")
                        break
                    
                    await asyncio.sleep(0.01)
                
                # Send audio-stop event
                stop_event = Event(type="audio-stop")
                await client.write_event(stop_event)
                logger.info("📤 Sent audio-stop event")
                
                # Wait for final transcript
                await asyncio.sleep(3)
                
                # Cancel transcript task
                transcript_task.cancel()
                try:
                    await transcript_task
                except asyncio.CancelledError:
                    pass
                
                logger.info(f"\n✅ Test complete. Sent {chunk_count} audio chunks")
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Interrupted by user")
                transcript_task.cancel()
                try:
                    await transcript_task
                except asyncio.CancelledError:
                    pass

async def main():
    parser = argparse.ArgumentParser(
        description="Real-time Flux streaming test client for Deepgram Wyoming Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic streaming test (press Ctrl+C to stop)
  python3 test_client_flux.py

  # With silence detection (auto-stops after 2s silence)
  python3 test_client_flux.py --silence-detection

  # Custom sample rate and chunk size
  python3 test_client_flux.py --sample-rate 16000 --chunk-size 2048
        """
    )
    parser.add_argument(
        "--uri",
        default="tcp://localhost:10301",
        help="Server URI (default: tcp://localhost:10301)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Audio chunk size in samples (default: 1024)"
    )
    parser.add_argument(
        "--silence-detection",
        action="store_true",
        help="Enable automatic silence detection (stops after 2s of silence)"
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.01,
        help="Silence detection threshold 0-1 (default: 0.01)"
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=2.0,
        help="Silence duration in seconds before auto-stop (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.silence_detection:
            await test_flux_with_silence_detection(
                args.uri,
                args.sample_rate,
                args.chunk_size,
                args.silence_threshold,
                args.silence_duration
            )
        else:
            await test_flux_streaming(
                args.uri,
                args.sample_rate,
                args.chunk_size
            )
    
    except ConnectionRefusedError:
        logger.error(f"❌ Connection refused. Is the server running on {args.uri}?")
        logger.error("   Start the server with: python3 deepgram_server.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

