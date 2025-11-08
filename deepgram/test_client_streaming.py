#!/usr/bin/env python3
"""
Test client for Deepgram streaming STT (like Home Assistant).
Tests the streaming mode with audio-start, audio-chunk, and audio-stop events.
"""

import asyncio
import argparse
import logging
import time
from pathlib import Path
from wyoming.client import AsyncClient
from wyoming.event import Event

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global flag for shutdown
shutdown_flag = False


class MicrophoneCapture:
    """Capture audio from microphone for real-time streaming"""
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = None
        self.stream = None
        
        # Try pyaudio first, then sounddevice
        try:
            import pyaudio
            self.audio = pyaudio.PyAudio()
            self.use_pyaudio = True
            logger.info("Using pyaudio for microphone input")
        except ImportError:
            try:
                import sounddevice as sd
                self.use_pyaudio = False
                logger.info("Using sounddevice for microphone input")
            except ImportError:
                raise ImportError("Neither pyaudio nor sounddevice is installed. Install one: pip install pyaudio OR pip install sounddevice")
    
    def __enter__(self):
        if self.use_pyaudio:
            self.stream = self.audio.open(
                format=self.audio.get_format_from_width(2),  # 16-bit
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        else:
            import sounddevice as sd
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.chunk_size
            )
            self.stream.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            if self.use_pyaudio:
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()
            else:
                self.stream.stop()
                self.stream.close()
    
    def read_chunk(self) -> bytes:
        """Read a chunk of audio data"""
        global shutdown_flag
        if shutdown_flag:
            return None
        
        try:
            if self.use_pyaudio:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                return data
            else:
                import sounddevice as sd
                data, overflowed = self.stream.read(self.chunk_size)
                if overflowed:
                    logger.warning("Audio buffer overflow")
                return data.tobytes()
        except OSError as e:
            if "Input overflowed" in str(e) or "Input not available" in str(e):
                logger.warning(f"Audio input error: {e}")
                return None
            raise
        except Exception as e:
            logger.error(f"Error reading audio: {e}")
            return None
    
    def get_audio_level(self, chunk: bytes) -> float:
        """Calculate RMS audio level (0.0 to 1.0)"""
        try:
            import numpy as np
            samples = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(samples**2))
            return rms / 32768.0  # Normalize to 0-1
        except ImportError:
            # Fallback without numpy
            import struct
            samples = struct.unpack(f'<{len(chunk)//2}h', chunk)
            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            return rms / 32768.0


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
        logger.error(f"Error sending audio: {e}", exc_info=True)
    finally:
        print()  # New line after audio level indicator
        logger.info(f"📤 Finished sending {chunk_count} audio chunks (max level: {max_level*100:.1f}%)")


async def send_audio_file(client: AsyncClient, audio_file: str, sample_rate: int = 16000, chunk_size: int = 4096):
    """Send audio file in chunks (for testing without microphone)"""
    audio_path = Path(audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    logger.info(f"Reading audio file: {audio_file}")
    audio_data = audio_path.read_bytes()
    logger.info(f"Read {len(audio_data)} bytes of audio")
    
    # Strip WAV header if present
    if audio_data[:4] == b'RIFF':
        # Find 'data' chunk
        data_chunk_pos = audio_data.find(b'data', 12)
        if data_chunk_pos != -1:
            audio_data = audio_data[data_chunk_pos + 8:]
            logger.info(f"Stripped WAV header, sending {len(audio_data)} bytes of PCM")
        else:
            audio_data = audio_data[44:]  # Fallback
    
    # Send in chunks
    total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
    logger.info(f"Sending audio in {total_chunks} chunk(s)...")
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        if chunk:
            audio_event = Event(
                type="audio-chunk",
                payload=chunk
            )
            await client.write_event(audio_event)
            await asyncio.sleep(0.01)  # Small delay to simulate real-time
    
    logger.info("📤 Finished sending audio file")


async def read_transcripts(client: AsyncClient):
    """Read and display transcript events as they arrive"""
    global shutdown_flag
    transcript_count = 0
    last_transcript = ""
    
    try:
        while True:
            try:
                # Wait for transcript event
                timeout = 1.0
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
                
                elif event:
                    logger.debug(f"Received event: {event.type}")
                    
            except asyncio.TimeoutError:
                if shutdown_flag:
                    # Wait a bit more after shutdown for final transcript
                    await asyncio.sleep(2.0)
                    break
                continue
            except Exception as e:
                logger.error(f"Error reading transcript: {e}", exc_info=True)
                if shutdown_flag:
                    break
                await asyncio.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error in transcript reader: {e}", exc_info=True)
    finally:
        if transcript_count == 0:
            logger.warning("⚠️  No transcripts received.")
        else:
            logger.info(f"📥 Finished reading {transcript_count} transcript(s)")


async def test_streaming(uri: str, audio_file: str = None, sample_rate: int = 16000, chunk_size: int = 1024):
    """Test streaming STT with microphone or audio file"""
    global shutdown_flag
    
    logger.info(f"Connecting to {uri}...")
    logger.info("🎤 Starting streaming STT test")
    logger.info("📝 This tests the streaming mode (audio-start → audio-chunk → audio-stop)")
    logger.info("⏹️  Press Ctrl+C to stop\n")
    
    async with AsyncClient.from_uri(uri) as client:
        logger.info("✅ Connected to server")
        
        # Send audio-start event (required for streaming mode)
        start_event = Event(
            type="audio-start",
            data={"rate": sample_rate, "width": 2, "channels": 1}
        )
        await client.write_event(start_event)
        logger.info("📤 Sent audio-start event")
        
        # Start transcript reader task
        transcript_task = asyncio.create_task(read_transcripts(client))
        
        try:
            if audio_file:
                # Test with audio file
                logger.info(f"📁 Using audio file: {audio_file}")
                await send_audio_file(client, audio_file, sample_rate, chunk_size)
            else:
                # Test with microphone
                logger.info("🎤 Using microphone input")
                logger.info("📝 Speak into your microphone. Transcripts will appear in real-time.\n")
                
                try:
                    with MicrophoneCapture(sample_rate=sample_rate, chunk_size=chunk_size) as mic:
                        # Test microphone
                        logger.info("🔍 Testing microphone...")
                        test_chunk = mic.read_chunk()
                        if test_chunk:
                            level = mic.get_audio_level(test_chunk)
                            if level < 0.001:
                                logger.warning("⚠️  Microphone appears silent. Check:")
                                logger.warning("   1. Microphone permissions (macOS: System Settings > Privacy & Security > Microphone)")
                                logger.warning("   2. Microphone is connected and not muted")
                                logger.warning("   3. Try speaking louder")
                            else:
                                logger.info(f"✅ Microphone working (level: {level*100:.1f}%)")
                        
                        logger.info("🎤 Microphone ready. Start speaking...\n")
                        
                        # Send audio chunks
                        audio_task = asyncio.create_task(send_audio_chunks(client, mic, show_levels=True))
                        
                        # Wait for user to stop (Ctrl+C)
                        try:
                            await audio_task
                        except KeyboardInterrupt:
                            logger.info("\n🛑 Interrupted by user")
                            shutdown_flag = True
                            
                except ImportError as e:
                    logger.error(f"❌ Microphone library not available: {e}")
                    logger.error("Install one of: pip install pyaudio OR pip install sounddevice")
                    return
                except Exception as e:
                    logger.error(f"❌ Microphone error: {e}", exc_info=True)
                    logger.error("\n📋 Troubleshooting:")
                    logger.error("   1. Check microphone permissions")
                    logger.error("   2. Ensure microphone is connected")
                    logger.error("   3. Try: python3 check_mic_permissions.py")
                    return
            
            # Send audio-stop event to finalize
            logger.info("\n📤 Sending audio-stop event...")
            stop_event = Event(type="audio-stop")
            await client.write_event(stop_event)
            shutdown_flag = True
            
            # Wait for transcript task to finish
            logger.info("⏳ Waiting for final transcript...")
            try:
                await asyncio.wait_for(transcript_task, timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning("⏱️  Timeout waiting for transcript")
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Interrupted by user")
            shutdown_flag = True
        finally:
            if not transcript_task.done():
                transcript_task.cancel()
                try:
                    await transcript_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("\n✅ Streaming test complete")


def main():
    parser = argparse.ArgumentParser(description="Test Deepgram streaming STT")
    parser.add_argument(
        "--uri",
        type=str,
        default="tcp://localhost:10301",
        help="Wyoming server URI (default: tcp://localhost:10301)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Audio file to test with (WAV format). If not provided, uses microphone."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Audio chunk size in bytes (default: 1024)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(test_streaming(
            uri=args.uri,
            audio_file=args.audio,
            sample_rate=args.sample_rate,
            chunk_size=args.chunk_size
        ))
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")


if __name__ == "__main__":
    main()

