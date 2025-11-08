import asyncio
import json
import os
import logging
import threading
import struct
from deepgram import DeepgramClient
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.info import Info, Describe, AsrProgram, AsrModel, Attribution

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Support both Docker and local testing
OPTIONS_FILE = os.getenv("OPTIONS_FILE", "/data/options.json")
# Fallback to local test file if Docker path doesn't exist
if not os.path.exists(OPTIONS_FILE):
    local_options = os.path.join(os.path.dirname(__file__), "data", "options.json")
    if os.path.exists(local_options):
        OPTIONS_FILE = local_options

class DeepgramSTT:
    def __init__(self, model: str = "nova-3"):
        deepgram_api_key = load_api_key()
        self.dg_client = DeepgramClient(api_key=deepgram_api_key)
        self.model = model
        self.is_flux = model.startswith("flux")

    async def transcribe(self, audio_data: bytes, sample_rate: int):
        """
        Send audio data to Deepgram and return transcription.
        Try prerecorded API first (faster), fallback to streaming for Flux if needed.
        """
        if self.is_flux:
            # Try prerecorded API first for speed (like nova-3)
            try:
                return await self._transcribe_prerecorded(audio_data, sample_rate)
            except Exception as e:
                logger.warning(f"Prerecorded API failed for Flux: {e}, falling back to streaming")
                return await self._transcribe_flux_streaming(audio_data, sample_rate)
        else:
            return await self._transcribe_prerecorded(audio_data, sample_rate)
    
    async def _transcribe_prerecorded(self, audio_data: bytes, sample_rate: int):
        """Use prerecorded API for non-Flux models."""
        # SDK 5.x uses media.transcribe_file with request as bytes
        # This is a synchronous method, so run it in executor
        loop = asyncio.get_event_loop()
        
        def transcribe_sync():
            # Check if audio_data is WAV (has RIFF header) or raw PCM
            is_wav = audio_data[:4] == b'RIFF'
            
            if is_wav:
                # WAV file - SDK auto-detects format
                response = self.dg_client.listen.v1.media.transcribe_file(
                    request=audio_data,
                    model=self.model,
                    smart_format=True,
                    language='en-US',
                    punctuate=True,
                )
            else:
                # Raw PCM from Wyoming - create WAV header
                # Wyoming sends raw PCM audio, so we need to wrap it in WAV format
                wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                    b'RIFF',
                    36 + len(audio_data),  # File size - 8
                    b'WAVE',
                    b'fmt ',
                    16,  # fmt chunk size
                    1,   # Audio format (PCM)
                    1,   # Number of channels (mono)
                    sample_rate,
                    sample_rate * 2,  # Byte rate
                    2,   # Block align
                    16,  # Bits per sample
                    b'data',
                    len(audio_data)  # Data chunk size
                )
                wav_data = wav_header + audio_data
                
                response = self.dg_client.listen.v1.media.transcribe_file(
                    request=wav_data,
                    model=self.model,
                    smart_format=True,
                    language='en-US',
                    punctuate=True,
                )
            return response
        
        try:
            response = await loop.run_in_executor(None, transcribe_sync)
            
            # SDK 5.x response structure - check the actual type
            if hasattr(response, 'results'):
                # Response object with .results attribute
                channels = response.results.channels
                if channels and len(channels) > 0:
                    alternatives = channels[0].alternatives
                    if alternatives and len(alternatives) > 0:
                        transcript = alternatives[0].transcript
                        return transcript
            elif isinstance(response, dict):
                # Dictionary response
                transcript = response.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
                return transcript
            
            logger.error(f"Unexpected response format: {type(response)}")
            return ""
        except Exception as e:
            logger.error(f"Error in prerecorded transcription: {e}", exc_info=True)
            raise
    
    async def _transcribe_flux_streaming(self, audio_data: bytes, sample_rate: int):
        """Use streaming API for Flux models (v2/listen endpoint)."""
        logger.info(f"Using Flux streaming API (v2/listen) with model: {self.model}")
        
        try:
            # Connect to v2/listen endpoint for Flux using SDK 5.x v2 API
            # Note: v2.connect() returns a sync context manager, so we use regular 'with'
            # but run the blocking operations in an executor
            loop = asyncio.get_event_loop()
            
            def run_flux_transcription():
                """Run the sync Flux transcription in a thread"""
                final_transcript = ""
                transcript_received = threading.Event()
                # Track last time we updated transcript to implement a short debounce
                import time
                last_update_time = time.monotonic()
                
                # Define message handler - accumulate all FINAL transcript segments
                def on_message(message):
                    nonlocal final_transcript, last_update_time
                    try:
                        # Check message type (per official docs)
                        msg_type_name = type(message).__name__
                        msg_type_attr = getattr(message, 'type', None)
                        logger.info(f"📨 Message received: class={msg_type_name}, type={msg_type_attr}")
                        
                        # Filter out connection and error events (these are handled separately)
                        if 'Connected' in msg_type_name or 'Error' in msg_type_name or 'Fatal' in msg_type_name:
                            if 'Error' in msg_type_name or 'Fatal' in msg_type_name:
                                # Log error details
                                error_code = getattr(message, 'code', None)
                                error_desc = getattr(message, 'description', None)
                                logger.error(f"❌ Deepgram error: code={error_code}, description={error_desc}")
                            return
                        
                        # Check if this is a final result (not interim)
                        is_final = getattr(message, 'is_final', None)
                        
                        # For Flux, we only want final transcripts, not interim ones
                        if is_final is False:
                            logger.debug("Ignoring interim transcript")
                            return
                        
                        # Try different message formats for transcript
                        transcript_text = None
                        
                        logger.debug(f"  Message attributes: {[attr for attr in dir(message) if not attr.startswith('_') and not callable(getattr(message, attr, None))]}")
                        
                        # Check for channel.alternatives pattern (most common for v2 API)
                        if hasattr(message, 'channel'):
                            channel = message.channel
                            logger.debug(f"  Found channel: {channel}")
                            if hasattr(channel, 'alternatives') and len(channel.alternatives) > 0:
                                alt = channel.alternatives[0]
                                logger.debug(f"  Alternative: {alt}")
                                transcript_text = getattr(alt, 'transcript', None)
                                logger.info(f"✅ Found transcript in channel.alternatives: {transcript_text}")
                        
                        # Check for direct transcript attribute
                        elif hasattr(message, 'transcript'):
                            transcript_text = message.transcript
                            logger.info(f"✅ Found transcript in message.transcript: {transcript_text}")
                        
                        # Check if message itself is a string or dict
                        elif isinstance(message, str):
                            transcript_text = message
                            logger.info(f"✅ Message is string: {transcript_text}")
                        elif isinstance(message, dict):
                            # Only use final results from dict
                            if message.get('is_final', True):
                                transcript_text = message.get('transcript') or message.get('text')
                                logger.info(f"✅ Message is dict (final): {transcript_text}")
                        
                        # For Flux, each final message is a REPLACEMENT/refinement of the previous transcript
                        # Flux sends progressive updates: "Hello" -> "Hello there" -> "Hello there. I'm Bob" etc.
                        # These are NOT segments to append - they are refinements that replace the previous one
                        if transcript_text and transcript_text.strip():
                            text = transcript_text.strip()
                            
                            # Only update if the new transcript is longer or significantly different
                            # This prevents replacing a longer, more complete transcript with a shorter fragment
                            old_length = len(final_transcript) if final_transcript else 0
                            new_length = len(text)
                            
                            # Update if:
                            # 1. We don't have a transcript yet (old_length == 0)
                            # 2. The new transcript is longer (refinement)
                            # 3. The new transcript is at least 80% of the old length (might be a correction)
                            should_update = (old_length == 0 or 
                                           new_length > old_length or 
                                           (new_length >= old_length * 0.8 and text != final_transcript))
                            
                            if should_update:
                                # Only update if new transcript is at least 80% of old length
                                # This prevents replacing a longer transcript with a short fragment
                                if old_length == 0 or new_length >= old_length * 0.8:
                                    final_transcript = text
                                    last_update_time = time.monotonic()
                                    
                                    if new_length > old_length:
                                        logger.info(f"✅ Transcript refined: {old_length} → {new_length} chars: {text[:100]}{'...' if len(text) > 100 else ''}")
                                    else:
                                        logger.debug(f"Transcript updated: {new_length} chars")
                                else:
                                    # New transcript is much shorter - ignore it, keep the longer one
                                    logger.debug(f"Ignoring shorter fragment: {new_length} chars (keeping {old_length} chars): {text[:50]}...")
                            else:
                                # Ignore shorter transcript fragments
                                logger.debug(f"Ignoring shorter transcript fragment: {new_length} chars (current: {old_length} chars)")
                            
                            # Don't set the event yet - wait for EndOfTurn or stream close
                        else:
                            logger.debug(f"No transcript found in message")
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}", exc_info=True)
                
                with self.dg_client.listen.v2.connect(
                    model=self.model,
                    encoding="linear16",
                    sample_rate=str(sample_rate),  # Deepgram expects string
                ) as connection:
                    logger.info("Connected to Deepgram Flux streaming API")
                    
                    # Set up event handlers using official EventType from documentation
                    def on_close(_):
                        # When connection closes, we have the final transcript
                        logger.debug("Connection closed, using final transcript")
                        transcript_received.set()
                    
                    def on_error(error):
                        error_code = getattr(error, 'code', None)
                        error_desc = getattr(error, 'description', None)
                        logger.error(f"Flux connection error: code={error_code}, description={error_desc}, error={error}")
                        transcript_received.set()
                    
                    # Use EventType from deepgram.core.events as per official docs
                    try:
                        from deepgram.core.events import EventType
                        connection.on(EventType.OPEN, lambda _: logger.debug("Flux connection opened"))
                        connection.on(EventType.MESSAGE, on_message)
                        connection.on(EventType.CLOSE, on_close)
                        connection.on(EventType.ERROR, on_error)
                        # Check for EndOfTurn if available
                        if hasattr(EventType, 'END_OF_TURN'):
                            connection.on(EventType.END_OF_TURN, lambda _: (logger.debug("EndOfTurn received"), transcript_received.set()))
                    except (ImportError, AttributeError) as e:
                        logger.debug(f"Using EventType failed: {e}, trying fallback")
                        # Fallback to string-based event names
                        connection.on("open", lambda _: logger.debug("Flux connection opened"))
                        connection.on("message", on_message)
                        connection.on("close", on_close)
                        connection.on("error", on_error)
                        connection.on("end_of_turn", lambda _: (logger.debug("EndOfTurn received"), transcript_received.set()))
                    
                    # Start listening for messages (per official docs, call before sending)
                    listening_thread = threading.Thread(target=connection.start_listening, daemon=True)
                    listening_thread.start()
                    # Give it a moment to start
                    time.sleep(0.1)
                    
                    # For streaming API, we need raw PCM (not WAV with header)
                    # Check if audio_data is WAV (has RIFF header) or raw PCM
                    is_wav = audio_data[:4] == b'RIFF'
                    if is_wav:
                        # Strip WAV header - find 'data' chunk and extract PCM data
                        data_chunk_pos = audio_data.find(b'data', 12)
                        if data_chunk_pos != -1:
                            # Skip 'data' marker (4 bytes) and size (4 bytes)
                            pcm_data = audio_data[data_chunk_pos + 8:]
                            logger.debug(f"Stripped WAV header, extracted {len(pcm_data)} bytes of raw PCM")
                        else:
                            # Fallback: assume data starts after header (usually at byte 44 for standard WAV)
                            pcm_data = audio_data[44:]
                            logger.debug(f"Using fallback WAV header skip, extracted {len(pcm_data)} bytes of raw PCM")
                    else:
                        # Already raw PCM
                        pcm_data = audio_data
                        logger.debug(f"Audio is already raw PCM: {len(pcm_data)} bytes")
                    
                    # Flux API REQUIRES 80ms chunks (per error message)
                    # Calculate chunk size: 80ms at sample_rate, 16-bit (2 bytes per sample)
                    chunk_size_bytes = int(sample_rate * 0.08 * 2)  # 80ms chunks
                    bytes_sent = 0
                    chunk_count = 0
                    
                    logger.info(f"Sending {len(pcm_data)} bytes of PCM audio in {len(pcm_data) // chunk_size_bytes + 1} chunks (80ms each)...")
                    for i in range(0, len(pcm_data), chunk_size_bytes):
                        chunk = pcm_data[i:i + chunk_size_bytes]
                        if chunk:
                            connection.send_media(chunk)
                            bytes_sent += len(chunk)
                            chunk_count += 1
                            logger.debug(f"Sent chunk {chunk_count}: {len(chunk)} bytes")
                    
                    logger.info(f"Finished sending {chunk_count} chunks, total {bytes_sent} bytes")
                    
                    # Send close control to signal end of stream
                    try:
                        connection.send_control({"type": "CloseStream"})
                    except Exception as e:
                        logger.debug(f"Could not send CloseStream: {e}")
                    
                    logger.info(f"Finished sending audio data, waiting for transcript...")
                    
                    # Wait for first message to arrive (with timeout)
                    # Reduced for prerecorded files since all audio is sent at once
                    initial_wait = 2.0  # Reduced from 3.0s for faster response
                    first_message_timeout = time.monotonic() + initial_wait
                    logger.info(f"Waiting up to {initial_wait}s for first transcript message...")
                    
                    # Wait for at least one message
                    while not final_transcript and time.monotonic() < first_message_timeout:
                        if transcript_received.is_set():
                            break
                        time.sleep(0.1)
                    
                    if not final_transcript:
                        logger.warning("No transcript messages received after initial wait")
                    else:
                        logger.info(f"First transcript received ({len(final_transcript)} chars): {final_transcript[:100]}...")
                    
                    # After receiving at least one message, debounce for final refinements
                    # For prerecorded files (all audio sent at once), use shorter debounce
                    # For real-time streaming, Flux continues refining longer
                    # Since we're sending all audio at once, use shorter debounce for faster response
                    # But keep max_total_wait high enough to allow complete processing of long audio files
                    debounce_seconds = 1.5  # Reduced to 1.5s for prerecorded files (was 4.0s)
                    max_total_wait = 60.0  # Increased to 60s to allow complete processing (was 15.0s, too short)
                    start_wait = time.monotonic()
                    last_logged_length = 0
                    logger.info("Waiting for EndOfTurn or transcript stabilization...")
                    
                    while True:
                        # Prioritize EndOfTurn event - this means Flux is done
                        if transcript_received.is_set():
                            logger.info("EndOfTurn/Close received; returning complete transcript")
                            break
                        
                        now = time.monotonic()
                        elapsed = now - start_wait
                        time_since_update = now - last_update_time if final_transcript else 999
                        
                        # Log when transcript length increases (but not every loop)
                        if final_transcript and len(final_transcript) > last_logged_length:
                            logger.info(f"Transcript updated: {len(final_transcript)} chars (update {time_since_update:.2f}s ago, total wait {elapsed:.1f}s)")
                            last_logged_length = len(final_transcript)
                        
                        # If we have a transcript, wait for debounce period of no updates
                        # This means Flux has stopped sending refinements
                        # Prioritize debounce check - if no updates for debounce period, return immediately
                        if final_transcript and time_since_update >= debounce_seconds:
                            logger.info(f"No updates for {debounce_seconds}s; transcript appears complete ({len(final_transcript)} chars)")
                            break
                        
                        # Safety timeout - only as a last resort if debounce never triggers
                        # This should rarely be hit if Flux is working properly
                        if elapsed >= max_total_wait:
                            logger.warning(f"Max wait time ({max_total_wait}s) reached; returning transcript ({len(final_transcript)} chars)")
                            break
                        
                        time.sleep(0.1)

                    if final_transcript:
                        logger.info(f"Returning complete transcript ({len(final_transcript)} chars)")
                        return final_transcript.strip()
                    else:
                        logger.warning("No transcript received after waiting")
                        return ""
            
            # Run the sync transcription in a thread
            result = await loop.run_in_executor(None, run_flux_transcription)
            
            if not result:
                logger.warning("No transcript received from Flux API")
                return ""
            
            logger.info(f"Flux transcription complete: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in Flux streaming transcription: {e}", exc_info=True)
            # Flux doesn't support prerecorded API, so re-raise the error
            raise

class State:
    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def set_session(self, session_id, data):
        self.sessions[session_id] = data

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

class EventHandler(AsyncEventHandler):
    WYOMING_INFO = Info(
        asr=[
            AsrProgram(
                name="Deepgram",
                description="A speech recognition toolkit",
                attribution=Attribution(
                    name="Deepgram",
                    url="https://deepgram.com",
                ),
                installed=True,
                version='1.0',
                models=[
                    AsrModel(
                        name='general-nova-3',
                        description='Nova 3',
                        attribution=Attribution(
                            name="Deepgram",
                            url="https://deepgram.com",
                        ),
                        installed=True,
                        version=None,
                        languages=['en'],
                    ),
                    AsrModel(
                        name='flux-general-en',
                        description='Flux General English (Streaming)',
                        attribution=Attribution(
                            name="Deepgram",
                            url="https://deepgram.com",
                        ),
                        installed=True,
                        version=None,
                        languages=['en'],
                    )
                ]
            )
        ]
    )

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize Wyoming event handler."""
        logger.info(f"✅ New connection")

        super().__init__(*args, **kwargs)

        state = State()
        model = load_config().get("model", "nova-3")
        logger.info(f"Using Deepgram model: {model}")
        self.stt = DeepgramSTT(model=model)
        self.audio_data = b""
        self.sample_rate = 16000  # Default sample rate; can be adjusted
        wyoming_info = self.WYOMING_INFO
        self.wyoming_info_event = wyoming_info.event()

    async def handle_event(self, event: Event) -> bool:
        """Process and log all incoming Wyoming protocol events."""
        if event.type == "describe":
            logger.info("📤 Responding to describe event.")
            await self.write_event(self.wyoming_info_event)
        elif event.type == "audio-chunk":
            self.audio_data += event.payload
        elif event.type == "audio-stop":
            logger.info(f"Received Wyoming event: {event.type} - Data: {event.data}")
            # Send to Deepgram and get transcription
            text = await self.stt.transcribe(self.audio_data, self.sample_rate)
            result_event = Event(type="transcript", data={"text": text})
            
            logger.info(f"Sending Transcript Event: {text}")
            
            await self.write_event(result_event)
            self.audio_data = b""  # Reset for next transcription
        else:
            logger.info(f"Received Wyoming event: {event.type} - Data: {event.data}")

        return True

def load_api_key():
    """Load the API key from the options.json file."""
    config = load_config()
    api_key = config.get("api_key", "")
    if api_key:
        logger.info(f"✅ API Key Loaded: {api_key[:4]}****")
    else:
        logger.error("⚠️ API key is missing! Set it in the Deepgram addon settings.")
    return api_key

def load_config():
    """Load configuration from the options.json file."""
    try:
        with open(OPTIONS_FILE, "r") as f:
            options = json.load(f)
        return options
    except Exception as e:
        logger.error(f"Error reading options.json: {e}")
        logger.error("⚠️ Configuration file not found or invalid!")
        return {}

async def main():
    """Starts the Wyoming Deepgram STT server using DeepgramServer."""
    server = AsyncServer.from_uri('tcp://0.0.0.0:10301')
    try:
        logger.info('Starting Wyoming Server')
        await server.run(EventHandler)
    except asyncio.CancelledError:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())
