import asyncio
import json
import os
import logging
import threading
import struct
import time
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.info import Info, Describe, AsrProgram, AsrModel, Attribution

# Support both Docker and local testing
OPTIONS_FILE = os.getenv("OPTIONS_FILE", "/data/options.json")
# Fallback to local test file if Docker path doesn't exist
if not os.path.exists(OPTIONS_FILE):
    local_options = os.path.join(os.path.dirname(__file__), "data", "options.json")
    if os.path.exists(local_options):
        OPTIONS_FILE = local_options

def load_config():
    """Load configuration from the options.json file."""
    try:
        with open(OPTIONS_FILE, "r") as f:
            options = json.load(f)
        return options
    except Exception as e:
        # Use print since logger might not be initialized yet
        print(f"Error reading options.json: {e}")
        print("⚠️ Configuration file not found or invalid!")
        return {}

# Load config early to set up logging level
_initial_config = load_config()
_debug_mode = _initial_config.get("debug", False)

# Configure logging based on debug mode
log_level = logging.DEBUG if _debug_mode else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if _debug_mode:
    logger.info("🐛 Debug mode ENABLED - verbose logging active")
else:
    logger.debug("Debug mode disabled - using INFO level logging")

class DeepgramSTT:
    def __init__(self, model: str = "nova-3"):
        deepgram_api_key = load_api_key()
        self.dg_client = DeepgramClient(api_key=deepgram_api_key)
        self.model = model
        self.is_flux = model.startswith("flux")
    
    async def start_streaming(self, sample_rate: int, transcript_callback):
        """
        Start a streaming transcription session.
        Returns a StreamingSession object that can be used to send audio chunks.
        """
        if not self.is_flux:
            raise ValueError("Streaming is only supported for Flux models")
        
        loop = asyncio.get_event_loop()
        
        # Create a session object to manage the connection
        session = StreamingSession(self.dg_client, self.model, transcript_callback, loop)
        await session.start()
        return session
    
    async def transcribe(self, audio_data: bytes, sample_rate: int):
        """
        Send audio data to Deepgram and return transcription.
        Flux models use streaming API (v2/listen), other models use prerecorded API (v1/listen).
        """
        if self.is_flux:
            # Flux models MUST use streaming API (v2/listen), not prerecorded
            logger.debug("Flux model detected, using streaming API")
            return await self._transcribe_flux_streaming(audio_data, sample_rate)
        else:
            # Non-Flux models use prerecorded API (v1/listen)
            return await self._transcribe_prerecorded(audio_data, sample_rate)
    
    async def _transcribe_prerecorded(self, audio_data: bytes, sample_rate: int):
        """Use prerecorded API for non-Flux models."""
        # SDK 5.x uses media.transcribe_file with request as bytes
        # This is a synchronous method, so run it in executor
        loop = asyncio.get_event_loop()
        
        def transcribe_sync():
            # Check if audio_data is WAV (has RIFF header) or raw PCM
            is_wav = audio_data[:4] == b'RIFF'
            
            # Try different API paths for Deepgram SDK 5.x
            # Based on logs: listen has 'prerecorded', 'rest', 'live', 'websocket'
            # Note: v2 is for streaming (Flux), v1 is for prerecorded (nova-3)
            api_path = None
            listen_obj = getattr(self.dg_client, 'listen', None)
            
            # Path 1: dg_client.listen.v1 (v1 API for prerecorded, similar to v2 for streaming)
            if listen_obj and hasattr(listen_obj, 'v1'):
                v1_obj = listen_obj.v1
                if hasattr(v1_obj, 'media') and hasattr(v1_obj.media, 'transcribe_file'):
                    api_path = v1_obj.media.transcribe_file
                    logger.debug("Using API path: listen.v1.media.transcribe_file")
                elif hasattr(v1_obj, 'prerecorded'):
                    prerecorded_obj = v1_obj.prerecorded
                    if hasattr(prerecorded_obj, 'transcribe_file'):
                        api_path = prerecorded_obj.transcribe_file
                        logger.debug("Using API path: listen.v1.prerecorded.transcribe_file")
            
            # Path 2: dg_client.listen.prerecorded (direct prerecorded access)
            if not api_path and listen_obj and hasattr(listen_obj, 'prerecorded'):
                prerecorded_obj = listen_obj.prerecorded
                # Check what methods are available on prerecorded
                prerecorded_attrs = [attr for attr in dir(prerecorded_obj) if not attr.startswith('_')]
                logger.debug(f"Prerecorded attributes: {prerecorded_attrs}")
                
                # Try common method names
                if hasattr(prerecorded_obj, 'transcribe_file'):
                    api_path = prerecorded_obj.transcribe_file
                    logger.debug("Using API path: listen.prerecorded.transcribe_file")
                elif hasattr(prerecorded_obj, 'transcribe'):
                    api_path = prerecorded_obj.transcribe
                    logger.debug("Using API path: listen.prerecorded.transcribe")
                elif hasattr(prerecorded_obj, 'sync'):
                    sync_obj = prerecorded_obj.sync
                    if hasattr(sync_obj, 'transcribe_file'):
                        api_path = sync_obj.transcribe_file
                        logger.debug("Using API path: listen.prerecorded.sync.transcribe_file")
                    elif hasattr(sync_obj, 'transcribe'):
                        api_path = sync_obj.transcribe
                        logger.debug("Using API path: listen.prerecorded.sync.transcribe")
            
            # Path 3: dg_client.listen.rest (alternative path)
            if not api_path and listen_obj and hasattr(listen_obj, 'rest'):
                rest_obj = listen_obj.rest
                if hasattr(rest_obj, 'v1'):
                    v1_obj = rest_obj.v1
                    if hasattr(v1_obj, 'listen') and hasattr(v1_obj.listen, 'media'):
                        if hasattr(v1_obj.listen.media, 'transcribe_file'):
                            api_path = v1_obj.listen.media.transcribe_file
                            logger.debug("Using API path: listen.rest.v1.listen.media.transcribe_file")
            
            # Path 4: dg_client.rest (direct rest access)
            if not api_path and hasattr(self.dg_client, 'rest'):
                rest_obj = self.dg_client.rest
                if hasattr(rest_obj, 'v1'):
                    v1_obj = rest_obj.v1
                    if hasattr(v1_obj, 'listen') and hasattr(v1_obj.listen, 'media'):
                        if hasattr(v1_obj.listen.media, 'transcribe_file'):
                            api_path = v1_obj.listen.media.transcribe_file
                            logger.debug("Using API path: rest.v1.listen.media.transcribe_file")
            
            if not api_path:
                # Log all available attributes for debugging
                dg_attrs = [attr for attr in dir(self.dg_client) if not attr.startswith('_')]
                logger.error(f"Available dg_client attributes: {dg_attrs}")
                if listen_obj:
                    logger.error(f"Available listen attributes: {[attr for attr in dir(listen_obj) if not attr.startswith('_')]}")
                    if hasattr(listen_obj, 'prerecorded'):
                        prerecorded_attrs = [attr for attr in dir(listen_obj.prerecorded) if not attr.startswith('_')]
                        logger.error(f"Available prerecorded attributes: {prerecorded_attrs}")
                raise AttributeError("Could not find Deepgram v1 API path. Please check SDK version and update code.")
            
            if is_wav:
                # WAV file - SDK auto-detects format
                response = api_path(
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
                
                response = api_path(
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
            
            # Fallback: try dict access
            if isinstance(response, dict):
                transcript = response.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
                return transcript
            
            logger.warning("Unexpected response format from Deepgram API")
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
                            
                            # Don't set the event yet - wait for EndOfTurn or stream close
                        else:
                            logger.debug(f"No transcript found in message")
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}", exc_info=True)
                
                # Connect to v2/listen endpoint for streaming using SDK 5.3.0+ pattern
                # SDK 5.3.0+ uses: client.listen.v2.connect() as per official docs
                connection_context = self.dg_client.listen.v2.connect(
                    model=self.model,
                    encoding="linear16",
                    sample_rate=str(sample_rate)
                )
                logger.debug("Using API path: listen.v2.connect (SDK 5.3.0+)")
                
                with connection_context as connection:
                    logger.info("Connected to Deepgram Flux streaming API")
                    
                    # Set up event handlers using EventType from SDK 5.3.0+
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
                    time.sleep(0.1)  # Give it a moment to start
                    
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
                            # SDK 5.3.0+ uses send_media() method
                            connection.send_media(chunk)
                            bytes_sent += len(chunk)
                            chunk_count += 1
                            logger.debug(f"Sent chunk {chunk_count}: {len(chunk)} bytes")
                    
                    
                    if hasattr(connection, 'finish'):
                        try:
                            connection.finish()
                            logger.debug("Called connection.finish() after sending audio")
                        except Exception as e:
                            logger.debug(f"connection.finish() raised error: {e}")
                    
                    logger.info(f"Finished sending {chunk_count} chunks, total {bytes_sent} bytes")
                    logger.info(f"Finished sending audio data, waiting for transcript...")
                    
                    # Wait for first message to arrive (with timeout)
                    # Reduced for prerecorded files since all audio is sent at once
                    initial_wait = 2.0  # Reduced from 3.0s for faster response
                    first_message_timeout = time.monotonic() + initial_wait
                    logger.info(f"Waiting up to {initial_wait}s for first transcript message...")
                    
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
            
            if result:
                logger.info(f"Flux transcription complete: {result}")
                return result
            else:
                logger.warning("No transcript received from Flux API")
                return ""
        except Exception as e:
            logger.error(f"Error in Flux streaming transcription: {e}", exc_info=True)
            raise


class StreamingSession:
    """Manages a streaming Deepgram connection"""
    def __init__(self, dg_client, model, transcript_callback, event_loop):
        self.dg_client = dg_client
        self.model = model
        self.transcript_callback = transcript_callback
        self.event_loop = event_loop
        self.connection = None
        self.connection_context = None
        self.final_transcript = ""
        self.last_sent_transcript = ""
        self.listening_thread = None
        self._lock = threading.Lock()
        self.end_of_turn = threading.Event()
        self.final_sent = False
        # Connection barrier to prevent race conditions
        self.streaming_ready = asyncio.Event()
    
    async def start(self):
        """Start the streaming connection - following faster-whisper pattern"""
        loop = asyncio.get_event_loop()
        self.end_of_turn.clear()
        
        def run_streaming():
            def on_message(message):
                """Handle incoming transcript messages - send progressive updates like faster-whisper"""
                logger.info(f"🔔 StreamingSession.on_message CALLED! Message type: {type(message).__name__}")
                try:
                    transcript_text = None
                    
                    # Check message type and extract transcript (same pattern as _transcribe_flux_streaming)
                    msg_type_name = type(message).__name__
                    msg_type_attr = getattr(message, 'type', None)
                    logger.info(f"📨 StreamingSession.on_message processing: class={msg_type_name}, type={msg_type_attr}")
                    logger.debug(f"Message attributes: {[attr for attr in dir(message) if not attr.startswith('_')]}")
                    
                    # Filter out connection and error events
                    if 'Connected' in msg_type_name or 'Error' in msg_type_name or 'Fatal' in msg_type_name:
                        if 'Connected' in msg_type_name:
                            # Connection is ready - signal the async event loop
                            logger.debug("Connection established, setting streaming_ready")
                            loop.call_soon_threadsafe(self.streaming_ready.set)
                        elif 'Error' in msg_type_name or 'Fatal' in msg_type_name:
                            error_code = getattr(message, 'code', None)
                            error_desc = getattr(message, 'description', None)
                            logger.error(f"❌ Deepgram streaming error: code={error_code}, description={error_desc}")
                        return
                    
                    # Check if this is a final result (not interim)
                    # Try multiple ways to detect is_final
                    is_final = None
                    if hasattr(message, 'is_final'):
                        is_final = getattr(message, 'is_final')
                    elif hasattr(message, 'channel') and hasattr(message.channel, 'alternatives') and len(message.channel.alternatives) > 0:
                        alt = message.channel.alternatives[0]
                        if hasattr(alt, 'is_final'):
                            is_final = getattr(alt, 'is_final')
                    
                    logger.debug(f"  is_final detection: {is_final} (type: {type(is_final)})")
                    # Process both interim and final - send all updates like Faster-Whisper
                    
                    # Try different message formats to extract transcript (matching _transcribe_flux_streaming)
                    logger.debug(f"  Message attributes: {[attr for attr in dir(message) if not attr.startswith('_') and not callable(getattr(message, attr, None))]}")
                    
                    # Check for channel.alternatives pattern (most common for v2 API)
                    if hasattr(message, 'channel'):
                        channel = message.channel
                        logger.debug(f"  Found channel: {channel}")
                        if hasattr(channel, 'alternatives') and len(channel.alternatives) > 0:
                            alt = channel.alternatives[0]
                            logger.debug(f"  Alternative: {alt}")
                            transcript_text = getattr(alt, 'transcript', None)
                            if transcript_text:
                                logger.info(f"✅ Found transcript in channel.alternatives: {transcript_text[:50]}...")
                    
                    # Check for direct transcript attribute
                    if not transcript_text and hasattr(message, 'transcript'):
                        transcript_text = message.transcript
                        logger.info(f"✅ Found transcript in message.transcript: {transcript_text[:50]}...")
                    
                    # Check if message itself is a string or dict
                    if not transcript_text:
                        if isinstance(message, str):
                            transcript_text = message
                        elif isinstance(message, dict):
                            if message.get('is_final', True):
                                transcript_text = message.get('transcript') or message.get('text')
                    
                    if transcript_text and transcript_text.strip():
                        text = transcript_text.strip()
                        with self._lock:
                            # Treat None as interim (send it), only False means skip
                            final_flag = bool(is_final) if is_final is not None else False
                            old_length = len(self.final_transcript) if self.final_transcript else 0
                            new_length = len(text)
                            
                            # Only update if new transcript is longer or significantly different
                            should_update = (old_length == 0 or 
                                           new_length > old_length or 
                                           (new_length >= old_length * 0.8 and text != self.final_transcript))
                            
                            if should_update:
                                self.final_transcript = text
                                
                                # Send ALL updates (both interim and final) with FULL transcript
                                # This ensures HA gets the complete sentence even if it closes after first event
                                if text != self.last_sent_transcript:
                                    self.last_sent_transcript = text
                                    try:
                                        asyncio.run_coroutine_threadsafe(
                                            self.transcript_callback(text, final_flag),
                                            self.event_loop
                                        )
                                        logger.info(f"📤 Sent streaming transcript ({'final' if final_flag else 'interim'}): {text[:50]}...")
                                    except Exception as e:
                                        logger.error(f"Error calling transcript callback: {e}", exc_info=True)
                                    
                                    if final_flag:
                                        self.final_sent = True
                            else:
                                logger.debug(f"Ignoring shorter fragment: {new_length} chars (keeping {old_length} chars)")
                except Exception as e:
                    logger.error(f"Error in streaming message handler: {e}", exc_info=True)
            
            def on_close(_):
                logger.debug("Streaming connection closed")
                self.end_of_turn.set()
            
            def on_error(error):
                error_code = getattr(error, 'code', None)
                error_desc = getattr(error, 'description', None)
                logger.error(f"Streaming connection error: code={error_code}, description={error_desc}, error={error}")
                self.end_of_turn.set()

            def on_end_of_turn(_):
                logger.info("🔚 EndOfTurn received - sending final transcript")
                self.end_of_turn.set()
                # Send the final transcript when EndOfTurn is received
                with self._lock:
                    if self.final_transcript and not self.final_sent:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self.transcript_callback(self.final_transcript, True),  # final=True
                                self.event_loop
                            )
                            self.final_sent = True
                            logger.info(f"📤 Sent FINAL transcript on EndOfTurn: {self.final_transcript[:80]}...")
                        except Exception as e:
                            logger.error(f"Error sending final transcript on EndOfTurn: {e}", exc_info=True)
            
            try:
                # Connect to v2/listen endpoint for streaming using SDK 5.3.0+ pattern
                # SDK 5.3.0+ uses: client.listen.v2.connect() as per official docs
                self.connection_context = self.dg_client.listen.v2.connect(
                    model=self.model,
                    encoding="linear16",
                    sample_rate="16000"
                )
                logger.debug("Using API path: listen.v2.connect (SDK 5.3.0+)")
                
                self.connection = self.connection_context.__enter__()
                logger.info("Connected to Deepgram streaming API")
                
                # Register event handlers BEFORE calling start_listening() (per official docs pattern)
                logger.info("Registering event handlers...")
                try:
                    self.connection.on(EventType.OPEN, lambda _: logger.info("🔵 Connection OPEN event received"))
                    self.connection.on(EventType.MESSAGE, on_message)
                    self.connection.on(EventType.CLOSE, on_close)
                    self.connection.on(EventType.ERROR, on_error)
                    if hasattr(EventType, 'END_OF_TURN'):
                        self.connection.on(EventType.END_OF_TURN, on_end_of_turn)
                    logger.info("✅ Registered handlers using EventType")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"EventType registration failed: {e}, trying string-based fallback")
                    # Fallback to string-based event names
                    self.connection.on("open", lambda _: logger.info("🔵 Connection OPEN event received"))
                    self.connection.on("message", on_message)
                    self.connection.on("close", on_close)
                    self.connection.on("error", on_error)
                    self.connection.on("end_of_turn", on_end_of_turn)
                    logger.info("✅ Registered handlers using string-based events")
                
                # Start listening for messages (per Deepgram docs: call AFTER registering handlers, BEFORE sending audio)
                logger.info("Starting message listener (blocking call in thread)...")
                self.listening_thread = threading.Thread(
                    target=self.connection.start_listening, 
                    daemon=True,
                    name="DeepgramListener"
                )
                self.listening_thread.start()
                time.sleep(0.1)  # Minimal wait for listener thread to start (reduced from 0.3s)
                
                # Verify listener thread is running
                if self.listening_thread.is_alive():
                    logger.info("✅ Streaming connection established and listening (ready for audio)")
                else:
                    logger.error("❌ Listener thread died immediately - check for errors above")
                    raise RuntimeError("Failed to start Deepgram listener thread")
                
                # Note: streaming_ready event is set when OPEN event is received (in on_message handler)
                # We don't set it here to avoid race conditions
            except Exception as e:
                logger.error(f"Error starting streaming connection: {e}", exc_info=True)
                raise
        
        # Run in executor since Deepgram SDK is sync
        await loop.run_in_executor(None, run_streaming)
    
    async def wait_for_ready(self, timeout: float = 5.0):
        """Wait for streaming connection to be ready before sending audio."""
        try:
            await asyncio.wait_for(self.streaming_ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Streaming connection not ready after {timeout}s timeout")
    
    def send_media(self, chunk: bytes):
        """Send audio chunk to the streaming connection (like faster-whisper processes chunks incrementally)"""
        if self.connection:
            try:
                # SDK 5.3.0+ uses send_media() method
                self.connection.send_media(chunk)
            except Exception as e:
                logger.error(f"Error sending chunk to streaming connection: {e}", exc_info=True)

    def wait_for_completion(self, timeout: float = 3.0) -> bool:
        """Block until Deepgram signals the end of the turn or timeout expires."""
        return self.end_of_turn.wait(timeout)

    def get_final_transcript(self) -> str:
        with self._lock:
            return self.final_transcript or ""
    
    def close(self):
        """Close the streaming connection"""
        if self.connection and hasattr(self.connection, 'finish'):
            try:
                self.connection.finish()
                logger.debug("Called connection.finish() before closing stream")
            except Exception as e:
                logger.debug(f"connection.finish() raised error: {e}")
        
        if self.connection_context and self.connection:
            try:
                # SDK 5.3.0+ uses context manager __exit__ to close
                self.connection_context.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing streaming connection: {e}")
            finally:
                self.connection = None
                self.connection_context = None


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
    async def _safe_write_event(self, event: Event) -> None:
        """Send event to Wyoming client, ignoring disconnect-related errors."""
        try:
            await self.write_event(event)
        except (BrokenPipeError, ConnectionResetError) as exc:
            logger.warning(f"Client disconnected before event delivery: {exc}")
        except TypeError as exc:
            # Happens when the underlying transport was already closed and asyncio Streams flips to None
            logger.warning(f"Transport already closed while sending event: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error sending event: {exc}", exc_info=True)
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
        model = load_config().get("model", "flux-general-en")  # Default to Flux for streaming
        logger.info(f"Using Deepgram model: {model}")
        self.stt = DeepgramSTT(model=model)
        self.audio_data = b""
        self.sample_rate = 16000  # Default sample rate; can be adjusted
        self.streaming_connection = None  # For streaming mode
        self.streaming_active = False
        self.streaming_transcript = ""
        self.streaming_ready = False  # Track if connection is ready (avoid repeated waits)
        wyoming_info = self.WYOMING_INFO
        self.wyoming_info_event = wyoming_info.event()

    async def handle_event(self, event: Event) -> bool:
        """Process and log all incoming Wyoming protocol events."""
        if event.type == "describe":
            logger.info("📤 Responding to describe event.")
            await self.write_event(self.wyoming_info_event)
        elif event.type == "audio-start":
            logger.info(f"Received Wyoming event: {event.type} - Data: {event.data}")
            # Extract sample rate from audio-start event
            if event.data:
                self.sample_rate = event.data.get("rate", 16000)
                logger.info(f"Audio stream started, sample rate: {self.sample_rate}")
            
            # For Flux models, start streaming connection
            if self.stt.is_flux:
                try:
                    self.streaming_transcript = ""
                    self.streaming_ready = False  # Reset ready flag for new connection
                    # Define callback to capture progressive transcripts (send immediately like faster-whisper)
                    async def send_transcript(text: str, is_final: bool):
                        self.streaming_transcript = text
                        result_event = Event(
                            type="transcript",
                            data={"text": text, "final": is_final},
                        )
                        await self._safe_write_event(result_event)
                        logger.info(f"📤 Sent streaming transcript ({'FINAL' if is_final else 'interim'}): {text[:80]}...")
                    
                    logger.info(f"🔄 Starting streaming connection for Flux model...")
                    self.streaming_connection = await self.stt.start_streaming(
                        self.sample_rate, 
                        send_transcript
                    )
                    self.streaming_active = True
                    # Wait for connection to be ready (only once at start)
                    # Reduced timeout since connection should be ready quickly after OPEN event
                    await self.streaming_connection.wait_for_ready(timeout=0.5)
                    self.streaming_ready = True
                    logger.info("✅ Started streaming transcription - ready to receive audio chunks")
                except Exception as e:
                    logger.error(f"❌ Failed to start streaming: {e}", exc_info=True)
                    self.streaming_active = False
                    # Fallback: will use batch mode on audio-stop
            else:
                # For non-Flux models, use batch mode
                self.streaming_active = False
                self.audio_data = b""
        elif event.type == "audio-chunk":
            if self.streaming_active and self.streaming_connection:
                # Streaming mode: send chunk immediately to Deepgram
                try:
                    # Only wait if connection not yet ready (should be ready from audio-start)
                    if not self.streaming_ready:
                        await self.streaming_connection.wait_for_ready(timeout=0.5)
                        self.streaming_ready = True
                    
                    chunk = event.payload
                    chunk_size = len(chunk) if chunk else 0
                    
                    # Flux requires 80ms chunks, but we'll send as received and let Deepgram handle it
                    # Send chunk directly (StreamingSession handles thread safety)
                    self.streaming_connection.send_media(chunk)
                    logger.debug(f"📤 Sent audio chunk to Deepgram: {chunk_size} bytes")
                except Exception as e:
                    logger.error(f"Error processing streaming audio chunk: {e}", exc_info=True)
            else:
                # Batch mode: accumulate audio data
                self.audio_data += event.payload
        elif event.type == "audio-stop":
            logger.info(f"Received Wyoming event: {event.type} - Data: {event.data}")
            
            if self.streaming_active and self.streaming_connection:
                # Streaming mode: finalize the stream
                try:
                    session = self.streaming_connection
                    loop = asyncio.get_event_loop()
                    
                    # Send final chunk if any remaining audio
                    if self.audio_data:
                        session.send_media(self.audio_data)
                    
                    # Tell Deepgram we're done sending audio (triggers final transcript messages)
                    def finish_stream():
                        try:
                            if session.connection and hasattr(session.connection, 'finish'):
                                session.connection.finish()
                                logger.debug("Called connection.finish() to signal end of audio")
                        except Exception as e:
                            logger.debug(f"connection.finish() raised error: {e}")
                    
                    # CRITICAL: Send latest transcript IMMEDIATELY to prevent Home Assistant timeout
                    # Home Assistant disconnects after ~3s if no transcript is received
                    # Send whatever we have NOW, then refine it later
                    latest_transcript = session.get_final_transcript().strip()
                    if not latest_transcript:
                        latest_transcript = self.streaming_transcript.strip()
                    
                    if latest_transcript:
                        # Send immediately to keep connection alive
                        result_event = Event(
                            type="transcript",
                            data={"text": latest_transcript, "final": False},  # Mark as interim initially
                        )
                        await self._safe_write_event(result_event)
                        logger.info(f"📤 Sent IMMEDIATE transcript (to prevent HA timeout): {latest_transcript[:80]}...")
                    
                    await loop.run_in_executor(None, finish_stream)
                    
                    # Wait for Deepgram to finish processing and send final transcript
                    # Reduced timeout to match faster-whisper's responsiveness (0.5s)
                    # We already send progressive updates, so we just need a brief wait for final refinement
                    def wait_for_end():
                        return session.wait_for_completion(timeout=0.5)  # Fast like faster-whisper

                    completed = await loop.run_in_executor(None, wait_for_end)
                    if not completed:
                        logger.debug("EndOfTurn not received (timeout), using latest transcript")
                    
                    # Get the final transcript after Deepgram has finished processing
                    final_transcript = session.get_final_transcript().strip()
                    if not final_transcript:
                        final_transcript = self.streaming_transcript.strip()
                    
                    logger.info(f"📋 Final transcript after processing: '{final_transcript}' (length: {len(final_transcript)})")
                    logger.info(f"📋 Last streaming transcript: '{self.streaming_transcript}' (length: {len(self.streaming_transcript)})")
                    
                    # Send final refined transcript (mark as final even if same as immediate)
                    if final_transcript:
                        if final_transcript != latest_transcript:
                            # Different transcript - send the refined version
                            result_event = Event(
                                type="transcript",
                                data={"text": final_transcript, "final": True},
                            )
                            logger.info(f"📤 Sending FINAL refined transcript: {final_transcript[:80]}...")
                            await self._safe_write_event(result_event)
                        elif latest_transcript:
                            # Same transcript - just mark the previous one as final
                            result_event = Event(
                                type="transcript",
                                data={"text": final_transcript, "final": True},
                            )
                            logger.info(f"📤 Marking transcript as FINAL: {final_transcript[:80]}...")
                            await self._safe_write_event(result_event)
                        session.final_sent = True
                    else:
                        logger.warning("No final transcript available after processing")
                    
                    # Close the connection
                    def close_stream():
                        try:
                            session.close()
                        except Exception as e:
                            logger.error(f"Error closing streaming connection: {e}")
                    
                    await loop.run_in_executor(None, close_stream)
                    self.streaming_active = False
                    self.streaming_ready = False  # Reset ready flag
                    self.audio_data = b""
                    logger.info("✅ Streaming transcription finalized")
                    
                    self.streaming_transcript = ""
                    self.streaming_connection = None
                except Exception as e:
                    logger.error(f"Error finalizing streaming: {e}", exc_info=True)
            else:
                # Batch mode: process accumulated audio
                text = await self.stt.transcribe(self.audio_data, self.sample_rate)
                result_event = Event(type="transcript", data={"text": text})
                
                logger.info(f"Sending Transcript Event: {text}")
                
                await self._safe_write_event(result_event)
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
