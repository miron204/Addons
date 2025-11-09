# Deepgram Speech-to-Text Add-on for Home Assistant

This Home Assistant add-on provides speech-to-text (STT) functionality using Deepgram's API with support for the Wyoming protocol.

## Features

- **Multiple Deepgram Models**: Support for Nova, Whisper, and Flux models
- **Configurable Processing Modes**: Choose between batch and streaming processing
- **Wyoming Protocol**: Compatible with Home Assistant's Assist pipeline
- **Real-time Transcription**: Progressive updates during speech recognition (streaming mode)
- **Debug Mode**: Verbose logging for troubleshooting

## Configuration

### Required Settings

- **`api_key`** (string, required): Your Deepgram API key
  - Get your API key from [Deepgram Console](https://console.deepgram.com/)

### Optional Settings

- **`model`** (string, default: `"flux-general-en"`): The Deepgram model to use
  - Options:
    - `flux-general-en` - Latest streaming model with enhanced accuracy
    - `nova-3` - Balanced accuracy and speed
    - `whisper-large` - OpenAI Whisper model

- **`streaming`** (boolean, default: `false`): Processing mode for audio
  - `false` (Batch mode - Recommended):
    - ✅ Accumulate all audio, process at end (like faster-whisper)
    - ✅ Better Home Assistant compatibility
    - ✅ One complete final transcript
    - ✅ Simpler, more predictable
    - ❌ No real-time interim updates
  - `true` (Streaming mode):
    - ✅ Send audio in real-time, get progressive updates
    - ✅ Real-time interim transcripts during speech
    - ✅ Faster initial response (~100ms faster)
    - ❌ More complex processing
    - ⚠️ May have timing issues with some setups

- **`debug`** (boolean, default: `false`): Enable debug logging
  - When `true`, shows detailed logs for troubleshooting
  - When `false`, shows only INFO level logs

## Example Configuration

### Recommended (Batch Mode)
```json
{
  "api_key": "your_deepgram_api_key_here",
  "model": "flux-general-en",
  "streaming": false
}
```

### Alternative (Streaming Mode)
```json
{
  "api_key": "your_deepgram_api_key_here",
  "model": "flux-general-en",
  "streaming": true
}
```

### With Debug Mode
```json
{
  "api_key": "your_deepgram_api_key_here",
  "model": "flux-general-en",
  "streaming": false,
  "debug": true
}
```

## How It Works

### Home Assistant → Wyoming Client → Your Server

Home Assistant always sends audio chunks in real-time via Wyoming protocol. The `streaming` setting controls how **your server** processes those chunks:

- **`streaming: false` (Batch)**:
  ```
  HA → [chunk, chunk, chunk] → audio-stop → [Process ALL chunks at once] → ONE final transcript → HA
  ```

- **`streaming: true`**:
  ```
  HA → chunk → [Process immediately] → interim transcript → HA
      → chunk → [Process immediately] → interim transcript → HA  
      → audio-stop → final transcript → HA
  ```

**Note:** Home Assistant's STT provider reads the FIRST transcript after `audio-stop`, so batch mode ensures that first transcript is complete and final.

## Troubleshooting

### Enable Debug Mode

If you're experiencing issues, enable debug mode to see detailed logs:

```json
{
  "streaming": false,
  "debug": true
}
```

### Incomplete Transcripts

If you're getting incomplete transcripts in Home Assistant:

1. **Use batch mode** (recommended):
   ```json
   "streaming": false
   ```

2. **Check your model**: Flux models work best with this add-on
   ```json
   "model": "flux-general-en"
   ```

### Connection Issues

- Verify your Deepgram API key is correct
- Check that port 10301 is not blocked
- Ensure Home Assistant can reach the add-on

## Testing

Test the add-on from command line:

```bash
# Test with audio file
python3 test_wyoming_client.py --file your_audio.wav

# Test with microphone (requires pyaudio/sounddevice)
python3 test_client_flux.py
```

## Performance Comparison

Based on testing with a 3-second audio sample:

| Mode | First Response | Final Response | Transcript Updates |
|------|---------------|----------------|-------------------|
| **Batch** (`false`) | ~2.4s | ~2.4s | 1 (final only) |
| **Streaming** (`true`) | ~1.3s | ~2.3s | 3 (2 interim + 1 final) |

**Recommendation:** Use batch mode (`streaming: false`) for Home Assistant - the tiny speed difference isn't worth the complexity.

## Support

- [GitHub Issues](https://github.com/brian-makes-things/home-assistant-addons/issues)
- [Deepgram Documentation](https://developers.deepgram.com/)
- [Home Assistant Assist](https://www.home-assistant.io/voice_control/)

## License

MIT License - see LICENSE file for details
