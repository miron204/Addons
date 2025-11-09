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

- **`streaming_mode`** (string, default: `"batch"`): Processing mode for audio
  - `"batch"`: Accumulate all audio, process at end (like faster-whisper)
    - ✅ Simpler, more predictable
    - ✅ Better Home Assistant compatibility
    - ✅ One complete final transcript
    - ❌ No real-time interim updates
  - `"streaming"`: Send audio in real-time, get progressive updates
    - ✅ Real-time interim transcripts during speech
    - ✅ Faster response times
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
  "streaming_mode": "batch",
  "debug": false
}
```

### Alternative (Streaming Mode)
```json
{
  "api_key": "your_deepgram_api_key_here",
  "model": "flux-general-en",
  "streaming_mode": "streaming",
  "debug": false
}
```

## Troubleshooting

### Enable Debug Mode

If you're experiencing issues, enable debug mode to see detailed logs:

```json
{
  "api_key": "your_deepgram_api_key_here",
  "model": "flux-general-en",
  "streaming_mode": "batch",
  "debug": true
}
```

### Incomplete Transcripts

If you're getting incomplete transcripts in Home Assistant:

1. **Switch to batch mode** (recommended):
   ```json
   "streaming_mode": "batch"
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

## Support

- [GitHub Issues](https://github.com/brian-makes-things/home-assistant-addons/issues)
- [Deepgram Documentation](https://developers.deepgram.com/)
- [Home Assistant Assist](https://www.home-assistant.io/voice_control/)

## License

MIT License - see LICENSE file for details
