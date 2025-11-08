# Testing Guide

## Quick Start

### 1. Start the Server

```bash
cd deepgram
./test_run.sh
```

Or manually:
```bash
# Install dependencies
pip install -r requirements.txt

# Create data directory and add API key
mkdir -p data
echo '{"api_key": "YOUR_DEEPGRAM_API_KEY"}' > data/options.json

# Run server
python3 deepgram_server.py
```

### 2. Test the Server

In another terminal, run the test client:

```bash
# Get server info
python3 test_client.py --describe

# Test transcription with silence (default test)
python3 test_client.py

# Test with an audio file
python3 test_client.py --audio path/to/audio.wav

# Interactive mode
python3 test_client.py --interactive
```

## Test Client Usage

### Command Line Options

- `--uri URI` - Server URI (default: `tcp://localhost:10301`)
- `--describe` - Get server information
- `--audio FILE` - Path to WAV audio file to transcribe
- `--sample-rate RATE` - Audio sample rate in Hz (default: 16000)
- `--interactive` - Run in interactive mode

### Examples

```bash
# Get server info
python3 test_client.py --describe

# Test with specific audio file
python3 test_client.py --audio test.wav --sample-rate 16000

# Connect to remote server
python3 test_client.py --uri tcp://192.168.1.100:10301

# Interactive mode (allows multiple commands)
python3 test_client.py --interactive
```

### Interactive Mode Commands

When running in interactive mode:
- `describe` - Get server information
- `test` - Send test audio for transcription
- `quit` or `q` - Exit

## Audio Format Requirements

- **Sample Rate**: 16000 Hz (default)
- **Channels**: 1 (mono)
- **Bit Depth**: 16-bit
- **Encoding**: Linear PCM (WAV format)

## Troubleshooting

### Connection Refused
- Make sure the server is running
- Check that the port (10301) is correct
- Verify the server URI is correct

### No Transcript Received
- Check that your Deepgram API key is valid
- Verify the audio format matches requirements
- Check server logs for errors

### API Key Error
- Ensure `data/options.json` exists and contains a valid API key
- Format: `{"api_key": "your-key-here"}`








