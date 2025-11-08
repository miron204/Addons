# Deepgram Models Configuration

## Available Models

### Nova Models (Prerecorded API - /v1/listen)
- `nova-3` - Latest Nova model (default)
- `nova-2` - Previous generation Nova
- `nova` - Original Nova model

### Base Models (Prerecorded API)
- `base` - Base model
- `enhanced` - Enhanced model

### Flux Models (Streaming API - /v2/listen)
- `flux-general-en` - Flux General English (streaming only)

**Important Notes for Flux:**
- Flux models **require** the `/v2/listen` streaming endpoint
- Flux models **do NOT work** with prerecorded `/v1/listen` endpoint
- Audio must be sent in **80ms chunks** for optimal performance
- Use `model=flux-general-en` (not just `flux`)

## Configuration

Edit `data/options.json` to change the model:

```json
{
  "api_key": "your-api-key-here",
  "model": "flux-general-en"
}
```

## Testing Different Models

1. **Update options.json** with the desired model
2. **Restart the server**
3. **Run the test client** to measure response times:

```bash
# Test Nova model
echo '{"api_key": "your-key", "model": "nova-3"}' > data/options.json
python3 deepgram_server.py  # restart server

# Test Flux model
echo '{"api_key": "your-key", "model": "flux-general-en"}' > data/options.json
python3 deepgram_server.py  # restart server

# Compare response times
python3 test_client.py --audio "Welcome BOTty.wav"
```

## Model Comparison

| Model | API Endpoint | Latency | Accuracy | Use Case |
|-------|-------------|---------|----------|----------|
| nova-3 | /v1/listen | Medium | High | General purpose |
| flux-general-en | /v2/listen | Low | High | Real-time streaming |








