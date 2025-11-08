#!/bin/bash

# Test script for running Deepgram server locally

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create test data directory if it doesn't exist
mkdir -p data

# Check if options.json exists
if [ ! -f "data/options.json" ]; then
    echo "⚠️  Creating test options.json file..."
    echo "Please add your Deepgram API key to data/options.json"
    echo '{"api_key": "YOUR_DEEPGRAM_API_KEY_HERE"}' > data/options.json
    echo ""
    echo "Edit data/options.json and add your API key, then run this script again."
    exit 1
fi

# Run the server
echo "Starting Deepgram server..."
echo "Server will listen on tcp://0.0.0.0:10301"
echo "Press Ctrl+C to stop"
echo ""
python3 deepgram_server.py








