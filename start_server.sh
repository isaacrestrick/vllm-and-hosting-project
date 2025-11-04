#!/bin/bash

# Startup script for vLLM FastAPI Server

echo "======================================"
echo "Starting vLLM FastAPI Server"
echo "======================================"

# Set default environment variables if not set
export LLM_MODEL_NAME="${LLM_MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"

echo ""
echo "Configuration:"
echo "  Model: $LLM_MODEL_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Prefix Caching: $ENABLE_PREFIX_CACHING"
echo ""
echo "Starting server..."
echo ""

# Start the server
python3 fastapi_server.py

