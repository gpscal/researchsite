#!/bin/bash
set -euo pipefail

# Docker entrypoint for researchsite
echo "🐳 Starting Research Site in Docker with Python venv..."

# Create necessary runtime directories
mkdir -p /tmp/hf_cache
mkdir -p /var/log/researchsite
mkdir -p /var/run/researchsite
mkdir -p /researchsite/data/uploads
mkdir -p /researchsite/data/vector_store

# Change to app directory
cd /researchsite

# Set environment variables
export PORT=${PORT:-80}
export PYTHONPATH="/researchsite"
export HF_HOME="/tmp/hf_cache"
export TRANSFORMERS_CACHE="/tmp/hf_cache"

echo "✅ Using Python venv at: /researchsite/.venv"
echo "🎯 Starting Gunicorn server on port ${PORT}..."
echo "📊 Access the app at: http://localhost:${PORT}"
echo "📝 Logs: /var/log/researchsite/"

# Start the application using venv's gunicorn directly
exec /researchsite/.venv/bin/gunicorn --config gunicorn.conf.py app:app