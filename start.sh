#!/bin/bash
set -euo pipefail

# Research Site Startup Script (for non-Docker environments)
echo "🚀 Starting Research Site..."

# Determine the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p /var/log/researchsite /var/run/researchsite
mkdir -p "${SCRIPT_DIR}/data/uploads" "${SCRIPT_DIR}/data/vector_store"

# Set environment variables
export PORT=${PORT:-80}
export PYTHONPATH="${SCRIPT_DIR}"
export HF_HOME="/tmp/hf_cache"
export TRANSFORMERS_CACHE="/tmp/hf_cache"
mkdir -p /tmp/hf_cache

# Create virtual environment if it doesn't exist
if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/.venv"
    echo "📚 Installing dependencies..."
    "${SCRIPT_DIR}/.venv/bin/pip" install --upgrade pip
    "${SCRIPT_DIR}/.venv/bin/pip" install -r requirements.txt
else
    echo "✅ Using existing virtual environment"
fi

echo "🎯 Starting Gunicorn server on port ${PORT}..."
echo "📊 Access the app at: http://localhost:${PORT}"
echo "📝 Logs: /var/log/researchsite/"

# Start the application using venv's gunicorn directly
exec "${SCRIPT_DIR}/.venv/bin/gunicorn" --config gunicorn.conf.py app:app