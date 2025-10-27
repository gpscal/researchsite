#!/bin/bash
set -euo pipefail

# Research Site Startup Script
echo "🚀 Starting Research Site..."

# Create necessary directories
mkdir -p /var/log/researchsite /var/run/researchsite /researchsite/data/uploads /researchsite/data/vector_store

# Change to app directory
cd /researchsite

# Set environment variables
export PORT=${PORT:-80}
export PYTHONPATH="/researchsite"

# Create virtual environment if it doesn't exist
if [ ! -d "/researchsite/.venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv /researchsite/.venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source /researchsite/.venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set Hugging Face cache to temp directory to avoid disk issues
export HF_HOME="/tmp/hf_cache"
export TRANSFORMERS_CACHE="/tmp/hf_cache"
mkdir -p /tmp/hf_cache

echo "🎯 Starting Gunicorn server on port ${PORT}..."
echo "📊 Access the app at: http://localhost:${PORT}"
echo "📝 Logs: /var/log/researchsite/"

# Start the application
exec gunicorn --config gunicorn.conf.py app:app