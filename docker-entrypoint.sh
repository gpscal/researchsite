#!/bin/bash
set -euo pipefail

# Docker entrypoint for researchsite
echo "🐳 Starting Research Site in Docker..."

# Create cache directories
mkdir -p /tmp/hf_cache

# Run the startup script
exec /researchsite/start.sh