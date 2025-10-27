FROM python:3.12-slim

# Set working directory
WORKDIR /researchsite

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create and activate virtual environment, install dependencies
RUN python3 -m venv /researchsite/.venv && \
    /researchsite/.venv/bin/pip install --upgrade pip && \
    /researchsite/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /var/log/researchsite \
    /var/run/researchsite \
    /researchsite/data/uploads \
    /researchsite/data/vector_store \
    /tmp/hf_cache

# Set permissions
RUN chmod +x /researchsite/start.sh && \
    chmod +x /researchsite/docker-entrypoint.sh

# Expose port
EXPOSE 80

# Set environment variables
ENV PORT=80 \
    PYTHONPATH=/researchsite \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    VIRTUAL_ENV=/researchsite/.venv \
    PATH="/researchsite/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Use the entrypoint script
ENTRYPOINT ["/researchsite/docker-entrypoint.sh"]