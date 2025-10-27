FROM python:3.12-slim

# Set working directory
WORKDIR /researchsite

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3 -m venv /researchsite/.venv
RUN /researchsite/.venv/bin/pip install --upgrade pip
RUN /researchsite/.venv/bin/pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /var/log/researchsite /var/run/researchsite /researchsite/data/uploads /researchsite/data/vector_store

# Set permissions
RUN chmod +x /researchsite/start.sh
RUN chmod +x /researchsite/docker-entrypoint.sh

# Expose port
EXPOSE 80

# Set environment variables
ENV PORT=80
ENV PYTHONPATH=/researchsite
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# Use the entrypoint script
ENTRYPOINT ["/researchsite/docker-entrypoint.sh"]