FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn8-runtime

# Prevent prompts from apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:1

WORKDIR /app

# System dependencies for desktop (XFCE), Xvfb, RustDesk, and PDF tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    xfce4 \
    xfce4-goodies \
    xfce4-terminal \
    xvfb \
    x11-utils \
    dbus-x11 \
    nginx \
    certbot \
    python3-certbot-nginx \
    unzip \
    build-essential \
    python3-dev \
    ninja-build \
    git \
    tesseract-ocr \
    poppler-utils \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install RustDesk
ARG RUSTDESK_VERSION=1.4.3
RUN set -eux; \
    wget -q https://github.com/rustdesk/rustdesk/releases/download/${RUSTDESK_VERSION}/rustdesk-${RUSTDESK_VERSION}-x86_64.deb; \
    apt-get update && apt-get install -y --no-install-recommends ./rustdesk-${RUSTDESK_VERSION}-x86_64.deb; \
    rm -f ./rustdesk-${RUSTDESK_VERSION}-x86_64.deb; \
    rm -rf /var/lib/apt/lists/*

# Install RustDesk rendezvous/relay servers (hbbs/hbbr)
ARG RUSTDESK_SERVER_VERSION=1.1.11
RUN set -eux; \
    cd /tmp; \
    # Try versioned assets; fall back to unversioned naming if necessary
    wget -q https://github.com/rustdesk/rustdesk-server/releases/download/${RUSTDESK_SERVER_VERSION}/hbbs-${RUSTDESK_SERVER_VERSION}-x86_64-unknown-linux-gnu.zip || \
      wget -q https://github.com/rustdesk/rustdesk-server/releases/download/${RUSTDESK_SERVER_VERSION}/hbbs-x86_64-unknown-linux-gnu.zip; \
    wget -q https://github.com/rustdesk/rustdesk-server/releases/download/${RUSTDESK_SERVER_VERSION}/hbbr-${RUSTDESK_SERVER_VERSION}-x86_64-unknown-linux-gnu.zip || \
      wget -q https://github.com/rustdesk/rustdesk-server/releases/download/${RUSTDESK_SERVER_VERSION}/hbbr-x86_64-unknown-linux-gnu.zip; \
    unzip -o hbbs*.zip -d /usr/local/bin; \
    unzip -o hbbr*.zip -d /usr/local/bin; \
    chmod +x /usr/local/bin/hbbs /usr/local/bin/hbbr; \
    rm -f hbbs*.zip hbbr*.zip

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure data directories exist
RUN mkdir -p /app/data/uploads /app/data/vector_store

# Expose API, web ports, and RustDesk server ports
EXPOSE 8080 80 443 21115 21116 21117

# Make startup script executable
RUN chmod +x /app/start.sh || true

# Default command
CMD ["/app/start.sh"]


