# Research Site Deployment Guide

## Overview

This application uses **Python venv** for dependency isolation in all deployment scenarios. The Docker deployment builds the virtual environment during image creation for optimal performance.

## Quick Start

### Option 1: Docker (Recommended)

The Docker deployment uses Python venv that's built into the image.

```bash
# Build the Docker image with Python venv
docker build -t researchsite .

# Run with environment variables
docker run -p 80:80 \
  -e ANTHROPIC_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  -e GOOGLE_CX=your_cx \
  -e HF_TOKEN=your_token \
  -v $(pwd)/data:/researchsite/data \
  researchsite

# Or run with .env file
docker run -p 80:80 \
  --env-file .env \
  -v $(pwd)/data:/researchsite/data \
  researchsite
```

**Docker Compose (Optional):**
```yaml
version: '3.8'
services:
  researchsite:
    build: .
    ports:
      - "80:80"
    env_file:
      - .env
    volumes:
      - ./data:/researchsite/data
    restart: unless-stopped
```

### Option 2: Direct Python with venv

```bash
# Make executable and run (creates venv automatically)
chmod +x start.sh
./start.sh
```

The script will:
1. Create a Python venv at `./.venv` if it doesn't exist
2. Install all dependencies from `requirements.txt`
3. Start the Gunicorn server

### Option 3: Systemd Service (Linux)

```bash
# Copy service file and enable
sudo cp researchsite.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable researchsite
sudo systemctl start researchsite
```

## Environment Variables

Create a `.env` file with:
```bash
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GOOGLE_CX=your_cx_here
HF_TOKEN=your_hf_token_here
PORT=80  # Optional, defaults to 80
```

## Architecture

### Docker Deployment
- **Base Image**: `python:3.12-slim`
- **Virtual Environment**: `/researchsite/.venv`
- **Working Directory**: `/researchsite`
- **Dependencies**: Installed during Docker build (no runtime installation)
- **Cache Directory**: `/tmp/hf_cache` (HuggingFace models)
- **Data Persistence**: `/researchsite/data` (mount as volume)

### Benefits of Python venv in Docker
✅ Isolated dependencies (no conflicts with system packages)  
✅ Faster container startup (deps installed at build time)  
✅ Reproducible environments across deployments  
✅ Easy to update dependencies (rebuild image)  
✅ Smaller attack surface (minimal system packages)

## Features

- ✅ Flask web application with research assistant
- ✅ PDF upload and processing (with pdf2image support)
- ✅ RAG (Retrieval Augmented Generation)
- ✅ Web search integration
- ✅ Gunicorn WSGI server (4 workers)
- ✅ Docker support with Python venv
- ✅ Health checks for container monitoring
- ✅ Systemd service support
- ✅ Optimized for low disk space

## Docker Commands

```bash
# Build image
docker build -t researchsite .

# Run container
docker run -d --name research -p 80:80 --env-file .env -v $(pwd)/data:/researchsite/data researchsite

# View logs
docker logs -f research

# Stop container
docker stop research

# Remove container
docker rm research

# Check health
docker inspect --format='{{.State.Health.Status}}' research

# Execute commands inside container
docker exec -it research /researchsite/.venv/bin/python --version
docker exec -it research /researchsite/.venv/bin/pip list
```

## Troubleshooting

### Disk Space Issues
- The app uses `/tmp/hf_cache` for model caching (cleared on restart)
- Models are downloaded on first use
- Consider mounting `/tmp/hf_cache` as a volume for persistence

### Port Conflicts
- Change `PORT` environment variable
- Update port mapping: `-p 8080:80`

### Memory Issues
- Reduce workers in `gunicorn.conf.py`
- Default: 4 workers (adjust based on available memory)

### Dependency Issues
- Rebuild Docker image to update dependencies
- For local deployment, delete `.venv` and run `./start.sh` again

### Permission Issues
```bash
# Fix data directory permissions
chmod -R 755 data/
chown -R 1000:1000 data/  # Adjust UID/GID as needed
```

## Logs

- **Application logs**: `/var/log/researchsite/`
- **Docker logs**: `docker logs researchsite`
- **System logs**: `journalctl -u researchsite`

## Production Recommendations

1. **Use Docker** for consistent deployments
2. **Mount volumes** for persistent data storage
3. **Set resource limits**:
   ```bash
   docker run --memory="2g" --cpus="2" ...
   ```
4. **Enable health checks** (already configured in Dockerfile)
5. **Use environment variables** for secrets (not hardcoded)
6. **Set up log rotation** for `/var/log/researchsite/`
7. **Use a reverse proxy** (nginx/traefik) for SSL/TLS
8. **Regular backups** of `/researchsite/data` volume