# Research Site Deployment Guide

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Build and run with Docker
docker build -t researchsite .
docker run -p 80:80 -v $(pwd)/data:/researchsite/data researchsite
```

### Option 2: Direct Python
```bash
# Make executable and run
chmod +x start.sh
./start.sh
```

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
```
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GOOGLE_CX=your_cx_here
HF_TOKEN=your_hf_token_here
```

## Features

- ✅ Flask web application with research assistant
- ✅ PDF upload and processing
- ✅ RAG (Retrieval Augmented Generation)
- ✅ Web search integration
- ✅ Gunicorn WSGI server
- ✅ Docker support
- ✅ Systemd service support
- ✅ Optimized for low disk space

## Troubleshooting

- **Disk space issues**: The app uses `/tmp/hf_cache` for model caching
- **Port conflicts**: Change `PORT` environment variable
- **Memory issues**: Reduce workers in `gunicorn.conf.py`

## Logs

- Application logs: `/var/log/researchsite/`
- System logs: `journalctl -u researchsite`