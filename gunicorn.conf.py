# Simplified Gunicorn configuration for researchsite
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:80"
backlog = 2048

# Worker processes - use 1 worker for qwen model (62GB GPU memory)
# Multiple workers would try to load the model and run out of GPU memory
workers = 1
worker_class = "gthread"
threads = 4  # Use more threads to handle concurrent requests
worker_connections = 1000
timeout = 300  # Increased timeout for model loading
keepalive = 2

# Restart workers after this many requests
max_requests = 500
max_requests_jitter = 50

# Logging
accesslog = "/var/log/researchsite/access.log"
errorlog = "/var/log/researchsite/error.log"
loglevel = "info"

# Process naming
proc_name = "researchsite"

# Server mechanics
daemon = False
pidfile = "/var/run/researchsite/researchsite.pid"

# Environment variables
raw_env = [
    "PORT=80",
    "PYTHONPATH=/researchsite",
]

# IMPORTANT: Disable preload to avoid CUDA forking issues
# CUDA cannot be re-initialized in forked subprocesses
preload_app = False

# Graceful timeout
graceful_timeout = 60

# Memory management
max_worker_memory = 1000  # MB - increased for ML models

# Worker lifecycle hooks for CUDA compatibility
def post_fork(server, worker):
    """Called after a worker has been forked."""
    # Ensure each worker has its own CUDA context
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
        worker.log.info(f"Worker {worker.pid}: CUDA initialized")