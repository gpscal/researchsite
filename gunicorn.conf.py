# Simplified Gunicorn configuration for researchsite
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:80"
backlog = 2048

# Worker processes - use fewer workers to save memory
workers = min(multiprocessing.cpu_count(), 4)  # Cap at 4 workers
worker_class = "gthread"
threads = 2
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

# Preload app for better performance
preload_app = True

# Graceful timeout
graceful_timeout = 60

# Memory management
max_worker_memory = 1000  # MB - increased for ML models