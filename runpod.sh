#!/bin/bash
set -euo pipefail

# Unified non-Docker installer/runner for RunPod or any Ubuntu host
# - Installs system deps (nginx, certbot, python3-venv)
# - Creates/uses a local Python venv and installs requirements
# - Configures nginx for DOMAIN (HTTP -> HTTPS) and obtains Let's Encrypt cert
# - Starts Gunicorn bound to 127.0.0.1:8080 (nginx terminates TLS)
# - Provides start/stop/status/renew helpers without systemd units

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="/var/run/researchsite"
LOG_DIR="/var/log/researchsite"
SITE_NAME="researchsite"
DOMAIN_DEFAULT="researchbuddy.org"

command_exists() { command -v "$1" >/dev/null 2>&1; }

ensure_dirs() {
  sudo mkdir -p "$RUN_DIR" "$LOG_DIR" "$APP_DIR/data/uploads" "$APP_DIR/data/vector_store"
  # Use current user or fallback to root
  local current_user="${USER:-$(whoami)}"
  sudo chown -R "$current_user":"$current_user" "$APP_DIR" || true
}

nginx_reload() {
  if command_exists systemctl; then
    sudo systemctl reload nginx || sudo systemctl restart nginx || sudo nginx -s reload || true
  else
    sudo service nginx reload || sudo service nginx restart || sudo nginx -s reload || true
  fi
}

nginx_start() {
  if command_exists systemctl; then
    sudo systemctl enable --now nginx || sudo systemctl start nginx || true
  else
    sudo service nginx start || true
  fi
}

create_http_only_site() {
  local domain="$1"
  local conf="/etc/nginx/sites-available/$SITE_NAME"
  sudo bash -c "cat > '$conf'" <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${domain} www.${domain};

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location /static/ { alias ${APP_DIR}/static/; }

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
    }
}
EOF
  sudo mkdir -p /var/www/certbot
  sudo ln -sf "$conf" "/etc/nginx/sites-enabled/$SITE_NAME"
  sudo rm -f /etc/nginx/sites-enabled/default || true
  sudo nginx -t
  nginx_start
  nginx_reload
}

obtain_cert() {
  local domain="$1" email="$2"
  sudo certbot --nginx -d "$domain" -d "www.$domain" --email "$email" --agree-tos --non-interactive || true
}

create_venv_and_install() {
  python3 -m venv "$APP_DIR/.venv" || true
  source "$APP_DIR/.venv/bin/activate"
  python -m pip install --upgrade pip
  pip install -r "$APP_DIR/requirements.txt"
}

start_gunicorn() {
  source "$APP_DIR/.venv/bin/activate"
  echo "[Start] Gunicorn on 127.0.0.1:8080"
  nohup gunicorn -w 5 -k gthread --threads 2 -b 127.0.0.1:8080 app:app --timeout 180 \
    > "$LOG_DIR/gunicorn.log" 2>&1 &
  echo $! | sudo tee "$RUN_DIR/gunicorn.pid" >/dev/null
}

stop_gunicorn() {
  if [ -f "$RUN_DIR/gunicorn.pid" ]; then
    PID=$(cat "$RUN_DIR/gunicorn.pid" || true)
    if [ -n "${PID:-}" ] && ps -p "$PID" >/dev/null 2>&1; then
      kill "$PID" || true
      sleep 1
    fi
    sudo rm -f "$RUN_DIR/gunicorn.pid"
  fi
}

wait_ready() {
  local host="${1:-127.0.0.1}" retries=30
  echo "[Wait] Checking readiness at http://${host}/ready"
  for i in $(seq 1 $retries); do
    if curl -fsS "http://${host}/ready" >/dev/null 2>&1; then
      echo "[Ready] Application responded"
      return 0
    fi
    sleep 2
  done
  echo "[Warn] App did not confirm readiness; continue anyway"
}

cmd_install() {
  local domain="${DOMAIN:-$DOMAIN_DEFAULT}" email="${CERTBOT_EMAIL:-admin@${DOMAIN_DEFAULT}}"
  while (( "$#" )); do
    case "$1" in
      --domain) domain="$2"; shift 2;;
      --email) email="$2"; shift 2;;
      *) echo "Unknown flag: $1"; exit 1;;
    esac
  done
  echo "[Install] Domain=$domain Email=$email"

  ensure_dirs
  sudo apt-get update
  sudo apt-get install -y nginx certbot python3-certbot-nginx python3-venv python3-pip \
    build-essential git curl

  create_venv_and_install
  create_http_only_site "$domain"
  obtain_cert "$domain" "$email" || true
  nginx_reload
}

cmd_start() {
  ensure_dirs
  start_gunicorn
  wait_ready "127.0.0.1:8080" || true
  nginx_reload
}

cmd_stop() {
  stop_gunicorn
}

cmd_restart() {
  cmd_stop || true
  cmd_start
}

cmd_status() {
  if [ -f "$RUN_DIR/gunicorn.pid" ]; then
    PID=$(cat "$RUN_DIR/gunicorn.pid")
    if ps -p "$PID" >/dev/null 2>&1; then
      echo "Gunicorn: running (pid $PID)"
    else
      echo "Gunicorn: pid file present but process not running"
    fi
  else
    echo "Gunicorn: not running"
  fi
  sudo nginx -t && echo "Nginx: config OK" || echo "Nginx: config ERROR"
}

cmd_renew() {
  sudo certbot renew --quiet || true
  nginx_reload
}

usage() {
  cat <<USAGE
Usage: $0 <command> [--domain DOMAIN --email EMAIL]
Commands:
  install    Install deps, create venv, configure nginx, obtain SSL
  start      Start Gunicorn (background)
  stop       Stop Gunicorn
  restart    Restart Gunicorn
  status     Show Gunicorn and nginx config status
  renew      Renew SSL certs and reload nginx

Examples:
  $0 install --domain calebgonsalves.com --email admin@calebgonsalves.com
  $0 start
USAGE
}

main() {
  local cmd="${1:-}"; shift || true
  case "$cmd" in
    install) cmd_install "$@";;
    start) cmd_start;;
    stop) cmd_stop;;
    restart) cmd_restart;;
    status) cmd_status;;
    renew) cmd_renew;;
    *) usage; exit 1;;
  esac
}

main "$@"


