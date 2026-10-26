#!/usr/bin/env bash
# Sets up pocketchange on a fresh Ubuntu 24.04 / Debian 12 VPS (needs Python 3.11+).
# Runs launch.py (bot + dashboard) under systemd so it survives reboots and crashes, and can host the
# dashboard through a dev tunnel. Run as root. Safe to re-run: it updates the checkout and keeps .env and data/.
#
#   bash setup-vps.sh [--branch main] [--tunnel] [--public]
#
#   --tunnel  install the devtunnel CLI, sign in with a device code, and host the dashboard (private to you)
#   --public  same, but anyone with the URL can open the dashboard (it has no login and can pause the bot)
set -euo pipefail

REPO=https://github.com/AbhiramV010/pocketChange.git
BRANCH=main
TUNNEL=0
PUBLIC=0
APP_USER=pocketchange
APP_DIR=/opt/pocketchange
TUNNEL_ID=pocketchange-vps
PORT=8501  # launch.py's default; nothing else listens on a fresh VPS

while [ $# -gt 0 ]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --tunnel) TUNNEL=1; shift ;;
    --public) TUNNEL=1; PUBLIC=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "run as root: sudo bash $0 $*" >&2; exit 1; }

say() { echo "[setup] $*"; }
as_app() { runuser -u "$APP_USER" -- env HOME="/home/$APP_USER" "$@"; }

# Small shapes (e.g. Oracle's 1 GB VM.Standard.E2.1.Micro) run out of memory installing pandas/scikit-learn.
if [ -z "$(swapon --noheadings)" ] && [ "$(awk '/MemTotal/ {print $2}' /proc/meminfo)" -lt 2000000 ]; then
  say "adding a 2 GB swap file (under 2 GB of RAM) ..."
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

say "installing system packages ..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3 python3-venv curl ca-certificates >/dev/null
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || { echo "Python 3.11+ is required; use Ubuntu 24.04 or Debian 12" >&2; exit 1; }

id "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"

if [ -d "$APP_DIR/.git" ]; then
  say "updating $APP_DIR to origin/$BRANCH ..."
  as_app git -C "$APP_DIR" fetch -q origin "$BRANCH"
  as_app git -C "$APP_DIR" checkout -q -B "$BRANCH" "origin/$BRANCH"
else
  say "cloning $REPO ($BRANCH) into $APP_DIR ..."
  install -d -o "$APP_USER" -g "$APP_USER" "$APP_DIR"
  as_app git clone -q --branch "$BRANCH" "$REPO" "$APP_DIR"
fi

# Builds .venv, creates .env from .env.example (DRY_RUN=true) and checks the price feed.
say "running launch.py pre-flight (first run installs dependencies, takes a few minutes) ..."
as_app python3 "$APP_DIR/launch.py" --check --no-browser

# launch.py turns SIGTERM into a clean stop and waits up to 150s for the bot to finish a swap.
cat > /etc/systemd/system/pocketchange.service <<EOF
[Unit]
Description=pocketchange bot + dashboard
After=network-online.target
Wants=network-online.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python launch.py --no-browser --port $PORT
Restart=always
RestartSec=30
KillMode=mixed
TimeoutStopSec=180

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q pocketchange.service
systemctl restart pocketchange.service
say "pocketchange.service started (logs: journalctl -u pocketchange -f, or $APP_DIR/data/)"

if [ "$TUNNEL" -eq 1 ]; then
  case "$(uname -m)" in
    x86_64) arch=x64 ;;
    aarch64) arch=arm64 ;;
    *) echo "no devtunnel build for $(uname -m)" >&2; exit 1 ;;
  esac
  if ! command -v devtunnel >/dev/null; then
    say "installing the devtunnel CLI ..."
    curl -fsSL -o /usr/local/bin/devtunnel "https://tunnelsassetsprod.blob.core.windows.net/cli/linux-$arch-devtunnel"
    chmod 755 /usr/local/bin/devtunnel
  fi
  if ! as_app devtunnel user show 2>/dev/null | grep -q "Logged in"; then
    say "sign in to dev tunnels: open the URL below and enter the code (it expires in about 15 minutes)"
    as_app devtunnel user login -d
  fi
  as_app devtunnel show "$TUNNEL_ID" >/dev/null 2>&1 || as_app devtunnel create "$TUNNEL_ID" >/dev/null
  as_app devtunnel port show "$TUNNEL_ID" -p "$PORT" >/dev/null 2>&1 \
    || as_app devtunnel port create "$TUNNEL_ID" -p "$PORT" --protocol http >/dev/null
  as_app devtunnel access reset "$TUNNEL_ID" >/dev/null
  if [ "$PUBLIC" -eq 1 ]; then
    as_app devtunnel access create "$TUNNEL_ID" --anonymous >/dev/null
  fi

  cat > /etc/systemd/system/pocketchange-tunnel.service <<EOF
[Unit]
Description=dev tunnel to the pocketchange dashboard
After=network-online.target pocketchange.service
Wants=network-online.target

[Service]
User=$APP_USER
ExecStart=/usr/local/bin/devtunnel host $TUNNEL_ID
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable -q pocketchange-tunnel.service
  systemctl restart pocketchange-tunnel.service
  sleep 10
  url=$(journalctl -u pocketchange-tunnel --since "-1min" --no-pager | grep -o "https://[^ ]*-$PORT\.[^ ]*devtunnels\.ms" | head -1 || true)
  say "tunnel ($([ "$PUBLIC" -eq 1 ] && echo public || echo private)): ${url:-see journalctl -u pocketchange-tunnel}"
fi

say "done. The dashboard listens on 127.0.0.1:$PORT only; from your PC without a tunnel: ssh -L $PORT:localhost:$PORT root@<vps-ip>"
