#!/usr/bin/env bash
# Installs pocketchange as an always-on service on a Raspberry Pi (Raspberry Pi OS 64-bit, Bookworm) or a Linux
# server (Ubuntu 24.04, Debian 12). Needs Python 3.11+. Run as root. Safe to re-run: it updates the checkout and keeps
# .env and data/.
#
#   sudo bash setup.sh [--branch main] [--no-dashboard] [--tailscale] [--tunnel] [--public]
#
#   --no-dashboard  run only the bot (least memory and CPU)
#   --tailscale     install Tailscale and serve the dashboard to your own devices only (recommended for remote access)
#   --tunnel        host the dashboard through a dev tunnel, private to your Microsoft/GitHub account
#   --public        same, but anyone with the URL can open it (sets DASHBOARD_PUBLIC=true: read-only, no controls)
#
# On a first install it also sets LOOP_INTERVAL_SECONDS=300 and SNIPER_INTERVAL_SECONDS=60 in .env: the strategy
# trades 4h candles, so a cycle a minute only spends CPU and network.
set -euo pipefail

REPO=https://github.com/AbhiramV010/pocketChange.git
BRANCH=main
DASHBOARD=1
TAILSCALE=0
TUNNEL=0
PUBLIC=0
APP_USER=pocketchange
APP_DIR=/opt/pocketchange
TUNNEL_ID=pocketchange
PORT=8501  # launch.py's default; nothing else listens on a fresh box

while [ $# -gt 0 ]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --no-dashboard) DASHBOARD=0; shift ;;
    --tailscale) TAILSCALE=1; shift ;;
    --tunnel) TUNNEL=1; shift ;;
    --public) TUNNEL=1; PUBLIC=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ "$DASHBOARD" -eq 0 ] && { [ "$TAILSCALE" -eq 1 ] || [ "$TUNNEL" -eq 1 ]; }; then
  echo "--tailscale and --tunnel serve the dashboard; drop --no-dashboard to use them" >&2; exit 2
fi

[ "$(id -u)" -eq 0 ] || { echo "run as root: sudo bash $0 $*" >&2; exit 1; }

say() { echo "[setup] $*"; }
as_app() { runuser -u "$APP_USER" -- env HOME="/home/$APP_USER" "$@"; }
set_env() {  # set KEY=VALUE in .env, replacing an existing line for KEY
  local key="$1" value="$2" env="$APP_DIR/.env"
  if grep -q "^${key}=" "$env"; then sed -i "s|^${key}=.*|${key}=${value}|" "$env"; else echo "${key}=${value}" >> "$env"; fi
}

# Small machines (a 1 GB Pi or VM) run out of memory installing pandas/scikit-learn without swap.
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
  || { echo "Python 3.11+ is required: use Raspberry Pi OS Bookworm (64-bit), Ubuntu 24.04 or Debian 12" >&2; exit 1; }

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

first_install=0
[ -f "$APP_DIR/.env" ] || first_install=1

# Builds .venv, creates .env from .env.example (DRY_RUN=true) and checks the price feed.
say "running launch.py pre-flight (the first run installs dependencies: several minutes, longer on a Pi) ..."
as_app python3 "$APP_DIR/launch.py" --check --no-browser

if [ "$first_install" -eq 1 ]; then
  say "first install: a cycle every 5 minutes and a sniper step every minute (edit $APP_DIR/.env to change)"
  set_env LOOP_INTERVAL_SECONDS 300
  set_env SNIPER_INTERVAL_SECONDS 60
fi
if [ "$PUBLIC" -eq 1 ]; then
  set_env DASHBOARD_PUBLIC true  # anyone with the link may look; nobody may press STOP, RISKY or change the timeframe
fi

LAUNCH_FLAGS="--no-browser --port $PORT"
[ "$DASHBOARD" -eq 1 ] || LAUNCH_FLAGS="$LAUNCH_FLAGS --no-dashboard"

# launch.py turns SIGTERM into a clean stop and waits up to 150s for the bot to finish a swap.
cat > /etc/systemd/system/pocketchange.service <<EOF
[Unit]
Description=pocketchange trading bot$([ "$DASHBOARD" -eq 1 ] && echo " + dashboard")
After=network-online.target
Wants=network-online.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python launch.py $LAUNCH_FLAGS
Restart=always
RestartSec=30
KillMode=mixed
TimeoutStopSec=180
Nice=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q pocketchange.service
systemctl restart pocketchange.service
say "pocketchange.service started (logs: journalctl -u pocketchange -f, or $APP_DIR/data/bot.log)"

if [ "$TAILSCALE" -eq 1 ]; then
  if ! command -v tailscale >/dev/null; then
    say "installing Tailscale (its official install script) ..."
    curl -fsSL https://tailscale.com/install.sh | sh
  fi
  if ! tailscale status >/dev/null 2>&1; then
    say "sign in to Tailscale: open the URL below with the account your phone or laptop uses"
    tailscale up
  fi
  tailscale serve --bg "$PORT"
  say "dashboard on your devices: $(tailscale serve status 2>/dev/null | grep -o 'https://[^ ]*' | head -1)"
  say "in the Tailscale admin console, choose 'Disable key expiry' for this machine, or it drops off in 180 days"
fi

if [ "$TUNNEL" -eq 1 ]; then
  case "$(uname -m)" in
    x86_64) arch=x64 ;;
    aarch64) arch=arm64 ;;
    *) echo "no devtunnel build for $(uname -m); use --tailscale" >&2; exit 1 ;;
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

say "done."
[ "$DASHBOARD" -eq 0 ] || say "the dashboard listens on 127.0.0.1:$PORT; without Tailscale or a tunnel: ssh -L $PORT:localhost:$PORT <user>@<this machine>"
