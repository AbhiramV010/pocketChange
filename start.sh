#!/usr/bin/env bash
# Starts the bot and the trading dashboard.
# launch.py does the heavy lifting: creates .venv on first run, installs
# requirements, creates .env from .env.example, starts both processes and
# restarts them on crash. Press Ctrl+C to stop (the bot finishes its cycle).
#
# Usage: ./start.sh [--no-browser] [--port N] [--check] [--yes]
#   --no-browser  do not open the dashboard in a browser
#   --port N      dashboard port (default 8501, first free from there)
#   --check       pre-flight checks only; start nothing
#   --yes         skip the live-mode confirmation prompt
set -euo pipefail
cd "$(dirname "$0")"

# launch.py rebuilds .venv whenever it does not match the running interpreter.
# From WSL over /mnt that means deleting the Windows venv through the 9P mount,
# which is slow and can leave it half-built: refuse instead.
if grep -qi microsoft /proc/version 2>/dev/null && [ "${PWD#/mnt/}" != "$PWD" ]; then
  echo "start.sh: refusing to run from WSL over /mnt (it would rebuild the Windows .venv)." >&2
  echo "Run it from PowerShell, or clone the repo inside the WSL filesystem." >&2
  exit 1
fi

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "start.sh: Python 3.11+ is required" >&2
  exit 1
fi

exec "$PY" launch.py "$@"
