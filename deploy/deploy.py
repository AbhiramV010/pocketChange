"""Deploy pocketchange to a Google Cloud e2-micro VM. Run from the repo root on Windows:

    python deploy/deploy.py up        create the VM if needed, upload, install, start everything
    python deploy/deploy.py push      upload the current code and .env, restart services
    python deploy/deploy.py url       print the public (read-only) dashboard URL
    python deploy/deploy.py status    service status on the VM
    python deploy/deploy.py logs      last bot log lines
    python deploy/deploy.py private   open the private dashboard (with pause) at http://localhost:8501 via SSH

Services on the VM (systemd, restart on failure and at boot):
    pocketchange-bot       python -m bot.run
    pocketchange-private   Streamlit on 127.0.0.1:8501, full dashboard, reachable only through SSH
    pocketchange-public    Streamlit on 127.0.0.1:8502 with DASHBOARD_PUBLIC=true (read-only)
    pocketchange-tunnel    Cloudflare quick tunnel -> public dashboard, prints a https://*.trycloudflare.com URL
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VM = "pocketchange"
ZONE = "us-central1-a"  # e2-micro is in the free tier in us-central1, us-west1 and us-east1
REMOTE_DIR = "pocketchange"
UPLOAD = ["bot", "dashboard", "requirements.txt", ".env"]
GCLOUD_FALLBACKS = [  # the Windows installer does not always update PATH for running terminals
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd",
    Path(os.environ.get("ProgramFiles(x86)", "")) / "Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd",
]

SETUP = r"""
set -euo pipefail
APP="$HOME/{remote_dir}"
export DEBIAN_FRONTEND=noninteractive

if ! command -v python3 >/dev/null || ! python3 -m venv --help >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip curl
fi
if [ ! -f /swapfile ]; then  # e2-micro has 1 GB RAM; swap keeps pip and two dashboards from OOMing
  sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i /tmp/cloudflared.deb >/dev/null
fi

mkdir -p "$APP"
tar -xzf "$HOME/pocketchange.tar.gz" -C "$APP"
rm -f "$HOME/pocketchange.tar.gz"
cd "$APP"
[ -f .env ] && chmod 600 .env
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

ST=".venv/bin/streamlit run dashboard/app.py --server.headless true --browser.gatherUsageStats false --client.toolbarMode viewer --server.address 127.0.0.1"
unit() {{  # name, description, command, extra Environment= lines
  sudo tee "/etc/systemd/system/$1.service" >/dev/null <<UNIT
[Unit]
Description=$2
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP
Environment=PYTHONUNBUFFERED=1
Environment=CANDLE_EXCHANGE=kraken
$4
ExecStart=$3
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
}}
unit pocketchange-bot "pocketchange trading loop" "$APP/.venv/bin/python -m bot.run" ""
unit pocketchange-private "pocketchange private dashboard" "$APP/$ST --server.port 8501" ""
unit pocketchange-public "pocketchange public read-only dashboard" "$APP/$ST --server.port 8502" "Environment=DASHBOARD_PUBLIC=true"
unit pocketchange-tunnel "Cloudflare quick tunnel to the public dashboard" "/usr/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8502" ""

sudo systemctl daemon-reload
for s in pocketchange-bot pocketchange-private pocketchange-public pocketchange-tunnel; do
  sudo systemctl enable "$s" >/dev/null 2>&1
  sudo systemctl restart "$s"
done
echo "SETUP_OK"
"""

URL_CMD = (
    "sudo journalctl -u pocketchange-tunnel --no-pager -n 200 "
    "| grep -o 'https://[a-z0-9-]*\\.trycloudflare\\.com' | tail -1"
)


def gcloud(*args: str, capture: bool = False, check: bool = True) -> str:
    exe = shutil.which("gcloud") or next((str(p) for p in GCLOUD_FALLBACKS if p.exists()), None)
    if not exe:
        sys.exit("gcloud is not installed or not on PATH. Install the Google Cloud CLI and run `gcloud init`.")
    env = dict(os.environ)
    # gcloud needs a Python; on Windows PATH may only have the Microsoft Store stub.
    env.setdefault("CLOUDSDK_PYTHON", getattr(sys, "_base_executable", sys.executable))
    result = subprocess.run([exe, *args], capture_output=capture, text=True, env=env)
    if check and result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        sys.exit(f"gcloud {' '.join(args[:3])} ... failed (exit {result.returncode}). {detail}")
    return (result.stdout or "").strip() if capture else ""


def ssh(command: str, capture: bool = False, check: bool = True) -> str:
    return gcloud("compute", "ssh", VM, "--zone", ZONE, "--quiet", "--command", command, capture=capture, check=check)


def project() -> str:
    proj = gcloud("config", "get-value", "project", capture=True)
    if not proj or proj == "(unset)":
        sys.exit("No Google Cloud project selected. Run `gcloud init` (or `gcloud config set project <id>`).")
    return proj


def vm_exists() -> bool:
    out = gcloud("compute", "instances", "list", f"--filter=name={VM}", "--format=value(name)", capture=True)
    return out.strip() == VM


def create_vm() -> None:
    print("Enabling the Compute Engine API (first time can take a minute)...")
    gcloud("services", "enable", "compute.googleapis.com")
    print(f"Creating e2-micro VM '{VM}' in {ZONE}...")
    gcloud(
        "compute", "instances", "create", VM,
        "--zone", ZONE,
        "--machine-type", "e2-micro",
        "--image-family", "debian-12",
        "--image-project", "debian-cloud",
        "--boot-disk-size", "30GB",
        "--boot-disk-type", "pd-standard",
    )
    print("Waiting for SSH to come up...")
    for _ in range(20):
        if "ready" in ssh("echo ready", capture=True, check=False):
            return
        time.sleep(10)
    sys.exit("VM did not become reachable over SSH.")


def bundle() -> Path:
    if not (ROOT / ".env").exists():
        sys.exit("No .env found. Copy .env.example to .env first (DRY_RUN=true is fine).")
    tmp = Path(tempfile.gettempdir()) / "pocketchange.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        for name in UPLOAD:
            tar.add(ROOT / name, arcname=name, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return tmp


def push() -> None:
    archive = bundle()
    print("Uploading code and .env...")
    gcloud("compute", "scp", "--zone", ZONE, "--quiet", str(archive), f"{VM}:pocketchange.tar.gz")
    archive.unlink()
    script = Path(tempfile.gettempdir()) / "pocketchange_setup.sh"
    script.write_bytes(SETUP.format(remote_dir=REMOTE_DIR).replace("\r\n", "\n").encode())
    gcloud("compute", "scp", "--zone", ZONE, "--quiet", str(script), f"{VM}:pocketchange_setup.sh")
    script.unlink()
    print("Installing and (re)starting services (a few minutes the first time)...")
    out = ssh("bash pocketchange_setup.sh && rm pocketchange_setup.sh", capture=True, check=False)
    if "SETUP_OK" not in out:
        print(out[-2000:])
        sys.exit("Setup on the VM failed; see output above.")


def url(wait: bool = True) -> str:
    for _ in range(12 if wait else 1):
        found = ssh(URL_CMD, capture=True, check=False).strip()
        if found.startswith("https://"):
            return found
        time.sleep(5)
    return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["up", "push", "url", "status", "logs", "private"])
    args = parser.parse_args(argv)
    print(f"Project: {project()}  VM: {VM}  zone: {ZONE}")

    if args.command == "up":
        if not vm_exists():
            create_vm()
        push()
    elif args.command == "push":
        push()
    elif args.command == "status":
        services = "pocketchange-bot pocketchange-private pocketchange-public pocketchange-tunnel"
        print(ssh(f"for s in {services}; do printf '%-22s ' $s; systemctl is-active $s; done", capture=True, check=False))
        return
    elif args.command == "logs":
        print(ssh("tail -n 40 ~/pocketchange/data/bot.log", capture=True, check=False))
        return
    elif args.command == "private":
        print("Private dashboard: http://localhost:8501  (Ctrl+C to close the tunnel)")
        gcloud("compute", "ssh", VM, "--zone", ZONE, "--", "-L", "8501:127.0.0.1:8501", "-N", check=False)
        return

    public = url()
    print()
    print(f"Public read-only dashboard: {public or '(not ready yet; run: python deploy/deploy.py url)'}")
    print("Private dashboard with pause: python deploy/deploy.py private")
    print("The quick-tunnel URL changes whenever the tunnel service restarts.")


if __name__ == "__main__":
    main()
