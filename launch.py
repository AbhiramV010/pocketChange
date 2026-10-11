from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from bot.config import Settings

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / "pocketchange-requirements.sha256"
MIN_PYTHON = (3, 11)

DEFAULT_PORT = 8501
HEALTH_TIMEOUT_S = 90
BOT_STOP_GRACE_S = 150  # longer than a swap's confirmation window: never kill the bot mid-swap
UI_STOP_GRACE_S = 10
MAX_RESTARTS = 5  # per process, within RESTART_WINDOW_S, before giving up
RESTART_WINDOW_S = 600
ALREADY_RUNNING_EXIT = 3  # must match bot.run.ALREADY_RUNNING_EXIT


def say(message: str) -> None:
    print(f"[launch] {message}", flush=True)


def fail(message: str) -> NoReturn:
    print(f"[launch] ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


# Step 0 runs under any Python, before the venv exists.

def in_project_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def ensure_venv() -> None:
    if not VENV_PY.exists():
        say("creating virtual environment in .venv ...")
        venv.EnvBuilder(with_pip=True, clear=VENV.exists()).create(VENV)
    if STAMP.exists() and STAMP.read_text().strip() == requirements_hash():
        return
    say("installing requirements (first run or requirements.txt changed) ...")
    for cmd in (
        [str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check", "-q", "--upgrade", "pip"],
        [str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", str(REQUIREMENTS)],
    ):
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            fail("pip install failed; see the output above")
    STAMP.write_text(requirements_hash())


def reexec_in_venv(argv: list[str]) -> None:
    proc = subprocess.Popen([str(VENV_PY), str(Path(__file__).resolve()), *argv], cwd=ROOT)
    while True:
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:
            continue  # the child got the same Ctrl+C and is shutting down; wait for it


# Steps 1-4 (pre-flight) run inside the venv.

def ensure_env_file() -> None:
    env, example = ROOT / ".env", ROOT / ".env.example"
    if env.exists():
        return
    if not example.exists():
        fail(".env and .env.example are both missing")
    shutil.copyfile(example, env)
    say("created .env from .env.example (dry run, no wallet key)")


def load_settings() -> "Settings":
    from bot.config import ConfigError, load_settings as load

    try:
        return load()
    except ConfigError as exc:
        fail(f"invalid setting in .env: {exc}")


def confirm_live(cfg, assume_yes: bool) -> None:
    from bot import chain

    print()
    print("  " + "=" * 64)
    print("   LIVE MODE: DRY_RUN=false. The bot will sign and send real swaps.")
    print(f"   Limits: MAX_TRADE_USDC={cfg.max_trade_usdc}  MAX_SLIPPAGE_BPS={cfg.max_slippage_bps}"
          f"  MIN_EQUITY_USD={cfg.min_equity_usd}")
    print("  " + "=" * 64)
    if not assume_yes:
        try:
            answer = input("  Type 'live' to continue, anything else to abort: ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "live":
            fail("aborted; set DRY_RUN=true in .env for a dry run")
    try:
        client = chain.ChainClient(cfg)
        balances = client.balances()
    except chain.ChainError as exc:
        fail(f"wallet check failed: {exc}")
    say(f"wallet {client.address}: USDC {balances.usdc:.2f} | wSOL {balances.wsol:.6f} | SOL (fees) {balances.sol:.4f}")
    if balances.sol < cfg.gas_reserve_sol:
        say(f"WARNING: SOL {balances.sol:.4f} is below GAS_RESERVE_SOL {cfg.gas_reserve_sol}; the bot will not trade")


def check_price_feed(cfg) -> None:
    from bot import data

    try:
        candles = data.fetch_recent(limit=3, exchange_id=cfg.candle_exchange)
        say(f"price feed OK: SOL/USDC {candles['close'].iloc[-1]:.2f}")
    except Exception as exc:
        say(f"WARNING: price feed unreachable right now ({type(exc).__name__}); the bot will keep retrying")


def init_database(cfg) -> None:
    from bot import db

    conn = db.connect(cfg.db_path)
    try:
        db.init_db(conn)
        if db.is_halted(conn):
            say(f"NOTE: the bot is HALTED ({db.halt_reason(conn)}). It will log balances but not trade.")
            say("      Clear it with: python -m bot.reset_halt")
        if db.is_paused(conn):
            say("NOTE: trading is PAUSED (toggle it in the dashboard)")
    finally:
        conn.close()


def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    fail(f"no free port found from {start}")


class Service:

    def __init__(self, name: str, cmd: list[str], log_file: Path | None = None, stop_grace: float = UI_STOP_GRACE_S):
        self.name, self.cmd, self.log_file, self.stop_grace = name, cmd, log_file, stop_grace
        self.proc: subprocess.Popen | None = None
        self.restarts: deque[float] = deque()

    def start(self) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        # Own process group: a console Ctrl+C reaches only the launcher, which then stops
        # each child deliberately (the bot first finishes its cycle) instead of all at once.
        group = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        )
        self.proc = subprocess.Popen(
            self.cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, **group,
        )
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen) -> None:
        sink = open(self.log_file, "a", encoding="utf-8") if self.log_file else None
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                print(f"[{self.name}] {line.rstrip()}", flush=True)
                if sink:
                    sink.write(line)
                    sink.flush()
        finally:
            if sink:
                sink.close()

    def exit_code(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def may_restart(self) -> bool:
        now = time.monotonic()
        while self.restarts and now - self.restarts[0] > RESTART_WINDOW_S:
            self.restarts.popleft()
        if len(self.restarts) >= MAX_RESTARTS:
            return False
        self.restarts.append(now)
        return True

    def request_stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        except OSError:
            pass

    def wait_stopped(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.wait(timeout=self.stop_grace)
        except subprocess.TimeoutExpired:
            say(f"{self.name} did not stop within {self.stop_grace:g}s; killing it")
            self.proc.kill()
            self.proc.wait()


def wait_for_dashboard(port: int, service: Service) -> bool:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if service.exit_code() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def supervise(services: list[Service]) -> int:
    while True:
        time.sleep(1)
        for svc in services:
            code = svc.exit_code()
            if code is None:
                continue
            if code == 0:  # a clean exit only happens on Ctrl+C / shutdown: never restart it
                say(f"{svc.name} stopped; shutting down")
                return 0
            if svc.name == "bot" and code == ALREADY_RUNNING_EXIT:
                say("another bot is already running on this database; stopping (only one may trade a wallet)")
                return 1
            if not svc.may_restart():
                say(f"{svc.name} crashed {MAX_RESTARTS} times in {RESTART_WINDOW_S // 60} min; giving up. "
                    f"See data/{svc.name}.log")
                return 1
            delay = 2 ** len(svc.restarts)
            say(f"{svc.name} exited with code {code}; restarting in {delay}s")
            time.sleep(delay)
            svc.start()


def run(args: argparse.Namespace) -> int:
    ensure_env_file()
    cfg = load_settings()
    mode = "DRY RUN (simulated, nothing on-chain)" if cfg.dry_run else "LIVE"
    say(f"mode: {mode} | strategy: {cfg.strategy} | cycle: {cfg.loop_interval_seconds:g}s | db: {cfg.db_path}")
    if not cfg.dry_run:
        confirm_live(cfg, args.yes)
    check_price_feed(cfg)
    init_database(cfg)
    port = free_port(args.port)
    if args.check:
        say(f"pre-flight OK; the dashboard would use port {port}")
        return 0

    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    bot = Service("bot", [py, "-m", "bot.run"], stop_grace=BOT_STOP_GRACE_S)  # bot.run writes data/bot.log itself
    ui = Service(
        "dashboard",
        [py, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
         "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        log_file=cfg.db_path.parent / "dashboard.log",
    )
    services = [bot, ui]
    try:
        bot.start()
        ui.start()
        url = f"http://localhost:{port}"
        if wait_for_dashboard(port, ui):
            say(f"dashboard ready: {url}")
            if not args.no_browser:
                webbrowser.open(url)
        else:
            say(f"WARNING: dashboard did not answer within {HEALTH_TIMEOUT_S}s; see data/dashboard.log")
        say("running. Press Ctrl+C to stop the bot and the dashboard.")
        return supervise(services)
    except KeyboardInterrupt:
        say("stopping ...")
        return 0
    finally:
        for svc in services:
            svc.request_stop()
        say("waiting for the bot to finish its current cycle (Ctrl+C again to force) ...")
        for svc in services:
            try:
                svc.wait_stopped()
            except KeyboardInterrupt:  # second Ctrl+C: stop waiting, force everything
                for other in services:
                    if other.proc and other.proc.poll() is None:
                        other.proc.kill()
                break
        say("stopped.")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Start the pocketchange bot and dashboard.")
    parser.add_argument("--no-browser", action="store_true", help="do not open the dashboard in a browser")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="dashboard port (first free from here)")
    parser.add_argument("--check", action="store_true", help="pre-flight checks only; start nothing")
    parser.add_argument("--yes", action="store_true", help="skip the live-mode confirmation prompt")
    args = parser.parse_args(argv)

    if sys.version_info < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, this is {sys.version.split()[0]}")
    os.chdir(ROOT)
    for name in ("SIGTERM", "SIGBREAK"):  # treat a terminate / Ctrl+Break like Ctrl+C, in both launcher stages
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.default_int_handler)
    if not in_project_venv():
        ensure_venv()
        reexec_in_venv(argv)
    ensure_venv()  # also inside the venv, in case requirements.txt changed since it was built
    sys.path.insert(0, str(ROOT))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
