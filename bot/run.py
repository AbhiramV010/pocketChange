"""The trading loop. Run it as its own process: python -m bot.run

Each cycle: top-of-cycle flags, balances, price, equity floor, signal, trade.
Errors are logged and the loop sleeps and carries on; it only stops on Ctrl+C.
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Callable, Protocol

import pandas as pd

from bot import chain, data, db
from bot.config import FEE_RATE, MIN_TRADE_USDC, SLIPPAGE_RATE, SMA_SLOW, TIMEFRAME, Settings, settings
from bot.db import Balances, Fill
from bot.strategy import Signal, dca_signal, size_trade, sma_crossover_signal

log = logging.getLogger("bot")

MAX_SWAP_RETRIES = 1  # hard rule: a failed swap is retried at most once per cycle
MAX_ATTEMPT_CYCLES = 3  # cycles that may try the same signal before giving up on it
def max_data_age(timeframe: str) -> pd.Timedelta:
    """Older candles than this mean the feed is stale."""
    return max(data.candle_delta(timeframe) * 2, pd.Timedelta(minutes=15))
CANDLES_WANTED = 120
ALREADY_RUNNING_EXIT = 3  # exit code when another bot holds the instance lock

_stop = threading.Event()


def request_stop(signum=None, frame=None) -> None:
    """Signal handler: finish the current cycle, then exit. A second signal stops immediately."""
    if _stop.is_set():
        raise KeyboardInterrupt
    _stop.set()
    log.info("stop requested; finishing the current cycle (signal again to force)")


def install_stop_handlers() -> None:
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):  # SIGBREAK: Ctrl+Break / console close on Windows
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)


def acquire_instance_lock(path: Path) -> IO | None:
    """Exclusive lock for the life of the process, so two bots never trade one wallet.

    Returns the open lock file (keep a reference to it) or None if another
    process holds it. The OS releases the lock if the process dies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


class Broker(Protocol):
    def balances(self) -> Balances: ...
    def swap(self, side: str, amount_in: float, ref_price: float) -> Fill: ...


class SimBroker:
    """Dry run: a simulated wallet replayed from simulated trades. Sends nothing on-chain."""

    def __init__(self, conn: sqlite3.Connection, cfg: Settings):
        self._conn, self._cfg = conn, cfg

    def balances(self) -> Balances:
        return db.sim_balances(self._conn, self._cfg.sim_start_usdc, self._cfg.sim_start_sol)

    def swap(self, side: str, amount_in: float, ref_price: float) -> Fill:
        have = self.balances()
        cap = self._cfg.max_trade_usdc
        if side == chain.BUY:
            if amount_in > cap + 1e-9:
                raise chain.SwapFailed(f"buy of {amount_in} USDC exceeds MAX_TRADE_USDC={cap}")
            if amount_in > have.usdc + 1e-9:
                raise chain.SwapFailed("insufficient simulated USDC")
            sol_out = amount_in * (1 - FEE_RATE) / (ref_price * (1 + SLIPPAGE_RATE))
            return Fill(amount_in, sol_out, amount_in / sol_out, None)
        if amount_in > have.wsol + 1e-12:
            raise chain.SwapFailed("insufficient simulated wSOL")
        usdc_out = amount_in * ref_price * (1 - SLIPPAGE_RATE) * (1 - FEE_RATE)
        if usdc_out > cap + 1e-9:
            raise chain.SwapFailed(f"sell of ~{usdc_out:.2f} USDC exceeds MAX_TRADE_USDC={cap}")
        return Fill(amount_in, usdc_out, usdc_out / amount_in, None)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Context:
    cfg: Settings
    conn: sqlite3.Connection
    broker: Broker
    feed: Callable[[str], pd.DataFrame]  # timeframe -> recent candles, including the one still forming
    clock: Callable[[], datetime] = field(default=_utc_now)


# --- one cycle ---------------------------------------------------------------

def run_cycle(ctx: Context) -> None:
    cfg, conn = ctx.cfg, ctx.conn
    now = ctx.clock()
    stamp = now.isoformat(timespec="seconds")
    db.set_state(conn, db.HEARTBEAT, stamp)
    db.set_state(conn, db.DRY_RUN, "1" if cfg.dry_run else "0")

    paused, halted = db.is_paused(conn), db.is_halted(conn)  # top of the cycle (hard rule 6)
    timeframe = db.get_timeframe(conn) or TIMEFRAME  # the dashboard can switch it live

    balances = ctx.broker.balances()
    candles, price = _market(ctx, now, timeframe)
    db.insert_balance(conn, balances, price, cfg.dry_run, timestamp=stamp)
    if price is None:
        log.warning("no usable price; not trading this cycle")
        return

    if not halted:
        halted = _enforce_equity_floor(ctx, balances, price)
    if halted:
        log.info("HALTED (%s); not trading. Clear with: python -m bot.reset_halt", db.halt_reason(conn))
        return
    if paused:
        log.info("paused; not trading (usdc=%.2f wsol=%.6f price=%.2f)", balances.usdc, balances.wsol, price)
        return
    _trade_if_signal(ctx, candles, balances, price, now, timeframe)


def _market(ctx: Context, now: datetime, timeframe: str) -> tuple[pd.DataFrame, float | None]:
    """Latest candles and the current price, or (empty, None) if the feed is down or stale."""
    try:
        candles = ctx.feed(timeframe)
    except Exception as exc:  # network, rate limit, exchange outage
        log.error("candle feed failed: %s: %s", type(exc).__name__, str(exc)[:200])
        return pd.DataFrame(), None
    if candles.empty or pd.Timestamp(now) - candles.index[-1] > max_data_age(timeframe):
        log.error("candle feed is stale (latest %s)", candles.index[-1] if len(candles) else "none")
        return candles, None
    return candles, float(candles["close"].iloc[-1])


def equity_usd(balances: Balances, price: float) -> float:
    """USDC + wSOL at `price`. Native SOL (the fee reserve) is never counted."""
    return balances.usdc + balances.wsol * price


def _enforce_equity_floor(ctx: Context, balances: Balances, price: float) -> bool:
    """Set the persistent halt flag if equity is under the floor. Returns True if halted."""
    equity = equity_usd(balances, price)
    if equity >= ctx.cfg.min_equity_usd:
        return False
    reason = f"equity ${equity:.2f} is below MIN_EQUITY_USD ${ctx.cfg.min_equity_usd:.2f}"
    db.set_halted(ctx.conn, reason)
    log.critical("HALTING: %s. The open position is not sold.", reason)
    return True


def _trade_if_signal(
    ctx: Context, candles: pd.DataFrame, balances: Balances, price: float, now: datetime, timeframe: str = TIMEFRAME
) -> None:
    cfg, conn = ctx.cfg, ctx.conn
    closed = data.closed_only(candles, now, timeframe)
    if cfg.strategy == "sma":
        if len(closed) < SMA_SLOW + 1:
            log.info("warming up: %d of %d closed candles", len(closed), SMA_SLOW + 1)
            return
        signal = sma_crossover_signal(closed)
    else:
        signal = dca_signal(closed, db.last_trade_time(conn, "buy", cfg.dry_run), cfg.dca_interval_hours, timeframe)
    equity = equity_usd(balances, price)
    if signal is Signal.HOLD:
        log.info("hold | %s | price=%.2f equity=%.2f usdc=%.2f wsol=%.6f", timeframe, price, equity, balances.usdc, balances.wsol)
        return

    key = f"{closed.index[-1].isoformat()}:{signal.value}"
    if db.get_state(conn, db.DONE_SIGNAL) == key:
        return
    attempts = int(db.get_state(conn, db.ATTEMPT_COUNT, "0") or 0) if db.get_state(conn, db.ATTEMPT_SIGNAL) == key else 0
    if attempts >= MAX_ATTEMPT_CYCLES:
        return  # gave up on this signal; the next one starts fresh

    amount = size_trade(
        signal, cfg.strategy, usdc=balances.usdc, wsol=balances.wsol, price=price,
        max_trade_usdc=cfg.max_trade_usdc, dca_amount_usdc=cfg.dca_amount_usdc, min_trade_usdc=MIN_TRADE_USDC,
    )
    if amount <= 0:
        log.info("%s signal, nothing to trade (usdc=%.2f wsol=%.6f)", signal.value, balances.usdc, balances.wsol)
        db.set_state(conn, db.DONE_SIGNAL, key)
        return
    if balances.sol < cfg.gas_reserve_sol:
        log.warning("%s signal skipped: native SOL %.6f is below GAS_RESERVE_SOL %s", signal.value, balances.sol, cfg.gas_reserve_sol)
        return

    db.set_state(conn, db.ATTEMPT_SIGNAL, key)
    db.set_state(conn, db.ATTEMPT_COUNT, str(attempts + 1))
    if db.is_paused(conn) or db.is_halted(conn):  # flipped while this cycle was running
        log.info("paused/halted during the cycle; not trading")
        return

    side = signal.value
    log.info("%s signal: trading %.9f %s at ~%.2f", side, amount, "USDC" if side == "buy" else "wSOL", price)
    fill = _swap_with_one_retry(ctx, side, amount, price)
    if fill is None:
        return

    db.insert_trade(conn, side, fill, cfg.dry_run, timestamp=ctx.clock().isoformat(timespec="seconds"))
    db.set_state(conn, db.ATTEMPT_COUNT, "0")
    left_usd = (balances.wsol - fill.amount_in) * price
    if side == "buy" or left_usd < MIN_TRADE_USDC:
        db.set_state(conn, db.DONE_SIGNAL, key)  # a capped sell of a big position continues next cycle
    log.info("%s filled: in=%.9f out=%.9f price=%.2f %s", side, fill.amount_in, fill.amount_out, fill.price, fill.tx_hash or "(simulated)")

    try:  # equity floor right after every trade (hard rule 8)
        after = ctx.broker.balances()
        db.insert_balance(conn, after, price, cfg.dry_run, timestamp=ctx.clock().isoformat(timespec="seconds"))
        _enforce_equity_floor(ctx, after, price)
    except Exception as exc:
        log.error("post-trade balance check failed, will re-check next cycle: %s", str(exc)[:200])


def _swap_with_one_retry(ctx: Context, side: str, amount: float, price: float) -> Fill | None:
    """At most one retry, and only when the swap definitely did not execute."""
    for attempt in range(1 + MAX_SWAP_RETRIES):
        try:
            return ctx.broker.swap(side, amount, price)
        except chain.SwapFailed as exc:
            log.error("%s failed (attempt %d of %d): %s", side, attempt + 1, 1 + MAX_SWAP_RETRIES, exc)
        except chain.SwapUncertain as exc:
            # A transaction may still land. Retrying could double-spend, so stop and ask for a human.
            reason = f"{side} outcome unknown: {exc}. Check the wallet on Solscan, then run bot.reset_halt"
            db.set_halted(ctx.conn, reason)
            log.critical("HALTING: %s", reason)
            return None
        except chain.ChainError as exc:  # RPC trouble before anything was sent; next cycle tries again
            log.error("%s aborted, no retry this cycle: %s", side, exc)
            return None
    return None


# --- process entry -------------------------------------------------------------

def configure_logging(cfg: Settings) -> None:
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(cfg.log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"),
    ]
    logging.Formatter.converter = time.gmtime  # UTC, same as the database
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%SZ",
        handlers=handlers,
        force=True,
    )
    for noisy in ("urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_loop(ctx: Context, once: bool = False) -> None:
    """Run cycles until a stop is requested. A stop never interrupts a cycle, so never a swap."""
    while not _stop.is_set():
        started = time.monotonic()
        try:
            run_cycle(ctx)
        except KeyboardInterrupt:
            raise
        except chain.ChainError as exc:  # message is already scrubbed of RPC URLs and keys
            log.error("cycle failed: %s", exc)
        except Exception:
            log.exception("cycle failed")
        if once:
            return
        _stop.wait(max(1.0, ctx.cfg.loop_interval_seconds - (time.monotonic() - started)))
    log.info("stopped")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="pocketchange trading loop")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args(argv)

    cfg = settings
    configure_logging(cfg)
    lock = acquire_instance_lock(cfg.db_path.with_suffix(".lock"))
    if lock is None:
        log.error("another bot is already running on %s; refusing to start a second one", cfg.db_path)
        sys.exit(ALREADY_RUNNING_EXIT)
    conn = db.connect(cfg.db_path)
    db.init_db(conn)
    try:
        broker: Broker = SimBroker(conn, cfg) if cfg.dry_run else chain.ChainClient(cfg)
    except chain.ChainError as exc:
        raise SystemExit(f"Cannot start: {exc}")
    log.info(
        "pocketchange starting | mode=%s strategy=%s | MAX_TRADE_USDC=%s MAX_SLIPPAGE_BPS=%s MIN_EQUITY_USD=%s | cycle=%ss",
        "DRY RUN (simulated)" if cfg.dry_run else "LIVE", cfg.strategy, cfg.max_trade_usdc,
        cfg.max_slippage_bps, cfg.min_equity_usd, cfg.loop_interval_seconds,
    )
    if not cfg.dry_run:
        log.warning("LIVE MODE: real transactions will be sent from %s", getattr(broker, "address", "?"))
    if db.is_halted(conn):
        log.warning("bot is HALTED: %s", db.halt_reason(conn))
    ctx = Context(
        cfg, conn, broker, feed=lambda tf: data.fetch_recent(CANDLES_WANTED, cfg.candle_exchange, timeframe=tf)
    )
    install_stop_handlers()
    try:
        run_loop(ctx, once=args.once)
    except KeyboardInterrupt:
        log.warning("forced stop")


if __name__ == "__main__":
    main()
