from __future__ import annotations

import argparse
import logging
import logging.handlers
import math
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

from bot import chain, data, db, ml
from bot.assets import ASSETS, PRIMARY, label
from bot.config import FEE_RATE, MIN_TRADE_USDC, SLIPPAGE_RATE, TIMEFRAME, Settings, settings
from bot.db import Balances, Fill
from bot.strategy import Params, Signal, bars_since, buy_budget, dca_signal, decide, momentum, regime_ok, size_trade

log = logging.getLogger("bot")

MAX_SWAP_RETRIES = 1  # hard rule: a failed swap is retried at most once per cycle
MAX_ATTEMPT_CYCLES = 3  # cycles that may try the same signal before giving up on it
MAX_SELL_CHUNKS = 10  # capped swaps one exit may take in a cycle; anything left continues next cycle
REGIME_ASSET = "BTC"  # the market filter looks at BTC
PROVEN_TIMEFRAME = "4h"  # the only candle size whose backtests beat fees (README "Results")
ALREADY_RUNNING_EXIT = 3  # exit code when another bot holds the instance lock


def max_data_age(timeframe: str) -> pd.Timedelta:
    return max(data.candle_delta(timeframe) * 2, pd.Timedelta(minutes=15))


def candles_wanted(p: Params, with_ml: bool = False) -> int:
    # 2 * slow: the trend-reset check looks back up to `slow` candles, each needing a slow average
    return max(120, 2 * p.slow + 5, p.regime_len + 5, ml.MIN_CANDLES + 30 if with_ml else 0)


_stop = threading.Event()


def request_stop(signum=None, frame=None) -> None:
    if _stop.is_set():
        raise KeyboardInterrupt
    _stop.set()
    log.info("stop requested; finishing the current cycle (signal again to force)")


def install_stop_handlers() -> None:
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):  # SIGBREAK: Ctrl+Break / console close on Windows
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)


def acquire_instance_lock(path: Path) -> IO | None:
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
    def swap(self, side: str, amount_in: float, ref_price: float, asset: str = PRIMARY) -> Fill: ...


class SimBroker:

    def __init__(self, conn: sqlite3.Connection, cfg: Settings):
        self._conn, self._cfg = conn, cfg

    def balances(self) -> Balances:
        return db.sim_balances(self._conn, self._cfg.sim_start_usdc, self._cfg.sim_start_sol)

    def swap(self, side: str, amount_in: float, ref_price: float, asset: str = PRIMARY) -> Fill:
        have = self.balances()
        cap = self._cfg.max_trade_usdc
        name = label(asset)
        if side == chain.BUY:
            if amount_in > cap + 1e-9:
                raise chain.SwapFailed(f"buy of {amount_in} USDC exceeds MAX_TRADE_USDC={cap}")
            if amount_in > have.usdc + 1e-9:
                raise chain.SwapFailed("insufficient simulated USDC")
            out = amount_in * (1 - FEE_RATE) / (ref_price * (1 + SLIPPAGE_RATE))
            return Fill(amount_in, out, amount_in / out, None)
        if amount_in > have.qty(asset) + 1e-12:
            raise chain.SwapFailed(f"insufficient simulated {name}")
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
    feed: Callable[[str], pd.DataFrame]  # timeframe -> recent SOL candles, including the one still forming
    clock: Callable[[], datetime] = field(default=_utc_now)
    extra_feeds: dict[str, Callable[[str], pd.DataFrame]] = field(default_factory=dict)  # the other assets
    params: Params = field(default_factory=Params)
    sniper: object | None = None  # bot.sniper.Sniper in dry runs, else None
    models: object | None = None  # bot.ml.ModelManager when STRATEGY=ml
    scores: dict[str, float] = field(default_factory=dict)  # this cycle's AI probabilities
    ml_notice: bool = False  # already said this cycle that there is no model


@dataclass
class Plan:
    asset: str
    signal: Signal
    price: float
    closed: pd.DataFrame
    rank: float
    budget: float = math.inf  # most USDC a buy may spend (risk sizing); MAX_TRADE_USDC still applies


def run_cycle(ctx: Context) -> None:
    cfg, conn = ctx.cfg, ctx.conn
    now = ctx.clock()
    stamp = now.isoformat(timespec="seconds")
    db.set_state(conn, db.HEARTBEAT, stamp)
    db.set_state(conn, db.DRY_RUN, "1" if cfg.dry_run else "0")

    paused, halted = db.is_paused(conn), db.is_halted(conn)  # top of the cycle (hard rule 6)
    risky_on = db.is_risky(conn, cfg.risky)
    timeframe = db.get_timeframe(conn) or TIMEFRAME  # the dashboard can switch it live

    ctx.scores, ctx.ml_notice = {}, False
    if ctx.models is not None:
        ctx.models.maybe_retrain(timeframe, now)  # background thread; never blocks the cycle
    balances = ctx.broker.balances()
    held = {a for a, qty in balances.positions().items() if qty > 0}
    assets = list(cfg.assets) + [a for a in cfg.risky_assets if risky_on or a in held]
    markets = _markets(ctx, assets, now, timeframe)
    prices = {a: price for a, (_, price) in markets.items() if price is not None}
    db.insert_balance(conn, balances, prices.get(PRIMARY), cfg.dry_run, timestamp=stamp, prices=prices)
    if not prices:
        log.warning("no usable price; not trading this cycle")
        return

    if not halted:
        halted = _enforce_equity_floor(ctx, balances, prices)
    if halted:
        log.info("HALTED (%s); not trading. Clear with: python -m bot.reset_halt", db.halt_reason(conn))
        return
    if paused:
        log.info("paused; not trading (usdc=%.2f positions=%s)", balances.usdc, _fmt_positions(balances, prices))
        return

    bench = _closed_bench(markets, now, timeframe)
    risk_on, why_off = _risk_on(ctx, markets, bench)
    db.set_state(conn, db.RISK_OFF, why_off)  # the dashboard says why nothing new is bought
    equity = portfolio_equity(balances, prices, db.last_marks(conn))
    plans = []
    for asset in assets:
        candles, price = markets.get(asset, (pd.DataFrame(), None))
        if price is None:
            if asset in held:
                log.warning("%s: no usable price this cycle", label(asset))
            continue
        entry_ok = asset in cfg.assets or risky_on
        plan = _plan(ctx, asset, candles, balances, price, now, timeframe, risk_on, entry_ok, bench, equity)
        if plan is not None:
            plans.append(plan)
    if ctx.models is not None:
        bundle = ctx.models.get(timeframe)
        info = None if bundle is None else {
            "timeframe": timeframe, "trained_at": bundle.trained_at, "oos_auc": bundle.oos_auc, "rows": bundle.rows,
            "enter": cfg.ml_enter, "exit": cfg.ml_exit, "horizon": bundle.params.horizon,
        }
        db.set_ml(conn, ctx.scores, info)

    sells = [p for p in plans if p.signal is Signal.SELL]
    buys = sorted((p for p in plans if p.signal is Signal.BUY), key=lambda p: p.rank, reverse=True)
    for plan in sells + buys:
        # A sell bigger than MAX_TRADE_USDC goes out in capped chunks, one swap each, all before the buys (as the
        # backtest fills them), so the buys see the cash the exit freed. _trade_if_signal stops at the last chunk.
        for _ in range(MAX_SELL_CHUNKS if plan.signal is Signal.SELL else 1):
            if db.is_halted(conn):
                break
            after = _trade_if_signal(ctx, plan, balances, prices, now, timeframe)
            if after is None:
                break
            balances = after


def _closed_bench(markets: dict[str, tuple[pd.DataFrame, float | None]], now: datetime, timeframe: str) -> pd.DataFrame | None:
    candles = markets.get(REGIME_ASSET, (pd.DataFrame(), None))[0]
    return None if candles.empty else data.closed_only(candles, now, timeframe)


def _risk_on(
    ctx: Context, markets: dict[str, tuple[pd.DataFrame, float | None]], bench: pd.DataFrame | None
) -> tuple[bool, str]:
    """The BTC filter on closed candles only, as in the backtest: (open new positions?, why not)."""
    p = ctx.params
    if p.regime_len <= 0:
        return True, ""
    closes = None if bench is None else bench["close"].to_numpy(float)
    if closes is None or not len(closes) or markets.get(REGIME_ASSET, (None, None))[1] is None:
        why = f"no fresh {REGIME_ASSET} candles to judge the market"  # do not open positions blind
    elif len(closes) < p.regime_len:
        why = f"only {len(closes)} of the {p.regime_len} {REGIME_ASSET} candles the market filter needs"
    elif regime_ok(closes, p):
        return True, ""
    else:
        why = (
            f"{REGIME_ASSET} {closes[-1]:,.0f} is below its {p.regime_len}-candle average "
            f"{closes[-p.regime_len:].mean():,.0f}"
        )
    log.info("risk-off: %s; no new positions", why)
    return False, why


def _fmt_positions(balances: Balances, prices: dict[str, float]) -> str:
    parts = [f"{a}={q:.6g}" for a, q in balances.positions().items() if q * prices.get(a, 0) >= 0.01]
    return " ".join(parts) or "flat"


def _market(
    feed: Callable[[str], pd.DataFrame], now: datetime, timeframe: str, name: str
) -> tuple[pd.DataFrame, float | None]:
    try:
        candles = feed(timeframe)
    except Exception as exc:  # network, rate limit, exchange outage
        log.error("%s candle feed failed: %s: %s", name, type(exc).__name__, str(exc)[:200])
        return pd.DataFrame(), None
    if candles.empty or pd.Timestamp(now) - candles.index[-1] > max_data_age(timeframe):
        log.error("%s candle feed is stale (latest %s)", name, candles.index[-1] if len(candles) else "none")
        return candles, None
    return candles, float(candles["close"].iloc[-1])


def _markets(
    ctx: Context, assets: list[str], now: datetime, timeframe: str
) -> dict[str, tuple[pd.DataFrame, float | None]]:
    need_bench = ctx.params.regime_len > 0 or ctx.cfg.strategy == "ml"
    wanted = list(dict.fromkeys(assets + ([REGIME_ASSET] if need_bench else [])))
    out = {}
    for asset in wanted:
        feed = ctx.feed if asset == PRIMARY else ctx.extra_feeds.get(asset)
        if feed is None:
            log.warning("%s: no price feed configured; skipping", asset)
            continue
        out[asset] = _market(feed, now, timeframe, asset)
    return out


def equity_usd(balances: Balances, price: float) -> float:
    return balances.usdc + balances.wsol * price


def portfolio_equity(
    balances: Balances, prices: dict[str, float], fallback: dict[str, float] | None = None
) -> float | None:
    """USDC plus every position at its price; None if a held asset has no price at all."""
    total = balances.usdc
    for asset, qty in balances.positions().items():
        if qty <= 0:
            continue
        price = prices.get(asset) or (fallback or {}).get(asset)
        if price is None:
            return None
        total += qty * price
    return total


def _enforce_equity_floor(ctx: Context, balances: Balances, prices: dict[str, float]) -> bool:
    equity = portfolio_equity(balances, prices, db.last_marks(ctx.conn))
    if equity is None:
        log.warning("equity floor not checked: a held asset has no price")
        return False
    if equity >= ctx.cfg.min_equity_usd:
        return False
    reason = f"equity ${equity:.2f} is below MIN_EQUITY_USD ${ctx.cfg.min_equity_usd:.2f}"
    db.set_halted(ctx.conn, reason)
    log.critical("HALTING: %s. The open position is not sold.", reason)
    return True


def _state_key(key: str, asset: str) -> str:
    return key if asset == PRIMARY else f"{key}:{asset}"  # SOL keeps the original keys


def _last_trade(conn: sqlite3.Connection, cfg: Settings, asset: str, closed: pd.DataFrame, timeframe: str):
    """(side, candles since) of the bot's latest trade in this coin, in this mode: 0 means during the forming
    candle, 1 during the latest closed one. (None, None) if it never traded."""
    last = db.last_trade(conn, cfg.dry_run, asset)
    return (None, None) if last is None else (last[0], bars_since(closed, last[1], timeframe))


def _exiting(side: str | None, ago: float | None, wallet_qty: float, log_qty: float) -> bool:
    """A sell capped by MAX_TRADE_USDC left part of the position: keep selling until flat.

    Only right after that sell (this candle or the last), and only if the wallet holds no more than the trade log
    says it left. Anything else (a buy that landed after an unknown outcome, a manual trade, an old sell) is not a
    leftover, so the normal rules decide. Read from the trade log, so a dry run never affects live."""
    return side == "sell" and ago is not None and ago <= 1 and 0 < wallet_qty <= log_qty * 1.001


def _one_decision_per_candle(signal: Signal, side: str | None, ago: float | None) -> Signal:
    """As in the backtest, a coin is judged once per closed candle: one bought during the forming candle is not
    sold in it, and one sold is not bought back in it. The loop runs every minute on the same closed candles, so
    without this a fresh fill is re-judged against a stale close (a stop) or a signal still showing (a re-buy),
    and the bot trades back and forth until the candle closes. Selling the rest of a capped sell is allowed."""
    if ago is None or ago >= 1:
        return signal
    if (signal is Signal.SELL and side == "buy") or (signal is Signal.BUY and side == "sell"):
        return Signal.HOLD
    return signal


def _budget(ctx: Context, closed: pd.DataFrame, equity: float | None) -> float:
    if equity is None:
        return ctx.cfg.max_trade_usdc  # a held coin has no price: fall back to the flat cap
    return buy_budget(
        equity, closed["high"].to_numpy(float), closed["low"].to_numpy(float), closed["close"].to_numpy(float),
        ctx.params, ctx.cfg.max_trade_usdc,
    )


def _plan(
    ctx: Context, asset: str, candles: pd.DataFrame, balances: Balances, price: float, now: datetime,
    timeframe: str, risk_on: bool, entry_ok: bool, bench: pd.DataFrame | None = None, equity: float | None = None,
) -> Plan | None:
    cfg, conn, p = ctx.cfg, ctx.conn, ctx.params
    closed = data.closed_only(candles, now, timeframe)
    name = label(asset)
    if cfg.strategy == "ml":
        return _plan_ml(ctx, asset, closed, balances, price, timeframe, risk_on and entry_ok, bench, equity)
    budget = math.inf
    side, ago = (None, None) if closed.empty else _last_trade(conn, cfg, asset, closed, timeframe)
    if cfg.strategy == "sma":
        if len(closed) < p.slow + 1:
            log.info("%s: warming up: %d of %d closed candles", name, len(closed), p.slow + 1)
            return None
        log_qty, entry_price, entry_time = db.open_position(conn, asset, cfg.dry_run)
        holding = balances.qty(asset) * price >= MIN_TRADE_USDC
        signal = decide(
            closed, p, holding=holding, entry_price=entry_price, entry_time=entry_time,
            last_sell_time=db.last_trade_time(conn, "sell", cfg.dry_run, asset), risk_on=risk_on,
            exiting=holding and _exiting(side, ago, balances.qty(asset), log_qty), timeframe=timeframe,
        )
        if signal is Signal.BUY and not holding:
            budget = _budget(ctx, closed, equity)
    else:
        signal = dca_signal(closed, db.last_trade_time(conn, "buy", cfg.dry_run, asset), cfg.dca_interval_hours, timeframe)
    signal = _one_decision_per_candle(signal, side, ago)
    if signal is Signal.BUY and not entry_ok:
        signal = Signal.HOLD  # risky asset with the switch off: exits only
    if signal is Signal.HOLD:
        log.info("hold | %s %s | price=%.6g usdc=%.2f qty=%.6g", name, timeframe, price, balances.usdc, balances.qty(asset))
        return None
    return Plan(asset, signal, price, closed, momentum(closed["close"].to_numpy(float), p.slow), budget)


def _plan_ml(
    ctx: Context, asset: str, closed: pd.DataFrame, balances: Balances, price: float, timeframe: str,
    entry_ok: bool, bench: pd.DataFrame | None, equity: float | None = None,
) -> Plan | None:
    """The AI's call through `ml.ml_decide`, the same rule the backtest runs. Without a usable probability it
    opens nothing, but the stops still guard what is held."""
    cfg, conn = ctx.cfg, ctx.conn
    name = label(asset)
    bundle = ctx.models.get(timeframe) if ctx.models is not None else None
    prob = None
    mp = ml.MLParams.from_settings(cfg)
    if bundle is None:
        if not ctx.ml_notice:
            log.info("AI: no trained model for %s yet (it trains in the background, or run: python -m bot.ml train)", timeframe)
            ctx.ml_notice = True
    elif len(closed) < ml.MIN_CANDLES:
        log.info("%s: warming up: %d of %d closed candles", name, len(closed), ml.MIN_CANDLES)
    else:
        prob = bundle.proba(ml.features(closed, bench).iloc[-1])
        if prob is None:
            log.info("%s: not enough clean history for the AI yet", name)
    if prob is not None and bundle is not None:
        ctx.scores[asset] = prob
        db.record_prediction(conn, asset, timeframe, closed.index[-1].isoformat(), prob, price,
                             bundle.params.horizon, bundle.params.target_pct)
        _grade_predictions(conn, asset, timeframe, closed)
        enter = _learned_enter(conn, timeframe, cfg.ml_enter, bundle.params.horizon) if cfg.ml_adapt else cfg.ml_enter
        mp = ml.MLParams(bundle.params.horizon, bundle.params.target_pct, enter, cfg.ml_exit)
    holding = balances.qty(asset) * price >= MIN_TRADE_USDC
    if (prob is None and not holding) or closed.empty:
        return None
    log_qty, entry_price, entry_time = db.open_position(conn, asset, cfg.dry_run)
    side, ago = _last_trade(conn, cfg, asset, closed, timeframe)
    freq = pd.Timedelta(data.candle_delta(timeframe))
    # candles closed since the buy (0 while it is still forming); with no recorded buy, the whole window, as decide()
    bars = int((closed.index >= pd.Timestamp(entry_time).floor(freq)).sum()) if entry_time is not None else len(closed)
    signal = ml.ml_decide(
        prob, closed["close"].to_numpy(float), closed["high"].to_numpy(float), closed["low"].to_numpy(float), mp,
        ctx.params, holding=holding, entry_price=entry_price, bars_held=bars,
        bars_since_sell=bars_since(closed, db.last_trade_time(conn, "sell", cfg.dry_run, asset), timeframe),
        risk_on=entry_ok, exiting=holding and _exiting(side, ago, balances.qty(asset), log_qty),
    )
    signal = _one_decision_per_candle(signal, side, ago)
    log.info("ai | %s %s p(win)=%s -> %s", name, timeframe, "-" if prob is None else f"{prob:.2f}", signal.value)
    if signal is Signal.HOLD:
        return None
    budget = _budget(ctx, closed, equity) if signal is Signal.BUY else math.inf
    return Plan(asset, signal, price, closed, prob or 0.0, budget)


def _grade_predictions(conn, asset: str, timeframe: str, closed: pd.DataFrame) -> None:
    """Score each open prediction once its horizon has closed: would buying then and selling now have kept more
    than the target after fees and slippage?"""
    closes = closed["close"]
    for row in db.open_predictions(conn, asset, timeframe):
        candle = pd.Timestamp(row["candle"])
        if candle not in closes.index:
            continue
        pos = closes.index.get_loc(candle) + row["horizon"]
        if pos >= len(closes):
            continue  # horizon not over yet
        net = ml.net_of_costs(closes.iloc[pos] / row["price0"] - 1)
        db.resolve_prediction(conn, row["id"], float(net), bool(net > row["target_pct"] / 100))


def _learned_enter(conn, timeframe: str, default: float, horizon: int) -> float:
    """The buy threshold the AI has proven on its own graded predictions (see `ml.best_threshold`)."""
    rows = db.resolved_predictions(conn, timeframe)
    calls = [(r["p"], r["fwd_return"], pd.Timestamp(r["candle"])) for r in rows]
    enter = ml.best_threshold(calls, default, data.candle_delta(timeframe) * horizon)
    db.set_ml_learn(conn, {"timeframe": timeframe, "enter": enter, "graded": len(calls)})
    return enter


def _open_positions(balances: Balances, prices: dict[str, float]) -> int:
    return sum(1 for a, q in balances.positions().items() if q * prices.get(a, 0.0) >= MIN_TRADE_USDC)


def _trade_if_signal(
    ctx: Context, plan: Plan, balances: Balances, prices: dict[str, float], now: datetime, timeframe: str
) -> Balances | None:
    """Execute one planned trade. Returns the balances after a fill, else None."""
    cfg, conn = ctx.cfg, ctx.conn
    asset, signal, price, closed = plan.asset, plan.signal, plan.price, plan.closed
    name = label(asset)
    done_key, attempt_key, count_key = (
        _state_key(k, asset) for k in (db.DONE_SIGNAL, db.ATTEMPT_SIGNAL, db.ATTEMPT_COUNT)
    )

    key = f"{closed.index[-1].isoformat()}:{signal.value}"
    if db.get_state(conn, done_key) == key:
        return None
    attempts = int(db.get_state(conn, count_key, "0") or 0) if db.get_state(conn, attempt_key) == key else 0
    if attempts >= MAX_ATTEMPT_CYCLES:
        return None  # gave up on this signal; the next one starts fresh

    held = balances.qty(asset)
    cap = min(cfg.max_trade_usdc, plan.budget) if signal is Signal.BUY else cfg.max_trade_usdc
    amount = size_trade(
        signal, cfg.strategy, usdc=balances.usdc, wsol=held, price=price,
        max_trade_usdc=cap, dca_amount_usdc=cfg.dca_amount_usdc, min_trade_usdc=MIN_TRADE_USDC,
    )
    if amount <= 0:
        log.info("%s %s signal, nothing to trade (usdc=%.2f qty=%.6g)", name, signal.value, balances.usdc, held)
        db.set_state(conn, done_key, key)
        return None
    if signal is Signal.BUY and _open_positions(balances, prices) >= cfg.max_positions:
        log.info("%s buy signal skipped: already holding MAX_POSITIONS=%d", name, cfg.max_positions)
        return None
    if balances.sol < cfg.gas_reserve_sol:
        log.warning(
            "%s %s signal skipped: native SOL %.6f is below GAS_RESERVE_SOL %s", name, signal.value, balances.sol,
            cfg.gas_reserve_sol,
        )
        return None

    db.set_state(conn, attempt_key, key)
    db.set_state(conn, count_key, str(attempts + 1))
    if db.is_paused(conn) or db.is_halted(conn):  # flipped while this cycle was running
        log.info("paused/halted during the cycle; not trading")
        return None

    side = signal.value
    log.info("%s %s signal: trading %.9f %s at ~%.6g", name, side, amount, "USDC" if side == "buy" else name, price)
    fill = _swap_with_one_retry(ctx, side, amount, price, asset)
    if fill is None:
        return None

    db.insert_trade(conn, side, fill, cfg.dry_run, timestamp=ctx.clock().isoformat(timespec="seconds"), asset=asset)
    db.set_state(conn, count_key, "0")
    left_usd = (held - fill.amount_in) * price
    if side == "buy" or left_usd < MIN_TRADE_USDC:
        db.set_state(conn, done_key, key)  # a capped sell of a big position continues next cycle (see _exiting)
    log.info(
        "%s %s filled: in=%.9f out=%.9f price=%.6g %s", name, side, fill.amount_in, fill.amount_out, fill.price,
        fill.tx_hash or "(simulated)",
    )

    try:  # equity floor right after every trade (hard rule 8)
        after = ctx.broker.balances()
        stamp = ctx.clock().isoformat(timespec="seconds")
        db.insert_balance(conn, after, prices.get(PRIMARY), cfg.dry_run, timestamp=stamp, prices=prices)
        _enforce_equity_floor(ctx, after, prices)
        return after
    except Exception as exc:
        log.error("post-trade balance check failed, will re-check next cycle: %s", str(exc)[:200])
        return None


def _swap_with_one_retry(ctx: Context, side: str, amount: float, price: float, asset: str = PRIMARY) -> Fill | None:
    for attempt in range(1 + MAX_SWAP_RETRIES):
        try:
            return ctx.broker.swap(side, amount, price, asset)
        except chain.SwapFailed as exc:
            log.error("%s %s failed (attempt %d of %d): %s", label(asset), side, attempt + 1, 1 + MAX_SWAP_RETRIES, exc)
        except chain.SwapUncertain as exc:
            # A transaction may still land. Retrying could double-spend, so stop and ask for a human.
            reason = f"{label(asset)} {side} outcome unknown: {exc}. Check the wallet on Solscan, then run bot.reset_halt"
            db.set_halted(ctx.conn, reason)
            log.critical("HALTING: %s", reason)
            return None
        except chain.ChainError as exc:  # RPC trouble before anything was sent; next cycle tries again
            log.error("%s %s aborted, no retry this cycle: %s", label(asset), side, exc)
            return None
    return None


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
        if ctx.sniper is not None:  # a separate sandbox: it can never break or delay the main strategy
            try:
                ctx.sniper.step()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("sniper step failed")
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
    params = Params()
    log.info(
        "pocketchange starting | mode=%s strategy=%s | MAX_TRADE_USDC=%s MAX_SLIPPAGE_BPS=%s MIN_EQUITY_USD=%s | cycle=%ss",
        "DRY RUN (simulated)" if cfg.dry_run else "LIVE", cfg.strategy, cfg.max_trade_usdc,
        cfg.max_slippage_bps, cfg.min_equity_usd, cfg.loop_interval_seconds,
    )
    log.info(
        "assets=%s risky=%s (switch %s) | %s %d/%d %s edge=%gbps confirm=%s stop=%gATR hard=%g%% regime=%d max_positions=%d",
        ",".join(cfg.assets), ",".join(cfg.risky_assets) or "-", "ON" if db.is_risky(conn, cfg.risky) else "off",
        "AI" if cfg.strategy == "ml" else "SMA", params.fast, params.slow, params.entry, params.edge_bps, params.confirm, params.stop_atr,
        params.hard_stop_pct, params.regime_len, cfg.max_positions,
    )
    timeframe = db.get_timeframe(conn) or TIMEFRAME
    if cfg.strategy != "dca" and timeframe != PROVEN_TIMEFRAME:
        log.warning("trading on %s candles: in backtests every setting below %s lost money after fees (README Results)",
                    timeframe, PROVEN_TIMEFRAME)
    if cfg.strategy == "sma" and params.entry == "trend":
        log.warning("ENTRY=trend lost money in every backtest; ENTRY=cross is the tested default (README Results)")
    if not cfg.dry_run:
        log.warning("LIVE MODE: real transactions will be sent from %s", getattr(broker, "address", "?"))
    if db.is_halted(conn):
        log.warning("bot is HALTED: %s", db.halt_reason(conn))

    limit = candles_wanted(params, cfg.strategy == "ml")

    def feed_for(symbol: str) -> Callable[[str], pd.DataFrame]:
        return lambda tf: data.fetch_recent(limit, cfg.candle_exchange, timeframe=tf, symbol=symbol)

    need_bench = params.regime_len > 0 or cfg.strategy == "ml"
    symbols = list(dict.fromkeys(cfg.assets + cfg.risky_assets + ((REGIME_ASSET,) if need_bench else ())))
    extra = {a: feed_for(ASSETS[a].market) for a in symbols if a != PRIMARY}
    sniper = None
    if cfg.dry_run and cfg.sniper:
        from bot.sniper import Sniper

        sniper = Sniper(cfg, conn)
    models = ml.ModelManager(cfg) if cfg.strategy == "ml" else None
    ctx = Context(
        cfg, conn, broker, feed=feed_for(ASSETS[PRIMARY].market), extra_feeds=extra, params=params, sniper=sniper,
        models=models,
    )
    install_stop_handlers()
    try:
        run_loop(ctx, once=args.once)
    except KeyboardInterrupt:
        log.warning("forced stop")


if __name__ == "__main__":
    main()
