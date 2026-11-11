"""Target-weight portfolio construction (research engine; the live loop still runs the per-coin rules in strategy.py).

Instead of binary cross events it holds each coin at a weight set by:
  continuous trend score  ->  top-k cross-sectional ranking  ->  graded market regime  ->  volatility target
  ->  drawdown governor   ->  rebalance only outside a no-trade band, idle cash earning a lending yield.

    python -m bot.portfolio_backtest --days 1500 --timeframe 4h --targets
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import FEE_RATE, FIXED_COST_USD, MIN_TRADE_USDC, SLIPPAGE_RATE


@dataclass(frozen=True)
class TargetCfg:
    lookbacks: tuple[int, ...] = (10, 20, 50, 100)  # bars; the trend score blends them
    top_k: int = 3  # coins held at once
    target_vol: float = 0.15  # annualised volatility of the whole book
    max_gross: float = 1.0  # most of equity in coins (no leverage)
    max_weight: float = 0.5  # most in one coin
    band: float = 0.05  # no trade unless a weight is this far (absolute) from its target
    regime_len: int = 200  # BTC average the graded regime measures against
    regime_width: float = 0.20  # BTC this far under its average = fully risk-off; this far over = fully on
    vol_window: int = 120  # bars for the covariance estimate
    ewma_halflife: float = 30.0
    cash_apr: float = 0.04  # lending yield on idle USDC
    dd_steps: tuple[tuple[float, float], ...] = ((0.05, 1.0), (0.10, 0.75), (0.15, 0.5), (1.0, 0.25))  # (drawdown <, scale)


def trend_score(close: np.ndarray, lookbacks: tuple[int, ...] = (10, 20, 50, 100)) -> float:
    """Blend of volatility-scaled returns over several lookbacks, in (-1, 1). NaN until the longest window exists."""
    if len(close) <= max(lookbacks) + 1:
        return float("nan")
    r = np.diff(np.log(close[-(max(lookbacks) + 1):]))
    sd = r.std()
    if not sd > 0:
        return 0.0
    zs = [np.log(close[-1] / close[-lb - 1]) / (sd * np.sqrt(lb)) for lb in lookbacks]
    return float(np.mean(np.tanh(zs)))


def regime_score(btc_close: np.ndarray, cfg: TargetCfg) -> float:
    """0 = risk-off, 1 = risk-on: how far BTC sits above its average, damped when short-term volatility spikes."""
    if len(btc_close) < cfg.regime_len + 1:
        return 0.0
    ratio = btc_close[-1] / btc_close[-cfg.regime_len:].mean() - 1
    trend = float(np.clip(0.5 + ratio / (2 * cfg.regime_width), 0.0, 1.0))
    r = np.diff(np.log(btc_close[-(cfg.regime_len + 1):]))
    short, long = r[-20:].std(), r.std()
    damp = float(np.clip(1.0 - max(0.0, (short / long - 1.0) if long > 0 else 0.0) * 0.5, 0.3, 1.0))
    return trend * damp


def ewma_cov(returns: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted covariance of (bars, assets) returns, per bar."""
    n = len(returns)
    w = 0.5 ** (np.arange(n)[::-1] / halflife)
    w /= w.sum()
    mu = (returns * w[:, None]).sum(axis=0)
    x = returns - mu
    return (x * w[:, None]).T @ x


def governor_scale(drawdown: float, steps: tuple[tuple[float, float], ...]) -> float:
    for limit, scale in steps:
        if drawdown < limit:
            return scale
    return steps[-1][1]


def target_weights(
    closes: dict[str, np.ndarray], btc_close: np.ndarray, cfg: TargetCfg, *, drawdown: float = 0.0, bars_per_year: float
) -> dict[str, float]:
    """Desired fraction of equity per coin (the rest is cash). Long-only, never above `max_gross`."""
    scores = {a: trend_score(c, cfg.lookbacks) for a, c in closes.items()}
    eligible = {a: s for a, s in scores.items() if not np.isnan(s) and s > 0}
    top = sorted(eligible, key=eligible.get, reverse=True)[: cfg.top_k]
    if not top:
        return dict.fromkeys(closes, 0.0)
    raw = np.array([eligible[a] for a in top])
    w = raw / raw.sum()  # conviction-weighted inside the book
    n = min(len(c) for c in closes.values())
    window = min(cfg.vol_window, n - 1)
    rets = np.column_stack([np.diff(np.log(closes[a][-(window + 1):])) for a in top])
    cov = ewma_cov(rets, cfg.ewma_halflife) * bars_per_year
    vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
    gross = min(cfg.max_gross, cfg.target_vol / vol)  # volatility target
    gross *= regime_score(btc_close, cfg) * governor_scale(drawdown, cfg.dd_steps)
    w = np.minimum(w * gross, cfg.max_weight)
    out = dict.fromkeys(closes, 0.0)
    out.update(dict(zip(top, map(float, w))))
    return out


def apply_band(current: dict[str, float], target: dict[str, float], band: float) -> dict[str, float]:
    """Keep today's weight unless the target is more than `band` away; a target of zero always closes the position."""
    out = {}
    for a, t in target.items():
        c = current.get(a, 0.0)
        out[a] = t if (abs(t - c) > band or (t == 0.0 and c > 0.0)) else c
    return out


def simulate_targets(
    candles: dict[str, pd.DataFrame], bench: pd.DataFrame, cfg: TargetCfg = TargetCfg(), *, label: str = "target weights",
    start_cash: float = 20.0, fee_rate: float = FEE_RATE, slippage_rate: float = SLIPPAGE_RATE,
    fixed_cost: float = FIXED_COST_USD,
):
    """Replay the target-weight book: decide at each close, fill at the next open, pay fees, slippage and a fixed
    cost per trade, accrue the cash yield. Returns a portfolio_backtest.Result."""
    from bot.portfolio_backtest import Result, align

    frames = align({**candles, "__b__": bench})
    b = frames.pop("__b__")
    assets = list(frames)
    o = {a: frames[a]["open"].to_numpy(float) for a in assets}
    c = {a: frames[a]["close"].to_numpy(float) for a in assets}
    bc = b["close"].to_numpy(float)
    n, index = len(b), b.index
    bpy = pd.Timedelta(days=365) / pd.Series(index).diff().median()
    warm = max(max(cfg.lookbacks), cfg.regime_len) + 2
    cash, qty = start_cash, dict.fromkeys(assets, 0.0)
    cost_basis = dict.fromkeys(assets, 0.0)
    equity, invested = np.full(n, start_cash), np.zeros(n)
    peak, trades, in_market = start_cash, 0, 0
    pnl, trade_pnl, fills = dict.fromkeys(assets, 0.0), [], []
    pending: dict[str, float] | None = None
    per_bar = (1 + cfg.cash_apr) ** (1 / bpy) - 1
    for i in range(n):
        if pending is not None:  # fill last close's decision at this open
            eq_now = cash + sum(qty[a] * o[a][i] for a in assets)
            for a in assets:  # sells first so the buys have the cash
                want = pending.get(a, 0.0) * eq_now
                have = qty[a] * o[a][i]
                if have - want >= MIN_TRADE_USDC or (pending.get(a, 0.0) == 0.0 and have > 0):
                    sell_qty = qty[a] if pending.get(a, 0.0) == 0.0 else (have - want) / o[a][i]
                    proceeds = max(0.0, sell_qty * o[a][i] * (1 - slippage_rate) * (1 - fee_rate) - fixed_cost)
                    basis = cost_basis[a] * sell_qty / qty[a]
                    cash += proceeds
                    cost_basis[a] -= basis
                    qty[a] -= sell_qty
                    pnl[a] += proceeds - basis
                    trade_pnl.append(proceeds - basis)
                    fills.append((index[i], "sell", a, sell_qty, proceeds))
                    trades += 1
            for a in sorted(assets, key=lambda x: pending.get(x, 0.0), reverse=True):
                spend = min(cash, pending.get(a, 0.0) * eq_now - qty[a] * o[a][i])
                if spend >= MIN_TRADE_USDC and spend > fixed_cost:
                    got = (spend - fixed_cost) * (1 - fee_rate) / (o[a][i] * (1 + slippage_rate))
                    qty[a] += got
                    cost_basis[a] += spend
                    cash -= spend
                    fills.append((index[i], "buy", a, spend, got))
                    trades += 1
            pending = None
        cash *= 1 + per_bar
        invested[i] = sum(qty[a] * c[a][i] for a in assets)
        equity[i] = cash + invested[i]
        peak = max(peak, equity[i])
        if i >= warm:
            in_market += invested[i] > MIN_TRADE_USDC
        if i < warm or i == n - 1:
            continue
        tgt = target_weights({a: c[a][: i + 1] for a in assets}, bc[: i + 1], cfg,
                             drawdown=1 - equity[i] / peak, bars_per_year=bpy)
        cur = {a: qty[a] * c[a][i] / equity[i] for a in assets}
        pending = apply_band(cur, tgt, cfg.band)
    curve = pd.Series(equity, index=index).iloc[warm:]
    wins = sum(x > 0 for x in trade_pnl)
    return Result(label, float(equity[-1]), start_cash, float(1 - (curve / curve.cummax()).min()), trades, wins,
                  in_market / max(1, n - warm), pnl, curve, pd.Series(invested, index=index).iloc[warm:], trade_pnl, fills)
