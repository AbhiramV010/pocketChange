"""Multi-asset backtest of the live strategy: shared USDC, capped positions, ranked entries, next-open fills.

It replays the live loop candle by candle with the same functions (`decide_core`, `ml_decide`, `buy_budget`,
`size_trade`) in the same order (exits first, then entries best-ranked first), so a result here is a result for the
bot as configured, not for a simplified copy of it.

    python -m bot.portfolio_backtest --days 365 --timeframe 1h          # the configured rules vs buy & hold
    python -m bot.portfolio_backtest --days 730 --timeframe 1h --ml     # the AI, scored only on candles it never saw
    python -m bot.portfolio_backtest --days 730 --timeframe 1h --grid   # robustness: first half vs second half
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from bot.config import FEE_RATE, MIN_TRADE_USDC, SLIPPAGE_RATE
from bot.strategy import Params, Signal, buy_budget, decide_core, momentum, regime_ok, size_trade

REGIME_ASSET = "BTC"  # the market filter looks at BTC, as in the live loop
YEAR = pd.Timedelta(days=365)


@dataclass
class Result:
    label: str
    final: float
    start: float
    max_drawdown: float
    trades: int
    wins: int
    exposure: float  # share of candles with a position open
    per_asset: dict[str, float] = field(default_factory=dict)  # realised profit per coin
    equity: pd.Series | None = None  # mark-to-market at each candle close
    invested: pd.Series | None = None  # value held in coins at each candle close
    trade_pnl: list[float] = field(default_factory=list)  # USDC profit of each closed position, after costs
    fills: list[tuple[pd.Timestamp, str, str, float, float]] = field(default_factory=list)  # (time, side, coin, in, out)

    @property
    def ret(self) -> float:
        return self.final / self.start - 1

    @property
    def win_rate(self) -> float:
        return self.wins / len(self.trade_pnl) if self.trade_pnl else float("nan")

    @property
    def profit_factor(self) -> float:
        gains = sum(x for x in self.trade_pnl if x > 0)
        losses = -sum(x for x in self.trade_pnl if x < 0)
        return gains / losses if losses else (math.inf if gains else float("nan"))

    @property
    def sharpe(self) -> float:
        return sharpe(self.equity)


def sharpe(equity: pd.Series | None) -> float:
    """Annualised Sharpe ratio of candle-to-candle equity returns (no risk-free rate). NaN without variation."""
    if equity is None or len(equity) < 3:
        return float("nan")
    returns = equity.pct_change().dropna()
    if not returns.std() > 0:
        return float("nan")
    per_year = YEAR / pd.Series(equity.index).diff().median()
    return float(returns.mean() / returns.std() * math.sqrt(per_year))


def align(candles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    index = None
    for df in candles.values():
        index = df.index if index is None else index.intersection(df.index)
    return {a: df.loc[index] for a, df in candles.items()}


# (asset, candle index, holding=, entry_price=, bars_held=, bars_since_sell=, risk_on=, exiting=) -> (signal, rank)
Decider = Callable[..., tuple[Signal, float]]


def sma_decider(c: dict[str, np.ndarray], h: dict[str, np.ndarray], lo: dict[str, np.ndarray], p: Params) -> Decider:
    """The live SMA rule; buys are ranked by momentum, as `run._plan` ranks them."""

    def decide(a: str, i: int, **state) -> tuple[Signal, float]:
        close = c[a][: i + 1]
        signal = decide_core(close, h[a][: i + 1], lo[a][: i + 1], p, **state)
        return signal, (momentum(close, p.slow) if signal is Signal.BUY else 0.0)

    return decide


def ml_decider(
    proba: dict[str, np.ndarray], c: dict[str, np.ndarray], h: dict[str, np.ndarray], lo: dict[str, np.ndarray],
    mp, p: Params,
) -> Decider:
    """The live AI rule on precomputed probabilities (walk-forward, so each one comes from a model that never saw
    that candle); buys are ranked by probability, as `run._plan_ml` ranks them."""
    from bot.ml import ml_decide

    def decide(a: str, i: int, **state) -> tuple[Signal, float]:
        prob = None if np.isnan(proba[a][i]) else float(proba[a][i])
        signal = ml_decide(prob, c[a][: i + 1], h[a][: i + 1], lo[a][: i + 1], mp, p, **state)
        return signal, prob or 0.0

    return decide


def simulate(
    candles: dict[str, pd.DataFrame],
    p: Params,
    *,
    decider: Callable[..., Decider] | None = None,
    bench: pd.DataFrame | None = None,
    label: str = "",
    start_cash: float = 20.0,
    trade_usdc: float = 5.0,
    max_positions: int = 3,
    fee_rate: float = FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE,
) -> Result:
    """Replay the live loop. Decisions at candle i use closes up to i; their orders fill at candle i+1's open.

    `decider(c, h, lo)` builds the per-asset rule from the aligned close/high/low arrays (default: the SMA rule).
    `bench` is the BTC series for the market filter; it defaults to the BTC candles if BTC is traded."""
    if bench is None:
        bench = candles.get(REGIME_ASSET)
    if p.regime_len > 0 and bench is None:
        raise ValueError(f"the {REGIME_ASSET} market filter (regime_len={p.regime_len}) needs {REGIME_ASSET} candles")
    frames = align({**candles, **({} if bench is None else {"__bench__": bench})})
    bench_close = frames.pop("__bench__")["close"].to_numpy(float) if bench is not None else None
    assets = list(frames)
    o, h, lo, c = ({a: frames[a][col].to_numpy(float) for a in assets} for col in ("open", "high", "low", "close"))
    n = len(frames[assets[0]])
    warm = p.slow + 2
    decide = (decider or (lambda c, h, lo: sma_decider(c, h, lo, p)))(c, h, lo)

    cash = start_cash
    qty, cost = dict.fromkeys(assets, 0.0), dict.fromkeys(assets, 0.0)
    entry_i: dict[str, int | None] = dict.fromkeys(assets)
    last_sell_i: dict[str, int | None] = dict.fromkeys(assets)
    exiting = dict.fromkeys(assets, False)
    pnl = dict.fromkeys(assets, 0.0)
    trade_pnl: list[float] = []
    round_trip: dict[str, float] = {}  # profit so far of positions being sold
    fills: list[tuple[pd.Timestamp, str, str, float, float]] = []
    index = frames[assets[0]].index
    trades = in_market = 0
    equity, invested = np.full(n, start_cash), np.zeros(n)
    sells_due: list[str] = []
    buys_due: list[tuple[float, str]] = []  # (rank, asset), best first

    def held(a: str, px: float) -> bool:
        return qty[a] * px >= MIN_TRADE_USDC  # dust below this is ignored, as the live bot ignores it

    for i in range(n):
        # 1. The orders decided at the previous close fill at this open: exits first, then entries. As live, a
        # buy is sized from the equity at the current price before this cycle's sells, and closed-candle volatility.
        equity_now = cash + sum(qty[a] * o[a][i] for a in assets)
        for a in sells_due:
            px = o[a][i]
            while True:  # capped chunks back to back, as the live loop sells them cycle after cycle in one candle
                amount = size_trade(
                    Signal.SELL, "sma", usdc=cash, wsol=qty[a], price=px, max_trade_usdc=trade_usdc,
                    dca_amount_usdc=0.0, min_trade_usdc=MIN_TRADE_USDC,
                )
                if amount <= 0:
                    break
                proceeds = amount * px * (1 - slippage_rate) * (1 - fee_rate)
                basis = cost[a] * amount / qty[a]
                cash += proceeds
                cost[a] -= basis
                qty[a] -= amount
                pnl[a] += proceeds - basis
                round_trip[a] = round_trip.get(a, 0.0) + proceeds - basis
                fills.append((index[i], "sell", a, amount, proceeds))
                trades += 1
                last_sell_i[a] = i
                exiting[a] = True  # as live: still held right after a sell means a capped sell left a remainder
            if a in round_trip and not held(a, px):
                trade_pnl.append(round_trip.pop(a))  # one result per position, however many chunks it took
        open_now = sum(held(a, o[a][i]) for a in assets)
        for _, a in buys_due:
            if open_now >= max_positions:
                break
            px = o[a][i]
            budget = buy_budget(equity_now, h[a][:i], lo[a][:i], c[a][:i], p, trade_usdc)
            spend = size_trade(
                Signal.BUY, "sma", usdc=cash, wsol=qty[a], price=px, max_trade_usdc=min(trade_usdc, budget),
                dca_amount_usdc=0.0, min_trade_usdc=MIN_TRADE_USDC,
            )
            if spend <= 0:
                continue
            bought = spend * (1 - fee_rate) / (px * (1 + slippage_rate))
            fills.append((index[i], "buy", a, spend, bought))
            qty[a] += bought
            cost[a] += spend
            cash -= spend
            entry_i[a] = i
            exiting[a] = False
            open_now += 1
            trades += 1

        # 2. Mark to market at this close.
        invested[i] = sum(qty[a] * c[a][i] for a in assets)
        equity[i] = cash + invested[i]
        if i >= warm:
            in_market += any(held(a, c[a][i]) for a in assets)

        # 3. Decide on this closed candle, exactly as one live cycle does.
        sells_due, wants = [], []
        if i < warm or i == n - 1:
            buys_due = []
            continue
        risk_on = bench_close is None or regime_ok(bench_close[: i + 1], p)
        for a in assets:
            holding = held(a, c[a][i])
            signal, rank = decide(
                a, i, holding=holding,
                entry_price=cost[a] / qty[a] if qty[a] > 0 else 0.0,
                bars_held=0 if entry_i[a] is None else i - entry_i[a] + 1,
                bars_since_sell=None if last_sell_i[a] is None else i - last_sell_i[a] + 1,
                risk_on=risk_on, exiting=holding and exiting[a] and i == last_sell_i[a],  # live: sold this candle
            )
            if holding and signal is Signal.SELL:
                sells_due.append(a)
            elif not holding and signal is Signal.BUY:
                wants.append((rank, a))
        buys_due = sorted(wants, key=lambda w: w[0], reverse=True)

    curve = pd.Series(equity, index=index).iloc[warm:]
    max_dd = float(1 - (curve / curve.cummax()).min()) if len(curve) else 0.0
    wins = sum(x > 0 for x in trade_pnl)
    return Result(
        label, float(equity[-1]), start_cash, max_dd, trades, wins, in_market / max(1, n - warm), pnl,
        curve, pd.Series(invested, index=index).iloc[warm:], trade_pnl, fills,
    )


def buy_and_hold(
    candles: dict[str, pd.DataFrame], start_cash: float = 20.0, trade_usdc: float = 5.0,
    fee_rate: float = FEE_RATE, slippage_rate: float = SLIPPAGE_RATE,
) -> Result:
    """Equal-weight buy at the second candle's open (the first fill the bot could get), marked at every close."""
    candles = align(candles)
    assets = list(candles)
    share = min(trade_usdc, start_cash / len(assets))
    cash = start_cash - share * len(assets)
    index = candles[assets[0]].index[1:]
    invested = sum(
        share * (1 - fee_rate) / (candles[a]["open"].iloc[1] * (1 + slippage_rate)) * candles[a]["close"].iloc[1:]
        for a in assets
    )
    curve = pd.Series(cash + np.asarray(invested, float), index=index)
    max_dd = float(1 - (curve / curve.cummax()).min())
    return Result("buy & hold (equal weight)", float(curve.iloc[-1]), start_cash, max_dd, len(assets), 0, 1.0, {}, curve)


def report(results: list[Result]) -> str:
    lines = [f"{'strategy':<44}{'return':>9}{'max DD':>8}{'sharpe':>8}{'trades':>8}{'win%':>6}{'PF':>6}{'in mkt':>8}"]
    for r in results:
        pf = r.profit_factor
        lines.append(
            f"{r.label:<44}{r.ret:>+9.1%}{r.max_drawdown:>8.1%}{r.sharpe:>8.2f}{r.trades:>8}"
            f"{r.win_rate:>6.0%}{pf:>6.2f}{r.exposure:>8.0%}".replace("nan%", "  - ").replace(" nan", "   -")
        )
    return "\n".join(lines)


GRID = (
    ("cross 20/50", dict(entry="cross", fast=20, slow=50)),
    ("trend 9/21", dict(entry="trend", fast=9, slow=21)),
    ("trend 20/50", dict(entry="trend", fast=20, slow=50)),
    ("trend 50/200", dict(entry="trend", fast=50, slow=200)),
)
FILTERS = (
    ("no filters", dict(edge_bps=0.0, confirm=False, stop_atr=0.0, hard_stop_pct=0.0, regime_len=0, risk_pct=0.0)),
    ("stops", dict(edge_bps=0.0, confirm=False, stop_atr=3.0, hard_stop_pct=10.0, regime_len=0, risk_pct=0.0)),
    ("stops+BTC", dict(edge_bps=0.0, confirm=False, stop_atr=3.0, hard_stop_pct=10.0, regime_len=400, risk_pct=0.0)),
    ("all+risk1%", dict(edge_bps=10.0, confirm=True, stop_atr=3.0, hard_stop_pct=10.0, regime_len=400, risk_pct=1.0)),
)


def halves(candles: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    aligned = align(candles)
    cut = len(next(iter(aligned.values()))) // 2
    return {a: df.iloc[:cut] for a, df in aligned.items()}, {a: df.iloc[cut:] for a, df in aligned.items()}


def robustness(
    candles: dict[str, pd.DataFrame], bench: pd.DataFrame, base: Params, **kw
) -> list[tuple[str, Result, Result]]:
    """Every grid setting on the first half and, untouched, on the second. A real edge holds up in both halves;
    one that only shows in the half you tuned on is curve fitting."""
    first, second = halves({**candles, "__bench__": bench})
    b1, b2 = first.pop("__bench__"), second.pop("__bench__")
    out = []
    for name, rule in GRID:
        for fname, filters in FILTERS:
            p = replace(base, **rule, **filters)
            out.append((f"{name} {fname}", simulate(first, p, bench=b1, **kw), simulate(second, p, bench=b2, **kw)))
    return out


def main(argv: list[str] | None = None) -> None:
    from bot import data, ml
    from bot.assets import ASSETS
    from bot.config import settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--assets", default=",".join(settings.assets))
    parser.add_argument("--max-positions", type=int, default=settings.max_positions)
    parser.add_argument("--ml", action="store_true", help="also backtest the AI on walk-forward probabilities")
    parser.add_argument("--grid", action="store_true", help="test a grid of settings on each half of the period")
    args = parser.parse_args(argv)
    symbols = [s.strip().upper() for s in args.assets.split(",")]
    fetch = lambda s: data.fetch_history(ASSETS[s].market, args.days, args.timeframe, settings.candle_exchange)  # noqa: E731
    candles = {s: fetch(s) for s in symbols}
    bench = candles.get(REGIME_ASSET)
    bench = fetch(REGIME_ASSET) if bench is None else bench
    kw = dict(trade_usdc=settings.max_trade_usdc, max_positions=args.max_positions)
    base = Params()
    if args.grid:
        print(f"{'setting':<30}{'1st half':>10}{'sharpe':>8}{'2nd half':>10}{'sharpe':>8}{'trades':>8}")
        for name, a, b in robustness(candles, bench, base, **kw):
            print(f"{name:<30}{a.ret:>+10.1%}{a.sharpe:>8.2f}{b.ret:>+10.1%}{b.sharpe:>8.2f}{a.trades + b.trades:>8}")
        return
    results = [buy_and_hold(candles), simulate(candles, base, bench=bench, label="configured (.env) rules", **kw)]
    if args.ml:
        mp = ml.MLParams.from_settings(settings)
        proba, summary = ml.walk_forward({**candles, ml.BENCH: bench}, mp, args.timeframe)
        oos = align({**{a: df[df.index >= summary["start"]] for a, df in candles.items()}, "__bench__": bench})
        oos_bench = oos.pop("__bench__")
        arrays = {a: proba[a].reindex(df.index).to_numpy(float) for a, df in oos.items()}  # same rows as simulate's
        decider = lambda c, h, lo: ml_decider(arrays, c, h, lo, mp, base)  # noqa: E731
        print(f"AI walk-forward: {summary['folds']} folds, out-of-sample AUC {summary['auc']:.3f}, "
              f"scored from {summary['start']:%Y-%m-%d}")
        results = [
            buy_and_hold(oos), simulate(oos, base, bench=oos_bench, label="configured rules (same period)", **kw),
            simulate(oos, base, decider=decider, bench=oos_bench, label="AI (out of sample)", **kw),
        ]
    print(report(results))


if __name__ == "__main__":
    main()
