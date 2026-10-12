"""Multi-asset backtest of the live strategy: shared USDC, one $5-capped position per asset, ranked entries.

    python -m bot.portfolio_backtest --days 45 --timeframe 5m --assets SOL,BTC,ETH,BNB,XRP,DOGE,AVAX
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace

from typing import Callable

import pandas as pd

from bot.config import FEE_RATE, MIN_TRADE_USDC, SLIPPAGE_RATE
from bot.strategy import Params, Signal, momentum, regime_ok, sma_signal, stop_hit


@dataclass
class Result:
    label: str
    final: float
    start: float
    max_drawdown: float
    trades: int
    wins: int
    exposure: float
    per_asset: dict[str, float] = field(default_factory=dict)

    @property
    def ret(self) -> float:
        return self.final / self.start - 1


def align(candles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    index = None
    for df in candles.values():
        index = df.index if index is None else index.intersection(df.index)
    return {a: df.loc[index] for a, df in candles.items()}


SignalFn = Callable[[str, int, bool, int], tuple[Signal, float]]  # (asset, i, holding, bars_held) -> (signal, rank)


def simulate(
    candles: dict[str, pd.DataFrame],
    p: Params,
    *,
    signals: SignalFn | None = None,
    label: str = "",
    start_cash: float = 20.0,
    trade_usdc: float = 5.0,
    max_positions: int = 3,
    fee_rate: float = FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE,
    regime_asset: str = "BTC",
) -> Result:
    candles = align(candles)
    assets = list(candles)
    regime_asset = regime_asset if regime_asset in candles else assets[0]
    o = {a: candles[a]["open"].to_numpy(float) for a in assets}
    h = {a: candles[a]["high"].to_numpy(float) for a in assets}
    lo = {a: candles[a]["low"].to_numpy(float) for a in assets}
    c = {a: candles[a]["close"].to_numpy(float) for a in assets}
    n = len(candles[assets[0]])
    warm = p.slow + 2

    def moving_average_signals(a: str, i: int, holding: bool, bars_held: int) -> tuple[Signal, float]:
        signal = sma_signal(c[a][: i + 1], p)
        return signal, (momentum(c[a][: i + 1], p.slow) if signal is Signal.BUY else 0.0)

    signal_for = signals or moving_average_signals

    cash = start_cash
    qty = dict.fromkeys(assets, 0.0)
    cost = dict.fromkeys(assets, 0.0)
    entry_i = dict.fromkeys(assets, -1)
    last_sell_i = dict.fromkeys(assets, -10**9)
    pnl = dict.fromkeys(assets, 0.0)
    trades = wins = 0
    peak = start_cash
    max_dd = 0.0
    in_market = 0

    def sell(a: str, i: int) -> None:
        nonlocal cash, trades, wins
        px = o[a][i + 1]
        sell_qty = min(qty[a], trade_usdc / px)
        proceeds = sell_qty * px * (1 - slippage_rate) * (1 - fee_rate)
        basis = cost[a] * (sell_qty / qty[a])
        cash += proceeds
        pnl[a] += proceeds - basis
        cost[a] -= basis
        qty[a] -= sell_qty
        trades += 1
        wins += proceeds > basis
        last_sell_i[a] = i
        if qty[a] * px < MIN_TRADE_USDC:  # dust: write it off, as the live bot ignores it
            qty[a], cost[a], entry_i[a] = 0.0, 0.0, -1

    for i in range(warm, n - 1):
        held = [a for a in assets if qty[a] * c[a][i] >= MIN_TRADE_USDC]
        for a in held:
            signal, _ = signal_for(a, i, True, i + 1 - entry_i[a])
            entry_price = cost[a] / qty[a]
            stopped = stop_hit(h[a][: i + 1], lo[a][: i + 1], c[a][: i + 1], entry_price, i + 1 - entry_i[a], p)
            if stopped or signal is Signal.SELL:
                sell(a, i)
        holding_now = sum(qty[a] * c[a][i] >= MIN_TRADE_USDC for a in assets)
        wants = []
        risk_on = regime_ok(c[regime_asset][: i + 1], p)
        for a in assets:
            if not risk_on:
                break
            if qty[a] * c[a][i] >= MIN_TRADE_USDC or i - last_sell_i[a] < p.reentry_candles:
                continue
            signal, rank = signal_for(a, i, False, 0)
            if signal is Signal.BUY:
                wants.append((rank, a))
        for _, a in sorted(wants, reverse=True):
            if holding_now >= max_positions or cash < MIN_TRADE_USDC:
                break
            spend = min(trade_usdc, cash)
            px = o[a][i + 1]
            qty[a] += spend * (1 - fee_rate) / (px * (1 + slippage_rate))
            cost[a] += spend
            entry_i[a] = i + 1
            cash -= spend
            holding_now += 1
            trades += 1
        equity = cash + sum(qty[a] * c[a][i] for a in assets)
        in_market += any(qty[a] > 0 for a in assets)
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)

    final = cash + sum(qty[a] * c[a][n - 2] for a in assets)
    return Result(label, final, start_cash, max_dd, trades, wins, in_market / max(1, n - warm - 1), pnl)


def buy_and_hold(
    candles: dict[str, pd.DataFrame], start_cash: float = 20.0, trade_usdc: float = 5.0,
    fee_rate: float = FEE_RATE, slippage_rate: float = SLIPPAGE_RATE,
) -> Result:
    candles = align(candles)
    assets = list(candles)
    share = min(trade_usdc, start_cash / len(assets))
    final = start_cash - share * len(assets)
    for a in assets:
        px0, px1 = candles[a]["open"].iloc[1], candles[a]["close"].iloc[-2]
        final += share * (1 - fee_rate) / (1 + slippage_rate) * px1 / px0 * (1 - slippage_rate) * (1 - fee_rate)
    return Result("buy & hold (equal weight)", final, start_cash, 0.0, len(assets), 0, 1.0)


def report(results: list[Result]) -> str:
    lines = [f"{'strategy':<44}{'return':>9}{'max DD':>8}{'trades':>8}{'win%':>6}{'in mkt':>8}"]
    for r in results:
        sells = max(1, r.trades // 2)
        lines.append(
            f"{r.label:<44}{r.ret:>+9.1%}{r.max_drawdown:>8.1%}{r.trades:>8}{r.wins / sells:>6.0%}{r.exposure:>8.0%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from bot import data
    from bot.assets import ASSETS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--assets", default="SOL,BTC,ETH,BNB,XRP,DOGE,AVAX")
    parser.add_argument("--max-positions", type=int, default=3)
    args = parser.parse_args(argv)
    symbols = [s.strip().upper() for s in args.assets.split(",")]
    candles = {s: data.fetch_history(ASSETS[s].market, args.days, args.timeframe, "binance") for s in symbols}
    base = Params()
    results = [buy_and_hold(candles)]
    for label, p in (
        ("configured (.env) strategy", base),
        ("+ edge 15bps, confirm, ATR stop 3", replace(base, edge_bps=15, confirm=True, stop_atr=3.0)),
    ):
        results.append(simulate(candles, p, label=label, max_positions=args.max_positions))
    print(report(results))


if __name__ == "__main__":
    main()
