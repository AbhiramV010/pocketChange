from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import pandas as pd

from bot import data, portfolio_backtest
from bot.assets import ASSETS, PRIMARY
from bot.config import (
    FEE_RATE,
    MIN_TRADE_USDC,
    SLIPPAGE_RATE,
    SYMBOL,
    TIMEFRAME,
    Settings,
    settings,
)
from bot.strategy import Params, Signal, dca_signal, size_trade

START_USDC = 20.0
DCA_WINDOW = 2  # candles the DCA rule looks at


@dataclass(frozen=True)
class Result:
    name: str
    start: float
    equity: pd.Series  # mark-to-market at each candle close
    exposure: float  # average share of equity held in SOL
    trades: int

    @property
    def final(self) -> float:
        return float(self.equity.iloc[-1])

    @property
    def total_return(self) -> float:
        return self.final / self.start - 1

    @property
    def max_drawdown(self) -> float:
        return float((self.equity / self.equity.cummax() - 1).min())

    @property
    def sharpe(self) -> float:
        return portfolio_backtest.sharpe(self.equity)


def _buy(cash: float, coin: float, spend: float, open_price: float, fee: float, slip: float):
    fill = open_price * (1 + slip)
    return cash - spend, coin + spend * (1 - fee) / fill


def _finish(name: str, start: float, candles: pd.DataFrame, cash: list, coin: list, trades: int) -> Result:
    closes = candles["close"].to_numpy()
    equity = pd.Series([c + q * p for c, q, p in zip(cash, coin, closes)], index=candles.index)
    exposure = float(pd.Series([q * p for q, p in zip(coin, closes)], index=candles.index).div(equity).mean())
    return Result(name, start, equity, exposure, trades)


def buy_and_hold(candles: pd.DataFrame, start: float = START_USDC, fee: float = FEE_RATE, slip: float = SLIPPAGE_RATE) -> Result:
    cash, coin = _buy(start, 0.0, start, float(candles["open"].iloc[0]), fee, slip)
    n = len(candles)
    return _finish("Buy & hold", start, candles, [cash] * n, [coin] * n, trades=1)


def simulate(
    name: str,
    candles: pd.DataFrame,
    strategy: str,
    *,
    start: float = START_USDC,
    max_trade_usdc: float,
    dca_amount_usdc: float,
    dca_interval_hours: float,
    fee: float = FEE_RATE,
    slip: float = SLIPPAGE_RATE,
    params: Params | None = None,
    bench: pd.DataFrame | None = None,
) -> Result:
    """One coin. SMA runs through the shared live-loop replay (stops, filters, sizing and all); DCA is simple."""
    if strategy == "sma":
        r = portfolio_backtest.simulate(
            {PRIMARY: candles}, params or Params(), bench=bench, label=name, start_cash=start,
            trade_usdc=max_trade_usdc, max_positions=1, fee_rate=fee, slippage_rate=slip,
        )
        assert r.equity is not None and r.invested is not None
        return Result(name, start, r.equity, float(r.invested.div(r.equity).mean()), r.trades)
    opens = candles["open"].to_numpy()
    cash, coin = start, 0.0
    cash_hist, coin_hist = [], []
    pending = Signal.HOLD
    last_buy: pd.Timestamp | None = None
    trades = 0
    for t in range(len(candles)):
        if pending is Signal.BUY:  # fill last candle's signal at this candle's open
            price = float(opens[t])
            amount = size_trade(
                pending, strategy, usdc=cash, wsol=coin, price=price, max_trade_usdc=max_trade_usdc,
                dca_amount_usdc=dca_amount_usdc, min_trade_usdc=MIN_TRADE_USDC,
            )
            if amount > 0:
                cash, coin = _buy(cash, coin, amount, price, fee, slip)
                last_buy = candles.index[t]
                trades += 1
        cash_hist.append(cash)
        coin_hist.append(coin)
        pending = dca_signal(candles.iloc[max(0, t + 1 - DCA_WINDOW) : t + 1], last_buy, dca_interval_hours)
    return _finish(name, start, candles, cash_hist, coin_hist, trades)


def run_all(
    candles: pd.DataFrame, cfg: Settings = settings, start: float = START_USDC, bench: pd.DataFrame | None = None
) -> list[Result]:
    common = dict(start=start, dca_amount_usdc=cfg.dca_amount_usdc, dca_interval_hours=cfg.dca_interval_hours)
    p = Params()
    rules = f"SMA {p.fast}/{p.slow} {p.entry}"
    return [
        buy_and_hold(candles, start),
        simulate(
            f"DCA ${cfg.dca_amount_usdc:g}/{cfg.dca_interval_hours:g}h", candles, "dca",
            max_trade_usdc=cfg.max_trade_usdc, **common,
        ),
        simulate(rules, candles, "sma", max_trade_usdc=cfg.max_trade_usdc, params=p, bench=bench, **common),
        simulate(
            f"{rules} all-in", candles, "sma", max_trade_usdc=math.inf, params=replace(p, risk_pct=0.0), bench=bench,
            **common,
        ),
    ]


def format_report(results: list[Result], candles: pd.DataFrame, cfg: Settings = settings) -> str:
    start = results[0].start
    rows = [
        f"{r.name:<26}{r.total_return:>+9.1%}{r.max_drawdown:>10.1%}{r.sharpe:>8.2f}{r.trades:>8d}{r.exposure:>11.0%}"
        f"{r.final:>10.2f}"
        for r in results
    ]
    header = f"{'Strategy':<26}{'Return':>9}{'Max DD':>10}{'Sharpe':>8}{'Trades':>8}{'Exposure':>11}{'Final $':>10}"
    return "\n".join(
        [
            f"{SYMBOL} {TIMEFRAME} backtest: {candles.index[0]:%Y-%m-%d %H:%M} -> {candles.index[-1]:%Y-%m-%d %H:%M} UTC "
            f"({len(candles)} candles)",
            f"Start ${start:.2f} USDC | cost per trade: {FEE_RATE:.2%} fee + {SLIPPAGE_RATE:.2%} slippage | "
            f"fills at next candle open",
            "",
            header,
            "-" * len(header),
            *rows,
            "",
            "SMA runs the configured rules (stops, filters, sizing) through the same code as the live loop, capped at",
            f"MAX_TRADE_USDC=${cfg.max_trade_usdc:g}. 'all-in' uses the same signals at buy & hold's exposure.",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest DCA and the configured SMA rules on cached candles.")
    parser.add_argument("--update", action="store_true", help="refresh the candle cache first")
    parser.add_argument("--days", type=int, default=None, help="only use the most recent N days")
    parser.add_argument("--start", type=float, default=START_USDC, help="starting USDC (default 20)")
    args = parser.parse_args(argv)

    candles = data.load_cache()
    if args.update or candles.empty:
        print("Updating candle cache...")
        candles = data.update_cache()
    candles = candles[candles.index + data.CANDLE <= pd.Timestamp.now(tz="UTC")]
    if args.days:
        candles = candles[candles.index >= candles.index[-1] - pd.Timedelta(days=args.days)]
    if len(candles) < Params().slow + 3:
        raise SystemExit(f"Not enough candles ({len(candles)}). Run: python -m bot.data")
    bench = None
    if Params().regime_len > 0:  # the BTC filter needs BTC candles over the same period
        days = (pd.Timestamp.now(tz="UTC") - candles.index[0]).days + 1
        bench = data.fetch_history(ASSETS[portfolio_backtest.REGIME_ASSET].market, days, TIMEFRAME, settings.candle_exchange)
    print(format_report(run_all(candles, settings, args.start, bench), candles, settings))


if __name__ == "__main__":
    main()
