"""Live versus backtest reconciliation: did the bot do what the backtest says it should have, at the prices modelled?

    python -m bot.reconcile --days 7     # replay the last week on the same candles and compare with the trade log
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd


@dataclass
class Reconciliation:
    matched: int
    missing_live: list  # the backtest traded, the bot did not
    extra_live: list  # the bot traded, the backtest did not
    slippage_bps: float  # mean live fill price vs backtest fill price (positive = live was worse)

    def alerts(self, max_mismatch: float = 0.2, max_slippage_bps: float = 50.0) -> list[str]:
        total = self.matched + len(self.missing_live) + len(self.extra_live)
        out = []
        if total and (len(self.missing_live) + len(self.extra_live)) / total > max_mismatch:
            out.append(f"{len(self.missing_live)} trades missing and {len(self.extra_live)} unexpected out of {total}")
        if self.slippage_bps == self.slippage_bps and self.slippage_bps > max_slippage_bps:
            out.append(f"live fills are {self.slippage_bps:.0f} bps worse than the backtest's")
        return out


def _price(side: str, amount_in: float, amount_out: float) -> float:
    return amount_in / amount_out if side == "buy" else amount_out / amount_in


def reconcile(live: pd.DataFrame, backtest_fills: list, freq: str, tolerance_candles: int = 1) -> Reconciliation:
    """`live`: db.read_trades rows (timestamp, side, amount_in, amount_out, asset). `backtest_fills`: Result.fills
    tuples (time, side, coin, in, out). A live trade matches a backtest fill of the same coin and side within
    `tolerance_candles` candles."""
    step = pd.Timedelta(freq)
    bt = [(pd.Timestamp(t), s, a, _price(s, i, o)) for t, s, a, i, o in backtest_fills]
    used, matched, slips, extra = set(), 0, [], []
    for _, row in live.iterrows():
        t = pd.Timestamp(row["timestamp"]).floor(freq)
        price = _price(row["side"], row["amount_in"], row["amount_out"])
        hit = next((k for k, (bt_t, s, a, _) in enumerate(bt)
                    if k not in used and a == row["asset"] and s == row["side"] and abs(bt_t - t) <= step * tolerance_candles), None)
        if hit is None:
            extra.append((row["timestamp"], row["side"], row["asset"]))
            continue
        used.add(hit)
        matched += 1
        ref = bt[hit][3]
        slips.append((price - ref) / ref * 10_000 * (1 if row["side"] == "buy" else -1))
    missing = [(t, s, a) for k, (t, s, a, _) in enumerate(bt) if k not in used]
    return Reconciliation(matched, missing, extra, float(sum(slips) / len(slips)) if slips else float("nan"))


def main(argv: list[str] | None = None) -> None:
    from bot import data, db
    from bot.assets import ASSETS
    from bot.config import settings
    from bot.portfolio_backtest import simulate
    from bot.strategy import Params

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--timeframe", default="4h")
    args = parser.parse_args(argv)
    conn = db.connect(settings.db_path, readonly=True)
    trades = db.read_trades(conn)
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)
    trades = trades[pd.to_datetime(trades["timestamp"], utc=True) >= since]
    symbols = list(dict.fromkeys(list(settings.assets) + ["BTC"]))
    candles = {s: data.fetch_history(ASSETS[s].market, args.days + 400, args.timeframe, settings.candle_exchange) for s in symbols}
    res = simulate(candles, Params(), bench=candles["BTC"], trade_usdc=settings.max_trade_usdc, max_positions=settings.max_positions)
    window = [f for f in res.fills if pd.Timestamp(f[0]) >= since]
    rec = reconcile(trades, window, data.candle_delta(args.timeframe))
    print(f"matched {rec.matched}, missing {len(rec.missing_live)}, unexpected {len(rec.extra_live)}, slippage {rec.slippage_bps:.1f} bps")
    for line in rec.alerts():
        print("ALERT:", line)


if __name__ == "__main__":
    main()
