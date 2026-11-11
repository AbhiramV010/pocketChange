"""Profit and loss attribution from the trade log: by coin, split into gross result and each cost component."""
from __future__ import annotations

import pandas as pd

from bot.config import FEE_RATE, FIXED_COST_USD, SLIPPAGE_RATE


def attribute(trades: pd.DataFrame, strategy: str = "sma") -> pd.DataFrame:
    """Average-cost realised P&L per coin. `trades` is `db.read_trades` output (side, amount_in, amount_out, asset).
    Costs are the modelled ones applied to each fill's notional, so gross + costs reconcile to net."""
    rows = []
    for asset, g in trades.sort_values("timestamp").groupby("asset"):
        qty = cost = 0.0
        net = fees = slip = fixed = 0.0
        n = 0
        for _, t in g.iterrows():
            n += 1
            if t["side"] == "buy":
                qty += t["amount_out"]
                cost += t["amount_in"]
                notional = t["amount_in"]
            else:
                frac = min(1.0, t["amount_in"] / qty) if qty > 0 else 0.0
                net += t["amount_out"] - cost * frac
                cost -= cost * frac
                qty -= t["amount_in"]
                notional = t["amount_out"]
            fees += notional * FEE_RATE
            slip += notional * SLIPPAGE_RATE
            fixed += FIXED_COST_USD
        rows.append({"strategy": strategy, "asset": asset, "trades": n, "net": net, "fees": fees, "slippage": slip,
                     "fixed": fixed, "gross": net + fees + slip + fixed})
    out = pd.DataFrame(rows, columns=["strategy", "asset", "trades", "net", "fees", "slippage", "fixed", "gross"])
    return out


def summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "no trades"
    t = table.sum(numeric_only=True)
    return (f"net {t['net']:+.2f} USDC = gross {t['gross']:+.2f} - fees {t['fees']:.2f} - slippage {t['slippage']:.2f} "
            f"- fixed {t['fixed']:.2f} over {int(t['trades'])} fills")
