"""Execution quality: what each swap really cost, a sanity check on quotes, and fee/slippage choices from market state.

    python -m bot.execution report        # implementation shortfall by coin and side from the trade-cost table
"""
from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np
import requests

HERMES = "https://hermes.pyth.network/v2/updates/price/latest"
PYTH_FEEDS = {  # Pyth price-feed ids (hex, no 0x) for the coins the bot trades
    "SOL": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "BTC": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
}
BASE_FEE_LAMPORTS = 5000


def implementation_shortfall_bps(side: str, arrival: float, fill: float) -> float:
    """Cost of a fill against the price when the decision was made, in basis points. Positive = worse than arrival."""
    if arrival <= 0 or fill <= 0:
        return float("nan")
    diff = (fill - arrival) if side == "buy" else (arrival - fill)
    return diff / arrival * 10_000


def choose_priority_level(recent_fees: Sequence[int], spike: float = 3.0) -> str:
    """Jupiter priority level from recent per-slot prioritisation fees: normally 'medium', 'high' when the busy end
    of the market (75th percentile) runs well above its typical (median) level. The lamport cap still applies."""
    fees = [f for f in recent_fees if f >= 0]
    if len(fees) < 10:
        return "medium"
    p50, p75 = np.percentile(fees, [50, 75])
    return "high" if p75 > spike * max(p50, 1) else "medium"


def dynamic_slippage_bps(vol_per_bar: float, *, base: float = 30.0, k: float = 0.5, cap: float = 100.0) -> int:
    """Slippage tolerance that widens with recent volatility (per-bar return std), never above the hard cap."""
    return int(min(cap, max(10.0, base + k * vol_per_bar * 10_000)))


def fee_usd(lamports: int, sol_price: float) -> float:
    return (lamports + BASE_FEE_LAMPORTS) / 1e9 * sol_price


def pyth_price(http: requests.Session, symbol: str, timeout: float = 5.0) -> float | None:
    """Latest Pyth price for a coin, or None if unavailable. A second, independent opinion on the price."""
    feed = PYTH_FEEDS.get(symbol)
    if feed is None:
        return None
    try:
        resp = http.get(HERMES, params={"ids[]": feed}, timeout=timeout)
        resp.raise_for_status()
        p = resp.json()["parsed"][0]["price"]
        return float(p["price"]) * 10 ** int(p["expo"])
    except (requests.RequestException, KeyError, IndexError, ValueError, TypeError):
        return None


def price_sane(price: float, other: float | None, tol_bps: float = 300.0) -> bool:
    """True if `price` is within `tol_bps` of an independent reference; True when there is no reference to compare."""
    if other is None or other <= 0:
        return True
    return abs(price / other - 1) * 10_000 <= tol_bps


def fit_impact(notional: Sequence[float], shortfall_bps: Sequence[float]) -> tuple[float, float]:
    """Linear impact model: shortfall_bps ~ fixed + slope * notional. Returns (fixed_bps, bps_per_usd)."""
    x, y = np.asarray(notional, float), np.asarray(shortfall_bps, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5 or np.ptp(x[ok]) == 0:
        return (float(np.nanmean(y)) if ok.any() else float("nan")), 0.0
    slope, fixed = np.polyfit(x[ok], y[ok], 1)
    return float(fixed), float(slope)


def report_text(rows) -> str:
    import pandas as pd

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return "no trades recorded yet"
    lines = [f"{'coin':<7}{'side':<6}{'n':>4}{'avg bps':>10}{'worst bps':>11}{'fees $':>9}"]
    for (asset, side), g in df.groupby(["asset", "side"]):
        lines.append(f"{asset:<7}{side:<6}{len(g):>4}{g['shortfall_bps'].mean():>10.1f}{g['shortfall_bps'].max():>11.1f}{g['fees_usd'].sum():>9.3f}")
    fixed, slope = fit_impact(df["notional"], df["shortfall_bps"])
    lines.append(f"impact model: {fixed:.1f} bps + {slope * 100:.2f} bps per $100 traded "
                 f"(modelled cost is 40 bps per side plus the fixed fee)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from bot import db
    from bot.config import settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["report"])
    parser.parse_args(argv)
    conn = db.connect(settings.db_path, readonly=True)
    print(report_text(db.read_tca(conn)))


if __name__ == "__main__":
    main()
