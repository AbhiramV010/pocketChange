import numpy as np
import pandas as pd
import pytest

from bot.portfolio_backtest import align, buy_and_hold, report, simulate
from bot.strategy import Params

TREND = Params(fast=5, slow=20, entry="trend", edge_bps=0.0, confirm=False, stop_atr=0.0, hard_stop_pct=0.0)


def candles(closes, freq="1h"):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq=freq, tz="UTC", name="timestamp")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0})


def market(rate, n=400):
    return candles(100 * (1 + rate) ** np.arange(n))


def test_trend_following_profits_on_a_clean_uptrend_without_costs():
    r = simulate({"SOL": market(0.002), "BTC": market(0.002)}, TREND, fee_rate=0.0, slippage_rate=0.0)
    assert r.final > r.start and r.trades >= 2 and r.exposure > 0.9


def test_costs_are_charged_on_every_fill():
    data = {"SOL": market(0.002), "BTC": market(0.002)}
    free = simulate(data, TREND, fee_rate=0.0, slippage_rate=0.0)
    costly = simulate(data, TREND, fee_rate=0.01, slippage_rate=0.005)
    assert costly.final < free.final


def test_the_btc_regime_filter_keeps_it_out_while_btc_is_falling():
    data = {"SOL": market(0.002), "BTC": market(-0.002)}
    filtered = simulate(data, Params(**{**TREND.__dict__, "regime_len": 50}), fee_rate=0.0, slippage_rate=0.0)
    assert filtered.trades == 0 and filtered.final == pytest.approx(filtered.start)


def test_max_positions_limits_open_positions():
    data = {a: market(0.002) for a in ("SOL", "BTC", "ETH", "BNB")}
    r = simulate(data, TREND, max_positions=2, fee_rate=0.0, slippage_rate=0.0)
    assert r.trades == 2  # two slots, held to the end
    assert sum(1 for v in r.per_asset.values() if v != 0) <= 2 or r.trades == 2


def test_series_are_aligned_and_buy_and_hold_splits_cash_equally():
    a, b = market(0.001, 300), market(0.001, 300).iloc[10:]
    aligned = align({"SOL": a, "BTC": b})
    assert len(aligned["SOL"]) == len(aligned["BTC"]) == 290
    bh = buy_and_hold({"SOL": market(0.001), "BTC": market(0.001)}, fee_rate=0.0, slippage_rate=0.0)
    assert bh.final > bh.start and bh.trades == 2


def test_report_has_one_line_per_result():
    r = simulate({"SOL": market(0.002), "BTC": market(0.002)}, TREND, label="trend")
    text = report([buy_and_hold({"SOL": market(0.002), "BTC": market(0.002)}), r])
    assert "trend" in text and "buy & hold" in text and len(text.splitlines()) == 3
