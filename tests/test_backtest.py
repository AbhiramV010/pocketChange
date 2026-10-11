import numpy as np
import pandas as pd
import pytest

from bot import backtest
from bot.backtest import Result, buy_and_hold, format_report, simulate
from bot.config import FEE_RATE, SLIPPAGE_RATE, load_settings

KW = dict(max_trade_usdc=5.0, dca_amount_usdc=1.0, dca_interval_hours=24)


def candles(closes, opens=None):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="1h", tz="UTC", name="timestamp")
    close = pd.Series(closes, index=idx, dtype=float)
    open_ = close if opens is None else pd.Series(opens, index=idx, dtype=float)
    return pd.DataFrame({"open": open_, "high": close, "low": close, "close": close, "volume": 1.0})


def test_buy_and_hold_pays_one_round_of_costs():
    r = buy_and_hold(candles([100.0] * 10), start=20.0)
    assert r.trades == 1
    assert r.final == pytest.approx(20.0 * (1 - FEE_RATE) / (1 + SLIPPAGE_RATE))
    assert r.total_return == pytest.approx(-0.004, abs=1e-4)
    assert r.exposure == pytest.approx(1.0)


def test_buy_and_hold_tracks_price():
    r = buy_and_hold(candles([100.0, 100.0, 200.0]), start=20.0)
    assert r.final == pytest.approx(2 * 20.0 * (1 - FEE_RATE) / (1 + SLIPPAGE_RATE))


def test_dca_buys_each_interval_and_stops_when_cash_is_gone():
    r = simulate("dca", candles([100.0] * 72), "dca", start=20.0, **KW)
    assert r.trades == 3  # fills at candle 1, 25 and 49
    assert 0 < r.exposure < 0.2
    out = simulate("dca", candles([100.0] * 24 * 30), "dca", start=3.0, **KW)
    assert out.trades == 3  # only $3 to spend


def test_flat_market_never_trades_sma():
    r = simulate("sma", candles([100.0] * 200), "sma", start=20.0, **KW)
    assert r.trades == 0 and r.final == 20.0 and r.exposure == 0


def test_signal_is_filled_at_next_open_not_the_signal_close():
    down = list(np.linspace(200, 100, 60))
    up = list(np.linspace(100, 300, 40))
    closes = down + up
    opens = [c * 1.5 for c in closes]  # next-open is far from the signal close
    flat = simulate("sma", candles(closes), "sma", start=20.0, **KW)
    gapped = simulate("sma", candles(closes, opens), "sma", start=20.0, **KW)
    assert flat.trades >= 1 and gapped.trades == flat.trades
    # with opens 50% above closes the same buy gets ~1/3 less coin, so exposure differs
    assert gapped.exposure != pytest.approx(flat.exposure)


def test_costs_scale_with_trade_count():
    wave = list(100 + 30 * np.sin(np.arange(600) / 12))  # repeated up/down swings -> many crossovers
    cheap = simulate("sma", candles(wave), "sma", start=20.0, fee=0.0, slip=0.0, **KW)
    costly = simulate("sma", candles(wave), "sma", start=20.0, fee=0.003, slip=0.001, **KW)
    assert cheap.trades >= 4 and costly.trades == cheap.trades
    assert costly.final < cheap.final


def test_result_metrics():
    eq = pd.Series([20.0, 25.0, 15.0, 30.0], index=pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"))
    r = Result("x", 20.0, eq, 0.5, 2)
    assert r.final == 30.0
    assert r.total_return == pytest.approx(0.5)
    assert r.max_drawdown == pytest.approx(-0.4)  # 25 -> 15


def test_report_lists_all_strategies_and_buy_and_hold():
    c = candles(list(np.linspace(200, 100, 80)) + list(np.linspace(100, 250, 80)))
    cfg = load_settings({})
    results = backtest.run_all(c, cfg)
    text = format_report(results, c, cfg)
    for name in ("Buy & hold", "DCA $1/24h", "SMA 20/50 cross", "SMA 20/50 cross all-in", "Max DD", "Return", "Trades"):
        assert name in text
    assert all(np.isfinite(r.equity).all() for r in results)
