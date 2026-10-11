import numpy as np
import pandas as pd
import pytest

from bot.strategy import Signal, dca_signal, size_trade, sma_crossover_signal


def candles(closes, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1h", tz="UTC", name="timestamp")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0})


def test_sma_needs_history():
    assert sma_crossover_signal(candles([100.0] * 50)) is Signal.HOLD


def test_sma_buy_on_cross_up_only_once():
    down = list(np.linspace(200, 100, 60))   # fast below slow
    up = list(np.linspace(100, 300, 40))
    signals = [sma_crossover_signal(candles(down + up[: i + 1])) for i in range(len(up))]
    assert signals.count(Signal.BUY) == 1
    assert Signal.SELL not in signals
    assert signals.index(Signal.BUY) > 0  # not on the very first candle of the rally


def test_sma_sell_on_cross_down_only_once():
    up = list(np.linspace(100, 200, 60))
    down = list(np.linspace(200, 50, 40))
    signals = [sma_crossover_signal(candles(up + down[: i + 1])) for i in range(len(down))]
    assert signals.count(Signal.SELL) == 1
    assert Signal.BUY not in signals


def test_sma_flat_market_holds():
    assert sma_crossover_signal(candles([100.0] * 80)) is Signal.HOLD


def test_sma_is_pure_and_ignores_old_history():
    data = candles(list(np.linspace(200, 100, 60)) + list(np.linspace(100, 300, 25)))
    before = data.copy()
    a = sma_crossover_signal(data)
    b = sma_crossover_signal(data.iloc[-60:])
    assert a is b
    pd.testing.assert_frame_equal(data, before)


def test_sma_rejects_bad_periods():
    with pytest.raises(ValueError):
        sma_crossover_signal(candles([1.0] * 100), fast=50, slow=20)


def test_dca_first_buy_and_interval():
    c = candles([100.0] * 30)  # last candle opens 2026-01-02 05:00, closes 06:00
    assert dca_signal(c, None, 24) is Signal.BUY
    assert dca_signal(c, pd.Timestamp("2026-01-01T06:00:00Z"), 24) is Signal.BUY       # exactly 24h
    assert dca_signal(c, pd.Timestamp("2026-01-01T06:40:00Z"), 24) is Signal.BUY       # lands mid-candle, same grid
    assert dca_signal(c, pd.Timestamp("2026-01-01T07:00:00Z"), 24) is Signal.HOLD      # 23h
    assert dca_signal(candles([]), None, 24) is Signal.HOLD


KW = dict(max_trade_usdc=5, dca_amount_usdc=2, min_trade_usdc=1)


def test_size_buy_caps():
    assert size_trade(Signal.BUY, "sma", usdc=20, wsol=0, price=2500, **KW) == 5
    assert size_trade(Signal.BUY, "sma", usdc=3, wsol=0, price=2500, **KW) == 3
    assert size_trade(Signal.BUY, "sma", usdc=0.5, wsol=0, price=2500, **KW) == 0
    assert size_trade(Signal.BUY, "dca", usdc=20, wsol=0, price=2500, **KW) == 2
    assert size_trade(Signal.BUY, "dca", usdc=20, wsol=0, price=2500, **{**KW, "dca_amount_usdc": 50}) == 5


def test_size_sma_does_not_stack_but_dca_does():
    assert size_trade(Signal.BUY, "sma", usdc=20, wsol=0.002, price=2500, **KW) == 0
    assert size_trade(Signal.BUY, "dca", usdc=20, wsol=0.002, price=2500, **KW) == 2


def test_size_sell_caps_at_limit_and_ignores_dust():
    assert size_trade(Signal.SELL, "sma", usdc=0, wsol=0.001, price=2500, **KW) == 0.001
    assert size_trade(Signal.SELL, "sma", usdc=0, wsol=0.01, price=2500, **KW) == pytest.approx(0.002)
    assert size_trade(Signal.SELL, "sma", usdc=0, wsol=0.0002, price=2500, **KW) == 0
    assert size_trade(Signal.HOLD, "sma", usdc=20, wsol=1, price=2500, **KW) == 0


def test_trend_entry_follows_the_regime_immediately():
    up = candles(list(np.linspace(100, 200, 80)))
    down = candles(list(np.linspace(200, 100, 80)))
    assert sma_crossover_signal(up, entry="trend") is Signal.BUY
    assert sma_crossover_signal(up, entry="cross") is Signal.HOLD  # cross happened long ago
    assert sma_crossover_signal(down, entry="trend") is Signal.SELL
    assert sma_crossover_signal(candles([100.0] * 80), entry="trend") is Signal.HOLD


def test_faster_periods_cross_more_often():
    wave = list(100 + np.cumsum(np.random.default_rng(7).normal(0, 1, 400)))  # seeded random walk
    def crosses(fast, slow):
        return sum(sma_crossover_signal(candles(wave[: i + 1]), fast, slow) is not Signal.HOLD for i in range(60, 400))
    assert crosses(9, 21) > crosses(20, 50)
