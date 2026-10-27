import numpy as np
import pandas as pd
import pytest

from bot.strategy import Params, Signal, atr, decide, momentum, regime_ok, sma_signal, stop_hit

TREND = Params(fast=5, slow=20, entry="trend", edge_bps=0.0, confirm=False, stop_atr=0.0, hard_stop_pct=0.0)


def frame(closes, spread=0.0, freq="1h"):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq=freq, tz="UTC", name="timestamp")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + spread, "low": c - spread, "close": c, "volume": 1.0})


def rising(n=60, step=0.5):
    return np.array([100 + step * i for i in range(n)])


def test_trend_mode_is_long_above_and_flat_below():
    assert sma_signal(rising(), TREND) is Signal.BUY
    assert sma_signal(rising()[::-1].copy(), TREND) is Signal.SELL
    assert sma_signal(np.full(60, 100.0), TREND) is Signal.HOLD


def test_edge_blocks_a_marginal_entry_but_never_an_exit():
    gentle = rising(step=0.01)  # fast barely above slow
    assert sma_signal(gentle, TREND) is Signal.BUY
    picky = Params(fast=5, slow=20, entry="trend", edge_bps=100.0)
    assert sma_signal(gentle, picky) is Signal.HOLD
    assert sma_signal(rising()[::-1].copy(), picky) is Signal.SELL


def test_confirm_needs_a_rising_slow_average_and_price_above_it():
    bounce = np.concatenate([np.full(40, 150.0), np.full(10, 90.0), np.full(10, 120.0)])  # dead-cat bounce
    plain = Params(fast=3, slow=20, entry="trend", confirm=False)
    strict = Params(fast=3, slow=20, entry="trend", confirm=True)
    assert sma_signal(bounce, plain) is Signal.BUY
    assert sma_signal(bounce, strict) is Signal.HOLD  # the slow average is still falling


def test_atr_and_hard_stop():
    f = frame(rising(30), spread=1.0)
    assert atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy()) == pytest.approx(2.0)
    hard = Params(hard_stop_pct=5.0)
    assert not stop_hit(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), 100.0, 10, hard)
    assert stop_hit(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), 200.0, 10, hard)


def test_trailing_stop_follows_the_peak():
    closes = np.concatenate([rising(30), np.linspace(114.5, 100, 8)])
    f = frame(closes, spread=0.5)
    h, l, c = f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy()
    trailing = Params(stop_atr=3.0)
    assert stop_hit(h, l, c, entry_price=100.0, bars_since_entry=len(c), p=trailing)  # gave back a lot from the peak
    assert not stop_hit(h[:30], l[:30], c[:30], entry_price=100.0, bars_since_entry=30, p=trailing)
    assert not stop_hit(h, l, c, entry_price=100.0, bars_since_entry=len(c), p=Params(stop_atr=0.0))


def test_momentum_and_regime():
    assert momentum(np.array([100.0, 110.0, 121.0]), 2) == pytest.approx(0.21)
    assert momentum(np.array([1.0, 2.0]), 5) == float("-inf")
    p = Params(regime_len=10)
    assert regime_ok(rising(30), p) and not regime_ok(rising(30)[::-1].copy(), p)
    assert not regime_ok(rising(5), p)  # too little history counts as risk-off
    assert regime_ok(np.array([1.0]), Params(regime_len=0))


def test_decide_cuts_a_position_on_a_stop_even_when_the_trend_says_buy():
    closes = np.concatenate([rising(55), [rising(55)[-1] * 0.9]])
    f = frame(closes, spread=0.3)
    p = Params(fast=5, slow=20, entry="trend", hard_stop_pct=5.0)
    assert decide(f, p, holding=True, entry_price=127.0, timeframe="1h") is Signal.SELL
    assert decide(f, p, holding=True, entry_price=100.0, timeframe="1h") in (Signal.BUY, Signal.HOLD, Signal.SELL)


def test_decide_blocks_entries_when_risk_off_and_right_after_an_exit():
    f = frame(np.concatenate([np.full(30, 100.0), rising(40) + 0.5]))  # flat, then a clean uptrend
    p = Params(fast=5, slow=20, entry="trend", reentry_candles=6)
    assert decide(f, p, holding=False, timeframe="1h") is Signal.BUY
    assert decide(f, p, holding=False, risk_on=False, timeframe="1h") is Signal.HOLD
    just_sold = f.index[-1] - pd.Timedelta(hours=2)
    assert decide(f, p, holding=False, last_sell_time=just_sold, timeframe="1h") is Signal.HOLD
    long_ago = f.index[-1] - pd.Timedelta(hours=45)  # sold while flat: the uptrend since is a fresh one
    assert decide(f, p, holding=False, last_sell_time=long_ago, timeframe="1h") is Signal.BUY
    assert decide(f, p, holding=True, risk_on=False, timeframe="1h") is Signal.BUY  # the filter never blocks exits


def test_trend_mode_waits_for_the_trend_to_reset_after_a_stop_out():
    f = frame(rising(120))  # one long uptrend: fast never dips below slow
    p = Params(fast=5, slow=40, entry="trend")
    stopped = f.index[-1] - pd.Timedelta(hours=10)  # stopped out ten candles ago, trend still up
    assert decide(f, p, holding=False, last_sell_time=stopped, timeframe="1h") is Signal.HOLD
    assert decide(f, p, holding=False, timeframe="1h") is Signal.BUY  # never sold: the trend is a fresh entry
    long_ago = f.index[-1] - pd.Timedelta(hours=45)  # over `slow` candles ago: a fresh trend by now
    assert decide(f, p, holding=False, last_sell_time=long_ago, timeframe="1h") is Signal.BUY
    dip = frame(np.concatenate([rising(50), rising(50)[-1] - np.arange(1, 9) * 2.0, rising(30, step=1.5) + 98]))
    after_dip = dip.index[-1] - pd.Timedelta(hours=30)  # the trend broke during the dip, then came back
    assert decide(dip, p, holding=False, last_sell_time=after_dip, timeframe="1h") is Signal.BUY
    cross = Params(fast=5, slow=40, entry="cross")  # cross mode needs a fresh crossover anyway
    assert decide(f, cross, holding=False, last_sell_time=stopped, timeframe="1h") is Signal.HOLD


def test_reentry_candles_counts_whole_candles_after_the_sell():
    from bot.strategy import gate_entry

    one = Params(reentry_candles=1)
    assert gate_entry(Signal.BUY, one, risk_on=True, bars_since_sell=1) is Signal.HOLD  # the sell's own candle
    assert gate_entry(Signal.BUY, one, risk_on=True, bars_since_sell=2) is Signal.BUY
    off = Params(reentry_candles=0)
    assert gate_entry(Signal.BUY, off, risk_on=True, bars_since_sell=1) is Signal.BUY
    assert gate_entry(Signal.SELL, Params(reentry_candles=9), risk_on=False, bars_since_sell=1) is Signal.SELL
