from __future__ import annotations

import math
from enum import Enum

import pandas as pd

from bot.config import ENTRY, SMA_FAST, SMA_SLOW, TIMEFRAME, candle_freq


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


def sma_crossover_signal(
    candles: pd.DataFrame, fast: int = SMA_FAST, slow: int = SMA_SLOW, entry: str = ENTRY
) -> Signal:
    if fast >= slow:
        raise ValueError("fast period must be shorter than slow period")
    closes = candles["close"].to_numpy(dtype=float)[-(slow + 1) :]
    if len(closes) < slow + 1:
        return Signal.HOLD
    prev_fast, prev_slow = closes[-fast - 1 : -1].mean(), closes[-slow - 1 : -1].mean()
    cur_fast, cur_slow = closes[-fast:].mean(), closes[-slow:].mean()
    if math.isnan(prev_fast + prev_slow + cur_fast + cur_slow):
        return Signal.HOLD
    if entry == "trend":
        return Signal.BUY if cur_fast > cur_slow else Signal.SELL if cur_fast < cur_slow else Signal.HOLD
    if prev_fast <= prev_slow and cur_fast > cur_slow:
        return Signal.BUY
    if prev_fast >= prev_slow and cur_fast < cur_slow:
        return Signal.SELL
    return Signal.HOLD


def dca_signal(
    candles: pd.DataFrame, last_buy_time: pd.Timestamp | None, interval_hours: float, timeframe: str = TIMEFRAME
) -> Signal:
    if candles.empty:
        return Signal.HOLD
    if last_buy_time is None:
        return Signal.BUY
    freq = candle_freq(timeframe)
    now = candles.index[-1] + pd.Timedelta(freq)  # close time of the latest closed candle
    last = pd.Timestamp(last_buy_time).floor(freq)  # buys land mid-candle; compare on the candle grid
    return Signal.BUY if now - last >= pd.Timedelta(hours=interval_hours) else Signal.HOLD


def size_trade(
    signal: Signal,
    strategy: str,
    *,
    usdc: float,
    wsol: float,
    price: float,
    max_trade_usdc: float,
    dca_amount_usdc: float,
    min_trade_usdc: float,
) -> float:
    position_usd = wsol * price
    if signal is Signal.BUY:
        if strategy == "sma" and position_usd >= min_trade_usdc:
            return 0.0  # already long: do not stack positions on a crossover
        want = dca_amount_usdc if strategy == "dca" else max_trade_usdc
        amount = min(want, max_trade_usdc, usdc)
        return amount if amount >= min_trade_usdc else 0.0
    if signal is Signal.SELL:
        if position_usd < min_trade_usdc:
            return 0.0  # nothing worth selling
        return min(wsol, max_trade_usdc / price)
    return 0.0
