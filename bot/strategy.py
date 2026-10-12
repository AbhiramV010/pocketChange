from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from bot.config import (
    EDGE_BPS, ENTRY, HARD_STOP_PCT, REENTRY_CANDLES, REGIME_LEN, SMA_FAST, SMA_SLOW, STOP_ATR, TIMEFRAME, TREND_CONFIRM,
    candle_freq,
)

ATR_PERIOD = 14


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class Params:
    """Everything the entry/exit rules depend on, shared by the live loop and the backtest."""

    fast: int = SMA_FAST
    slow: int = SMA_SLOW
    entry: str = ENTRY
    edge_bps: float = EDGE_BPS  # fast must exceed slow by this much to enter
    confirm: bool = TREND_CONFIRM  # also need close > slow SMA and the slow SMA rising
    stop_atr: float = STOP_ATR  # trailing stop, in ATRs below the highest close since entry (0 = off)
    hard_stop_pct: float = HARD_STOP_PCT  # exit if price is this % under the entry price (0 = off)
    reentry_candles: int = REENTRY_CANDLES  # after any sell, wait this many candles before buying again
    regime_len: int = REGIME_LEN  # only open new positions while BTC is above its SMA of this length (0 = off)


# Array cores. The live loop and the backtest both call these, so they cannot drift apart.

def sma_signal(closes: np.ndarray, p: Params) -> Signal:
    if p.fast >= p.slow:
        raise ValueError("fast period must be shorter than slow period")
    closes = closes[-(p.slow + 1):]
    if len(closes) < p.slow + 1:
        return Signal.HOLD
    prev_fast, prev_slow = closes[-p.fast - 1:-1].mean(), closes[-p.slow - 1:-1].mean()
    cur_fast, cur_slow = closes[-p.fast:].mean(), closes[-p.slow:].mean()
    if math.isnan(prev_fast + prev_slow + cur_fast + cur_slow):
        return Signal.HOLD
    edge = 1 + p.edge_bps / 10_000
    confirmed = not p.confirm or (closes[-1] > cur_slow and cur_slow > prev_slow)
    if p.entry == "trend":
        if cur_fast < cur_slow:
            return Signal.SELL
        return Signal.BUY if cur_fast > cur_slow * edge and confirmed else Signal.HOLD
    if prev_fast <= prev_slow and cur_fast > cur_slow * edge and confirmed:
        return Signal.BUY
    if prev_fast >= prev_slow and cur_fast < cur_slow:
        return Signal.SELL
    return Signal.HOLD


def regime_ok(btc_closes: np.ndarray, p: Params) -> bool:
    """Market filter: risk-on only while BTC trades above its long moving average."""
    if p.regime_len <= 0:
        return True
    if len(btc_closes) < p.regime_len:
        return False  # not enough history to call it risk-on
    return bool(btc_closes[-1] > btc_closes[-p.regime_len:].mean())


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = ATR_PERIOD) -> float:
    if len(close) < 2:
        return float("nan")
    prev_close = close[:-1]
    true_range = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)])
    return float(true_range[-period:].mean())


def stop_hit(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, entry_price: float, bars_since_entry: int, p: Params
) -> bool:
    """True if the position should be cut now: trailing ATR stop or hard stop under the entry."""
    last = close[-1]
    if p.hard_stop_pct > 0 and last < entry_price * (1 - p.hard_stop_pct / 100):
        return True
    if p.stop_atr > 0 and len(close) > ATR_PERIOD:
        window = max(1, min(bars_since_entry, len(close)))
        peak = max(close[-window:].max(), entry_price)
        level = peak - p.stop_atr * atr(high, low, close)
        if last < level:
            return True
    return False


def momentum(closes: np.ndarray, lookback: int) -> float:
    """Rate of change over `lookback` candles; used to rank buy candidates."""
    if len(closes) <= lookback or closes[-lookback - 1] <= 0:
        return float("-inf")
    return float(closes[-1] / closes[-lookback - 1] - 1)


def decide(
    candles: pd.DataFrame,
    p: Params,
    *,
    holding: bool,
    entry_price: float = 0.0,
    entry_time: pd.Timestamp | None = None,
    last_sell_time: pd.Timestamp | None = None,
    risk_on: bool = True,
    timeframe: str = TIMEFRAME,
) -> Signal:
    """Signal for one asset from its closed candles, given what we hold of it."""
    closes = candles["close"].to_numpy(dtype=float)
    signal = sma_signal(closes, p)
    if holding:
        if entry_time is not None:
            bars = int((candles.index >= pd.Timestamp(entry_time).floor(candle_freq(timeframe))).sum())
        else:
            bars = len(candles)
        if stop_hit(
            candles["high"].to_numpy(dtype=float), candles["low"].to_numpy(dtype=float), closes, entry_price, bars, p
        ):
            return Signal.SELL
        return signal
    if signal is Signal.BUY and not risk_on:
        return Signal.HOLD
    if signal is Signal.BUY and last_sell_time is not None and p.reentry_candles > 0:
        wait = pd.Timedelta(candle_freq(timeframe)) * p.reentry_candles
        if candles.index[-1] + pd.Timedelta(candle_freq(timeframe)) - pd.Timestamp(last_sell_time) < wait:
            return Signal.HOLD
    return signal


def sma_crossover_signal(
    candles: pd.DataFrame, fast: int = SMA_FAST, slow: int = SMA_SLOW, entry: str = ENTRY, **kwargs
) -> Signal:
    return sma_signal(
        candles["close"].to_numpy(dtype=float),
        Params(fast=fast, slow=slow, entry=entry, **{"edge_bps": 0.0, "confirm": False, **kwargs}),
    )


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
    """Amount to trade: USDC to spend on a buy, units of the asset (`wsol`) to sell on a sell."""
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
