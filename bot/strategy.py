from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from bot.config import (
    EDGE_BPS, ENTRY, HARD_STOP_PCT, REENTRY_CANDLES, REGIME_LEN, RISK_PCT, SMA_FAST, SMA_SLOW, STOP_ATR, TIMEFRAME,
    TREND_CONFIRM, candle_freq,
)

ATR_PERIOD = 14
SIZE_ATR = 3.0  # the move risk sizing guards against when neither stop is set
PEAK_LOOKBACK = 100  # candles the trailing stop looks back for its peak (the live loop fetches at least 119 closed)


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
    risk_pct: float = RISK_PCT  # size buys so hitting the stop loses this % of equity (0 = flat MAX_TRADE_USDC)


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
    high, low, close = high[-period - 1:], low[-period - 1:], close[-period - 1:]  # only the last `period` ranges count
    prev_close = close[:-1]
    true_range = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)])
    return float(true_range.mean())


def stop_hit(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, entry_price: float, bars_since_entry: int, p: Params
) -> bool:
    """True if the position should be cut now: trailing ATR stop or hard stop under the entry.

    The trailing peak is the highest close since entry within the last PEAK_LOOKBACK candles, a window the live
    loop always fetches, so a long hold is judged the same live as in the backtest's full history."""
    last = close[-1]
    if p.hard_stop_pct > 0 and last < entry_price * (1 - p.hard_stop_pct / 100):
        return True
    if p.stop_atr > 0 and len(close) > ATR_PERIOD:
        window = max(1, min(bars_since_entry, len(close), PEAK_LOOKBACK))
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


def gate_entry(signal: Signal, p: Params, *, risk_on: bool, bars_since_sell: float | None) -> Signal:
    """Entry filters shared by every strategy: the BTC regime and the wait after a sell. Exits never pass here."""
    if signal is not Signal.BUY:
        return signal
    if not risk_on:
        return Signal.HOLD
    if p.reentry_candles > 0 and bars_since_sell is not None and bars_since_sell <= p.reentry_candles:
        return Signal.HOLD  # the candle the sell landed in, then reentry_candles - 1 more, sit out
    return signal


def decide_core(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    p: Params,
    *,
    holding: bool,
    entry_price: float = 0.0,
    bars_held: int = 0,
    bars_since_sell: float | None = None,
    risk_on: bool = True,
    exiting: bool = False,
) -> Signal:
    """The whole SMA rule for one asset on closed candles. The live loop (via `decide`) and the backtest call this.

    `exiting` means an earlier sell was capped by MAX_TRADE_USDC and left a position: keep selling until flat."""
    signal = sma_signal(close, p)
    if holding:
        if exiting or stop_hit(high, low, close, entry_price, bars_held, p):
            return Signal.SELL
        return signal
    if signal is Signal.BUY and p.entry == "trend" and not trend_reset_since(close, p, bars_since_sell):
        return Signal.HOLD  # stopped out of a trend that never broke: buying straight back in just churns
    return gate_entry(signal, p, risk_on=risk_on, bars_since_sell=bars_since_sell)


def trend_reset_since(close: np.ndarray, p: Params, bars_since_sell: float | None) -> bool:
    """True if fast was at or below slow on some candle from the one that decided the last sell up to now.

    A sell on the trend signal passes at once. A stop-out in a trend that is still up does not, until the
    trend breaks; without this, trend mode re-buys on the next candle and pays the round trip again and again.
    A sell more than `slow` candles ago passes too: the slow average is all post-exit history by then. The fixed
    look-back keeps the answer the same for the live loop's short window and the backtest's full history."""
    if bars_since_sell is None or bars_since_sell > p.slow:
        return True
    window = close[-(int(math.ceil(bars_since_sell)) + p.slow):]  # the decision candle onward, plus slow-1 history
    if len(window) < p.slow:
        return True
    csum = np.cumsum(np.insert(window.astype(float), 0, 0.0))
    slow = (csum[p.slow:] - csum[: -p.slow]) / p.slow
    fast = ((csum[p.fast:] - csum[: -p.fast]) / p.fast)[-len(slow):]
    return bool((fast <= slow).any())


def bars_since(candles: pd.DataFrame, when: pd.Timestamp | None, timeframe: str) -> float | None:
    """Candles from the one a fill landed in up to the latest closed one: the backtest's count, since fills land
    mid-candle live but at the open in the backtest."""
    if when is None:
        return None
    freq = pd.Timedelta(candle_freq(timeframe))
    return (candles.index[-1] + freq - pd.Timestamp(when).floor(freq)) / freq


def decide(
    candles: pd.DataFrame,
    p: Params,
    *,
    holding: bool,
    entry_price: float = 0.0,
    entry_time: pd.Timestamp | None = None,
    last_sell_time: pd.Timestamp | None = None,
    risk_on: bool = True,
    exiting: bool = False,
    timeframe: str = TIMEFRAME,
) -> Signal:
    """Signal for one asset from its closed candles, given what we hold of it."""
    if entry_time is not None:
        bars = int((candles.index >= pd.Timestamp(entry_time).floor(candle_freq(timeframe))).sum())
    else:
        bars = len(candles)
    return decide_core(
        candles["close"].to_numpy(dtype=float), candles["high"].to_numpy(dtype=float),
        candles["low"].to_numpy(dtype=float), p, holding=holding, entry_price=entry_price, bars_held=bars,
        bars_since_sell=bars_since(candles, last_sell_time, timeframe), risk_on=risk_on, exiting=exiting,
    )


def buy_budget(
    equity: float, high: np.ndarray, low: np.ndarray, close: np.ndarray, p: Params, max_trade_usdc: float
) -> float:
    """USDC for one new position: hitting the stop should cost `risk_pct` of equity, never more than the cap.

    The stop distance is the nearer of the trailing ATR stop and the hard stop; with neither set it is SIZE_ATR ATRs.
    Volatile coins get smaller positions, so one memecoin swing risks no more than a BTC one."""
    if p.risk_pct <= 0:
        return max_trade_usdc
    vol = atr(high, low, close)
    if not vol > 0 or close[-1] <= 0:
        return max_trade_usdc  # no usable volatility estimate yet
    stops = [d for d in (p.stop_atr * vol / close[-1], p.hard_stop_pct / 100) if d > 0]
    distance = min(stops) if stops else SIZE_ATR * vol / close[-1]
    return min(max_trade_usdc, equity * p.risk_pct / 100 / distance)


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
