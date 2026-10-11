"""Candle download (ccxt public endpoints, no key) and CSV cache.

Candle frames are indexed by UTC candle *open* time with columns
open, high, low, close, volume.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

from bot.config import SYMBOL, TIMEFRAME, settings

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]
CANDLE = pd.Timedelta(TIMEFRAME)
PAGE_LIMIT = 1000
MAX_PAGES = 200
# Tried in order after the configured exchange. All list SOL/USDC.
FALLBACK_EXCHANGES = ("binance", "bybit", "okx", "kraken")

_exchanges: dict[str, ccxt.Exchange] = {}


class DataError(RuntimeError):
    """No exchange could provide candles."""


def _exchange(exchange_id: str) -> ccxt.Exchange:
    if exchange_id not in _exchanges:
        _exchanges[exchange_id] = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 15000})
    return _exchanges[exchange_id]


def _exchange_order(preferred: str | None) -> list[str]:
    first = (preferred or settings.candle_exchange).lower()
    return [first] + [e for e in FALLBACK_EXCHANGES if e != first]


def to_frame(rows: list[list[float]]) -> pd.DataFrame:
    """ccxt OHLCV rows -> candle frame (sorted, de-duplicated, no NaNs)."""
    df = pd.DataFrame(rows, columns=["ts", *COLUMNS])
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ms", utc=True), name="timestamp")
    df = df.astype(float).dropna()
    return df[~df.index.duplicated(keep="last")].sort_index()


def download(since: datetime, exchange_id: str, until: datetime | None = None) -> pd.DataFrame:
    """Page through one exchange's history from `since`."""
    ex = _exchange(exchange_id)
    until_ms = int((until or datetime.now(timezone.utc)).timestamp() * 1000)
    cursor = int(since.timestamp() * 1000)
    step = int(CANDLE.total_seconds() * 1000)
    rows: list[list[float]] = []
    for _ in range(MAX_PAGES):
        page = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, since=cursor, limit=PAGE_LIMIT)
        if not page:
            break
        rows.extend(page)
        last = int(page[-1][0])
        if last + step <= cursor or last >= until_ms:
            break  # no progress, or caught up
        cursor = last + step
    if not rows:
        raise DataError(f"{exchange_id} returned no candles")
    return to_frame(rows)


def _download_with_fallback(since: datetime, preferred: str | None) -> pd.DataFrame:
    errors = []
    for exchange_id in _exchange_order(preferred):
        try:
            df = download(since, exchange_id)
            log.debug("candles: %d rows from %s", len(df), exchange_id)
            return df
        except Exception as exc:  # network, rate limit, geo-block, unknown market
            errors.append(f"{exchange_id}: {type(exc).__name__}: {str(exc)[:80]}")
            log.warning("candle source %s failed: %s", exchange_id, errors[-1])
    raise DataError("all candle sources failed: " + "; ".join(errors))


def fetch_recent(limit: int = 120, exchange_id: str | None = None) -> pd.DataFrame:
    """Latest `limit` candles including the still-forming one."""
    since = datetime.now(timezone.utc) - CANDLE * (limit + 1)
    return _download_with_fallback(since, exchange_id).tail(limit)


def closed_only(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop the candle that is still forming."""
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    return df[df.index + CANDLE <= now_ts]


def load_cache(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or settings.candles_path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df = pd.read_csv(path)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("timestamp"), utc=True), name="timestamp")
    return df[COLUMNS].astype(float).sort_index()


def save_cache(df: pd.DataFrame, path: Path | None = None) -> None:
    path = Path(path or settings.candles_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[COLUMNS].to_csv(tmp, index_label="timestamp", date_format="%Y-%m-%dT%H:%M:%S%z")
    tmp.replace(path)  # atomic: a crash never leaves a half-written cache


def gap_count(df: pd.DataFrame) -> int:
    """Number of missing candles inside the cached range."""
    if len(df) < 2:
        return 0
    expected = int((df.index[-1] - df.index[0]) / CANDLE) + 1
    return expected - len(df)


def update_cache(days: int = 365, path: Path | None = None, exchange_id: str | None = None) -> pd.DataFrame:
    """Download new closed candles into the cache (full pull if empty)."""
    cached = load_cache(path)
    if cached.empty:
        since = datetime.now(timezone.utc) - timedelta(days=days)
    else:
        since = cached.index[-1].to_pydatetime()
    fresh = closed_only(_download_with_fallback(since, exchange_id))
    merged = pd.concat([cached, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    save_cache(merged, path)
    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Download {SYMBOL} {TIMEFRAME} candles to the CSV cache.")
    parser.add_argument("--days", type=int, default=365, help="history to pull when the cache is empty")
    parser.add_argument("--exchange", default=None, help="ccxt exchange id (default: CANDLE_EXCHANGE)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.time()
    df = update_cache(args.days, exchange_id=args.exchange)
    print(
        f"{len(df)} candles, {df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M} UTC, "
        f"{gap_count(df)} missing, cached at {settings.candles_path} ({time.time() - started:.0f}s)"
    )


if __name__ == "__main__":
    main()
