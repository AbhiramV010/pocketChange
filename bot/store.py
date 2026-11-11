"""Versioned research data store: one Parquet file per coin and timeframe, with integrity checks.

    python -m bot.store update --days 1500 --timeframe 4h      # download / extend every configured coin
    python -m bot.store check  --timeframe 4h                  # gaps, duplicates, bad candles, outliers
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bot.config import DATA_DIR, candle_freq

STORE_DIR = DATA_DIR / "store"
COLUMNS = ["open", "high", "low", "close", "volume"]


def path_for(symbol: str, timeframe: str):
    return STORE_DIR / f"{symbol}_{timeframe}.parquet"


def validate(df: pd.DataFrame, timeframe: str, outlier_move: float = 0.5) -> dict:
    """Counts of the problems that quietly corrupt a backtest. All zeros means clean."""
    if df.empty:
        return {"rows": 0, "gaps": 0, "duplicates": 0, "unsorted": 0, "bad_ohlc": 0, "outliers": 0}
    step = pd.Timedelta(candle_freq(timeframe))
    idx = df.index
    expected = int((idx.max() - idx.min()) / step) + 1
    bad = ((df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))
           | (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["volume"] < 0))
    moves = df["close"].pct_change().abs()
    return {
        "rows": len(df), "gaps": max(0, expected - len(idx.unique())), "duplicates": int(idx.duplicated().sum()),
        "unsorted": int((np.diff(idx.asi8) < 0).sum()), "bad_ohlc": int(bad.sum()), "outliers": int((moves > outlier_move).sum()),
    }


def save_frame(symbol: str, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
    """Merge `df` into the stored history (new rows win on a clash) and write it back."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    old = load_frame(symbol, timeframe)
    merged = pd.concat([old, df[COLUMNS]]) if not old.empty else df[COLUMNS]
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    tmp = path_for(symbol, timeframe).with_suffix(".tmp")
    merged.to_parquet(tmp)
    tmp.replace(path_for(symbol, timeframe))  # atomic, like the CSV cache
    return merged


def load_frame(symbol: str, timeframe: str) -> pd.DataFrame:
    p = path_for(symbol, timeframe)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    return pd.read_parquet(p)


def first_listed(symbol: str, timeframe: str) -> pd.Timestamp | None:
    df = load_frame(symbol, timeframe)
    return None if df.empty else df.index[0]


def main(argv: list[str] | None = None) -> None:
    from bot import data
    from bot.assets import ASSETS
    from bot.config import settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["update", "check"])
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--assets", default=",".join(dict.fromkeys(list(settings.assets) + list(settings.risky_assets) + ["BTC"])))
    args = parser.parse_args(argv)
    for sym in (s.strip().upper() for s in args.assets.split(",")):
        if args.command == "update":
            df = save_frame(sym, args.timeframe, data.fetch_history(ASSETS[sym].market, args.days, args.timeframe, settings.candle_exchange))
        else:
            df = load_frame(sym, args.timeframe)
        print(sym, validate(df, args.timeframe), f"from {df.index[0]:%Y-%m-%d}" if len(df) else "empty")


if __name__ == "__main__":
    main()
