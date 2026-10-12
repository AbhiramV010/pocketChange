"""The AI strategy: a gradient-boosted classifier trained on candle history.

For every coin and closed candle it estimates the probability that price will be more than `target_pct` percent
higher `horizon` candles later (enough to clear trading costs). The bot buys when that probability is high, sells
when it fades, and keeps every other safety rule (stops, caps, equity floor, STOP button) untouched.

    python -m bot.ml train --days 400 --timeframe 1h     # fit on all history and save the model
    python -m bot.ml eval  --days 400 --timeframe 1h     # walk-forward test on months the model never saw
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from bot.config import DATA_DIR, Settings, candle_freq
from bot.strategy import Signal

log = logging.getLogger("bot.ml")

MODEL_DIR = DATA_DIR / "models"
MAX_TRAIN_CANDLES = 25_000  # per coin; keeps training on short timeframes to a few minutes
MIN_CANDLES = 130  # history a feature row needs (the longest window is BTC's 100-candle average, plus slack)
BENCH = "BTC"  # market context for every coin


@dataclass(frozen=True)
class MLParams:
    horizon: int = 6  # candles ahead the model looks
    target_pct: float = 0.4  # "worth it": the move must beat this many percent
    enter: float = 0.6  # buy when P(up) is at least this
    exit: float = 0.45  # sell when it falls to this or below

    @classmethod
    def from_settings(cls, cfg: Settings) -> "MLParams":
        return cls(cfg.ml_horizon, cfg.ml_target_pct, cfg.ml_enter, cfg.ml_exit)


# --- features ----------------------------------------------------------------------------------------------

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def features(candles: pd.DataFrame, bench: pd.DataFrame | None) -> pd.DataFrame:
    """One row per candle, computed from that candle and earlier ones only (no lookahead)."""
    c, h, lo, v = candles["close"], candles["high"], candles["low"], candles["volume"]
    ret1 = c.pct_change()
    f = pd.DataFrame(index=candles.index)
    for n in (1, 3, 6, 12, 24, 48):
        f[f"ret{n}"] = c.pct_change(n)
    for n in (12, 48):
        f[f"vol{n}"] = ret1.rolling(n).std()
    true_range = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    f["atr14"] = true_range.rolling(14).mean() / c
    for n in (9, 21, 50):
        f[f"sma{n}"] = c / c.rolling(n).mean() - 1
    f["sma9_21"] = c.rolling(9).mean() / c.rolling(21).mean() - 1
    f["rsi14"] = _rsi(c)
    f["volume_rel"] = v / v.rolling(24).mean() - 1
    span = (h.rolling(24).max() - lo.rolling(24).min()).replace(0, np.nan)
    f["range_pos"] = (c - lo.rolling(24).min()) / span
    hour = candles.index.hour + candles.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    dow = candles.index.dayofweek
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    if bench is not None and not bench.empty:
        bc = bench["close"].reindex(candles.index, method="ffill")
        f["btc_ret1"], f["btc_ret6"], f["btc_ret24"] = bc.pct_change(), bc.pct_change(6), bc.pct_change(24)
        f["btc_sma100"] = bc / bc.rolling(100).mean() - 1
        f["btc_vol24"] = bc.pct_change().rolling(24).std()
    else:
        for col in ("btc_ret1", "btc_ret6", "btc_ret24", "btc_sma100", "btc_vol24"):
            f[col] = np.nan
    return f.replace([np.inf, -np.inf], np.nan)


FEATURE_NAMES = list(features(
    pd.DataFrame(
        {k: np.ones(MIN_CANDLES + 5) for k in ("open", "high", "low", "close", "volume")},
        index=pd.date_range("2026-01-01", periods=MIN_CANDLES + 5, freq="1h", tz="UTC"),
    ),
    None,
).columns)


def labels(candles: pd.DataFrame, p: MLParams) -> pd.Series:
    """1 if the close `horizon` candles ahead is more than target_pct above this close. NaN where unknown."""
    fwd = candles["close"].shift(-p.horizon) / candles["close"] - 1
    return (fwd > p.target_pct / 100).where(fwd.notna()).astype(float)


def dataset(frames: dict[str, pd.DataFrame], p: MLParams) -> pd.DataFrame:
    """All coins stacked: feature columns, `y`, `asset`, and the forward return `fwd`."""
    bench = frames.get(BENCH)
    parts = []
    for asset, candles in frames.items():
        f = features(candles, bench)
        f["y"] = labels(candles, p)
        f["fwd"] = candles["close"].shift(-p.horizon) / candles["close"] - 1
        f["asset"] = asset
        parts.append(f)
    return pd.concat(parts)


# --- model -------------------------------------------------------------------------------------------------

def new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=4, min_samples_leaf=300, l2_regularization=2.0, random_state=0
    )


def fit(rows: pd.DataFrame) -> HistGradientBoostingClassifier:
    rows = rows.dropna(subset=FEATURE_NAMES + ["y"])
    model = new_model()
    model.fit(rows[FEATURE_NAMES], rows["y"].astype(int))
    return model


@dataclass
class Bundle:
    model: HistGradientBoostingClassifier
    params: MLParams
    timeframe: str
    trained_at: str
    rows: int
    oos_auc: float | None = None

    def proba(self, feature_row: pd.Series) -> float | None:
        values = feature_row[FEATURE_NAMES]
        if values.isna().any():
            return None
        return float(self.model.predict_proba(values.to_frame().T.astype(float))[0, 1])

    def proba_series(self, f: pd.DataFrame) -> pd.Series:
        """P(win) for each row; NaN where the features are incomplete. Works on stacked frames (repeated index)."""
        ok = f[FEATURE_NAMES].notna().all(axis=1).to_numpy()
        out = np.full(len(f), np.nan)
        if ok.any():
            out[ok] = self.model.predict_proba(f.loc[ok, FEATURE_NAMES].astype(float))[:, 1]
        return pd.Series(out, index=f.index)


def model_path(timeframe: str) -> Path:
    return MODEL_DIR / f"ml_{timeframe}.joblib"


def save(bundle: Bundle) -> Path:
    path = model_path(bundle.timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump({**bundle.__dict__, "params": asdict(bundle.params), "features": FEATURE_NAMES}, tmp)
    tmp.replace(path)  # atomic: a crash never leaves a half-written model
    return path


def load(timeframe: str) -> Bundle | None:
    path = model_path(timeframe)
    if not path.exists():
        return None
    try:
        blob = joblib.load(path)
        if blob.get("features") != FEATURE_NAMES:
            log.warning("model %s was trained on different features; retrain it", path.name)
            return None
        return Bundle(blob["model"], MLParams(**blob["params"]), blob["timeframe"], blob["trained_at"], blob["rows"], blob.get("oos_auc"))
    except Exception as exc:
        log.warning("could not load model %s: %s", path.name, type(exc).__name__)
        return None


def train(frames: dict[str, pd.DataFrame], p: MLParams, timeframe: str, oos_auc: float | None = None) -> Bundle:
    rows = dataset(frames, p)
    model = fit(rows)
    n = int(rows.dropna(subset=FEATURE_NAMES + ["y"]).shape[0])
    return Bundle(model, p, timeframe, datetime.now(timezone.utc).isoformat(timespec="seconds"), n, oos_auc)


# --- trading rule (shared by the live loop and the backtest) --------------------------------------------------

def ml_signal(prob: float | None, holding: bool, bars_held: int, p: MLParams) -> Signal:
    if prob is None:
        return Signal.HOLD
    if not holding:
        return Signal.BUY if prob >= p.enter else Signal.HOLD
    if prob <= p.exit or (bars_held >= p.horizon and prob < p.enter):
        return Signal.SELL  # the edge faded, or the horizon passed without the model still liking it
    return Signal.HOLD


# --- walk-forward test -------------------------------------------------------------------------------------

def walk_forward(
    frames: dict[str, pd.DataFrame], p: MLParams, timeframe: str, *, min_train_days: int = 180, test_days: int = 30
) -> tuple[dict[str, pd.Series], dict]:
    """Out-of-sample probabilities: each block is scored by a model trained only on the time before it."""
    rows = dataset(frames, p)
    step = pd.Timedelta(candle_freq(timeframe))
    start = rows.index.min() + pd.Timedelta(days=min_train_days)
    end = rows.index.max()
    proba = {a: pd.Series(np.nan, index=frames[a].index) for a in frames}
    auc_parts, cursor = [], start
    while cursor < end:
        block_end = cursor + pd.Timedelta(days=test_days)
        train_rows = rows[rows.index < cursor - step * p.horizon]  # embargo: labels must be fully in the past
        test_rows = rows[(rows.index >= cursor) & (rows.index < block_end)]
        if len(train_rows.dropna(subset=FEATURE_NAMES + ["y"])) > 5000 and len(test_rows):
            bundle = Bundle(fit(train_rows), p, timeframe, "", 0)
            scores = bundle.proba_series(test_rows)
            for asset in frames:
                mask = (test_rows["asset"] == asset).to_numpy()
                proba[asset].loc[test_rows.index[mask]] = scores.to_numpy()[mask]
            scored = test_rows.assign(p=scores.to_numpy()).dropna(subset=["p", "y"])
            if scored["y"].nunique() == 2:
                auc_parts.append((len(scored), roc_auc_score(scored["y"], scored["p"]), scored["y"].mean()))
        cursor = block_end
    total = sum(n for n, _, _ in auc_parts) or 1
    summary = {
        "folds": len(auc_parts),
        "auc": sum(n * a for n, a, _ in auc_parts) / total if auc_parts else float("nan"),
        "base_rate": sum(n * r for n, _, r in auc_parts) / total if auc_parts else float("nan"),
        "start": start,
    }
    return proba, summary


# --- live model management ---------------------------------------------------------------------------------

class ModelManager:
    """Holds the current model per timeframe and retrains it in a background thread when it gets stale."""

    def __init__(self, cfg: Settings, history: "callable | None" = None):
        self._cfg = cfg
        self._history = history  # (timeframe) -> {asset: candles}; injectable for tests
        self._bundles: dict[str, Bundle | None] = {}
        self._lock = threading.Lock()
        self._training: set[str] = set()

    def get(self, timeframe: str) -> Bundle | None:
        with self._lock:
            if timeframe not in self._bundles:
                self._bundles[timeframe] = load(timeframe)
            return self._bundles[timeframe]

    def maybe_retrain(self, timeframe: str, now: datetime | None = None) -> None:
        bundle = self.get(timeframe)
        now = now or datetime.now(timezone.utc)
        if bundle is not None:
            trained = datetime.fromisoformat(bundle.trained_at)
            same = (bundle.params.horizon, bundle.params.target_pct) == (self._cfg.ml_horizon, self._cfg.ml_target_pct)
            if same and now - trained < timedelta(hours=self._cfg.ml_retrain_hours):
                return
        with self._lock:
            if timeframe in self._training:
                return
            self._training.add(timeframe)
        threading.Thread(target=self._retrain, args=(timeframe,), name=f"ml-train-{timeframe}", daemon=True).start()

    def _retrain(self, timeframe: str) -> None:
        try:
            started = time.monotonic()
            frames = self._history(timeframe) if self._history else fetch_frames(self._cfg, timeframe)
            bundle = train(frames, MLParams.from_settings(self._cfg), timeframe, self.get(timeframe).oos_auc if self.get(timeframe) else None)
            save(bundle)
            with self._lock:
                self._bundles[timeframe] = bundle
            log.info("ML model for %s retrained on %d rows in %.0fs", timeframe, bundle.rows, time.monotonic() - started)
        except Exception as exc:
            log.warning("ML retrain for %s failed: %s: %s", timeframe, type(exc).__name__, str(exc)[:150])
        finally:
            with self._lock:
                self._training.discard(timeframe)


def fetch_frames(cfg: Settings, timeframe: str, days: int | None = None) -> dict[str, pd.DataFrame]:
    from bot import data
    from bot.assets import ASSETS

    symbols = list(dict.fromkeys(list(cfg.assets) + list(cfg.risky_assets) + [BENCH]))
    days = days or cfg.ml_train_days
    per_day = pd.Timedelta(days=1) / pd.Timedelta(candle_freq(timeframe))
    days = max(2, min(days, int(MAX_TRAIN_CANDLES / per_day)))  # 1m candles: weeks of history, not a year
    return {s: data.fetch_history(ASSETS[s].market, days, timeframe, cfg.candle_exchange) for s in symbols}


# --- command line ------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from bot.config import settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--days", type=int, default=settings.ml_train_days)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frames = fetch_frames(settings, args.timeframe, args.days)
    params = MLParams.from_settings(settings)
    if args.command == "eval":
        _, summary = walk_forward(frames, params, args.timeframe)
        print(f"walk-forward: {summary['folds']} folds, out-of-sample AUC {summary['auc']:.3f} "
              f"(0.5 = coin flip), base rate {summary['base_rate']:.1%}")
        return
    _, summary = walk_forward(frames, params, args.timeframe)
    bundle = train(frames, params, args.timeframe, summary["auc"])
    path = save(bundle)
    print(f"trained on {bundle.rows} rows ({len(frames)} coins), out-of-sample AUC {summary['auc']:.3f}; saved {path}")


if __name__ == "__main__":
    main()
