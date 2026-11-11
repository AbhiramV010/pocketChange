"""Model-building tools beyond the fixed-horizon classifier in bot/ml.py: better labels, honest sample weights, purged
cross-validation, calibration, a promotion gate for retrained models, and sizing from a calibrated edge."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from bot.config import ROUND_TRIP_COST


# --- labels ------------------------------------------------------------------------------------------------

def vol_scaled_target(candles: pd.DataFrame, horizon: int, window: int = 48) -> pd.Series:
    """Forward return after costs divided by the volatility expected over the horizon: a magnitude-aware target
    (the binary 'beat the target' label throws the size of the move away)."""
    c = candles["close"]
    fwd = (1 + (c.shift(-horizon) / c - 1)) * (1 - ROUND_TRIP_COST) - 1
    sd = c.pct_change().rolling(window).std() * np.sqrt(horizon)
    return (fwd / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def triple_barrier(candles: pd.DataFrame, horizon: int, up_atr: float = 2.0, dn_atr: float = 1.5, period: int = 14) -> pd.Series:
    """1 if the price reaches +up_atr ATRs before -dn_atr ATRs within `horizon` bars (the barriers a stop-managed
    trade really faces); if neither is touched, 1 when the horizon return beats round-trip costs. NaN at the tail."""
    c, h, lo = (candles[k].to_numpy(float) for k in ("close", "high", "low"))
    n = len(c)
    tr = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])])
    atr = np.full(n, np.nan)
    atr[period:] = pd.Series(tr).rolling(period).mean().to_numpy()[period - 1:]
    up, dn = c + up_atr * atr, c - dn_atr * atr
    out = np.full(n, np.nan)
    live = np.ones(n, bool)
    live[n - horizon:] = False
    live &= ~np.isnan(atr)
    state = np.full(n, -1)  # -1 undecided, 0 lower first, 1 upper first
    for k in range(1, horizon + 1):
        hi_k, lo_k = np.full(n, np.nan), np.full(n, np.nan)
        hi_k[: n - k], lo_k[: n - k] = h[k:], lo[k:]
        hit_dn = (lo_k <= dn) & (state == -1)
        hit_up = (hi_k >= up) & (state == -1)
        state[hit_dn] = 0  # a candle touching both counts against us: the stop is assumed to fire first
        state[hit_up & ~hit_dn] = 1
    fin = np.full(n, np.nan)
    fin[: n - horizon] = c[horizon:] / c[: n - horizon] - 1
    timeout = (1 + fin) * (1 - ROUND_TRIP_COST) - 1 > 0
    out[live] = np.where(state[live] == -1, timeout[live], state[live] == 1).astype(float)
    return pd.Series(out, index=candles.index)


# --- sample weights and validation -------------------------------------------------------------------------

def uniqueness_weights(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Average uniqueness of each label span [start, end]: a label overlapping many others carries less new
    information. Weights are normalised to mean 1. Rows with a missing end get weight 1."""
    s = pd.to_datetime(start).to_numpy("datetime64[ns]").astype("int64")
    e = pd.to_datetime(end).to_numpy("datetime64[ns]").astype("int64")
    ok = ~np.isnan(e.astype(float)) & (e > np.iinfo("int64").min)
    times = np.unique(s[ok])
    starts = np.sort(s[ok])
    ends = np.sort(e[ok])
    conc = np.searchsorted(starts, times, side="right") - np.searchsorted(ends, times, side="left")  # spans are inclusive
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    prefix = np.concatenate([[0.0], np.cumsum(inv)])
    w = np.ones(len(s))
    lo_i = np.searchsorted(times, s, side="left")
    hi_i = np.searchsorted(times, e, side="right")
    span = np.maximum(hi_i - lo_i, 1)
    w[ok] = ((prefix[hi_i] - prefix[lo_i]) / span)[ok]
    return w / w.mean()


def purged_kfold(index: pd.Index, t_end: pd.Series, k: int = 5, embargo: pd.Timedelta = pd.Timedelta(0)):
    """Yield (train_positions, test_positions): contiguous time blocks as tests; training rows whose label span
    reaches into the test block (or the embargo after it) are purged, so no training label sees test prices."""
    t = pd.DatetimeIndex(index)
    order = np.argsort(t.to_numpy())
    blocks = np.array_split(order, k)
    ends = pd.DatetimeIndex(t_end).to_numpy()
    starts = t.to_numpy()
    for block in blocks:
        lo, hi = starts[block].min(), starts[block].max()
        test = np.zeros(len(t), bool)
        test[block] = True
        overlap = (ends >= lo) & (starts <= hi + embargo.to_timedelta64())
        yield np.where(~test & ~overlap)[0], np.where(test)[0]


def information_coefficient(score, realised) -> float:
    """Rank correlation between predicted scores and realised forward returns."""
    df = pd.DataFrame({"s": np.asarray(score, float), "r": np.asarray(realised, float)}).dropna()
    return float(df["s"].rank().corr(df["r"].rank())) if len(df) > 20 else float("nan")


def rank_normalize(f: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectional percentile rank of each feature at every timestamp (index may repeat across coins)."""
    out = f.copy()
    out[cols] = f.groupby(level=0)[cols].rank(pct=True)
    return out


# --- calibration, promotion, sizing ------------------------------------------------------------------------

class CalibratedModel:
    """HistGradientBoosting whose probabilities are recalibrated by isotonic regression on the latest slice of
    the training data (time-ordered, never the slice the trees saw)."""

    def __init__(self, base: HistGradientBoostingClassifier, calib_frac: float = 0.2):
        self.base, self.calib_frac, self.iso = base, calib_frac, IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        cut = int(len(X) * (1 - self.calib_frac))
        sw = None if sample_weight is None else np.asarray(sample_weight)
        self.base.fit(X.iloc[:cut], y[:cut], sample_weight=None if sw is None else sw[:cut])
        raw = self.base.predict_proba(X.iloc[cut:])[:, 1]
        self.iso.fit(raw, np.asarray(y[cut:], float), sample_weight=None if sw is None else sw[cut:])
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.clip(self.iso.predict(self.base.predict_proba(X)[:, 1]), 0.0, 1.0)
        return np.column_stack([1 - p, p])


def should_promote(candidate_ic: float, incumbent_ic: float | None, margin: float = 0.0) -> bool:
    """A retrained model replaces the live one only if its out-of-sample information coefficient is better."""
    if incumbent_ic is None or np.isnan(incumbent_ic):
        return True  # nothing to protect: take the new one
    return bool(not np.isnan(candidate_ic) and candidate_ic > incumbent_ic + margin)


def holdout_gate(rows: pd.DataFrame, incumbent, fit, features: list[str], days: int = 30, margin: float = 0.0):
    """Score challenger and incumbent on the newest `days` of rows. The challenger is trained only on rows whose
    labels settled before that window; the incumbent is taken as it is (so it may have seen some of it, which only
    makes the test harder for the challenger). Returns (promote, challenger_ic, incumbent_ic)."""
    cut = rows.index.max() - pd.Timedelta(days=days)
    train, test = rows[rows["t_end"] < cut], rows[rows.index >= cut].dropna(subset=features + ["fwd"])
    if incumbent is None or len(test) < 50 or len(train.dropna(subset=features + ["y"])) < 500:
        return True, float("nan"), float("nan")
    cand = fit(train)
    ic_c = information_coefficient(cand.predict_proba(test[features].astype(float))[:, 1], test["fwd"])
    ic_i = information_coefficient(incumbent.predict_proba(test[features].astype(float))[:, 1], test["fwd"])
    return should_promote(ic_c, ic_i, margin), ic_c, ic_i


def kelly_fraction(p: float, win: float, loss: float, fraction: float = 0.25, cap: float = 1.0) -> float:
    """Fractional Kelly stake (share of equity) for a binary bet: win `win`, lose `loss` (both positive fractions),
    from a calibrated P(win). Zero when the edge is not positive."""
    if not (0 <= p <= 1) or win <= 0 or loss <= 0:
        return 0.0
    full = p / loss - (1 - p) / win  # maximises long-run growth; stake is a share of equity
    return float(np.clip(full * fraction, 0.0, cap)) if full > 0 else 0.0
