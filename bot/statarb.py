"""Pairs / statistical-arbitrage maths (research; trading it needs a venue that can short, see changes.md Phase 3).

A pair is tradable when log(y) - beta*log(x) is stationary and mean-reverts fast enough to pay for the costs.
"""
from __future__ import annotations

import math

import numpy as np

ADF_5PCT = -3.34  # Engle-Granger critical value (two series, constant) at the 5% level


def hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS of log y on log x: (beta, intercept)."""
    ly, lx = np.log(y), np.log(x)
    beta, alpha = np.polyfit(lx, ly, 1)
    return float(beta), float(alpha)


def spread(y: np.ndarray, x: np.ndarray, beta: float, alpha: float = 0.0) -> np.ndarray:
    return np.log(y) - beta * np.log(x) - alpha


def _ar1(s: np.ndarray) -> tuple[float, float]:
    """(phi, t-stat of phi - 1) from s[t] = c + phi * s[t-1] + e."""
    a, b = s[:-1], s[1:]
    X = np.column_stack([np.ones(len(a)), a])
    coef, *_ = np.linalg.lstsq(X, b, rcond=None)
    resid = b - X @ coef
    sigma2 = resid @ resid / max(len(b) - 2, 1)
    var_phi = sigma2 * np.linalg.inv(X.T @ X)[1, 1]
    return float(coef[1]), float((coef[1] - 1) / math.sqrt(var_phi)) if var_phi > 0 else float("nan")


def half_life(s: np.ndarray) -> float:
    """Bars for a deviation to halve (inf if the spread does not mean-revert)."""
    phi, _ = _ar1(s)
    return math.inf if not 0 < phi < 1 else -math.log(2) / math.log(phi)


def cointegrated(y: np.ndarray, x: np.ndarray) -> tuple[bool, float]:
    """Engle-Granger: stationary residual at the 5% level? Returns (decision, t-stat)."""
    beta, alpha = hedge_ratio(y, x)
    _, t = _ar1(spread(y, x, beta, alpha))
    return bool(t < ADF_5PCT), t


def pair_signal(
    y: np.ndarray, x: np.ndarray, *, lookback: int = 200, entry_z: float = 2.0, exit_z: float = 0.5,
    max_half_life: float = 60.0, position: int = 0,
) -> int:
    """Target position in the spread on the latest bar: +1 long y / short x, -1 the reverse, 0 flat.
    Only trades pairs that pass the cointegration test and revert faster than `max_half_life` bars."""
    if len(y) < lookback or len(x) < lookback:
        return 0
    y, x = y[-lookback:], x[-lookback:]
    form_y, form_x = y[:-1], x[:-1]  # fit and test on history only: the bar being judged must not shape its own test
    ok, _ = cointegrated(form_y, form_x)
    beta, alpha = hedge_ratio(form_y, form_x)
    base = spread(form_y, form_x, beta, alpha)
    if not ok or half_life(base) > max_half_life or not base.std() > 0:
        return 0
    z = (spread(y[-1:], x[-1:], beta, alpha)[0] - base.mean()) / base.std()
    if position == 0:
        return -1 if z > entry_z else 1 if z < -entry_z else 0
    return 0 if abs(z) < exit_z else position
