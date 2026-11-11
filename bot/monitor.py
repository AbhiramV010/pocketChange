"""Operational monitoring: is the bot alive, is its data fresh, is its model decaying, is the book in drawdown?

`check` returns alerts (level, message); the bot logs them every cycle and the dashboard can show them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from bot import db


@dataclass(frozen=True)
class Alert:
    level: str  # "info" | "warn" | "critical"
    message: str


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of a feature between training and live data. Under 0.1 stable, over 0.25 drifted."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) < bins or len(a) < bins:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(train: pd.DataFrame, live: pd.DataFrame, columns: list[str], threshold: float = 0.25) -> dict[str, float]:
    """Features whose PSI exceeds `threshold`."""
    out = {c: psi(train[c].to_numpy(), live[c].to_numpy()) for c in columns if c in train and c in live}
    return {c: v for c, v in out.items() if v > threshold}


def drawdown(equity: pd.Series) -> float:
    return float(1 - equity.iloc[-1] / equity.cummax().iloc[-1]) if len(equity) else 0.0


def model_decay(resolved: list[tuple[float, bool]], enter: float, recent: int = 100, floor: float = 0.4) -> float | None:
    """Hit rate of the model's recent confident calls (None when there are too few to say). Below `floor` is decay."""
    calls = [win for p, win in resolved[-recent * 5:] if p >= enter][-recent:]
    return float(np.mean(calls)) if len(calls) >= 20 else None


def check(conn, cfg, now: datetime, *, max_drawdown: float = 0.15, ml_enter: float | None = None) -> list[Alert]:
    alerts: list[Alert] = []
    beat = db.get_state(conn, db.HEARTBEAT)
    if beat is None:
        alerts.append(Alert("warn", "no heartbeat yet"))
    else:
        age = (pd.Timestamp(now) - pd.Timestamp(beat)).total_seconds()
        if age > cfg.loop_interval_seconds * 5:
            alerts.append(Alert("critical", f"bot heartbeat is {age / 60:.0f} min old; is it running?"))
    if db.is_halted(conn):
        alerts.append(Alert("critical", f"HALTED: {db.halt_reason(conn)}"))
    bal = db.read_balances(conn)
    if len(bal):
        eq = pd.Series([db.row_equity(r) for _, r in bal.iterrows()], dtype=float).dropna()
        if len(eq) and (dd := drawdown(eq)) > max_drawdown:
            alerts.append(Alert("warn", f"drawdown {dd:.0%} from the peak"))
    info = db.get_ml(conn)[1] if cfg.strategy == "ml" else {}
    if info and info.get("trained_at"):
        age_h = (pd.Timestamp(now) - pd.Timestamp(info["trained_at"])).total_seconds() / 3600
        if age_h > cfg.ml_retrain_hours * 3:
            alerts.append(Alert("warn", f"AI model is {age_h:.0f} h old (retrain every {cfg.ml_retrain_hours:g} h)"))
    if cfg.strategy == "ml" and info:
        rows = db.resolved_predictions(conn, info.get("timeframe", "4h"))
        rate = model_decay([(r["p"], bool(r["win"])) for r in rows], ml_enter or cfg.ml_enter) if rows and "win" in rows[0].keys() else None
        if rate is not None and rate < 0.4:
            alerts.append(Alert("warn", f"AI hit rate on recent confident calls is {rate:.0%}"))
    return alerts
