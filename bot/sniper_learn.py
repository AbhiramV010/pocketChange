"""What the simulated sniper learns from its own closed trades: it tightens its entry filters, and only on proof.

Each filter (liquidity, holder count, top-holder share, dev mints, token age, price impact) has a grid of stricter
settings. A setting is adopted only if at least MIN_SAMPLES closed snipes pass it and their mean return, minus Z
standard errors, beats what the sniper already earns without it (or zero, if that is negative). Z grows with the
number of settings tested (Bonferroni), so trying many thresholds does not let a lucky one through. Learning only
tightens: no filter ever gets looser than its .env value. Exits, sizes and the simulation itself are not learned.

    python -m bot.sniper_learn      # relearn from the database and print the result
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass

from bot import db

MIN_SAMPLES = 30  # closed snipes that must pass a setting before it may be adopted
ALPHA = 0.05  # chance, across every setting tested, of adopting one that only got lucky


@dataclass(frozen=True)
class Rule:
    name: str  # also the snipes column holding what the sniper saw at entry
    kind: str  # "floor": value >= limit passes; "ceiling": value <= limit passes
    grid: tuple[float, ...]

    def passes(self, value: float | None, limit: float) -> bool:
        if value is None:
            return False  # not recorded (an older snipe): never counts as passing a stricter setting
        return value >= limit if self.kind == "floor" else value <= limit

    def stricter(self, limit: float, than: float) -> bool:
        return limit > than if self.kind == "floor" else limit < than

    def tightest(self, a: float, b: float) -> float:
        return max(a, b) if self.kind == "floor" else min(a, b)


RULES = (
    Rule("liquidity_usd", "floor", (5_000, 10_000, 20_000, 50_000)),
    Rule("holders", "floor", (10, 20, 50, 100)),
    Rule("top_holders_pct", "ceiling", (30, 20, 10)),
    Rule("dev_mints", "ceiling", (5, 2, 0)),
    Rule("age_min", "ceiling", (10, 5, 2)),
    Rule("impact", "ceiling", (0.02, 0.01, 0.005)),
)


def z_for(tests: int) -> float:
    """One-sided z for a family of `tests` settings at ALPHA overall (Bonferroni)."""
    return statistics.NormalDist().inv_cdf(1 - ALPHA / max(tests, 1))


def lower_bound(returns: list[float], z: float) -> float:
    n = len(returns)
    if n < 2:
        return -math.inf
    return statistics.fmean(returns) - z * statistics.stdev(returns) / math.sqrt(n)


def best_limit(rule: Rule, samples: list[tuple[float | None, float]], base: float, bar: float, z: float) -> float:
    """The stricter setting with the best proven return, or `base` if none beats `bar`.

    `samples` are (what the sniper saw for this filter, return) per closed snipe."""
    best, best_score = base, bar
    for limit in rule.grid:
        if not rule.stricter(limit, base):
            continue
        picked = [ret for value, ret in samples if rule.passes(value, limit)]
        if len(picked) < MIN_SAMPLES:
            continue
        score = lower_bound(picked, z)
        if score > best_score:
            best, best_score = limit, score
    return best


def learn(conn: sqlite3.Connection, base: dict[str, float]) -> dict:
    """Relearn every filter from all closed snipes and store the result for the sniper and the dashboard.

    `base` holds each filter's .env (or built-in) limit, by rule name."""
    columns = ", ".join(rule.name for rule in RULES)
    rows = conn.execute(f"SELECT {columns}, entry_usdc, exit_usdc FROM snipes WHERE status = 'closed'").fetchall()
    returns = [row["exit_usdc"] / row["entry_usdc"] - 1 for row in rows]
    z = z_for(sum(1 for rule in RULES for limit in rule.grid if rule.stricter(limit, base[rule.name])))
    bar = max(0.0, lower_bound(returns, z)) if len(returns) >= MIN_SAMPLES else 0.0
    limits = {
        rule.name: best_limit(rule, [(row[rule.name], ret) for row, ret in zip(rows, returns)], base[rule.name], bar, z)
        for rule in RULES
    }
    state = {
        "limits": {name: value for name, value in limits.items() if value != base[name]},  # only what it learned
        "closed": len(rows),
        "wins": sum(ret > 0 for ret in returns),
        "mean_return": statistics.fmean(returns) if returns else 0.0,
        "z": round(z, 2),
    }
    db.set_state(conn, db.SNIPER_LEARN, json.dumps(state))
    return state


def limits(conn: sqlite3.Connection, base: dict[str, float]) -> dict[str, float]:
    """The limits the sniper uses now: what it learned, never looser than `base`."""
    raw = db.get_state(conn, db.SNIPER_LEARN)
    try:
        learned = json.loads(raw).get("limits", {}) if raw else {}
    except ValueError:
        learned = {}
    return {rule.name: rule.tightest(float(learned.get(rule.name, base[rule.name])), base[rule.name]) for rule in RULES}


def main() -> None:
    from bot.config import settings
    from bot.sniper import base_limits

    conn = db.connect(settings.db_path)
    db.init_db(conn)
    state = learn(conn, base_limits(settings))
    closed = state["closed"]
    win_rate = f"{state['wins'] / closed:.0%}" if closed else "n/a"
    print(f"closed snipes {closed}, won {win_rate}, mean return {state['mean_return']:+.1%}, bar z={state['z']}")
    print("learned: " + (", ".join(f"{k} {v:g}" for k, v in state["limits"].items()) or "nothing proven yet"))


if __name__ == "__main__":
    main()
