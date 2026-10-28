"""What the simulated sniper learns from its own closed trades: it tightens its entry filters, and only when they prove it.

For each filter (liquidity, holder count) it looks at the closed snipes that clear a stricter floor. A floor is adopted
only if at least MIN_SAMPLES snipes clear it and their mean return minus Z standard errors is above zero. Learning only
tightens: a floor never drops below its .env value. Exits, position sizes and the simulation itself are not learned.

    python -m bot.sniper_learn      # relearn from the database and print the result
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics

from bot import db
from bot.config import Settings

LIQUIDITY_GRID = (3_000, 5_000, 10_000, 20_000, 50_000)  # USD
HOLDERS_GRID = (5, 10, 20, 50, 100)
MIN_SAMPLES = 30  # closed snipes that must clear a floor before it may be adopted
Z = 2.0  # standard errors the mean return must clear


def lower_bound(returns: list[float]) -> float:
    n = len(returns)
    if n < 2:
        return -math.inf
    return statistics.fmean(returns) - Z * statistics.stdev(returns) / math.sqrt(n)


def best_floor(samples: list[tuple[float | None, float]], default: float, grid: tuple[float, ...]) -> float:
    """The floor from `grid` with the best proven return, or `default` if no stricter floor is proven.

    `samples` are (feature value, return) per closed snipe; a value of None never clears a floor. Only floors above
    `default` are considered. A floor is proven when enough snipes clear it and their lower bound is above zero."""
    best, best_score = default, 0.0
    for floor in grid:
        if floor <= default:
            continue
        picked = [ret for value, ret in samples if value is not None and value >= floor]
        if len(picked) < MIN_SAMPLES:
            continue
        score = lower_bound(picked)
        if score > best_score:
            best, best_score = floor, score
    return best


def learn(conn: sqlite3.Connection, cfg: Settings) -> dict:
    """Relearn both floors from every closed snipe and store the result for the sniper to read."""
    rows = conn.execute(
        "SELECT liquidity_usd, holders, entry_usdc, exit_usdc FROM snipes WHERE status = 'closed'"
    ).fetchall()
    returns = [row["exit_usdc"] / row["entry_usdc"] - 1 for row in rows]
    state = {
        "liquidity_usd": best_floor(
            [(row["liquidity_usd"], ret) for row, ret in zip(rows, returns)], cfg.sniper_min_liquidity_usd, LIQUIDITY_GRID
        ),
        "holders": best_floor(
            [(row["holders"], ret) for row, ret in zip(rows, returns)], cfg.sniper_min_holders, HOLDERS_GRID
        ),
        "closed": len(rows),
        "wins": sum(ret > 0 for ret in returns),
        "mean_return": statistics.fmean(returns) if returns else 0.0,
    }
    db.set_state(conn, db.SNIPER_LEARN, json.dumps(state))
    return state


def floors(conn: sqlite3.Connection, cfg: Settings) -> tuple[float, int]:
    """The liquidity and holder floors the sniper uses now: the learned ones, never below the .env values."""
    raw = db.get_state(conn, db.SNIPER_LEARN)
    if not raw:
        return cfg.sniper_min_liquidity_usd, cfg.sniper_min_holders
    state = json.loads(raw)
    return max(state["liquidity_usd"], cfg.sniper_min_liquidity_usd), max(state["holders"], cfg.sniper_min_holders)


def main() -> None:
    from bot.config import settings

    conn = db.connect(settings.db_path)
    db.init_db(conn)
    state = learn(conn, settings)
    closed = state["closed"]
    win_rate = f"{state['wins'] / closed:.0%}" if closed else "n/a"
    print(
        f"closed snipes {closed}, won {win_rate}, mean return {state['mean_return']:+.1%} | "
        f"liquidity floor ${state['liquidity_usd']:,.0f} | holder floor {state['holders']}"
    )


if __name__ == "__main__":
    main()
