from dataclasses import replace

from bot import db, sniper_learn
from bot.config import load_settings

CFG = replace(load_settings({}), sniper_min_liquidity_usd=3000.0, sniper_min_holders=5)


def wave(n, mean):
    """Returns that average `mean` but vary, so the standard error is not zero."""
    return [mean + (0.02 if i % 2 else -0.02) for i in range(n)]


def test_no_stricter_floor_without_enough_snipes():
    samples = [(20_000.0, 0.5)] * (sniper_learn.MIN_SAMPLES - 1)
    assert sniper_learn.best_floor(samples, 3000.0, sniper_learn.LIQUIDITY_GRID) == 3000.0


def test_a_stricter_floor_is_adopted_when_the_snipes_above_it_prove_it():
    bad = [(4_000.0, -0.2)] * 40  # below 5,000: loses
    good = [(12_000.0, r) for r in wave(40, 0.2)]
    assert sniper_learn.best_floor(bad + good, 3000.0, sniper_learn.LIQUIDITY_GRID) == 5_000.0


def test_a_profitable_band_with_no_proof_is_not_adopted():
    noisy = [(12_000.0, r) for r in wave(40, 0.0)]  # the mean is zero: the lower bound is negative
    assert sniper_learn.best_floor(noisy, 3000.0, sniper_learn.LIQUIDITY_GRID) == 3000.0


def test_learning_never_loosens_a_floor_below_the_env_value():
    good_small = [(3_500.0, r) for r in wave(40, 0.2)]
    assert sniper_learn.best_floor(good_small, 5_000.0, sniper_learn.LIQUIDITY_GRID) == 5_000.0


def test_learn_stores_the_floors_and_the_sniper_reads_them_with_the_env_minimum(conn):
    for i in range(80):
        good = i % 2 == 0
        db.open_snipe(
            conn, mint=f"M{i}", symbol="S", decimals=6, entry_usdc=2.0, qty_raw=1,
            liquidity_usd=20_000.0 if good else 4_000.0, holders=30 if good else 6,
        )
        ret = wave(1, 0.2)[0] if good else -0.2
        conn.execute(
            "UPDATE snipes SET status = 'closed', exit_usdc = ?, reason = 'test', closed_at = ? WHERE mint = ?",
            (2.0 * (1 + ret), "2026-01-01T00:00:00+00:00", f"M{i}"),
        )
    conn.commit()
    state = sniper_learn.learn(conn, CFG)
    assert state["closed"] == 80 and state["wins"] == 40
    assert state["liquidity_usd"] == 5_000.0 and state["holders"] == 10
    assert sniper_learn.floors(conn, CFG) == (5_000.0, 10)
    assert sniper_learn.floors(conn, replace(CFG, sniper_min_liquidity_usd=50_000.0))[0] == 50_000.0


def test_no_learned_state_means_the_env_floors(conn):
    assert sniper_learn.floors(conn, CFG) == (3000.0, 5)


def test_the_entry_features_are_stored_with_the_snipe(conn):
    db.open_snipe(
        conn, mint="X", symbol="X", decimals=6, entry_usdc=2.0, qty_raw=1, liquidity_usd=9_000.0,
        holders=42, top_holders_pct=12.5, dev_mints=2, age_min=3.5, impact=0.01, round_trip=0.97,
    )
    row = conn.execute("SELECT holders, top_holders_pct, dev_mints, age_min, impact, round_trip FROM snipes").fetchone()
    assert tuple(row) == (42, 12.5, 2, 3.5, 0.01, 0.97)
