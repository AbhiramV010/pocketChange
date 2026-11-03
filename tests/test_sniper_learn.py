from bot import db, sniper_learn
from bot.sniper_learn import RULES, Rule, best_limit, z_for

BASE = {"liquidity_usd": 3000.0, "holders": 5, "top_holders_pct": 40.0, "dev_mints": 10, "age_min": 15.0, "impact": 0.03}
LIQUIDITY = next(r for r in RULES if r.name == "liquidity_usd")
TOP = next(r for r in RULES if r.name == "top_holders_pct")
Z = 2.8


def wave(n, mean):
    """Returns that average `mean` but vary, so the standard error is not zero."""
    return [mean + (0.02 if i % 2 else -0.02) for i in range(n)]


def test_no_stricter_setting_without_enough_snipes():
    samples = [(20_000.0, 0.5)] * (sniper_learn.MIN_SAMPLES - 1)
    assert best_limit(LIQUIDITY, samples, 3000.0, 0.0, Z) == 3000.0


def test_a_stricter_floor_is_adopted_when_the_snipes_passing_it_prove_it():
    bad = [(4_000.0, -0.2)] * 40  # under 5,000: loses
    good = [(12_000.0, r) for r in wave(40, 0.2)]
    assert best_limit(LIQUIDITY, bad + good, 3000.0, 0.0, Z) == 5_000.0


def test_a_ceiling_tightens_downwards():
    concentrated = [(35.0, -0.3)] * 40  # a few wallets hold 35%: dumped on
    spread = [(8.0, r) for r in wave(40, 0.15)]
    assert best_limit(TOP, concentrated + spread, 40.0, 0.0, Z) == 30.0


def test_no_proof_means_no_change():
    noisy = [(12_000.0, r) for r in wave(40, 0.0)]  # mean zero: the lower bound is negative
    assert best_limit(LIQUIDITY, noisy, 3000.0, 0.0, Z) == 3000.0


def test_a_stricter_setting_must_beat_what_the_sniper_already_earns():
    good = [(12_000.0, r) for r in wave(40, 0.2)]
    assert best_limit(LIQUIDITY, good, 3000.0, bar=0.5, z=Z) == 3000.0  # positive, but worse than not filtering


def test_learning_never_loosens_and_unrecorded_values_never_pass():
    assert best_limit(LIQUIDITY, [(3_500.0, r) for r in wave(40, 0.2)], 50_000.0, 0.0, Z) == 50_000.0
    assert not Rule("holders", "floor", (10,)).passes(None, 10)


def test_the_bar_rises_with_the_number_of_settings_tested():
    assert z_for(1) < z_for(5) < z_for(20) and 2.7 < z_for(20) < 2.9


def seed(conn, n_good=40, n_bad=40):
    for i in range(n_good + n_bad):
        good = i < n_good
        db.open_snipe(
            conn, mint=f"M{i}", symbol="S", decimals=6, entry_usdc=2.0, qty_raw=1,
            liquidity_usd=20_000.0 if good else 4_000.0, holders=30 if good else 6, top_holders_pct=15.0,
            dev_mints=1, age_min=3.0, impact=0.01,
        )
        ret = wave(2, 0.2)[i % 2] if good else -0.25
        conn.execute(
            "UPDATE snipes SET status = 'closed', exit_usdc = ?, reason = 'test', closed_at = ? WHERE mint = ?",
            (2.0 * (1 + ret), "2026-01-01T00:00:00+00:00", f"M{i}"),
        )
    conn.commit()


def test_learn_stores_what_it_learned_and_limits_apply_it_never_looser_than_env(conn):
    seed(conn)
    state = sniper_learn.learn(conn, BASE)
    assert state["closed"] == 80 and state["wins"] == 40
    assert state["limits"] == {"liquidity_usd": 5_000, "holders": 10}  # the two filters that separate the groups
    lim = sniper_learn.limits(conn, BASE)
    assert lim["liquidity_usd"] == 5_000 and lim["holders"] == 10 and lim["impact"] == 0.03
    assert sniper_learn.limits(conn, {**BASE, "liquidity_usd": 50_000.0})["liquidity_usd"] == 50_000.0


def test_no_learned_state_or_a_garbled_one_means_the_env_limits(conn):
    assert sniper_learn.limits(conn, BASE) == BASE
    db.set_state(conn, db.SNIPER_LEARN, "{not json")
    assert sniper_learn.limits(conn, BASE) == BASE
    db.set_state(conn, db.SNIPER_LEARN, '{"liquidity_usd": 5000, "holders": 10}')  # the older format
    assert sniper_learn.limits(conn, BASE) == BASE


def test_the_entry_features_are_stored_with_the_snipe(conn):
    db.open_snipe(
        conn, mint="X", symbol="X", decimals=6, entry_usdc=2.0, qty_raw=1, liquidity_usd=9_000.0,
        holders=42, top_holders_pct=12.5, dev_mints=2, age_min=3.5, impact=0.01, round_trip=0.97, fee_usd=0.015,
    )
    row = conn.execute(
        "SELECT holders, top_holders_pct, dev_mints, age_min, impact, round_trip, fee_usd FROM snipes"
    ).fetchone()
    assert tuple(row) == (42, 12.5, 2, 3.5, 0.01, 0.97, 0.015)
