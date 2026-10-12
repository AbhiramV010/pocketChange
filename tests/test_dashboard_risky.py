from dataclasses import replace

import pytest

import bot.config
from bot import db
from tests.test_dashboard import app, public_app, run_app, seed  # noqa: F401  (fixtures)


@pytest.fixture
def multi_app(app, monkeypatch):
    monkeypatch.setattr(
        bot.config, "settings",
        replace(bot.config.settings, assets=("SOL", "BTC"), risky_assets=("BONK",)),
    )
    return app


def seed_multi(path):
    conn = db.connect(path)
    db.init_db(conn)
    db.set_state(conn, db.DRY_RUN, "1")
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.0001, 50_000.0, None), True, asset="BTC")
    db.insert_balance(
        conn, db.Balances(15.0, 0.05, 0.0, {"BTC": 0.0001}), 100.0, True, prices={"BTC": 60_000.0, "SOL": 100.0}
    )
    conn.close()


def test_risky_switch_writes_only_the_risky_flag(app):
    seed(app)
    conn = db.connect(app)
    before = dict(conn.execute("SELECT key, value FROM state").fetchall())
    at = run_app()
    assert at.button(key="risky_off").label == "RISKY" and not at.button(key="risky_off").disabled
    at.button(key="risky_off").click().run()
    assert db.is_risky(conn)
    assert at.button(key="risky_on").label == "RISKY ON"
    after = dict(conn.execute("SELECT key, value FROM state").fetchall())
    assert {k for k in set(before) | set(after) if before.get(k) != after.get(k)} == {db.RISKY}
    at.button(key="risky_on").click().run()
    assert not db.is_risky(conn)


def test_risky_switch_sits_beside_stop_and_both_work_independently(app):
    seed(app)
    at = run_app()
    at.button(key="risky_off").click().run()
    at.button(key="emergency_stop").click().run()
    conn = db.connect(app)
    assert db.is_risky(conn) and db.is_paused(conn)


def test_public_view_has_no_risky_switch(public_app):
    seed(public_app)
    at = run_app()
    assert len(at.button) == 0


def test_multi_asset_layout_shows_positions_and_equity(multi_app):
    seed_multi(multi_app)
    at = run_app()
    labels = {m.label: m.value for m in at.metric}
    assert labels["Equity (USDC + positions)"] == "$21.00"  # 15 + 0.0001 BTC at 60,000
    assert labels["In positions"] == "$6.00" and labels["Mode"] == "Top coins"
    table = next(d.value for d in at.dataframe if "asset" in d.value.columns and "unrealised" in d.value.columns)
    row = table[table["asset"] == "wBTC"].iloc[0]
    assert row["qty"] == pytest.approx(0.0001) and row["avg entry"] == pytest.approx(50_000.0)
    assert row["unrealised"] == pytest.approx(1.0)
    log = next(d.value for d in at.dataframe if "tx" in d.value.columns)
    assert list(log["asset"]) == ["wBTC"]


def test_mode_label_follows_the_switch(multi_app):
    seed_multi(multi_app)
    conn = db.connect(multi_app)
    db.set_risky(conn, True)
    at = run_app()
    assert {m.label: m.value for m in at.metric}["Mode"] == "RISKY"


def seed_snipes(path):
    conn = db.connect(path)
    db.init_db(conn)
    db.set_risky(conn, True)
    for mint, symbol in (("M1", "WIN"), ("M2", "LOSE"), ("M3", "LIVE")):
        db.open_snipe(conn, mint=mint, symbol=symbol, decimals=6, entry_usdc=2.0, qty_raw=10, liquidity_usd=5000.0)
    snipes = db.read_snipes(conn)
    db.close_snipe(conn, int(snipes.loc[0, "id"]), 3.0, "take profit", "2026-01-01T00:00:00+00:00")
    db.close_snipe(conn, int(snipes.loc[1, "id"]), 1.0, "stop loss", "2026-01-01T00:05:00+00:00")
    db.mark_snipe(conn, int(snipes.loc[2, "id"]), 2.4, 0, "2026-01-01T00:06:00+00:00")


def test_the_sniper_is_folded_into_the_top_equity_bar(app):
    seed(app)  # wallet: $15 USDC + 0.002 SOL at $2,600 = $20.20
    seed_snipes(app)  # sniper: $8.00 free + one open position worth $2.40 = $10.40
    at = run_app()
    metrics = {m.label: m for m in at.metric}
    assert metrics["Total equity (wallet + sniper)"].value == "$30.60"
    assert "wallet $20.20 + sniper $10.40" in metrics["Total equity (wallet + sniper)"].delta
    assert metrics["USDC (wallet + sniper)"].value == "23.00"  # 15 + 8
    assert metrics["In positions"].value == "$7.60" and "sniper $2.40" in metrics["In positions"].delta  # 5.20 + 2.40
    assert "Sniper bankroll free" not in metrics and "Win rate" not in metrics  # no separate sniper bar any more
    captions = " ".join(c.value for c in at.caption)
    assert "1 open, 2 closed, win rate 50%" in captions and "realised +$0.00" in captions and "unrealised +$0.40" in captions
    assert "never touches the wallet" in captions


def test_the_curve_caption_keeps_the_halt_floor_about_the_wallet_only(app):
    seed(app)
    seed_snipes(app)
    at = run_app()
    caption = next(c.value for c in at.caption if "halt floor" in c.value)
    assert "Total $30.60 (wallet $20.20 + sniper $10.40)" in caption
    assert "wallet halt floor $15.00 | headroom $5.20" in caption  # the sniper's money never props up the floor


def test_an_unused_sniper_is_not_part_of_the_totals(multi_app):
    seed_multi(multi_app)  # RISKY off and no snipes yet
    at = run_app()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Equity (USDC + positions)"] == "$21.00" and "Total equity (wallet + sniper)" not in metrics


def load_dashboard_module(monkeypatch):
    import importlib.util

    import streamlit as st
    from tests.test_dashboard import APP

    spec = importlib.util.spec_from_file_location("dashboard_under_test", APP)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(st, "fragment", lambda **kw: (lambda fn: fn))
    spec.loader.exec_module(module)
    return module


def test_sniper_helpers_account_for_every_dollar(app, monkeypatch):
    import pandas as pd

    module = load_dashboard_module(monkeypatch)
    seed_snipes(app)
    snipes = db.read_snipes(db.connect(app))
    totals = module.sniper_totals(snipes, 10.0)
    assert totals == {"free": 8.0, "open_value": 2.4, "realised": 0.0, "unrealised": pytest.approx(0.4),
                      "open": 1, "closed": 2, "wins": 1}
    assert module.sniper_totals(snipes.iloc[0:0], 10.0)["free"] == 10.0  # no snipes: the full bankroll

    times = pd.Series(pd.to_datetime(["2025-12-31T00:00:00Z", "2026-01-01T00:02:00Z", "2026-01-01T00:10:00Z"]))
    curve = module.sniper_realised_over_time(snipes, times, unrealised_now=0.4)
    # before any close: 0; after the +$1 win only: +1; after the -$1 loss too: 0, plus today's unrealised +0.4
    assert list(curve) == pytest.approx([0.0, 1.0, 0.4])


def test_sniper_hint_when_off_and_empty(app):
    seed(app)
    at = run_app()
    assert any("Turn on RISKY" in c.value for c in at.caption)


@pytest.fixture
def ai_app(multi_app, monkeypatch):
    monkeypatch.setattr(bot.config, "settings", replace(bot.config.settings, strategy="ml"))
    return multi_app


def test_ai_scores_and_model_info_are_shown(ai_app):
    seed_multi(ai_app)
    conn = db.connect(ai_app)
    info = {"timeframe": "1h", "trained_at": "2026-10-11T22:40:00+00:00", "oos_auc": 0.533, "rows": 98000,
            "enter": 0.6, "exit": 0.45, "horizon": 6}
    db.set_ml(conn, {"SOL": 0.2, "BTC": 0.71}, info)
    at = run_app()
    table = next(d.value for d in at.dataframe if "AI p(win)" in d.value.columns)
    assert table.set_index("asset")["AI p(win)"]["wBTC"] == pytest.approx(0.71)
    caption = " ".join(c.value for c in at.caption)
    assert "AI model (1h)" in caption and "0.533" in caption and "0.6" in caption


def test_dashboard_says_when_the_ai_has_no_model_yet(ai_app):
    seed_multi(ai_app)
    at = run_app()
    assert any("none for this timeframe yet" in c.value for c in at.caption)


def test_no_ai_column_when_the_ai_is_not_the_strategy(multi_app):
    seed_multi(multi_app)
    at = run_app()
    assert not any("AI p(win)" in d.value.columns for d in at.dataframe)
