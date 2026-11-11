from dataclasses import replace

import pandas as pd
import pytest

import bot.config
from bot import db
from dashboard import view
from tests.test_dashboard import cfg, client, db_path, seed, state  # noqa: F401  (fixtures)


@pytest.fixture
def multi_cfg(cfg):
    return replace(cfg, assets=("SOL", "BTC"), risky_assets=("BONK",))


@pytest.fixture
def multi_client(multi_cfg):
    from dashboard.server import create_app
    from fastapi.testclient import TestClient

    return TestClient(create_app(multi_cfg))


def seed_multi(path):
    conn = db.connect(path)
    db.init_db(conn)
    db.set_state(conn, db.DRY_RUN, "1")
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.0001, 50_000.0, None), True, asset="BTC")
    db.insert_balance(
        conn, db.Balances(15.0, 0.05, 0.0, {"BTC": 0.0001}), 100.0, True, prices={"BTC": 60_000.0, "SOL": 100.0}
    )
    conn.close()


def test_risky_switch_writes_only_the_risky_flag(client, cfg, db_path):
    seed(db_path)
    conn = db.connect(db_path)
    before = dict(conn.execute("SELECT key, value FROM state").fetchall())
    assert state(client)["risky"] is False
    assert client.post("/api/risky", json={"risky": True}).status_code == 200
    assert db.is_risky(conn)
    assert state(client)["risky"] is True
    after = dict(conn.execute("SELECT key, value FROM state").fetchall())
    assert {k for k in set(before) | set(after) if before.get(k) != after.get(k)} == {db.RISKY}
    assert client.post("/api/risky", json={"risky": False}).status_code == 200
    assert not db.is_risky(conn)


def test_risky_and_stop_work_independently(client, cfg, db_path):
    seed(db_path)
    client.post("/api/risky", json={"risky": True})
    client.post("/api/pause", json={"paused": True})
    conn = db.connect(db_path)
    assert db.is_risky(conn) and db.is_paused(conn)


def test_public_view_refuses_the_risky_switch(public_client_fixture=None):
    pass  # covered in test_dashboard.test_public_mode_refuses_every_write_and_reveals_nothing


def test_multi_asset_positions_and_equity(multi_client, multi_cfg, db_path):
    seed_multi(db_path)
    s = state(multi_client)
    assert s["equity"] == pytest.approx(21.00)  # 15 + 0.0001 BTC at 60,000
    assert s["invested"] == pytest.approx(6.00) and s["mode_label"] == "Top coins"
    assert s["multi"] is True
    btc = next(row for row in s["positions"] if row["asset"] == "wBTC")
    assert btc["qty"] == pytest.approx(0.0001) and btc["avg_entry"] == pytest.approx(50_000.0)
    assert btc["unrealised"] == pytest.approx(1.0)
    log = multi_client.get("/api/trades").json()["rows"]
    assert [row["asset"] for row in log] == ["wBTC"]


def test_mode_label_follows_the_switch(multi_client, multi_cfg, db_path):
    seed_multi(db_path)
    conn = db.connect(db_path)
    db.set_risky(conn, True)
    assert state(multi_client)["mode_label"] == "RISKY"


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
    conn.close()


def test_the_sniper_is_folded_into_the_top_equity_bar(client, cfg, db_path):
    seed(db_path)  # wallet: $15 USDC + 0.002 SOL at $2,600 = $20.20
    seed_snipes(db_path)  # sniper: $8.00 free + one open position worth $2.40 = $10.40
    s = state(client)
    assert s["sniper_active"] is True
    assert s["total_equity"] == pytest.approx(30.60)
    assert s["usdc_total"] == pytest.approx(23.00)  # 15 + 8
    assert s["invested_total"] == pytest.approx(7.60)  # 5.20 + 2.40
    assert "Total $30.60 (wallet $20.20 + sniper $10.40)" in s["floor_caption"]
    assert "wallet halt floor $15.00 | headroom $5.20" in s["floor_caption"]  # the sniper never props up the floor
    totals = s["sniper"]["totals"]
    assert totals == {
        "free": 8.0, "open_value": 2.4, "realised": 0.0, "unrealised": pytest.approx(0.4),
        "open": 1, "closed": 2, "wins": 1,
    }
    assert "never touches the wallet" in s["sniper"]["caption"]


def test_an_unused_sniper_is_not_part_of_the_totals(multi_client, multi_cfg, db_path):
    seed_multi(db_path)  # RISKY off and no snipes yet
    s = state(multi_client)
    assert s["equity"] == pytest.approx(21.00) and s["total_equity"] == pytest.approx(21.00)
    assert s["sniper_active"] is False


def test_the_sniper_hint_shows_when_it_is_off_and_empty(client, cfg, db_path):
    seed(db_path)
    sn = state(client)["sniper"]
    assert sn["active"] is False and "Turn on RISKY" in sn["caption"]


def test_view_sniper_helpers_account_for_every_dollar(cfg, db_path):
    seed(db_path)
    seed_snipes(db_path)
    snipes = db.read_snipes(db.connect(db_path))
    totals = view.sniper_totals(snipes, 10.0)
    assert totals == {"free": 8.0, "open_value": 2.4, "realised": 0.0, "unrealised": pytest.approx(0.4),
                      "open": 1, "closed": 2, "wins": 1}
    assert view.sniper_totals(snipes.iloc[0:0], 10.0)["free"] == 10.0  # no snipes: the full bankroll

    times = pd.Series(pd.to_datetime(["2025-12-31T00:00:00Z", "2026-01-01T00:02:00Z", "2026-01-01T00:10:00Z"]))
    curve = view.sniper_realised_over_time(snipes, times, unrealised_now=0.4)
    # before any close: 0; after the +$1 win only: +1; after the -$1 loss too: 0, plus today's unrealised +0.4
    assert list(curve) == pytest.approx([0.0, 1.0, 0.4])


def test_series_carries_the_sniper_overlay(client, cfg, db_path):
    seed(db_path)
    seed_snipes(db_path)
    body = client.get("/api/series?range=7d").json()
    assert body["sniper_active"] is True
    # the overlay adds the sniper's P&L, not its bankroll: wallet $20.20 + $0.40 unrealised
    assert body["points"][-1]["total"] == pytest.approx(20.60)


def test_ticker_lists_marks_with_change(client, cfg, db_path):
    seed(db_path)
    rows = state(client)["ticker"]
    assert rows and rows[0]["symbol"] == "SOL" and rows[0]["price"] == 2600.0
    assert rows[0]["change"] is None  # a single snapshot has no baseline yet


# --- AI ---------------------------------------------------------------------------------------------

@pytest.fixture
def ai_cfg(multi_cfg):
    return replace(multi_cfg, strategy="ml")


@pytest.fixture
def ai_client(ai_cfg):
    from dashboard.server import create_app
    from fastapi.testclient import TestClient

    return TestClient(create_app(ai_cfg))


def test_ai_scores_and_model_info_are_shown(ai_client, ai_cfg, db_path):
    seed_multi(db_path)
    conn = db.connect(db_path)
    info = {"timeframe": "1h", "trained_at": "2026-10-11T22:40:00+00:00", "oos_auc": 0.533, "rows": 98000,
            "enter": 0.6, "exit": 0.45, "horizon": 6}
    db.set_ml(conn, {"SOL": 0.2, "BTC": 0.71}, info)
    s = state(ai_client)
    btc = next(row for row in s["positions"] if row["asset"] == "wBTC")
    assert btc["p_win"] == pytest.approx(0.71)
    assert "AI model (1h)" in s["ml_caption"] and "0.533" in s["ml_caption"] and "0.6" in s["ml_caption"]
    assert s["ai"] is True


def test_the_ai_says_when_it_has_no_model_yet(ai_client, ai_cfg, db_path):
    seed_multi(db_path)
    assert "none for this timeframe yet" in state(ai_client)["ml_caption"]


def test_no_ai_column_when_the_ai_is_not_the_strategy(multi_client, multi_cfg, db_path):
    seed_multi(db_path)
    s = state(multi_client)
    assert s["ai"] is False and s["ml_caption"] is None
    assert all("p_win" not in row for row in s["positions"])
