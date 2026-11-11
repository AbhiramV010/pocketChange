import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import bot.config
from bot import db
from dashboard.server import create_app

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "dashboard" / "server.py"
VIEW = ROOT / "dashboard" / "view.py"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "dash.db"


@pytest.fixture
def cfg(db_path):
    return replace(bot.config.settings, db_path=db_path, candles_path=db_path.parent / "candles.csv")


@pytest.fixture
def client(cfg):
    return TestClient(create_app(cfg))


def seed(path, dry_run=True, halted=False, paused=False):
    conn = db.connect(path)
    db.init_db(conn)
    db.set_state(conn, db.DRY_RUN, "1" if dry_run else "0")
    db.set_state(conn, db.HEARTBEAT, db.utc_now_iso())
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.002, 2500.0, None if dry_run else "0xabc"), dry_run)
    db.insert_balance(conn, db.Balances(15.0, 0.002, 0.002), 2600.0, dry_run)
    if halted:
        db.set_halted(conn, "equity $12.00 is below MIN_EQUITY_USD $15.00")
    if paused:
        db.set_paused(conn, True)
    conn.close()


def state(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    return r.json()


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_index_and_static_are_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "pocketchange" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_no_database_reports_a_hint(client):
    body = state(client)
    assert body["ready"] is False and "No database yet" in body["message"]


def test_state_shows_mode_equity_and_trades(client, cfg, db_path):
    seed(db_path)
    s = state(client)
    assert s["ready"] is True and s["dry_run"] is True
    assert s["equity"] == pytest.approx(20.20) and s["usdc"] == 15.0 and s["wsol"] == 0.002
    assert s["headroom"] == pytest.approx(5.20)
    assert "halt floor $15.00" in s["floor_caption"] and "headroom $5.20" in s["floor_caption"]
    assert s["halted"] is False and s["paused"] is False
    trades = client.get("/api/trades").json()["rows"]
    assert len(trades) == 1 and trades[0]["side"] == "buy" and trades[0]["asset"] == "wSOL"


def test_series_and_floor(client, cfg, db_path):
    seed(db_path)
    body = client.get("/api/series?range=7d").json()
    assert body["floor"] == 15.0
    assert body["points"] and body["points"][-1]["equity"] == pytest.approx(20.20)


def test_candles_endpoint_handles_a_missing_cache(client):
    body = client.get("/api/candles").json()
    assert body["candles"] == []


def test_halted_shows_banner_and_disables_stop(client, cfg, db_path):
    seed(db_path, halted=True)
    s = state(client)
    assert s["halted"] is True
    assert "BOT HALTED" in s["halt_banner"] and "MIN_EQUITY_USD" in s["halt_banner"]
    assert "reset_halt" in s["halt_banner"]  # the operator is told the dashboard cannot clear it
    conn = db.connect(db_path)
    assert db.is_halted(conn)


def test_stop_and_resume_write_only_the_pause_flag(client, cfg, db_path):
    seed(db_path)
    conn = db.connect(db_path)
    before = dict(conn.execute("SELECT key, value FROM state").fetchall())
    trades_before = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    assert client.post("/api/pause", json={"paused": True}).status_code == 200
    assert db.is_paused(conn)
    assert client.post("/api/pause", json={"paused": False}).status_code == 200
    assert not db.is_paused(conn)

    after = dict(conn.execute("SELECT key, value FROM state").fetchall())
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed <= {db.PAUSED}
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == trades_before


def test_state_mirrors_a_flag_set_elsewhere(client, cfg, db_path):
    seed(db_path, paused=True)
    assert state(client)["paused"] is True


def test_heartbeat_is_reported(client, cfg, db_path):
    seed(db_path)
    s = state(client)
    assert s["heartbeat_age_s"] is not None and s["heartbeat_age_s"] < 60


def test_private_key_is_only_referenced_in_chain_module():
    offenders = []
    for path in list((ROOT / "bot").glob("*.py")) + list((ROOT / "dashboard").glob("*.py")):
        if path.name != "chain.py" and "PRIVATE_KEY" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []


def test_dashboard_imports_nothing_that_can_trade_and_writes_only_the_three_flags():
    allowed_writes = {"set_paused", "set_risky", "set_timeframe"}
    banned_writes = {"set_halted", "clear_halt", "insert_trade", "insert_balance", "set_state", "init_db"}
    banned_imports = {"bot.chain", "bot.run", "solders", "requests", "bot.reset_halt"}
    for path in (SERVER, VIEW):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        assert not banned_imports & imported, path.name
        called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not banned_writes & called, path.name
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert allowed_writes <= called


def test_pocketchange_has_no_hardcoded_secrets():
    pattern = re.compile(r"0x[0-9a-fA-F]{64}\b")
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        assert not pattern.search(path.read_text(encoding="utf-8")), path


# --- public mode ----------------------------------------------------------------------------------

@pytest.fixture
def public_cfg(db_path):
    return replace(
        bot.config.settings, db_path=db_path, candles_path=db_path.parent / "candles.csv", dashboard_public=True
    )


@pytest.fixture
def public_client(public_cfg):
    return TestClient(create_app(public_cfg))


def test_public_mode_refuses_every_write_and_reveals_nothing(public_client, public_cfg, db_path):
    seed(db_path, dry_run=False)
    conn = db.connect(db_path)
    db.insert_trade(conn, "buy", db.Fill(1.0, 0.01, 100.0, "5SecretWalletRevealingSig"), simulated=False)
    snapshot = lambda: (
        dict(conn.execute("SELECT key, value FROM state").fetchall()),
        conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
    )
    before = snapshot()
    for path, body in (
        ("/api/pause", {"paused": True}),
        ("/api/risky", {"risky": True}),
        ("/api/timeframe", {"timeframe": "4h"}),
    ):
        r = public_client.post(path, json=body)
        assert r.status_code == 403, path
    assert snapshot() == before
    s = state(public_client)
    assert s["public"] is True
    rows = public_client.get("/api/trades").json()["rows"]
    assert all(row["tx"] is None for row in rows)
    assert "5SecretWalletRevealingSig" not in public_client.get("/api/trades").text


def test_public_mode_halt_banner_has_no_operator_commands(public_client, public_cfg, db_path):
    seed(db_path, halted=True)
    banner = state(public_client)["halt_banner"]
    assert banner and "reset_halt" not in banner


# --- timeframe -------------------------------------------------------------------------------------

def test_timeframe_switch_writes_only_the_timeframe(client, cfg, db_path):
    seed(db_path)
    conn = db.connect(db_path)
    before = dict(conn.execute("SELECT key, value FROM state").fetchall())
    options = state(client)["tf_options"]
    assert options == ["1h", "4h"]  # the one in use plus 4h; nothing shorter is offered
    assert client.post("/api/timeframe", json={"timeframe": "4h"}).status_code == 200
    after = dict(conn.execute("SELECT key, value FROM state").fetchall())
    assert db.get_timeframe(conn) == "4h"
    assert {k for k in set(before) | set(after) if before.get(k) != after.get(k)} == {db.TIMEFRAME}


def test_timeframe_switch_rejects_everything_outside_the_dashboard_choices(client, cfg, db_path):
    seed(db_path)
    conn = db.connect(db_path)
    r = client.post("/api/timeframe", json={"timeframe": "1m"})
    assert r.status_code == 400
    assert db.get_timeframe(conn) is None
