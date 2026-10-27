import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bot.config
from bot import db

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = tmp_path / "dash.db"
    monkeypatch.setattr(bot.config, "settings", replace(bot.config.settings, db_path=path))
    return path


def run_app():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


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


def text(at):
    return " ".join(m.value for m in at.markdown) + " ".join(e.value for e in at.error) + " ".join(w.value for w in at.warning)


def test_no_database_shows_hint(app):
    at = run_app()
    assert any("No database yet" in i.value for i in at.info)


def test_shows_dry_run_indicator_balances_and_trade_log(app):
    seed(app)
    at = run_app()
    labels = {m.label: m.value for m in at.metric}
    assert labels["USDC"] == "15.00" and labels["wSOL"] == "0.002000"
    assert labels["Equity (USDC + wSOL)"] == "$20.20"
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 1  # the trade log
    assert not any("BOT HALTED" in m.value for m in at.markdown)


def test_status_bar_is_gone(app):
    seed(app)
    at = run_app()
    assert len(at.toggle) == 0
    assert not any("DRY RUN" in m.value or "Bot alive" in m.value for m in at.markdown)
    assert len(at.get("divider")) >= 2

def test_says_why_nothing_is_bought_while_the_btc_filter_is_off(app):
    seed(app)
    conn = db.connect(app)
    db.set_state(conn, db.RISK_OFF, "BTC 80,262 is below its 400-candle average 81,861")
    at = run_app()
    assert any("Not opening new positions: BTC 80,262 is below" in i.value for i in at.info)
    db.set_state(conn, db.RISK_OFF, "")
    assert not run_app().info
    db.set_state(conn, db.RISK_OFF, "BTC 80,262 is below its 400-candle average 81,861")
    db.set_paused(conn, True)
    assert not run_app().info  # paused has its own control; one message at a time


def test_halted_shows_red_banner_and_cannot_be_cleared(app):
    seed(app, halted=True)
    at = run_app()
    banner = [m.value for m in at.markdown if "BOT HALTED" in m.value]
    assert banner and "#c62828" in banner[0] and "MIN_EQUITY_USD" in banner[0]
    assert at.button(key="emergency_stop").disabled
    conn = db.connect(app)
    assert db.is_halted(conn)


def test_stop_and_resume_write_only_the_pause_flag(app):
    seed(app)
    conn = db.connect(app)
    before_state = dict(conn.execute("SELECT key, value FROM state").fetchall())
    trades_before = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    at = run_app()
    at.button(key="emergency_stop").click().run()
    assert db.is_paused(conn)
    at.button(key="resume").click().run()
    assert not db.is_paused(conn)

    after_state = dict(conn.execute("SELECT key, value FROM state").fetchall())
    changed = {k for k in set(before_state) | set(after_state) if before_state.get(k) != after_state.get(k)}
    assert changed <= {db.PAUSED}
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == trades_before

def test_button_mirrors_flag_set_elsewhere(app):
    seed(app, paused=True)
    at = run_app()
    assert at.button(key="resume").label == "RESUME"


def test_private_key_is_only_referenced_in_chain_module():
    offenders = []
    for path in list((ROOT / "bot").glob("*.py")) + list((ROOT / "dashboard").glob("*.py")):
        if path.name != "chain.py" and "PRIVATE_KEY" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []


def test_dashboard_imports_nothing_that_can_trade_and_writes_only_pause():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not {"bot.chain", "bot.run", "solders", "requests", "bot.reset_halt"} & imported
    called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    writers = {"set_halted", "clear_halt", "insert_trade", "insert_balance", "set_state", "init_db"}
    assert not writers & called
    assert "set_paused" in called


def test_pocketchange_has_no_hardcoded_secrets():
    pattern = re.compile(r"0x[0-9a-fA-F]{64}\b")
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_equity_curve_and_headroom_caption_render(app):
    seed(app)
    at = run_app()
    assert at.get("vega_lite_chart") or at.get("arrow_vega_lite_chart")
    assert any("halt floor" in c.value and "headroom" in c.value for c in at.caption)


@pytest.fixture
def public_app(tmp_path, monkeypatch):
    path = tmp_path / "public.db"
    monkeypatch.setattr(bot.config, "settings", replace(bot.config.settings, db_path=path, dashboard_public=True))
    return path


def test_public_mode_has_no_controls_and_writes_nothing(public_app):
    seed(public_app, dry_run=False)
    conn = db.connect(public_app)
    db.insert_trade(conn, "buy", db.Fill(1.0, 0.01, 100.0, "5SecretWalletRevealingSig"), simulated=False)
    snapshot = lambda: (
        conn.execute("SELECT key, value FROM state ORDER BY key").fetchall(),
        conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
    )
    before = snapshot()
    at = run_app()
    assert len(at.toggle) == 0 and len(at.button) == 0
    shown = at.dataframe[0].value
    assert "tx" not in shown.columns and "tx_hash" not in shown.columns
    assert "5SecretWalletRevealingSig" not in shown.to_string()
    assert snapshot() == before


def test_public_mode_halt_banner_has_no_operator_commands(public_app):
    seed(public_app, halted=True)
    at = run_app()
    banner = next(m.value for m in at.markdown if "BOT HALTED" in m.value)
    assert "reset_halt" not in banner


def test_pause_callback_refuses_in_public_mode(public_app, monkeypatch):
    seed(public_app)
    import importlib.util
    import streamlit as st

    spec = importlib.util.spec_from_file_location("public_dashboard", APP)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(st, "fragment", lambda **kw: (lambda fn: fn))
    spec.loader.exec_module(module)
    module.set_paused(True)
    assert not db.is_paused(db.connect(public_app))

def test_stop_button_pauses_trading(app):
    seed(app)
    at = run_app()
    stop = at.button(key="emergency_stop")
    assert stop.label == "STOP" and not stop.disabled
    stop.click().run()
    assert db.is_paused(db.connect(app))
    assert at.button(key="resume").label == "RESUME"  # same spot now resumes



def test_timeframe_switch_writes_only_the_timeframe(app):
    seed(app)
    conn = db.connect(app)
    at = run_app()
    switch = at.get("button_group")[0]
    assert switch.value == "1h"
    before = dict(conn.execute("SELECT key, value FROM state").fetchall())
    switch.set_value("5m").run()
    after = dict(conn.execute("SELECT key, value FROM state").fetchall())
    assert db.get_timeframe(conn) == "5m"
    assert {k for k in set(before) | set(after) if before.get(k) != after.get(k)} == {db.TIMEFRAME}


def test_public_timeframe_switch_is_read_only(public_app):
    seed(public_app)
    at = run_app()
    assert at.get("button_group")[0].disabled
