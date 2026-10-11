from bot import db
from bot.db import Balances, Fill


def test_schema_has_required_columns(conn):
    cols = lambda t: [r["name"] for r in conn.execute(f"PRAGMA table_info({t})")]
    for needed in ("timestamp", "side", "amount_in", "amount_out", "price", "tx_hash", "simulated"):
        assert needed in cols("trades")
    for needed in ("timestamp", "usdc", "sol", "wsol"):
        assert needed in cols("balances")
    assert cols("state") == ["key", "value"]


def test_pause_and_halt_flags(conn):
    assert not db.is_paused(conn) and not db.is_halted(conn)
    db.set_paused(conn, True)
    assert db.is_paused(conn)
    db.set_paused(conn, False)
    assert not db.is_paused(conn)
    db.set_halted(conn, "equity below floor")
    assert db.is_halted(conn) and db.halt_reason(conn) == "equity below floor"
    db.clear_halt(conn)
    assert not db.is_halted(conn) and db.halt_reason(conn) == ""


def test_halt_survives_reopen(tmp_path):
    path = tmp_path / "a.db"
    c1 = db.connect(path)
    db.init_db(c1)
    db.set_halted(c1, "why")
    c1.close()
    c2 = db.connect(path)
    assert db.is_halted(c2)


def test_trades_balances_roundtrip(conn):
    db.insert_trade(conn, "buy", Fill(5.0, 0.002, 2500.0, None), simulated=True)
    db.insert_balance(conn, Balances(15.0, 0.002, 0.002), 2500.0, simulated=True)
    trades = db.read_trades(conn)
    assert trades.loc[0, "side"] == "buy" and bool(trades.loc[0, "simulated"])
    assert db.latest_balance(conn)["usdc"] == 15.0
    assert len(db.read_balances(conn)) == 1


def test_sim_balances_replay(conn):
    db.insert_trade(conn, "buy", Fill(5.0, 0.002, 2500.0, None), simulated=True)
    db.insert_trade(conn, "sell", Fill(0.001, 2.6, 2600.0, None), simulated=True)
    db.insert_trade(conn, "buy", Fill(9.0, 1.0, 9.0, "0xabc"), simulated=False)  # live trade is ignored
    b = db.sim_balances(conn, start_usdc=20.0, start_sol=0.002)
    assert round(b.usdc, 6) == 17.6
    assert round(b.wsol, 6) == 0.001
    assert b.sol == 0.002


def test_last_trade_time_filters_by_mode(conn):
    assert db.last_trade_time(conn, "buy", True) is None
    db.insert_trade(conn, "buy", Fill(1, 1, 1, None), True, timestamp="2026-01-01T00:00:00+00:00")
    assert db.last_trade_time(conn, "buy", True).year == 2026
    assert db.last_trade_time(conn, "buy", False) is None


def test_readonly_connection_cannot_write(tmp_path):
    path = tmp_path / "ro.db"
    c = db.connect(path)
    db.init_db(c)
    ro = db.connect(path, readonly=True)
    import sqlite3, pytest
    with pytest.raises(sqlite3.OperationalError):
        db.set_paused(ro, True)
