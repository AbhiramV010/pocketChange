import sqlite3

import pytest

from bot import db
from bot.db import Balances, Fill


def test_old_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, side TEXT NOT NULL,
            amount_in REAL NOT NULL, amount_out REAL NOT NULL, price REAL NOT NULL, tx_hash TEXT,
            simulated INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE balances (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, usdc REAL NOT NULL,
            sol REAL NOT NULL, wsol REAL NOT NULL, price REAL, simulated INTEGER NOT NULL DEFAULT 1);
        INSERT INTO trades (timestamp, side, amount_in, amount_out, price, simulated)
            VALUES ('2026-01-01T00:00:00+00:00', 'buy', 5, 0.05, 100, 1);
        INSERT INTO balances (timestamp, usdc, sol, wsol, price) VALUES ('2026-01-01T00:00:00+00:00', 15, 0.05, 0.05, 100);
        """
    )
    old.commit()
    old.close()

    conn = db.connect(path)
    db.init_db(conn)
    db.init_db(conn)  # idempotent
    trades = db.read_trades(conn)
    assert trades.loc[0, "asset"] == "SOL"  # the old row defaults to SOL
    assert db.sim_balances(conn, 20.0, 0.05).wsol == pytest.approx(0.05)
    assert db.row_equity(db.latest_balance(conn)) == pytest.approx(20.0)


def test_sim_balances_track_each_asset_separately(conn):
    db.insert_trade(conn, "buy", Fill(5.0, 0.05, 100.0, None), True, asset="SOL")
    db.insert_trade(conn, "buy", Fill(5.0, 0.0001, 50_000.0, None), True, asset="BTC")
    db.insert_trade(conn, "sell", Fill(0.00004, 2.4, 60_000.0, None), True, asset="BTC")
    b = db.sim_balances(conn, 20.0, 0.05)
    assert b.usdc == pytest.approx(20 - 5 - 5 + 2.4)
    assert b.wsol == pytest.approx(0.05)
    assert b.extra == {"BTC": pytest.approx(0.00006)}
    assert b.qty("BTC") == pytest.approx(0.00006) and b.qty("ETH") == 0.0
    assert set(b.positions()) == {"SOL", "BTC"}


def test_holdings_roundtrip_equity_and_marks(conn):
    b = Balances(10.0, 0.05, 0.05, {"BTC": 0.0001, "ETH": 0.0})
    db.insert_balance(conn, b, 100.0, True, prices={"BTC": 80_000.0, "ETH": 4_000.0, "SOL": 100.0})
    row = db.latest_balance(conn)
    assert db.holdings_of(row["holdings"]) == {"BTC": (0.0001, 80_000.0), "ETH": (0.0, 4_000.0)}
    assert db.row_equity(row) == pytest.approx(10 + 5 + 8)
    assert db.last_marks(conn) == {"SOL": 100.0, "BTC": 80_000.0, "ETH": 4_000.0}


def test_equity_is_unknown_when_sol_is_held_without_a_price(conn):
    db.insert_balance(conn, Balances(10.0, 0.05, 0.05), None, True)
    assert db.row_equity(db.latest_balance(conn)) is None
    db.insert_balance(conn, Balances(10.0, 0.05, 0.0), None, True)
    assert db.row_equity(db.latest_balance(conn)) == 10.0


def test_open_position_replays_the_trade_log(conn):
    assert db.open_position(conn, "BTC", True) == (0.0, 0.0, None)
    db.insert_trade(conn, "buy", Fill(5.0, 0.0001, 50_000.0, None), True, "2026-01-01T00:00:00+00:00", "BTC")
    db.insert_trade(conn, "buy", Fill(5.0, 0.0001, 50_000.0, None), True, "2026-01-02T00:00:00+00:00", "BTC")
    db.insert_trade(conn, "sell", Fill(0.0001, 6.0, 60_000.0, None), True, "2026-01-03T00:00:00+00:00", "BTC")
    qty, avg, since = db.open_position(conn, "BTC", True)
    assert qty == pytest.approx(0.0001) and avg == pytest.approx(50_000.0)
    assert since.isoformat() == "2026-01-02T00:00:00+00:00"
    assert db.open_position(conn, "ETH", True)[0] == 0.0
    assert db.last_trade_time(conn, "sell", True, "BTC").isoformat() == "2026-01-03T00:00:00+00:00"
    assert db.last_trade_time(conn, "sell", True, "SOL") is None


def test_risky_flag_defaults_to_the_config_and_persists(conn):
    assert db.is_risky(conn) is False and db.is_risky(conn, default=True) is True
    db.set_risky(conn, True)
    assert db.is_risky(conn, default=False) is True  # the stored choice beats the default
    db.set_risky(conn, False)
    assert db.is_risky(conn, default=True) is False


def test_snipe_lifecycle_and_bankroll(conn):
    assert db.sniper_free_usdc(conn, 10.0) == 10.0
    db.open_snipe(conn, mint="MintA", symbol="AAA", decimals=6, entry_usdc=2.0, qty_raw=1000, liquidity_usd=5000.0)
    assert db.sniper_free_usdc(conn, 10.0) == 8.0
    db.open_snipe(conn, mint="MintB", symbol="BBB", decimals=6, entry_usdc=2.0, qty_raw=1000, liquidity_usd=5000.0)
    with pytest.raises(sqlite3.IntegrityError):  # one entry per token, ever
        db.open_snipe(conn, mint="MintA", symbol="AAA", decimals=6, entry_usdc=2.0, qty_raw=1, liquidity_usd=None)
    snipes = db.read_snipes(conn)
    db.close_snipe(conn, int(snipes.loc[0, "id"]), 3.0, "take profit", "2026-01-01T00:00:00+00:00")
    assert db.sniper_free_usdc(conn, 10.0) == pytest.approx(10 + 1.0 - 2.0)
    assert list(db.read_snipes(conn)["status"]) == ["closed", "open"]
