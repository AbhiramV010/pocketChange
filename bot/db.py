"""SQLite schema and helpers, shared by the bot and the dashboard.

The bot writes everything. The dashboard reads, and writes exactly one thing:
the pause flag (`set_paused`).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from bot.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    side      TEXT    NOT NULL CHECK (side IN ('buy', 'sell')),
    amount_in  REAL   NOT NULL,   -- buy: USDC spent, sell: wSOL sold
    amount_out REAL   NOT NULL,   -- buy: wSOL received, sell: USDC received
    price     REAL    NOT NULL,   -- USDC per SOL actually obtained
    tx_hash   TEXT,
    simulated INTEGER NOT NULL DEFAULT 1 CHECK (simulated IN (0, 1))
);
CREATE TABLE IF NOT EXISTS balances (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    usdc      REAL    NOT NULL,
    sol       REAL    NOT NULL,   -- native SOL (fees, never traded)
    wsol      REAL    NOT NULL,   -- wrapped SOL: the trading position
    price     REAL,               -- SOL/USD mark at the time, NULL if unavailable
    simulated INTEGER NOT NULL DEFAULT 1 CHECK (simulated IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_balances_timestamp ON balances (timestamp);
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# state keys
PAUSED = "paused"
HALTED = "halted"
HALT_REASON = "halt_reason"
HALTED_AT = "halted_at"
DRY_RUN = "dry_run"
HEARTBEAT = "heartbeat"
TIMEFRAME = "timeframe"
DONE_SIGNAL = "done_signal"
ATTEMPT_SIGNAL = "attempt_signal"
ATTEMPT_COUNT = "attempt_count"


class Balances(NamedTuple):
    usdc: float
    sol: float  # native SOL, kept for fees and never traded
    wsol: float  # wrapped SOL, the trading position


class Fill(NamedTuple):
    amount_in: float
    amount_out: float
    price: float  # USDC per SOL
    tx_hash: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str | None = None, *, readonly: bool = False) -> sqlite3.Connection:
    """Open the database. WAL mode lets the bot write while the dashboard reads."""
    path = Path(path) if path is not None else settings.db_path
    if readonly:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
        conn.execute("PRAGMA query_only = ON")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10)
        conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(SCHEMA)


# --- state / flags -----------------------------------------------------------

def get_state(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO state (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


def _flag(conn: sqlite3.Connection, key: str) -> bool:
    return get_state(conn, key, "0") == "1"


def is_paused(conn: sqlite3.Connection) -> bool:
    return _flag(conn, PAUSED)


def set_paused(conn: sqlite3.Connection, paused: bool) -> None:
    """The only write the dashboard is allowed to make."""
    set_state(conn, PAUSED, "1" if paused else "0")


def get_timeframe(conn: sqlite3.Connection) -> str | None:
    """Timeframe chosen on the dashboard, or None to use the TIMEFRAME setting."""
    from bot.config import TIMEFRAMES

    value = get_state(conn, TIMEFRAME)
    return value if value in TIMEFRAMES else None


def set_timeframe(conn: sqlite3.Connection, timeframe: str) -> None:
    """The dashboard's second write: the candle timeframe the bot trades on."""
    from bot.config import TIMEFRAMES

    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
    set_state(conn, TIMEFRAME, timeframe)


def is_halted(conn: sqlite3.Connection) -> bool:
    return _flag(conn, HALTED)


def halt_reason(conn: sqlite3.Connection) -> str:
    return get_state(conn, HALT_REASON, "") or ""


def set_halted(conn: sqlite3.Connection, reason: str) -> None:
    """Persistent halt. Survives restarts; only bot.reset_halt clears it."""
    with conn:
        for key, value in ((HALTED, "1"), (HALT_REASON, reason), (HALTED_AT, utc_now_iso())):
            conn.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )


def clear_halt(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("DELETE FROM state WHERE key IN (?, ?, ?)", (HALTED, HALT_REASON, HALTED_AT))


# --- writes ------------------------------------------------------------------

def insert_trade(
    conn: sqlite3.Connection, side: str, fill: Fill, simulated: bool, timestamp: str | None = None
) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO trades (timestamp, side, amount_in, amount_out, price, tx_hash, simulated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp or utc_now_iso(),
                side,
                fill.amount_in,
                fill.amount_out,
                fill.price,
                fill.tx_hash,
                int(simulated),
            ),
        )
    return cur.lastrowid or 0


def insert_balance(
    conn: sqlite3.Connection,
    balances: Balances,
    price: float | None,
    simulated: bool,
    timestamp: str | None = None,
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO balances (timestamp, usdc, sol, wsol, price, simulated) VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp or utc_now_iso(), balances.usdc, balances.sol, balances.wsol, price, int(simulated)),
        )


# --- reads -------------------------------------------------------------------

def read_trades(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY id", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["simulated"] = df["simulated"].astype(bool)
    return df


def read_balances(conn: sqlite3.Connection, since: datetime | None = None) -> pd.DataFrame:
    if since is None:
        df = pd.read_sql_query("SELECT * FROM balances ORDER BY id", conn)
    else:
        df = pd.read_sql_query(
            "SELECT * FROM balances WHERE timestamp >= ? ORDER BY id", conn, params=(since.isoformat(),)
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["simulated"] = df["simulated"].astype(bool)
    return df


def latest_balance(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM balances ORDER BY id DESC LIMIT 1").fetchone()


def last_trade_time(conn: sqlite3.Connection, side: str, simulated: bool) -> pd.Timestamp | None:
    row = conn.execute(
        "SELECT timestamp FROM trades WHERE side = ? AND simulated = ? ORDER BY id DESC LIMIT 1",
        (side, int(simulated)),
    ).fetchone()
    return pd.Timestamp(row["timestamp"]) if row else None


def sim_balances(conn: sqlite3.Connection, start_usdc: float, start_sol: float) -> Balances:
    """Simulated wallet = starting balances replayed through the simulated trades."""
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN side = 'buy'  THEN -amount_in  ELSE amount_out END), 0) AS usdc_delta,
          COALESCE(SUM(CASE WHEN side = 'buy'  THEN  amount_out ELSE -amount_in  END), 0) AS wsol_delta
        FROM trades WHERE simulated = 1
        """
    ).fetchone()
    usdc = max(start_usdc + row["usdc_delta"], 0.0)
    wsol = max(row["wsol_delta"], 0.0)
    return Balances(usdc=usdc, sol=start_sol, wsol=wsol)
