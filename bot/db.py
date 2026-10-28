from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, NamedTuple

import pandas as pd

from bot.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    side      TEXT    NOT NULL CHECK (side IN ('buy', 'sell')),
    amount_in  REAL   NOT NULL,   -- buy: USDC spent, sell: asset sold
    amount_out REAL   NOT NULL,   -- buy: asset received, sell: USDC received
    price     REAL    NOT NULL,   -- USDC per unit of the asset actually obtained
    tx_hash   TEXT,
    simulated INTEGER NOT NULL DEFAULT 1 CHECK (simulated IN (0, 1)),
    asset     TEXT    NOT NULL DEFAULT 'SOL'
);
CREATE TABLE IF NOT EXISTS balances (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    usdc      REAL    NOT NULL,
    sol       REAL    NOT NULL,   -- native SOL (fees, never traded)
    wsol      REAL    NOT NULL,   -- wrapped SOL: the trading position
    price     REAL,               -- SOL/USD mark at the time, NULL if unavailable
    simulated INTEGER NOT NULL DEFAULT 1 CHECK (simulated IN (0, 1)),
    holdings  TEXT                -- JSON {symbol: [qty, price]} for every non-SOL asset held or marked
);
CREATE INDEX IF NOT EXISTS idx_balances_timestamp ON balances (timestamp);
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ai_predictions (  -- every AI score, graded once its horizon has passed
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    asset      TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,
    candle     TEXT    NOT NULL,   -- timestamp of the last closed candle the AI saw
    p          REAL    NOT NULL,
    price0     REAL    NOT NULL,
    horizon    INTEGER NOT NULL,
    target_pct REAL    NOT NULL,
    resolved   INTEGER NOT NULL DEFAULT 0,
    fwd_return REAL,
    win        INTEGER,
    UNIQUE (asset, timeframe, candle)
);
CREATE INDEX IF NOT EXISTS idx_ai_open ON ai_predictions (timeframe, resolved);
CREATE TABLE IF NOT EXISTS snipes (   -- simulated new-token sniper: a sandbox that never touches the wallet
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    mint          TEXT    NOT NULL UNIQUE,
    symbol        TEXT    NOT NULL,
    decimals      INTEGER NOT NULL,
    opened_at     TEXT    NOT NULL,
    entry_usdc    REAL    NOT NULL,
    qty_raw       INTEGER NOT NULL,
    liquidity_usd REAL,
    status        TEXT    NOT NULL CHECK (status IN ('open', 'closed')),
    closed_at     TEXT,
    exit_usdc     REAL,
    reason        TEXT,
    last_value    REAL,           -- latest executable sell value of an open snipe
    last_checked  TEXT,
    fail_count    INTEGER NOT NULL DEFAULT 0
);
"""

MIGRATIONS = (  # columns added after the first release: (table, column, definition)
    ("trades", "asset", "TEXT NOT NULL DEFAULT 'SOL'"),
    ("balances", "holdings", "TEXT"),
    ("snipes", "holders", "INTEGER"),  # what the sniper saw at entry, so it can learn which filters pay
    ("snipes", "top_holders_pct", "REAL"),
    ("snipes", "dev_mints", "INTEGER"),
    ("snipes", "age_min", "REAL"),
    ("snipes", "impact", "REAL"),
    ("snipes", "round_trip", "REAL"),
)

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
RISKY = "risky"
ML_SCORES = "ml_scores"  # latest P(win) per coin, written by the bot for the dashboard
ML_LEARN = "ml_learn"  # what the AI has learned from its own predictions
ML_INFO = "ml_info"  # which model is live: when trained, out-of-sample AUC, thresholds
SNIPER_LEARN = "sniper_learn"  # the entry floors the sniper learned from its closed snipes
RISK_OFF = "risk_off"  # why the BTC filter blocks new positions this cycle; empty while it allows them


class Balances(NamedTuple):
    usdc: float
    sol: float  # native SOL, kept for fees and never traded
    wsol: float  # wrapped SOL, the trading position
    extra: Mapping[str, float] = {}  # every other traded asset, by symbol

    def qty(self, asset: str) -> float:
        return self.wsol if asset == "SOL" else self.extra.get(asset, 0.0)

    def positions(self) -> dict[str, float]:
        return {"SOL": self.wsol, **self.extra}


class Fill(NamedTuple):
    amount_in: float
    amount_out: float
    price: float  # USDC per SOL
    tx_hash: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str | None = None, *, readonly: bool = False) -> sqlite3.Connection:
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
        for table, column, definition in MIGRATIONS:
            columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


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
    set_state(conn, PAUSED, "1" if paused else "0")


def get_timeframe(conn: sqlite3.Connection) -> str | None:
    from bot.config import TIMEFRAMES

    value = get_state(conn, TIMEFRAME)
    return value if value in TIMEFRAMES else None


def set_timeframe(conn: sqlite3.Connection, timeframe: str) -> None:
    from bot.config import TIMEFRAMES

    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
    set_state(conn, TIMEFRAME, timeframe)


def is_risky(conn: sqlite3.Connection, default: bool = False) -> bool:
    return get_state(conn, RISKY, "1" if default else "0") == "1"


def set_risky(conn: sqlite3.Connection, risky: bool) -> None:
    set_state(conn, RISKY, "1" if risky else "0")


def record_prediction(
    conn: sqlite3.Connection, asset: str, timeframe: str, candle: str, p: float, price0: float, horizon: int, target_pct: float
) -> None:
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO ai_predictions (asset, timeframe, candle, p, price0, horizon, target_pct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (asset, timeframe, candle, p, price0, horizon, target_pct),
        )


def open_predictions(conn: sqlite3.Connection, asset: str, timeframe: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM ai_predictions WHERE asset = ? AND timeframe = ? AND resolved = 0 ORDER BY id", (asset, timeframe)
    ).fetchall()


def resolve_prediction(conn: sqlite3.Connection, prediction_id: int, fwd_return: float, win: bool) -> None:
    with conn:
        conn.execute(
            "UPDATE ai_predictions SET resolved = 1, fwd_return = ?, win = ? WHERE id = ?",
            (fwd_return, int(win), prediction_id),
        )


def resolved_predictions(conn: sqlite3.Connection, timeframe: str, limit: int = 5000) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM ai_predictions WHERE timeframe = ? AND resolved = 1 ORDER BY id DESC LIMIT ?", (timeframe, limit)
    ).fetchall()


def set_ml_learn(conn: sqlite3.Connection, info: Mapping[str, object]) -> None:
    set_state(conn, ML_LEARN, json.dumps(info))


def set_ml(conn: sqlite3.Connection, scores: Mapping[str, float], info: Mapping[str, object] | None) -> None:
    set_state(conn, ML_SCORES, json.dumps({a: round(p, 4) for a, p in scores.items()}))
    if info is not None:
        set_state(conn, ML_INFO, json.dumps(info))


def get_ml(conn: sqlite3.Connection) -> tuple[dict[str, float], dict]:
    scores, info = get_state(conn, ML_SCORES), get_state(conn, ML_INFO)
    return (json.loads(scores) if scores else {}), (json.loads(info) if info else {})


def is_halted(conn: sqlite3.Connection) -> bool:
    return _flag(conn, HALTED)


def halt_reason(conn: sqlite3.Connection) -> str:
    return get_state(conn, HALT_REASON, "") or ""


def set_halted(conn: sqlite3.Connection, reason: str) -> None:
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


def insert_trade(
    conn: sqlite3.Connection,
    side: str,
    fill: Fill,
    simulated: bool,
    timestamp: str | None = None,
    asset: str = "SOL",
) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO trades (timestamp, side, amount_in, amount_out, price, tx_hash, simulated, asset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp or utc_now_iso(),
                side,
                fill.amount_in,
                fill.amount_out,
                fill.price,
                fill.tx_hash,
                int(simulated),
                asset,
            ),
        )
    return cur.lastrowid or 0


def insert_balance(
    conn: sqlite3.Connection,
    balances: Balances,
    price: float | None,
    simulated: bool,
    timestamp: str | None = None,
    prices: Mapping[str, float] | None = None,
) -> None:
    prices = prices or {}
    holdings = {
        a: [qty, prices.get(a)] for a, qty in balances.extra.items() if qty > 0 or prices.get(a) is not None
    }
    with conn:
        conn.execute(
            "INSERT INTO balances (timestamp, usdc, sol, wsol, price, simulated, holdings) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp or utc_now_iso(),
                balances.usdc,
                balances.sol,
                balances.wsol,
                price,
                int(simulated),
                json.dumps(holdings) if holdings else None,
            ),
        )


def read_trades(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY id", conn)
    if "asset" not in df:  # database from before multi-asset support, not yet migrated by the bot
        df["asset"] = "SOL"
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
    if "holdings" not in df:
        df["holdings"] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["simulated"] = df["simulated"].astype(bool)
    return df


def holdings_of(raw: str | None) -> dict[str, tuple[float, float | None]]:
    """Decode a balances.holdings JSON cell into {symbol: (qty, price)}."""
    if not raw or not isinstance(raw, str):
        return {}
    return {a: (float(qty), None if price is None else float(price)) for a, (qty, price) in json.loads(raw).items()}


def holdings_value(raw: str | None) -> float:
    return sum(qty * price for qty, price in holdings_of(raw).values() if price)


def row_equity(row) -> float | None:
    """USDC plus every position at its recorded mark. None if SOL is held without a price."""
    sol_value = 0.0
    if row["wsol"]:
        if row["price"] is None:
            return None
        sol_value = row["wsol"] * row["price"]
    return row["usdc"] + sol_value + holdings_value(row["holdings"] if "holdings" in row.keys() else None)


def last_marks(conn: sqlite3.Connection) -> dict[str, float]:
    """Latest known price per symbol from the newest snapshots (for the floor during feed outages)."""
    marks: dict[str, float] = {}
    for row in conn.execute("SELECT price, holdings FROM balances ORDER BY id DESC LIMIT 200"):
        if row["price"] is not None:
            marks.setdefault("SOL", row["price"])
        for asset, (_, price) in holdings_of(row["holdings"]).items():
            if price is not None:
                marks.setdefault(asset, price)
    return marks


def latest_balance(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM balances ORDER BY id DESC LIMIT 1").fetchone()


def last_trade_time(conn: sqlite3.Connection, side: str, simulated: bool, asset: str = "SOL") -> pd.Timestamp | None:
    row = conn.execute(
        "SELECT timestamp FROM trades WHERE side = ? AND simulated = ? AND asset = ? ORDER BY id DESC LIMIT 1",
        (side, int(simulated), asset),
    ).fetchone()
    return pd.Timestamp(row["timestamp"]) if row else None


def last_trade(conn: sqlite3.Connection, simulated: bool, asset: str = "SOL") -> tuple[str, pd.Timestamp] | None:
    """(side, time) of the latest trade in this coin, in this mode."""
    row = conn.execute(
        "SELECT side, timestamp FROM trades WHERE simulated = ? AND asset = ? ORDER BY id DESC LIMIT 1",
        (int(simulated), asset),
    ).fetchone()
    return (row["side"], pd.Timestamp(row["timestamp"])) if row else None


def open_position(conn: sqlite3.Connection, asset: str, simulated: bool) -> tuple[float, float, pd.Timestamp | None]:
    """(quantity, average entry price, time of the latest buy) replayed from the trade log."""
    qty = cost = 0.0
    since = None
    rows = conn.execute(
        "SELECT timestamp, side, amount_in, amount_out FROM trades WHERE simulated = ? AND asset = ? ORDER BY id",
        (int(simulated), asset),
    )
    for row in rows:
        if row["side"] == "buy":
            qty += row["amount_out"]
            cost += row["amount_in"]
            since = pd.Timestamp(row["timestamp"])
        elif qty > 0:
            avg = cost / qty
            cost -= avg * row["amount_in"]
            qty -= row["amount_in"]
    if qty <= 1e-12:
        return 0.0, 0.0, None
    return qty, cost / qty, since


def sim_balances(conn: sqlite3.Connection, start_usdc: float, start_sol: float) -> Balances:
    rows = conn.execute(
        """
        SELECT
          asset,
          COALESCE(SUM(CASE WHEN side = 'buy' THEN -amount_in  ELSE amount_out END), 0) AS usdc_delta,
          COALESCE(SUM(CASE WHEN side = 'buy' THEN  amount_out ELSE -amount_in  END), 0) AS qty_delta
        FROM trades WHERE simulated = 1 GROUP BY asset
        """
    ).fetchall()
    usdc = max(start_usdc + sum(r["usdc_delta"] for r in rows), 0.0)
    qty = {r["asset"]: max(r["qty_delta"], 0.0) for r in rows}
    return Balances(usdc, start_sol, qty.get("SOL", 0.0), {a: q for a, q in qty.items() if a != "SOL"})


# Simulated new-token sniper: a sandbox with its own bankroll that never touches the wallet.

def open_snipe(
    conn: sqlite3.Connection, *, mint: str, symbol: str, decimals: int, entry_usdc: float, qty_raw: int,
    liquidity_usd: float | None, timestamp: str | None = None, holders: int | None = None,
    top_holders_pct: float | None = None, dev_mints: int | None = None, age_min: float | None = None,
    impact: float | None = None, round_trip: float | None = None,
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO snipes (mint, symbol, decimals, opened_at, entry_usdc, qty_raw, liquidity_usd, status, "
            "holders, top_holders_pct, dev_mints, age_min, impact, round_trip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?)",
            (mint, symbol, decimals, timestamp or utc_now_iso(), entry_usdc, qty_raw, liquidity_usd,
             holders, top_holders_pct, dev_mints, age_min, impact, round_trip),
        )


def mark_snipe(conn: sqlite3.Connection, snipe_id: int, value: float | None, fail_count: int, timestamp: str) -> None:
    with conn:
        conn.execute(
            "UPDATE snipes SET last_value = COALESCE(?, last_value), fail_count = ?, last_checked = ? WHERE id = ?",
            (value, fail_count, timestamp, snipe_id),
        )


def close_snipe(conn: sqlite3.Connection, snipe_id: int, exit_usdc: float, reason: str, timestamp: str) -> None:
    with conn:
        conn.execute(
            "UPDATE snipes SET status = 'closed', exit_usdc = ?, reason = ?, closed_at = ?, last_value = ? WHERE id = ?",
            (exit_usdc, reason, timestamp, exit_usdc, snipe_id),
        )


def read_snipes(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        df = pd.read_sql_query("SELECT * FROM snipes ORDER BY id", conn)
    except Exception:  # database from before the sniper existed
        return pd.DataFrame(columns=["id", "mint", "symbol", "status"])
    for col in ("opened_at", "closed_at", "last_checked"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def sniper_free_usdc(conn: sqlite3.Connection, bankroll: float) -> float:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(CASE WHEN status = 'open' THEN -entry_usdc
                                 ELSE COALESCE(exit_usdc, 0) - entry_usdc END), 0) AS delta FROM snipes
        """
    ).fetchone()
    return bankroll + row["delta"]
