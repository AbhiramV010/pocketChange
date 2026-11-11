"""Read-only view layer for the trading terminal: SQLite rows in, JSON-ready dicts out.

It imports no trading code and never writes anything; the server exposes exactly three
writes (pause, timeframe, RISKY) and each goes through one validated bot.db helper.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bot import db
from bot.assets import ASSETS, PRIMARY, label
from bot.config import DASHBOARD_TIMEFRAMES, Settings

RANGES = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30), "All": None}
DEFAULT_RANGE = "7d"


def config_timeframe() -> str:
    import bot.config as bc

    return bc.TIMEFRAME


def timeframe_options(conn) -> list[str]:
    """The one in use plus the dashboard's choices; nothing shorter is offered."""
    current = db.get_timeframe(conn) or config_timeframe()
    return list(dict.fromkeys((current, *DASHBOARD_TIMEFRAMES)))


def dry_run_flag(conn, cfg: Settings) -> bool:
    value = db.get_state(conn, db.DRY_RUN)
    return cfg.dry_run if value is None else value == "1"


def signed_usd(value: float) -> str:
    return f"{'-' if value < 0 else '+'}${abs(value):,.2f}"


# --- shapes from the latest balance snapshot -------------------------------------------------

def marks_from(latest) -> dict[str, float]:
    marks: dict[str, float] = {}
    if latest is None:
        return marks
    if latest["price"]:
        marks["SOL"] = float(latest["price"])
    for asset, (_, price) in db.holdings_of(latest["holdings"] if "holdings" in latest.keys() else None).items():
        if price:
            marks[asset] = price
    return marks


def quantities_from(latest) -> dict[str, float]:
    if latest is None:
        return {}
    qty = {"SOL": float(latest["wsol"])}
    for asset, (amount, _) in db.holdings_of(latest["holdings"] if "holdings" in latest.keys() else None).items():
        qty[asset] = amount
    return qty


def position_summary(trades: pd.DataFrame) -> tuple[float, float, float]:
    """(qty, average entry, realised) for one coin's trade rows."""
    qty = cost = realised = 0.0
    for side, amount_in, amount_out in zip(trades["side"], trades["amount_in"], trades["amount_out"]):
        if side == "buy":
            qty += amount_out
            cost += amount_in
        elif qty > 0:
            avg = cost / qty
            realised += amount_out - avg * amount_in
            cost -= avg * amount_in
            qty -= amount_in
    qty = max(qty, 0.0)
    return qty, (cost / qty if qty > 1e-12 else 0.0), realised


def positions(conn, cfg: Settings) -> list[dict]:
    latest = db.latest_balance(conn)
    if latest is None:
        return []
    trades = db.read_trades(conn)
    trades = trades[trades["simulated"] == dry_run_flag(conn, cfg)]
    scores, _ = db.get_ml(conn)
    marks, quantities = marks_from(latest), quantities_from(latest)
    rows = []
    for asset in ASSETS:
        qty = quantities.get(asset, 0.0)
        mine = trades[trades["asset"] == asset]
        _, avg_entry, realised = position_summary(mine)
        if qty * marks.get(asset, 0.0) < 0.01 and mine.empty and asset not in scores:
            continue
        price = marks.get(asset)
        row = {
            "symbol": asset,
            "asset": label(asset),
            "qty": qty,
            "price": price,
            "value": qty * price if price else None,
            "avg_entry": avg_entry or None,
            "unrealised": (price - avg_entry) * qty if price and qty and avg_entry else 0.0,
            "realised": realised,
        }
        if cfg.strategy == "ml" and asset in scores:
            row["p_win"] = float(scores[asset])
        rows.append(row)
    return rows


# --- AI --------------------------------------------------------------------------------------

def ml_caption(conn, cfg: Settings) -> str | None:
    if cfg.strategy != "ml":
        return None
    _, info = db.get_ml(conn)
    if info and info.get("trained_at"):
        auc = info.get("oos_auc")
        enter = info.get("enter", cfg.ml_enter)
        exit_ = info.get("exit", cfg.ml_exit)
        return (
            f"AI model ({info['timeframe']}): trained {str(info['trained_at'])[:16].replace('T', ' ')} UTC on "
            f"{int(info['rows']):,} rows"
            + (f", out-of-sample AUC {auc:.3f} (0.5 is a coin flip)" if auc else "")
            + f". Buys at p(win) >= {enter:g}, sells at <= {exit_:g}."
        )
    return "AI model: none for this timeframe yet. It trains in the background on first start (a minute or two)."


# --- sniper -----------------------------------------------------------------------------------

def sniper_totals(snipes: pd.DataFrame, bankroll: float) -> dict:
    """The sniper sandbox: free cash, open positions at their latest executable value, and P&L."""
    if snipes.empty:
        return {"free": bankroll, "open_value": 0.0, "realised": 0.0, "unrealised": 0.0, "open": 0, "closed": 0, "wins": 0}
    closed, opened = snipes[snipes["status"] == "closed"], snipes[snipes["status"] == "open"]
    open_cost = float(opened["entry_usdc"].sum())
    open_value = float(opened["last_value"].fillna(opened["entry_usdc"]).sum())
    realised = float((closed["exit_usdc"] - closed["entry_usdc"]).sum())
    return {
        "free": bankroll + realised - open_cost, "open_value": open_value, "realised": realised,
        "unrealised": open_value - open_cost, "open": len(opened), "closed": len(closed),
        "wins": int((closed["exit_usdc"] > closed["entry_usdc"]).sum()),
    }


def sniper_realised_over_time(snipes: pd.DataFrame, times: pd.Series, unrealised_now: float) -> np.ndarray:
    """Sniper P&L at each snapshot time: closed snipes count when they close; open ones only at the latest point."""
    closed = snipes[snipes["status"] == "closed"].sort_values("closed_at") if not snipes.empty else snipes
    out = np.zeros(len(times))
    if len(closed):
        stamps = closed["closed_at"].astype("int64").to_numpy()
        running = (closed["exit_usdc"] - closed["entry_usdc"]).cumsum().to_numpy()
        pos = np.searchsorted(stamps, pd.to_datetime(times, utc=True).astype("int64").to_numpy(), side="right")
        out = np.where(pos > 0, running[np.maximum(pos - 1, 0)], 0.0)
    if len(out):
        out[-1] += unrealised_now
    return out


def sniper_section(conn, cfg: Settings, now: datetime, snipes: pd.DataFrame | None = None) -> dict | None:
    """None when the section is hidden entirely (sniper off, or live with no snipe history)."""
    if not cfg.sniper:
        return None
    dry = dry_run_flag(conn, cfg)
    if snipes is None:
        snipes = db.read_snipes(conn)
    if not dry and snipes.empty:
        return None
    risky = db.is_risky(conn, cfg.risky)
    active = dry and (risky or not snipes.empty)
    totals = sniper_totals(snipes, cfg.sniper_bankroll_usdc)
    open_rows, closed_rows = [], []
    for _, r in snipes[snipes["status"] == "open"].iterrows():
        entry, value = float(r["entry_usdc"]), r["last_value"]
        value = None if pd.isna(value) else float(value)
        ref = value if value is not None else entry
        age = (pd.Timestamp(now) - r["opened_at"]).total_seconds() / 60
        open_rows.append({"symbol": r["symbol"], "age_min": round(age, 1), "entry": entry, "value": value,
                          "pnl_pct": (ref / entry - 1) * 100 if entry else None})
    closed = snipes[snipes["status"] == "closed"].sort_values("closed_at", ascending=False)
    for _, r in closed.iterrows():
        entry, exit_ = float(r["entry_usdc"]), float(r["exit_usdc"])
        closed_rows.append({"symbol": r["symbol"], "closed_at": r["closed_at"].isoformat(), "entry": entry,
                            "exit": exit_, "pnl_pct": (exit_ / entry - 1) * 100 if entry else None,
                            "reason": r["reason"]})
    caption = (
        f"Real Jupiter quotes, simulated money: a separate bankroll that never touches the wallet, counted in the "
        f"totals above. Entries and exits include a {cfg.sniper_latency_slippage:.0%} latency haircut per leg."
        if active
        else "Off. Turn on RISKY to start sniping brand-new tokens in simulation. It never touches the wallet."
    )
    return {"active": active, "totals": totals, "open": open_rows, "closed": closed_rows, "caption": caption}


# --- page pieces -------------------------------------------------------------------------------

def floor_caption(equity: float | None, cfg: Settings, sniper_active: bool, sn: dict | None) -> str | None:
    if equity is None:
        return None
    floor = float(cfg.min_equity_usd)
    if sniper_active and sn is not None:
        sniper_equity = sn["free"] + sn["open_value"]
        return (
            f"Total ${equity + sniper_equity:,.2f} (wallet ${equity:,.2f} + sniper ${sniper_equity:,.2f}) "
            f"| wallet halt floor ${floor:,.2f} | headroom ${equity - floor:,.2f}"
        )
    return f"Equity ${equity:,.2f} | halt floor ${floor:,.2f} | headroom ${equity - floor:,.2f}"


def halt_banner(conn, cfg: Settings) -> str | None:
    if not db.is_halted(conn):
        return None
    text = (
        f"BOT HALTED: {db.halt_reason(conn)}. Trading is stopped and the open position was left as-is."
    )
    if not cfg.dashboard_public:
        text += " Only `python -m bot.reset_halt` can clear this; the dashboard cannot."
    return text


def ticker(conn, cfg: Settings, now: datetime) -> list[dict]:
    """Every configured coin's latest mark and its change over the trailing 24 h of snapshots."""
    latest = db.latest_balance(conn)
    if latest is None:
        return []
    dry = dry_run_flag(conn, cfg)
    now_marks = marks_from(latest)
    bal = db.read_balances(conn, since=now - timedelta(hours=24))
    bal = bal[bal["simulated"] == dry]
    bal = bal.iloc[:-1] if len(bal) else bal  # baseline must be strictly before the latest snapshot
    base = marks_from(bal.iloc[0]) if len(bal) else {}
    order = [PRIMARY] + [a for a in dict.fromkeys((*cfg.assets, *cfg.risky_assets)) if a != PRIMARY]
    order += [a for a in now_marks if a not in order]
    rows = []
    for symbol in dict.fromkeys(order):
        price = now_marks.get(symbol)
        if price is None:
            continue
        ref = base.get(symbol)
        rows.append({"symbol": symbol, "asset": label(symbol), "price": price,
                     "change": price / ref - 1 if ref else None})
    return rows


def state(conn, cfg: Settings, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    latest = db.latest_balance(conn)
    dry = dry_run_flag(conn, cfg)
    paused, halted = db.is_paused(conn), db.is_halted(conn)
    risky = db.is_risky(conn, cfg.risky)
    snipes = db.read_snipes(conn)
    sn = sniper_section(conn, cfg, now, snipes)
    sniper_active = bool(sn and sn["active"])
    sniper_shown = sn is not None

    equity = usdc = sol_fees = wsol = sol_price = invested = None
    if latest is not None:
        equity = db.row_equity(latest)
        marks = marks_from(latest)
        quantities = quantities_from(latest)
        usdc, sol_fees, wsol = float(latest["usdc"]), float(latest["sol"]), float(latest["wsol"])
        sol_price = None if latest["price"] is None else float(latest["price"])
        invested = sum(q * marks.get(a, 0.0) for a, q in quantities.items())
    sniper_equity = (sn["totals"]["free"] + sn["totals"]["open_value"]) if sn else 0.0
    total_equity = None if equity is None else (equity + sniper_equity if sniper_active else equity)
    usdc_total = None if usdc is None else (usdc + (sn["totals"]["free"] if sn else 0.0))
    invested_total = None if invested is None else (invested + (sn["totals"]["open_value"] if sn else 0.0))
    floor = float(cfg.min_equity_usd)

    beat = db.get_state(conn, db.HEARTBEAT)
    heartbeat_age = None if beat is None else (pd.Timestamp(now) - pd.Timestamp(beat)).total_seconds()

    return {
        "ready": True,
        "now": now.isoformat(timespec="seconds"),
        "public": cfg.dashboard_public,
        "dry_run": dry,
        "paused": paused,
        "halted": halted,
        "halt_reason": db.halt_reason(conn),
        "halt_exits_ok": db.halt_allows_exits(conn),
        "halt_banner": halt_banner(conn, cfg),
        "risky": risky,
        "timeframe": db.get_timeframe(conn) or config_timeframe(),
        "tf_options": timeframe_options(conn),
        "heartbeat_age_s": heartbeat_age,
        "loop_interval_s": cfg.loop_interval_seconds,
        "multi": len(cfg.assets) + len(cfg.risky_assets) > 1,
        "ai": cfg.strategy == "ml",
        "strategy": cfg.strategy,
        "equity": equity,
        "total_equity": total_equity,
        "usdc": usdc,
        "usdc_total": usdc_total,
        "wsol": wsol,
        "sol_price": sol_price,
        "sol_fees": sol_fees,
        "invested": invested,
        "invested_total": invested_total,
        "floor": floor,
        "headroom": None if equity is None else equity - floor,
        "floor_caption": floor_caption(equity, cfg, sniper_active, sn["totals"] if sn else None),
        "mode_label": "RISKY" if risky else "Top coins",
        "positions": positions(conn, cfg),
        "ticker": ticker(conn, cfg, now),
        "ml_caption": ml_caption(conn, cfg),
        "sniper": sn,
        "sniper_shown": sniper_shown,
        "sniper_active": sniper_active,
    }


# --- chart and log data -------------------------------------------------------------------------

def series(conn, cfg: Settings, range_key: str, now: datetime | None = None) -> dict:
    """Wallet equity at each priced snapshot in the range; the sniper's realised overlay when it is active."""
    now = now or datetime.now(timezone.utc)
    window = RANGES[range_key] if range_key in RANGES else RANGES[DEFAULT_RANGE]
    since = now - window if window else None
    dry = dry_run_flag(conn, cfg)
    bal = db.read_balances(conn, since=since)
    bal = bal[bal["simulated"] == dry]
    snipes = db.read_snipes(conn)
    sniper_active = bool(cfg.sniper and dry and (db.is_risky(conn, cfg.risky) or not snipes.empty))
    sn = sniper_totals(snipes, cfg.sniper_bankroll_usdc) if sniper_active else None
    rows = []
    if len(bal):
        stamps = pd.to_datetime(bal["timestamp"], utc=True)
        equity = [db.row_equity(r) for _, r in bal.iterrows()]
        keep = [i for i, e in enumerate(equity) if e is not None and bal.iloc[i]["price"] is not None]
        overlay = (
            sniper_realised_over_time(snipes, stamps.iloc[keep].reset_index(drop=True), sn["unrealised"])
            if sniper_active else None
        )
        for k, i in enumerate(keep):
            point = {"t": stamps.iloc[i].isoformat(), "equity": float(equity[i])}
            if overlay is not None:
                point["total"] = float(equity[i] + overlay[k])
            rows.append(point)
    return {"points": rows, "floor": float(cfg.min_equity_usd), "sniper_active": sniper_active}


def trades(conn, cfg: Settings, limit: int = 100) -> dict:
    dry = dry_run_flag(conn, cfg)
    df = db.read_trades(conn)
    df = df[df["simulated"] == dry].sort_values("timestamp", ascending=False).head(limit)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "timestamp": r["timestamp"].isoformat(),
            "side": r["side"],
            "asset": label(r["asset"]),
            "symbol": r["asset"],
            "amount_in": float(r["amount_in"]),
            "amount_out": float(r["amount_out"]),
            "price": float(r["price"]),
            "simulated": bool(r["simulated"]),
            "tx": None if cfg.dashboard_public or not r["tx_hash"] else f"https://solscan.io/tx/{r['tx_hash']}",
        })
    return {"rows": rows, "total": int(len(df))}


def candles(cfg: Settings, limit: int = 400) -> dict:
    """OHLCV from the local candle cache; the terminal never touches the network."""
    path = Path(cfg.candles_path)
    out = {"candles": [], "timeframe": config_timeframe()}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path)
    except Exception:
        return out
    if df.empty or "timestamp" not in df.columns:
        return out
    df["timestamp"] = pd.to_datetime(df.pop("timestamp"), utc=True)
    df = df.sort_values("timestamp").tail(limit)
    out["candles"] = [
        {"t": ts.isoformat(), "o": float(r["open"]), "h": float(r["high"]), "l": float(r["low"]),
         "c": float(r["close"]), "v": float(r["volume"])}
        for ts, r in df.iterrows()
    ]
    return out
