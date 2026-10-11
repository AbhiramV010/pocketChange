"""Streamlit dashboard. Run: streamlit run dashboard/app.py

Read-only view of the bot's SQLite database. The one thing it writes is the
pause flag. It cannot clear a halt and never touches the wallet key.

With DASHBOARD_PUBLIC=true it is safe to share: there is no pause control,
nothing is ever written, and transaction links (which reveal the wallet) are
hidden.
"""
import html
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

# `streamlit run dashboard/app.py` puts dashboard/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from bot import db  # noqa: E402
from bot.config import settings  # noqa: E402

PUBLIC = settings.dashboard_public
REFRESH = "5s"
MAX_CHART_POINTS = 2000
RANGES = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30), "All": None}

st.set_page_config(page_title="pocketchange", page_icon=":moneybag:", layout="wide")


def set_pause_from_toggle() -> None:
    """Callback for the pause toggle: the dashboard's only write. Never runs in public mode."""
    if PUBLIC:
        return
    with closing(db.connect(settings.db_path)) as conn:
        db.set_paused(conn, bool(st.session_state["pause_toggle"]))


def position_summary(trades: pd.DataFrame) -> tuple[float, float, float]:
    """(wSOL held, average entry price, realised P&L in USDC) by moving-average cost."""
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


def signed_usd(value: float) -> str:
    return f"{'-' if value < 0 else '+'}${abs(value):,.2f}"


def seconds_since(iso: str | None) -> float | None:
    if not iso:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()


def banner(text: str) -> None:
    st.markdown(
        "<div style='background:#c62828;color:#fff;padding:14px 18px;border-radius:6px;"
        f"font-weight:600;margin-bottom:12px'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def render_pause_toggle(paused: bool, halted: bool) -> None:
    st.session_state["pause_toggle"] = paused  # always mirror the database
    st.toggle(
        "Pause trading (takes effect within one bot cycle)",
        key="pause_toggle",
        on_change=set_pause_from_toggle,
        disabled=halted,
        help="Disabled while halted. A paused bot keeps logging balances.",
    )


@st.fragment(run_every=REFRESH)
def live_view() -> None:
    if not settings.db_path.exists():
        st.info("The bot has not started yet." if PUBLIC else f"No database yet at `{settings.db_path}`. Start the bot with `python -m bot.run`.")
        return
    with closing(db.connect(settings.db_path, readonly=True)) as conn:
        halted, reason = db.is_halted(conn), db.halt_reason(conn)
        paused = db.is_paused(conn)
        mode_flag = db.get_state(conn, db.DRY_RUN)
        heartbeat = db.get_state(conn, db.HEARTBEAT)
        trades_all = db.read_trades(conn)
        range_choice = st.session_state.get("range", "7d")
        window = RANGES[range_choice]
        since = datetime.now(timezone.utc) - window if window else None
        balances_all = db.read_balances(conn, since=since)
        latest = db.latest_balance(conn)

    dry_run = settings.dry_run if mode_flag is None else mode_flag == "1"

    if halted:
        banner(
            f"BOT HALTED: {reason}. Trading is stopped and the open position was left as-is."
            if PUBLIC
            else f"BOT HALTED: {reason}. Trading is stopped and the open position was left as-is. "
            "Only `python -m bot.reset_halt` can clear this; the dashboard cannot."
        )

    st.title("pocketchange")
    top = st.columns([2, 2, 3])
    with top[0]:
        if dry_run:
            st.warning("**DRY RUN**: simulated balances and trades, nothing on-chain", icon=":material/science:")
        else:
            st.error("**LIVE**: real funds on Solana", icon=":material/bolt:")
    with top[1]:
        age = seconds_since(heartbeat)
        if age is None:
            st.info("Bot has not reported yet")
        elif age > max(180, settings.loop_interval_seconds * 3):
            st.error(f"Bot silent for {age / 60:.0f} min." + ("" if PUBLIC else " Is `python -m bot.run` running?"))
        else:
            st.success(f"Bot alive, last cycle {age:.0f}s ago")
    with top[2]:
        if PUBLIC:
            st.info("Trading paused" if paused else "Trading active", icon=":material/visibility:")
        else:
            render_pause_toggle(paused, halted)

    trades = trades_all[trades_all["simulated"] == dry_run]
    balances = balances_all[balances_all["simulated"] == dry_run]

    if latest is None:
        st.info("Waiting for the first balance snapshot.")
    else:
        price = latest["price"]
        wsol_usd = latest["wsol"] * price if price else None
        equity = latest["usdc"] + (wsol_usd or 0.0)
        cols = st.columns(5)
        cols[0].metric("Equity (USDC + wSOL)", f"${equity:,.2f}", help=f"Trading halts below ${settings.min_equity_usd:g}")
        cols[1].metric("USDC", f"{latest['usdc']:,.2f}")
        cols[2].metric("wSOL", f"{latest['wsol']:.6f}", f"${wsol_usd:,.2f}" if wsol_usd is not None else None, delta_color="off")
        cols[3].metric("SOL (fees, not traded)", f"{latest['sol']:.5f}")
        cols[4].metric("SOL price", f"${price:,.2f}" if price else "n/a")

        st.subheader("Current position")
        qty, avg_entry, realised = position_summary(trades)
        unrealised = (price - avg_entry) * qty if price and qty else 0.0
        p = st.columns(4)
        p[0].metric("wSOL held", f"{qty:.6f}")
        p[1].metric("Average entry", f"${avg_entry:,.2f}" if qty else "flat")
        p[2].metric("Unrealised P&L", signed_usd(unrealised))
        p[3].metric("Realised P&L", signed_usd(realised))

    st.subheader("Equity curve")
    st.radio("Range", list(RANGES), index=1, key="range", horizontal=True, label_visibility="collapsed")
    curve = balances.dropna(subset=["price"]).sort_values("timestamp").copy()
    if curve.empty:
        st.caption("No priced balance snapshots in this range yet.")
    else:
        # Plot change vs the start of the range: a $20 wallet moves cents, and a chart
        # of absolute equity would be a flat line squashed against the top.
        curve["equity"] = curve["usdc"] + curve["wsol"] * curve["price"]
        curve["change"] = curve["equity"] - curve["equity"].iloc[0]
        step = max(1, len(curve) // MAX_CHART_POINTS)
        st.line_chart(curve.iloc[::step].set_index("timestamp")["change"], y_label="Equity change vs start of range (USD)")
        now_equity = curve["equity"].iloc[-1]
        st.caption(
            f"Equity ${now_equity:,.2f} | halt floor ${settings.min_equity_usd:,.2f} | "
            f"headroom ${now_equity - settings.min_equity_usd:,.2f}"
        )

    st.subheader("Trade log")
    if trades.empty:
        st.caption("No trades yet.")
    else:
        log = trades.sort_values("timestamp", ascending=False).copy()
        columns = ["timestamp", "side", "amount_in", "amount_out", "price", "simulated"]
        if not PUBLIC:  # a transaction link reveals the wallet address and its whole history
            log["tx"] = log["tx_hash"].map(lambda h: f"https://solscan.io/tx/{h}" if h else None)
            columns.insert(5, "tx")
        st.dataframe(
            log[columns],
            hide_index=True,
            width="stretch",
            column_config={"tx": st.column_config.LinkColumn("tx", display_text="Solscan")},
        )


live_view()
