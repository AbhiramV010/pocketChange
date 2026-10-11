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
from bot.config import DASHBOARD_TIMEFRAMES, TIMEFRAME, settings  # noqa: E402

PUBLIC = settings.dashboard_public
REFRESH = "5s"
MAX_CHART_POINTS = 2000
RANGES = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30), "All": None}

st.set_page_config(page_title="pocketchange", page_icon=":moneybag:", layout="wide")
st.markdown(  # the STOP button is always red, whatever the theme
    """<style>
    .st-key-emergency_stop button { background:#d32f2f; border-color:#d32f2f; color:#fff; font-weight:700;
                                    letter-spacing:.12em; }
    .st-key-emergency_stop button:hover { background:#b71c1c; border-color:#b71c1c; color:#fff; }
    .st-key-emergency_stop button:disabled { background:#5c2a2a; border-color:#5c2a2a; color:#d9b3b3; }
    .st-key-resume button { background:#2e7d32; border-color:#2e7d32; color:#fff; font-weight:700; letter-spacing:.12em; }
    .st-key-resume button:hover { background:#1b5e20; border-color:#1b5e20; color:#fff; }
    </style>""",
    unsafe_allow_html=True,
)


def set_paused(paused: bool) -> None:
    if PUBLIC:
        return
    with closing(db.connect(settings.db_path)) as conn:
        db.set_paused(conn, paused)
    st.toast("Trading stopped." if paused else "Trading resumed.", icon=":material/block:" if paused else ":material/play_arrow:")


def set_timeframe() -> None:
    choice = st.session_state.get("timeframe")
    if PUBLIC or choice not in DASHBOARD_TIMEFRAMES:
        return
    with closing(db.connect(settings.db_path)) as conn:
        db.set_timeframe(conn, choice)
    st.toast(f"Trading on {choice} candles from the next cycle.", icon=":material/schedule:")


def position_summary(trades: pd.DataFrame) -> tuple[float, float, float]:
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


def banner(text: str) -> None:
    st.markdown(
        "<div style='background:#c62828;color:#fff;padding:14px 18px;border-radius:6px;"
        f"font-weight:600;margin-bottom:12px'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


@st.fragment(run_every=REFRESH)
def live_view() -> None:
    if not settings.db_path.exists():
        st.info("The bot has not started yet." if PUBLIC else f"No database yet at `{settings.db_path}`. Start the bot with `python -m bot.run`.")
        return
    with closing(db.connect(settings.db_path, readonly=True)) as conn:
        halted, reason = db.is_halted(conn), db.halt_reason(conn)
        paused = db.is_paused(conn)
        timeframe = db.get_timeframe(conn) or TIMEFRAME
        mode_flag = db.get_state(conn, db.DRY_RUN)
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

    title, button = st.columns([5, 1], vertical_alignment="center")
    title.title("pocketchange")
    if not PUBLIC:
        if paused:
            button.button(
                "RESUME", key="resume", on_click=set_paused, args=(False,), disabled=halted, width="stretch",
                help="Trading is stopped. Resume it." if not halted else "Halted: only bot.reset_halt can clear this.",
            )
        else:
            button.button(
                "STOP", key="emergency_stop", on_click=set_paused, args=(True,), disabled=halted, width="stretch",
                help="Emergency stop: pauses trading now. The bot checks this right before any swap.",
            )
    st.divider()
    st.session_state["timeframe"] = timeframe  # always mirror the database
    st.segmented_control(
        "Timeframe",
        DASHBOARD_TIMEFRAMES,
        key="timeframe",
        on_change=set_timeframe,
        disabled=PUBLIC,
        label_visibility="collapsed",
        help="Candle size the bot trades on. Shorter means more trades and more fees.",
    )
    st.divider()

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
