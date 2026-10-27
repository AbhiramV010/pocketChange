import html
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

# `streamlit run dashboard/app.py` puts dashboard/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from bot import db  # noqa: E402
from bot.assets import ASSETS, label  # noqa: E402
from bot.config import DASHBOARD_TIMEFRAMES, TIMEFRAME, settings  # noqa: E402

PUBLIC = settings.dashboard_public
REFRESH = "5s"
MAX_CHART_POINTS = 2000
RANGES = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30), "All": None}
AI = settings.strategy == "ml"
MULTI = len(settings.assets) + len(settings.risky_assets) > 1  # more than the original SOL-only layout

st.set_page_config(page_title="pocketchange", page_icon=":moneybag:", layout="wide")
st.markdown(  # the STOP button is always red and the RISKY switch always yellow, whatever the theme
    """<style>
    .st-key-emergency_stop button { background:#d32f2f; border-color:#d32f2f; color:#fff; font-weight:700;
                                    letter-spacing:.12em; }
    .st-key-emergency_stop button:hover { background:#b71c1c; border-color:#b71c1c; color:#fff; }
    .st-key-emergency_stop button:disabled { background:#5c2a2a; border-color:#5c2a2a; color:#d9b3b3; }
    .st-key-resume button { background:#2e7d32; border-color:#2e7d32; color:#fff; font-weight:700; letter-spacing:.12em; }
    .st-key-resume button:hover { background:#1b5e20; border-color:#1b5e20; color:#fff; }
    .st-key-risky_off button { background:transparent; border:2px solid #fbc02d; color:#fbc02d; font-weight:700;
                               letter-spacing:.12em; }
    .st-key-risky_off button:hover { background:rgba(251,192,45,.16); border-color:#fbc02d; color:#fbc02d; }
    .st-key-risky_on button { background:#fbc02d; border:2px solid #fbc02d; color:#1a1a1a; font-weight:700;
                              letter-spacing:.12em; box-shadow:0 0 14px rgba(251,192,45,.55); }
    .st-key-risky_on button:hover { background:#f9a825; border-color:#f9a825; color:#1a1a1a; }
    .st-key-risky_on button p, .st-key-risky_off button p { font-weight:700; }
    </style>""",
    unsafe_allow_html=True,
)


def set_paused(paused: bool) -> None:
    if PUBLIC:
        return
    with closing(db.connect(settings.db_path)) as conn:
        db.set_paused(conn, paused)
    st.toast("Trading stopped." if paused else "Trading resumed.", icon=":material/block:" if paused else ":material/play_arrow:")


def set_risky(risky: bool) -> None:
    if PUBLIC:
        return
    with closing(db.connect(settings.db_path)) as conn:
        db.set_risky(conn, risky)
    st.toast(
        "RISKY mode ON: memecoins are in play." if risky else "RISKY mode off: top coins only. Open memecoin positions exit.",
        icon=":material/priority_high:",
    )


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


def marks_from(latest) -> dict[str, float]:
    marks = {}
    if latest["price"]:
        marks["SOL"] = latest["price"]
    for asset, (_, price) in db.holdings_of(latest["holdings"] if "holdings" in latest.keys() else None).items():
        if price:
            marks[asset] = price
    return marks


def quantities_from(latest) -> dict[str, float]:
    qty = {"SOL": latest["wsol"]}
    for asset, (amount, _) in db.holdings_of(latest["holdings"] if "holdings" in latest.keys() else None).items():
        qty[asset] = amount
    return qty


def positions_table(trades: pd.DataFrame, latest, scores: dict[str, float] | None = None) -> pd.DataFrame:
    marks, quantities = marks_from(latest), quantities_from(latest)
    rows = []
    for asset in ASSETS:
        qty = quantities.get(asset, 0.0)
        mine = trades[trades["asset"] == asset]
        _, avg_entry, realised = position_summary(mine)
        if qty * marks.get(asset, 0.0) < 0.01 and mine.empty and asset not in (scores or {}):
            continue
        price = marks.get(asset)
        rows.append(
            {
                "asset": label(asset),
                **({"AI p(win)": scores.get(asset)} if scores is not None else {}),
                "qty": qty,
                "price": price,
                "value": qty * price if price else None,
                "avg entry": avg_entry or None,
                "unrealised": (price - avg_entry) * qty if price and qty and avg_entry else 0.0,
                "realised": realised,
            }
        )
    return pd.DataFrame(rows)


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


def sniper_section(snipes: pd.DataFrame, active: bool, totals: dict) -> None:
    st.subheader("Memecoin sniper (simulated)")
    if not active:
        st.caption("Off. Turn on RISKY to start sniping brand-new tokens in simulation. It never touches the wallet.")
        return
    st.caption(
        "Real Jupiter quotes, simulated money: a separate bankroll that never touches the wallet, counted in the totals above. "
        "Entries and exits include a 2% latency haircut per leg."
    )
    win_rate = f"{totals['wins'] / totals['closed']:.0%}" if totals["closed"] else "n/a"
    st.caption(
        f"{totals['open']} open, {totals['closed']} closed, win rate {win_rate}, realised {signed_usd(totals['realised'])}, "
        f"unrealised {signed_usd(totals['unrealised'])}, bankroll free ${totals['free']:,.2f}"
    )
    closed = snipes[snipes["status"] == "closed"]
    opened = snipes[snipes["status"] == "open"]
    now = datetime.now(timezone.utc)
    if len(opened):
        table = pd.DataFrame(
            {
                "token": opened["symbol"],
                "age (min)": ((now - opened["opened_at"]).dt.total_seconds() / 60).round(1),
                "entry $": opened["entry_usdc"],
                "value $": opened["last_value"],
                "P&L %": ((opened["last_value"] / opened["entry_usdc"] - 1) * 100).round(1),
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
    if len(closed):
        table = pd.DataFrame(
            {
                "token": closed["symbol"],
                "closed": closed["closed_at"],
                "entry $": closed["entry_usdc"],
                "exit $": closed["exit_usdc"],
                "P&L %": ((closed["exit_usdc"] / closed["entry_usdc"] - 1) * 100).round(1),
                "reason": closed["reason"],
            }
        ).sort_values("closed", ascending=False)
        st.dataframe(table, hide_index=True, width="stretch")


@st.fragment(run_every=REFRESH)
def live_view() -> None:
    if not settings.db_path.exists():
        st.info("The bot has not started yet." if PUBLIC else f"No database yet at `{settings.db_path}`. Start the bot with `python -m bot.run`.")
        return
    with closing(db.connect(settings.db_path, readonly=True)) as conn:
        halted, reason = db.is_halted(conn), db.halt_reason(conn)
        paused = db.is_paused(conn)
        risky = db.is_risky(conn, settings.risky)
        timeframe = db.get_timeframe(conn) or TIMEFRAME
        mode_flag = db.get_state(conn, db.DRY_RUN)
        trades_all = db.read_trades(conn)
        range_choice = st.session_state.get("range", "7d")
        window = RANGES[range_choice]
        since = datetime.now(timezone.utc) - window if window else None
        balances_all = db.read_balances(conn, since=since)
        latest = db.latest_balance(conn)
        snipes = db.read_snipes(conn)
        ml_scores, ml_info = db.get_ml(conn)
        risk_off = db.get_state(conn, db.RISK_OFF) or ""

    dry_run = settings.dry_run if mode_flag is None else mode_flag == "1"
    # The sniper counts once it has been used (or RISKY is on); until then it is not part of the totals.
    sniper_active = settings.sniper and dry_run and (risky or not snipes.empty)
    sn = sniper_totals(snipes, settings.sniper_bankroll_usdc)
    sniper_equity = sn["free"] + sn["open_value"]

    if halted:
        banner(
            f"BOT HALTED: {reason}. Trading is stopped and the open position was left as-is."
            if PUBLIC
            else f"BOT HALTED: {reason}. Trading is stopped and the open position was left as-is. "
            "Only `python -m bot.reset_halt` can clear this; the dashboard cannot."
        )

    title, risky_col, button = st.columns([5, 1, 1], vertical_alignment="center")
    title.title("pocketchange")
    if not PUBLIC:
        if risky:
            risky_col.button(
                "RISKY ON", key="risky_on", icon=":material/priority_high:", on_click=set_risky, args=(False,),
                width="stretch", help="Memecoins are in play. Click to go back to the top coins only.",
            )
        else:
            risky_col.button(
                "RISKY", key="risky_off", icon=":material/priority_high:", on_click=set_risky, args=(True,),
                width="stretch",
                help="Turn on memecoin trading (BONK, WIF, TRUMP, PENGU and the simulated new-token sniper). "
                "Off: top coins only.",
            )
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
        help="Candle size the bot trades on. Shorter means more trades and more fees: in backtests every setting "
        "below 4h lost money after costs.",
    )
    st.divider()
    if risk_off and not (halted or paused):
        st.info(
            f"Not opening new positions: {risk_off}. Buys resume when the market filter clears; stops and exits "
            "still run.",
            icon=":material/trending_down:",
        )

    trades = trades_all[trades_all["simulated"] == dry_run]
    balances = balances_all[balances_all["simulated"] == dry_run]

    if latest is None:
        st.info("Waiting for the first balance snapshot.")
    else:
        price = latest["price"]
        equity = db.row_equity(latest)
        if MULTI or sniper_active:
            marks, quantities = marks_from(latest), quantities_from(latest)
            invested = sum(q * marks.get(a, 0.0) for a, q in quantities.items())
            cols = st.columns(5)
            if sniper_active:
                total = equity + sniper_equity if equity is not None else None
                cols[0].metric(
                    "Total equity (wallet + sniper)", f"${total:,.2f}" if total is not None else "n/a",
                    f"wallet ${equity:,.2f} + sniper ${sniper_equity:,.2f}" if equity is not None else None,
                    delta_color="off", help=f"Only the wallet counts toward the halt floor: trading halts below ${settings.min_equity_usd:g}",
                )
                cols[1].metric("USDC (wallet + sniper)", f"{latest['usdc'] + sn['free']:,.2f}")
                cols[2].metric("In positions", f"${invested + sn['open_value']:,.2f}", f"sniper ${sn['open_value']:,.2f}", delta_color="off")
            else:
                cols[0].metric("Equity (USDC + positions)", f"${equity:,.2f}" if equity is not None else "n/a",
                               help=f"Trading halts below ${settings.min_equity_usd:g}")
                cols[1].metric("USDC", f"{latest['usdc']:,.2f}")
                cols[2].metric("In positions", f"${invested:,.2f}")
            cols[3].metric("SOL (fees, not traded)", f"{latest['sol']:.5f}")
            cols[4].metric("Mode", "RISKY" if risky else "Top coins")
            st.subheader("Positions")
            table = positions_table(trades, latest, ml_scores if AI else None)
            if table.empty:
                st.caption("No positions yet.")
            else:
                st.dataframe(
                    table, hide_index=True, width="stretch",
                    column_config={
                        "AI p(win)": st.column_config.ProgressColumn(
                            "AI p(win)", min_value=0.0, max_value=1.0, format="%.2f",
                            help=f"The AI's probability that this coin rises more than {settings.ml_target_pct:g}% "
                            f"over the next {settings.ml_horizon} candles. Buys at {settings.ml_enter:g}+, sells at {settings.ml_exit:g} or below.",
                        ),
                        "qty": st.column_config.NumberColumn(format="%.6g"),
                        "price": st.column_config.NumberColumn(format="$%.6g"),
                        "value": st.column_config.NumberColumn(format="$%.2f"),
                        "avg entry": st.column_config.NumberColumn(format="$%.6g"),
                        "unrealised": st.column_config.NumberColumn(format="%+.2f"),
                        "realised": st.column_config.NumberColumn(format="%+.2f"),
                    },
                )
        else:
            wsol_usd = latest["wsol"] * price if price else None
            cols = st.columns(5)
            cols[0].metric("Equity (USDC + wSOL)", f"${equity:,.2f}" if equity is not None else "n/a",
                           help=f"Trading halts below ${settings.min_equity_usd:g}")
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

    if AI:
        if ml_info:
            auc = ml_info.get("oos_auc")
            st.caption(
                f"AI model ({ml_info['timeframe']}): trained {str(ml_info['trained_at'])[:16].replace('T', ' ')} UTC on "
                f"{ml_info['rows']:,} rows"
                + (f", out-of-sample AUC {auc:.3f} (0.5 is a coin flip)" if auc else "")
                + f". Buys at p(win) >= {ml_info['enter']:g}, sells at <= {ml_info['exit']:g}."
            )
        else:
            st.caption("AI model: none for this timeframe yet. It trains in the background on first start (a minute or two).")

    st.subheader("Equity curve")
    st.radio("Range", list(RANGES), index=1, key="range", horizontal=True, label_visibility="collapsed")
    curve = balances.dropna(subset=["price"]).sort_values("timestamp").copy()
    if curve.empty:
        st.caption("No priced balance snapshots in this range yet.")
    else:
        # Plot change vs the start of the range: a $20 wallet moves cents, and a chart
        # of absolute equity would be a flat line squashed against the top.
        curve["equity"] = curve["usdc"] + curve["wsol"] * curve["price"] + curve["holdings"].map(db.holdings_value)
        curve["plotted"] = curve["equity"]
        if sniper_active:
            curve["plotted"] = curve["equity"] + sniper_realised_over_time(snipes, curve["timestamp"], sn["unrealised"])
        curve["change"] = curve["plotted"] - curve["plotted"].iloc[0]
        step = max(1, len(curve) // MAX_CHART_POINTS)
        st.line_chart(curve.iloc[::step].set_index("timestamp")["change"], y_label="Equity change vs start of range (USD)")
        now_equity = curve["equity"].iloc[-1]
        floor_text = f"halt floor ${settings.min_equity_usd:,.2f} | headroom ${now_equity - settings.min_equity_usd:,.2f}"
        if sniper_active:
            st.caption(
                f"Total ${now_equity + sniper_equity:,.2f} (wallet ${now_equity:,.2f} + sniper ${sniper_equity:,.2f}) "
                f"| wallet {floor_text}"
            )
        else:
            st.caption(f"Equity ${now_equity:,.2f} | {floor_text}")

    if settings.sniper and (dry_run or not snipes.empty):
        sniper_section(snipes, sniper_active, sn)

    st.subheader("Trade log")
    if trades.empty:
        st.caption("No trades yet.")
    else:
        log = trades.sort_values("timestamp", ascending=False).copy()
        columns = ["timestamp", "side", "amount_in", "amount_out", "price", "simulated"]
        if MULTI or (trades["asset"] != "SOL").any():
            log["asset"] = log["asset"].map(label)
            columns.insert(1, "asset")
        if not PUBLIC:  # a transaction link reveals the wallet address and its whole history
            log["tx"] = log["tx_hash"].map(lambda h: f"https://solscan.io/tx/{h}" if h else None)
            columns.insert(columns.index("simulated") + 1, "tx")
        st.dataframe(
            log[columns],
            hide_index=True,
            width="stretch",
            column_config={"tx": st.column_config.LinkColumn("tx", display_text="Solscan")},
        )


live_view()
