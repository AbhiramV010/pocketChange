"""changes.md Phase 0: fixed costs, state-based exits, the timeframe freeze and the retired defaults."""
import numpy as np
import pytest

from bot import chain, db, portfolio_backtest
from bot.config import DASHBOARD_TIMEFRAMES, load_settings
from bot.run import SimBroker
from bot.strategy import Params, Signal, decide_core, sma_signal
from tests.test_run import CFG, SETUP, Rig, candle_frame, sell_signal_series

CROSS = Params(
    fast=20, slow=50, entry="cross", edge_bps=0.0, confirm=False, stop_atr=0.0, hard_stop_pct=0.0, regime_len=0,
    risk_pct=0.0,
)


def test_cross_mode_exits_on_any_candle_below_not_only_the_cross():
    crossed = sell_signal_series()[:-1]  # ends on the candle where fast crossed under slow
    late = np.array(crossed + [crossed[-1] - 1, crossed[-1] - 2, crossed[-1] - 3])  # the bot missed that candle
    assert sma_signal(late, CROSS) is Signal.HOLD  # no fresh cross to act on
    assert decide_core(late, late, late, CROSS, holding=True, entry_price=late[-1]) is Signal.SELL
    assert decide_core(late, late, late, CROSS, holding=False) is Signal.HOLD  # flat: nothing to exit, no entry


def test_a_held_position_is_kept_while_fast_is_above_slow():
    up = np.linspace(100, 200, 80)
    assert decide_core(up, up, up, CROSS, holding=True, entry_price=100.0) is not Signal.SELL


def test_sim_broker_charges_the_fixed_cost_on_both_legs(conn, monkeypatch):
    monkeypatch.setattr("bot.run.FIXED_COST_USD", 0.05)
    sim = SimBroker(conn, CFG)
    buy = sim.swap("buy", 5.0, 100.0)
    assert buy.amount_out == pytest.approx(4.95 * 0.997 / 100.1)
    db.insert_trade(conn, "buy", buy, simulated=True, timestamp=SETUP)
    sell = sim.swap("sell", buy.amount_out, 100.0)
    assert sell.amount_out == pytest.approx(buy.amount_out * 100.0 * 0.999 * 0.997 - 0.05)
    with pytest.raises(chain.SwapFailed):
        sim.swap("buy", 0.05, 100.0)  # the whole order would go to the network


def test_backtest_fixed_cost_is_paid_per_fill():
    closes = list(np.linspace(200, 100, 60)) + list(np.linspace(100, 300, 60)) + list(np.linspace(300, 100, 60))
    frames = {"SOL": candle_frame(closes)}
    free = portfolio_backtest.simulate(frames, CROSS, max_positions=1, fixed_cost=0.0)
    costly = portfolio_backtest.simulate(frames, CROSS, max_positions=1, fixed_cost=0.05)
    assert free.trades == costly.trades >= 2  # one round trip at least, same decisions either way
    assert costly.final < free.final
    hold_free = portfolio_backtest.buy_and_hold(frames, fixed_cost=0.0)
    hold_costly = portfolio_backtest.buy_and_hold(frames, fixed_cost=0.05)
    assert hold_costly.final < hold_free.final


def test_timeframe_switch_waits_until_the_bot_is_flat(conn):
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.05, 100.0, None), simulated=True, timestamp=SETUP)  # a $5 position
    rig = Rig(conn, [100.0] * 72)  # flat market: no signal, the position just sits there
    seen = []
    original = rig._feed
    rig.ctx.feed = lambda tf: seen.append(tf) or original(tf)
    rig.cycle()
    db.set_timeframe(conn, "4h")
    rig.cycle()
    assert seen == ["1h", "1h"]  # still holding: the old candles keep deciding
    assert db.get_state(conn, db.ACTIVE_TIMEFRAME) == "1h" and db.get_timeframe(conn) == "4h"
    db.insert_trade(conn, "sell", db.Fill(0.05, 5.0, 100.0, None), simulated=True, timestamp=SETUP)  # now flat
    rig.cycle()
    assert seen[-1] == "4h" and db.get_state(conn, db.ACTIVE_TIMEFRAME) == "4h"


def test_retired_defaults():
    assert load_settings({}).sniper is False  # opt-in only
    assert load_settings({"SNIPER": "true"}).sniper is True
    assert DASHBOARD_TIMEFRAMES == ("4h",)  # nothing under 4h on the switch
