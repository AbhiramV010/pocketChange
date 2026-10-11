from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot import chain, db
from bot.config import load_settings
from bot.run import Context, SimBroker, run_cycle, run_loop
from bot.strategy import Signal, sma_crossover_signal

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
CFG = replace(load_settings({}), min_equity_usd=15.0)


def candle_frame(closes, start=START):
    idx = pd.date_range(start, periods=len(closes), freq="1h", tz="UTC", name="timestamp")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0})


def buy_signal_series():
    """Closes where the last *closed* candle is an SMA cross-up, plus one forming candle."""
    closes = list(np.linspace(200, 100, 60)) + list(np.linspace(100, 300, 60))
    for i in range(60, len(closes)):
        if sma_crossover_signal(candle_frame(closes[: i + 1])) is Signal.BUY:
            return closes[: i + 1] + [closes[i]]  # forming candle repeats the last close
    raise AssertionError("no cross found")


def sell_signal_series():
    closes = list(np.linspace(100, 200, 60)) + list(np.linspace(200, 50, 60))
    for i in range(60, len(closes)):
        if sma_crossover_signal(candle_frame(closes[: i + 1])) is Signal.SELL:
            return closes[: i + 1] + [closes[i]]
    raise AssertionError("no cross found")


class Rig:
    """A Context wired to an in-memory-ish DB, a scripted feed and a fixed clock."""

    def __init__(self, conn, closes, cfg=CFG, broker=None):
        self.frame = candle_frame(closes)
        self.now = self.frame.index[-1].to_pydatetime() + timedelta(minutes=5)  # forming candle just opened
        self.feed_calls = 0
        self.feed_error: Exception | None = None
        self.ctx = Context(cfg, conn, broker or SimBroker(conn, cfg), feed=self._feed, clock=lambda: self.now)

    def _feed(self):
        self.feed_calls += 1
        if self.feed_error:
            raise self.feed_error
        return self.frame

    def cycle(self):
        run_cycle(self.ctx)


def trades(conn):
    return db.read_trades(conn)


# --- trading -----------------------------------------------------------------

def test_sma_buy_in_dry_run_is_simulated_and_not_repeated(conn):
    rig = Rig(conn, buy_signal_series())
    rig.cycle()
    t = trades(conn)
    assert len(t) == 1 and t.loc[0, "side"] == "buy" and bool(t.loc[0, "simulated"])
    assert t.loc[0, "amount_in"] == CFG.max_trade_usdc and t.loc[0, "tx_hash"] is None
    rig.cycle()
    rig.cycle()
    assert len(trades(conn)) == 1  # same signal candle: never traded twice
    b = db.latest_balance(conn)
    assert b["usdc"] == pytest.approx(CFG.sim_start_usdc - CFG.max_trade_usdc) and b["wsol"] > 0
    assert b["price"] is not None and db.get_state(conn, db.DRY_RUN) == "1"


def test_sma_sell_exits_in_capped_chunks_until_dust(conn):
    closes = sell_signal_series()
    price = closes[-1]
    db.insert_trade(conn, "buy", db.Fill(15.0, 15.0 / price, price, None), simulated=True)  # ~$15 position
    rig = Rig(conn, closes)
    for _ in range(6):
        rig.cycle()
    t = trades(conn)
    sells = t[t["side"] == "sell"]
    assert len(sells) == 3  # $5 + $5 + remainder, one chunk per cycle
    assert (sells["amount_out"] <= CFG.max_trade_usdc).all()
    assert db.latest_balance(conn)["wsol"] * price < 1.0  # only dust left
    rig.cycle()
    assert len(trades(conn)) == len(t)  # signal satisfied: nothing more to do


def test_dca_buys_on_interval_only(conn):
    cfg = replace(CFG, strategy="dca", dca_amount_usdc=2.0, dca_interval_hours=24)
    rig = Rig(conn, [100.0] * 72, cfg=cfg)
    rig.cycle()
    assert len(trades(conn)) == 1 and trades(conn).loc[0, "amount_in"] == 2.0
    rig.cycle()
    assert len(trades(conn)) == 1
    rig.now += timedelta(hours=24)
    rig.frame = candle_frame([100.0] * 96)
    rig.cycle()
    assert len(trades(conn)) == 2


def test_trade_sizes_never_exceed_cap(conn):
    cfg = replace(CFG, max_trade_usdc=3.0)
    Rig(conn, buy_signal_series(), cfg=cfg).cycle()
    assert trades(conn)["amount_in"].max() <= 3.0


# --- pause and halt -------------------------------------------------------------

def test_pause_blocks_trading_but_keeps_logging_balances(conn):
    rig = Rig(conn, buy_signal_series())
    db.set_paused(conn, True)
    rig.cycle()
    rig.cycle()
    assert len(trades(conn)) == 0
    assert len(db.read_balances(conn)) == 2
    db.set_paused(conn, False)  # resume: trades on the next cycle
    rig.cycle()
    assert len(trades(conn)) == 1


def test_pause_flipped_mid_cycle_still_stops_the_swap(conn):
    rig = Rig(conn, buy_signal_series())
    original = rig._feed

    def feed_then_pause():
        frame = original()
        db.set_paused(conn, True)  # user clicks pause after the top-of-cycle check
        return frame

    rig.ctx.feed = feed_then_pause
    rig.cycle()
    assert len(trades(conn)) == 0


def test_equity_below_floor_halts_and_stays_halted_across_restart(conn):
    cfg = replace(CFG, min_equity_usd=25.0)  # simulated equity is 20
    rig = Rig(conn, buy_signal_series(), cfg=cfg)
    rig.cycle()
    assert db.is_halted(conn) and "below MIN_EQUITY_USD" in db.halt_reason(conn)
    assert len(trades(conn)) == 0

    restarted = Rig(conn, buy_signal_series(), cfg=replace(CFG, min_equity_usd=1.0))  # new process, floor now fine
    restarted.cycle()
    assert db.is_halted(conn) and len(trades(conn)) == 0  # still halted until reset_halt

    db.clear_halt(conn)
    restarted.cycle()
    assert not db.is_halted(conn) and len(trades(conn)) == 1


def test_halt_does_not_sell_the_open_position(conn):
    db.insert_trade(conn, "buy", db.Fill(10.0, 0.1, 100.0, None), simulated=True)
    cfg = replace(CFG, min_equity_usd=1000.0)
    Rig(conn, sell_signal_series(), cfg=cfg).cycle()
    assert db.is_halted(conn)
    assert trades(conn)["side"].tolist() == ["buy"]  # no sell despite a SELL signal


def test_floor_is_checked_right_after_a_trade(conn):
    # equity 20.00 before; fees+slippage on a $5 buy leave ~19.98, below a 19.99 floor
    cfg = replace(CFG, min_equity_usd=19.99)
    rig = Rig(conn, buy_signal_series(), cfg=cfg)
    rig.cycle()
    assert len(trades(conn)) == 1
    assert db.is_halted(conn)
    rig.cycle()
    assert len(trades(conn)) == 1


def test_native_sol_is_not_counted_in_equity(conn):
    cfg = replace(CFG, sim_start_usdc=10.0, sim_start_sol=5.0, min_equity_usd=15.0)  # lots of SOL, little USDC
    Rig(conn, [100.0] * 72, cfg=cfg).cycle()
    assert db.is_halted(conn)


# --- failures --------------------------------------------------------------------

class FailingBroker:
    def __init__(self, inner, error):
        self.inner, self.error, self.swaps = inner, error, 0

    def balances(self):
        return self.inner.balances()

    def swap(self, side, amount_in, ref_price):
        self.swaps += 1
        raise self.error


def test_failed_swap_retried_at_most_once_per_cycle(conn):
    broker = FailingBroker(SimBroker(conn, CFG), chain.SwapFailed("reverted"))
    rig = Rig(conn, buy_signal_series(), broker=broker)
    rig.cycle()
    assert broker.swaps == 2  # original + exactly one retry
    assert len(trades(conn)) == 0
    for _ in range(5):
        rig.cycle()
    assert broker.swaps == 6  # three cycles of 2 attempts, then the signal is abandoned


def test_non_retryable_chain_error_is_not_retried_in_cycle(conn):
    broker = FailingBroker(SimBroker(conn, CFG), chain.ChainError("rpc down"))
    rig = Rig(conn, buy_signal_series(), broker=broker)
    rig.cycle()
    assert broker.swaps == 1 and not db.is_halted(conn)


def test_uncertain_swap_halts_and_is_never_retried(conn):
    broker = FailingBroker(SimBroker(conn, CFG), chain.SwapUncertain("tx 5abc not confirmed"))
    rig = Rig(conn, buy_signal_series(), broker=broker)
    rig.cycle()
    rig.cycle()
    assert broker.swaps == 1
    assert db.is_halted(conn) and "outcome unknown" in db.halt_reason(conn)


def test_feed_failure_logs_balances_without_price_and_does_not_trade(conn):
    rig = Rig(conn, buy_signal_series())
    rig.feed_error = ConnectionError("exchange down")
    rig.cycle()
    assert len(trades(conn)) == 0
    assert db.latest_balance(conn)["price"] is None


def test_stale_feed_does_not_trade_or_halt(conn):
    rig = Rig(conn, buy_signal_series())
    rig.now += timedelta(hours=5)
    rig.cycle()
    assert len(trades(conn)) == 0 and not db.is_halted(conn)


def test_broker_error_propagates_to_loop_which_survives(conn):
    class Boom(SimBroker):
        calls = 0

        def balances(self):
            Boom.calls += 1
            if Boom.calls == 1:
                raise chain.ChainError("rpc timeout")
            return super().balances()

    rig = Rig(conn, buy_signal_series(), broker=Boom(conn, CFG))
    with pytest.raises(chain.ChainError):
        rig.cycle()
    run_loop(rig.ctx, once=True)  # the loop catches, logs and returns instead of crashing
    run_loop(rig.ctx, once=True)
    assert len(trades(conn)) == 1


def test_low_native_sol_blocks_trade(conn):
    cfg = replace(CFG, sim_start_sol=0.0001, gas_reserve_sol=0.0005)
    Rig(conn, buy_signal_series(), cfg=cfg).cycle()
    assert len(trades(conn)) == 0


# --- simulator -----------------------------------------------------------------------

def test_sim_broker_charges_costs_and_enforces_limits(conn):
    sim = SimBroker(conn, CFG)
    fill = sim.swap("buy", 5.0, 2500.0)
    assert fill.amount_out == pytest.approx(5.0 * 0.997 / 2502.5)
    with pytest.raises(chain.SwapFailed):
        sim.swap("buy", CFG.max_trade_usdc + 1, 2500.0)
    with pytest.raises(chain.SwapFailed):
        sim.swap("sell", 1.0, 2500.0)  # no wSOL
