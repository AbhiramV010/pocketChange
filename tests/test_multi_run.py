from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot import chain, db
from bot.config import load_settings
from bot.run import Context, SimBroker, run_cycle
from bot.strategy import Params

SETUP = "2026-01-01T00:00:00+00:00"  # setup trades predate every candle in these tests

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
TREND = Params(
    fast=5, slow=20, entry="trend", edge_bps=0.0, confirm=False, stop_atr=0.0, hard_stop_pct=0.0,
    reentry_candles=0, regime_len=0,
)
BASE = replace(load_settings({}), assets=("SOL", "BTC", "ETH"), min_equity_usd=1.0)


def frame(closes):
    closes = list(closes) + [closes[-1]]  # the candle still forming
    idx = pd.date_range(START, periods=len(closes), freq="1h", tz="UTC", name="timestamp")
    c = pd.Series(closes, dtype=float, index=idx)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0})


def series(rate, n=80, start=100.0):
    return np.array([start * (1 + rate) ** i for i in range(n)])


class SpyBroker:
    def __init__(self, inner):
        self.inner, self.swaps = inner, []

    def balances(self):
        return self.inner.balances()

    def swap(self, side, amount_in, ref_price, asset="SOL"):
        fill = self.inner.swap(side, amount_in, ref_price, asset)
        self.swaps.append((side, asset))
        return fill


class Market:
    def __init__(self, conn, closes: dict, cfg=BASE, params=TREND, missing=()):
        self.conn, self.cfg = conn, cfg
        self.frames = {a: frame(c) for a, c in closes.items()}
        self.errors: dict[str, Exception] = {}
        self.now = self.frames["SOL"].index[-1].to_pydatetime() + timedelta(minutes=5)
        self.broker = SpyBroker(SimBroker(conn, cfg))
        feeds = {a: self._feed(a) for a in closes if a != "SOL" and a not in missing}
        self.ctx = Context(
            cfg, conn, self.broker, feed=self._feed("SOL"), clock=lambda: self.now, extra_feeds=feeds, params=params
        )

    def _feed(self, asset):
        def feed(timeframe="1h"):
            if asset in self.errors:
                raise self.errors[asset]
            return self.frames[asset]

        return feed

    def cycle(self):
        run_cycle(self.ctx)


def test_each_asset_gets_its_own_position_and_signal_state(conn):
    m = Market(conn, {"SOL": series(0.003), "BTC": series(0.003), "ETH": series(0.003)}, cfg=replace(BASE, max_positions=3))
    m.cycle()
    trades = db.read_trades(conn)
    assert sorted(trades["asset"]) == ["BTC", "ETH", "SOL"] and (trades["amount_in"] == 5.0).all()
    b = db.latest_balance(conn)
    assert b["usdc"] == pytest.approx(5.0) and db.holdings_of(b["holdings"]).keys() == {"BTC", "ETH"}
    for key in (db.DONE_SIGNAL, f"{db.DONE_SIGNAL}:BTC", f"{db.DONE_SIGNAL}:ETH"):
        assert db.get_state(conn, key)
    m.cycle()
    assert len(db.read_trades(conn)) == 3  # same candles: nothing is bought twice


def test_buys_are_ranked_by_momentum_and_capped_by_max_positions(conn):
    cfg = replace(BASE, max_positions=1)
    m = Market(conn, {"SOL": series(0.0), "BTC": series(0.001), "ETH": series(0.005)}, cfg=cfg)
    m.cycle()
    assert m.broker.swaps == [("buy", "ETH")]  # the stronger trend wins the single slot
    m.cycle()
    assert m.broker.swaps == [("buy", "ETH")]


def test_sells_run_before_buys_so_the_freed_cash_is_used(conn):
    cfg = replace(BASE, sim_start_usdc=5.0, max_positions=3)
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.05, 100.0, None), True, asset="SOL", timestamp=SETUP)  # all cash is in SOL
    m = Market(conn, {"SOL": series(-0.004), "BTC": series(0.004), "ETH": series(0.0)}, cfg=cfg)
    m.cycle()
    assert m.broker.swaps == [("sell", "SOL"), ("buy", "BTC")]
    assert db.sim_balances(conn, 5.0, 0.05).qty("BTC") > 0


def test_regime_filter_blocks_new_entries_until_btc_turns_up(conn):
    params = replace(TREND, regime_len=30)
    m = Market(conn, {"SOL": series(0.0), "BTC": series(-0.003), "ETH": series(0.004)}, params=params)
    m.cycle()
    assert m.broker.swaps == []  # ETH trends up but BTC is below its average: risk-off
    m.frames["BTC"] = frame(series(0.003))
    m.cycle()
    assert ("buy", "ETH") in m.broker.swaps


def test_regime_never_blocks_an_exit(conn):
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.05, 100.0, None), True, asset="ETH", timestamp=SETUP)
    params = replace(TREND, regime_len=30)
    m = Market(conn, {"SOL": series(0.0), "BTC": series(-0.003), "ETH": series(-0.004)}, params=params)
    m.cycle()
    assert m.broker.swaps == [("sell", "ETH")]


def test_risky_assets_trade_only_while_the_switch_is_on_but_always_exit(conn):
    cfg = replace(BASE, assets=("SOL",), risky_assets=("BONK",), risky=False)
    m = Market(conn, {"SOL": series(0.0), "BONK": series(0.004)}, cfg=cfg)
    m.cycle()
    assert m.broker.swaps == []  # switch off: top coins only
    db.set_risky(conn, True)
    m.cycle()
    assert m.broker.swaps == [("buy", "BONK")]
    db.set_risky(conn, False)
    m.now += timedelta(hours=1)  # a candle later the memecoin has reversed, with the switch off
    m.frames.update(SOL=frame(series(0.0, n=81)), BONK=frame(series(-0.01, n=81)))
    m.cycle()
    assert m.broker.swaps == [("buy", "BONK"), ("sell", "BONK")]  # the exit still runs; no position is stranded
    m.now += timedelta(hours=1)
    m.frames.update(SOL=frame(series(0.0, n=82)), BONK=frame(series(0.01, n=82)))
    m.cycle()
    assert len(m.broker.swaps) == 2  # and nothing is re-bought


def test_hard_stop_cuts_a_holding_even_when_the_trend_is_up(conn):
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.0166, 300.0, None), True, asset="BTC", timestamp=SETUP)  # bought at 300
    params = replace(TREND, hard_stop_pct=10.0)
    m = Market(conn, {"SOL": series(0.0), "BTC": series(0.004, start=100.0), "ETH": series(0.0)}, params=params)
    m.cycle()
    assert m.broker.swaps == [("sell", "BTC")]


def test_equity_floor_counts_every_position(conn):
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.05, 100.0, None), True, asset="BTC", timestamp=SETUP)  # worth $5 at entry
    cfg = replace(BASE, min_equity_usd=19.0)
    m = Market(conn, {"SOL": series(0.0), "BTC": series(-0.03), "ETH": series(0.0)}, cfg=cfg)
    m.cycle()  # BTC has collapsed, so equity is 15 + a little
    assert db.is_halted(conn) and "below MIN_EQUITY_USD" in db.halt_reason(conn)
    assert m.broker.swaps == []


def test_a_missing_or_failing_feed_only_skips_that_asset(conn):
    m = Market(conn, {"SOL": series(0.004), "BTC": series(0.004), "ETH": series(0.004)}, missing=("ETH",))
    m.errors["BTC"] = ConnectionError("exchange down")
    m.cycle()
    assert m.broker.swaps == [("buy", "SOL")]
    assert not db.is_halted(conn)


def test_swap_errors_are_per_asset_and_retried_once(conn):
    class Flaky(SpyBroker):
        def swap(self, side, amount_in, ref_price, asset="SOL"):
            if asset == "BTC":
                self.swaps.append(("attempt", asset))
                raise chain.SwapFailed("reverted")
            return super().swap(side, amount_in, ref_price, asset)

    m = Market(conn, {"SOL": series(0.004), "BTC": series(0.004), "ETH": series(0.004)})
    m.broker = Flaky(SimBroker(conn, BASE))
    m.ctx.broker = m.broker
    m.cycle()
    attempts = [s for s in m.broker.swaps if s[0] == "attempt"]
    assert len(attempts) == 2  # one try plus exactly one retry for BTC
    assert {a for s, a in m.broker.swaps if s == "buy"} == {"SOL", "ETH"}  # the others still filled


def test_the_sniper_runs_after_each_cycle_and_can_never_break_it(conn):
    from bot.run import run_loop

    class Boom:
        calls = 0

        def step(self):
            Boom.calls += 1
            raise RuntimeError("sniper bug")

    m = Market(conn, {"SOL": series(0.004), "BTC": series(0.004), "ETH": series(0.0)})
    m.ctx.sniper = Boom()
    run_loop(m.ctx, once=True)
    assert Boom.calls == 1 and ("buy", "SOL") in m.broker.swaps  # the main strategy traded regardless
