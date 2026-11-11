from dataclasses import replace

import numpy as np
import pytest

from bot import targets
from bot.portfolio_backtest import simulate
from bot.strategy import Params, Signal, decide_core, intrabar_hit
from tests.test_portfolio_backtest import TREND, candles

CFG = targets.TargetCfg()


def walk(n, drift, vol=0.01, seed=0):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(drift, vol, n)))


def test_trend_score_signs_and_bounds():
    assert targets.trend_score(walk(300, 0.004, 0.005)) > 0.5
    assert targets.trend_score(walk(300, -0.004, 0.005)) < -0.5
    assert np.isnan(targets.trend_score(walk(50, 0.0)))


def test_regime_score_graded_and_bounded():
    up, down = 100 * 1.002 ** np.arange(400), 100 * 0.998 ** np.arange(400)
    assert targets.regime_score(up, CFG) > 0.9 and targets.regime_score(down, CFG) < 0.05
    assert targets.regime_score(up[:50], CFG) == 0.0


def test_governor_steps_down_with_drawdown():
    s = [targets.governor_scale(d, CFG.dd_steps) for d in (0.0, 0.07, 0.12, 0.4)]
    assert s == [1.0, 0.75, 0.5, 0.25]


def test_target_weights_respect_caps_and_vol_target():
    closes = {a: walk(400, 0.003, 0.01, seed=i) for i, a in enumerate("ABCD")}
    btc = 100 * 1.002 ** np.arange(400)
    w = targets.target_weights(closes, btc, CFG, bars_per_year=2190)
    assert sum(w.values()) <= CFG.max_gross + 1e-9 and max(w.values()) <= CFG.max_weight + 1e-9
    assert sum(1 for v in w.values() if v > 0) <= CFG.top_k
    calm = targets.target_weights({a: walk(400, 0.003, 0.002, seed=i) for i, a in enumerate("ABCD")}, btc, CFG, bars_per_year=2190)
    assert sum(calm.values()) >= sum(w.values())  # lower volatility lets it hold more


def test_target_weights_are_flat_when_btc_is_risk_off_or_drawdown_is_deep():
    closes = {a: walk(400, 0.003, 0.01, seed=i) for i, a in enumerate("ABC")}
    off = targets.target_weights(closes, 100 * 0.998 ** np.arange(400), CFG, bars_per_year=2190)
    assert sum(off.values()) < 0.1
    on = 100 * 1.002 ** np.arange(400)
    full = sum(targets.target_weights(closes, on, CFG, bars_per_year=2190).values())
    deep = sum(targets.target_weights(closes, on, CFG, drawdown=0.5, bars_per_year=2190).values())
    assert deep == pytest.approx(full * 0.25)


def test_band_holds_small_changes_and_always_closes_exits():
    out = targets.apply_band({"A": 0.30, "B": 0.20, "C": 0.0}, {"A": 0.32, "B": 0.0, "C": 0.30}, band=0.05)
    assert out == {"A": 0.30, "B": 0.0, "C": 0.30}


def test_ewma_cov_is_symmetric_psd():
    r = np.random.default_rng(1).normal(0, 0.01, (200, 4))
    c = targets.ewma_cov(r, 30)
    assert np.allclose(c, c.T) and np.linalg.eigvalsh(c).min() > -1e-12


def test_simulate_targets_makes_money_on_a_trend_and_pays_costs():
    d = {"SOL": candles(walk(700, 0.002, 0.004, 1), "4h"), "ETH": candles(walk(700, 0.002, 0.004, 2), "4h")}
    btc = candles(walk(700, 0.002, 0.004, 3), "4h")
    r = targets.simulate_targets(d, btc, replace(CFG, target_vol=0.6, cash_apr=0.0), start_cash=100.0)
    assert r.final > r.start and r.trades > 0 and 0 <= r.max_drawdown < 0.5
    free = targets.simulate_targets(d, btc, replace(CFG, target_vol=0.6, cash_apr=0.0), start_cash=100.0,
                                    fee_rate=0.0, slippage_rate=0.0, fixed_cost=0.0)
    assert free.final > r.final


def test_cash_yield_accrues_when_flat():
    d = {"SOL": candles(walk(700, -0.002, 0.004, 1), "4h")}
    btc = candles(walk(700, -0.002, 0.004, 3), "4h")
    r = targets.simulate_targets(d, btc, replace(CFG, cash_apr=0.10), start_cash=100.0)
    assert r.final > r.start and r.trades == 0


# --- intrabar hard stop -------------------------------------------------------------------------------------

def test_intrabar_hit_uses_the_live_price():
    p = Params(hard_stop_pct=10.0, intrabar=True)
    assert intrabar_hit(89.9, 100.0, p) and not intrabar_hit(95.0, 100.0, p) and not intrabar_hit(None, 100.0, p)
    assert not intrabar_hit(50.0, 100.0, replace(p, intrabar=False))


def test_decide_core_sells_on_the_live_price_before_the_close_shows_it():
    close = np.linspace(80, 100, 120)
    p = Params(fast=5, slow=20, entry="cross", edge_bps=0, confirm=False, stop_atr=0.0, hard_stop_pct=10.0, regime_len=0)
    args = dict(holding=True, entry_price=100.0, bars_held=5)
    assert decide_core(close, close, close, p, **args) is not Signal.SELL
    assert decide_core(close, close, close, p, live_price=88.0, **args) is Signal.SELL


def test_backtest_stop_fills_at_the_level_not_the_close():
    px = 100 * 1.003 ** np.arange(120)  # steady uptrend, so only the crash candle's low can trigger an exit
    crash = candles(px, "4h")
    crash.iloc[70, crash.columns.get_loc("low")] = px[69] * 0.5  # one candle's low crashes, the close recovers
    p = replace(TREND, hard_stop_pct=10.0, regime_len=0, intrabar=True, risk_pct=0.0)
    data = {"SOL": crash, "BTC": crash}
    on = simulate(data, p, fee_rate=0, slippage_rate=0, fixed_cost=0)
    off = simulate(data, replace(p, intrabar=False), fee_rate=0, slippage_rate=0, fixed_cost=0)
    hit = [f for f in on.fills if f[1] == "sell" and f[0] == crash.index[70] and f[2] == "SOL"]
    assert hit and hit[0][4] / hit[0][3] < px[70]  # sold at the stop level, below where the candle closed
    assert not [f for f in off.fills if f[1] == "sell" and f[0] == crash.index[70]]


def test_kept_weights_are_not_traded_even_with_a_large_account():
    d = {"SOL": candles(walk(700, 0.002, 0.004, 1), "4h"), "ETH": candles(walk(700, 0.002, 0.004, 2), "4h")}
    btc = candles(walk(700, 0.002, 0.004, 3), "4h")
    wide = targets.simulate_targets(d, btc, replace(CFG, target_vol=0.6, band=0.9), start_cash=1_000_000.0)
    tight = targets.simulate_targets(d, btc, replace(CFG, target_vol=0.6, band=0.01), start_cash=1_000_000.0)
    assert wide.trades < tight.trades and wide.trades <= 6  # a huge band: little more than the opening buys
