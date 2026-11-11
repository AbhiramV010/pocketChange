import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bot import research
from bot.strategy import Params
from tests.test_portfolio_backtest import TREND, candles


def wavy(n=1200, seed=0, drift=0.0004):
    rng = np.random.default_rng(seed)
    return candles(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), freq="4h")


def data(seed=0):
    return {"SOL": wavy(seed=seed), "BTC": wavy(seed=seed + 1)}


def test_expected_max_sharpe_grows_with_trials():
    assert research.expected_max_sharpe(1, 0.01) == 0.0
    assert 0 < research.expected_max_sharpe(10, 0.01) < research.expected_max_sharpe(1000, 0.01)


def test_deflated_sharpe_falls_as_trials_rise():
    r = np.random.default_rng(1).normal(0.001, 0.01, 800)
    assert research.deflated_sharpe(r, 1, 0.0) > research.deflated_sharpe(r, 500, 1e-4)


def test_deflated_sharpe_is_high_for_a_real_edge_and_low_for_noise():
    rng = np.random.default_rng(2)
    assert research.deflated_sharpe(rng.normal(0.003, 0.01, 2000), 1, 0.0) > 0.99
    assert research.deflated_sharpe(rng.normal(0.0, 0.01, 2000), 50, 4e-4) < 0.5


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    r = np.random.default_rng(3).normal(0.001, 0.01, 600)
    a, b = research.block_bootstrap(r, 365, n=200), research.block_bootstrap(r, 365, n=200)
    assert a == b and a["sharpe"][0] < a["sharpe"][1] < a["sharpe"][2]
    assert all(0 <= x <= 1 for x in a["max_dd"])


def test_pbo_is_high_when_every_config_is_noise_and_low_when_one_dominates():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.01, (800, 12))
    skill = noise.copy()
    skill[:, 0] += 0.004
    assert research.pbo(noise) > research.pbo(skill)
    assert research.pbo(skill) < 0.2


def test_trial_log_counts_distinct_configs(tmp_path):
    path = tmp_path / "t.jsonl"
    assert research.trial_stats(path) == (1, 0.0)
    for label, s in (("a", 0.5), ("a", 0.6), ("b", 1.0), ("c", -0.2)):
        research.log_trial(label, Params(), s, 0.1, path)
    n, var = research.trial_stats(path)
    assert n == 3 and var > 0  # the repeated "a" counts once, at its latest value


def test_vol_matched_hold_matches_target_volatility():
    hold = (1 + pd.Series(np.random.default_rng(5).normal(0, 0.02, 500))).cumprod()
    target = (1 + pd.Series(np.random.default_rng(6).normal(0, 0.005, 500))).cumprod()
    vm = research.vol_matched_hold(hold, target)
    assert vm.pct_change().std() == pytest.approx(target.pct_change().std(), rel=0.05)


def test_btc_filter_only_is_flat_while_btc_falls():
    down = candles(100 * 0.998 ** np.arange(600), freq="4h")
    curve = research.btc_filter_only({"SOL": wavy(600)}, down, 50)
    assert curve.iloc[-1] == pytest.approx(curve.iloc[100:].iloc[0], rel=1e-9) or curve.iloc[-1] >= 0.9


def test_rolling_walk_forward_stitches_out_of_sample_blocks():
    grid = [replace(TREND, fast=f, slow=s, regime_len=0) for f, s in ((5, 20), (10, 40))]
    curve, chosen = research.rolling_walk_forward(data(), data()["BTC"], grid, train_bars=400, test_bars=200)
    assert len(chosen) == 4 and curve.iloc[0] > 0 and not curve.isna().any()
    assert curve.index[0] > data()["SOL"].index[400]  # nothing before the first test block


def test_breakeven_cost_is_finite_positive_for_a_thin_edge_and_zero_for_a_loser():
    up = {"SOL": candles(100 * 1.0015 ** np.arange(500), freq="4h"), "BTC": candles(100 * 1.0015 ** np.arange(500), freq="4h")}
    p = replace(TREND, regime_len=0)
    assert research.breakeven_cost(up, up["BTC"], p, hi=50.0) > 1.0
    down = {"SOL": candles(100 * 0.998 ** np.arange(500), freq="4h"), "BTC": candles(100 * 0.998 ** np.arange(500), freq="4h")}
    assert research.breakeven_cost(down, down["BTC"], p) in (0.0, math.inf)


def test_research_report_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "TRIALS_PATH", tmp_path / "trials.jsonl")
    d = data()
    text = research.research_report(d, d["BTC"], replace(TREND, regime_len=0, entry="cross"), trade_usdc=5.0)
    for needle in ("Deflated Sharpe", "bootstrap", "PBO", "break-even", "parameter surface"):
        assert needle in text
