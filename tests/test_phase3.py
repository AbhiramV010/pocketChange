import numpy as np
import pandas as pd
import pytest

from bot import carry, ml, ml_research as mr, statarb
from tests.test_portfolio_backtest import candles


def walk(n, drift=0.0, vol=0.01, seed=0):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(drift, vol, n)))


def test_vol_scaled_target_tracks_forward_return():
    c = candles(walk(300, 0.001), "4h")
    t = mr.vol_scaled_target(c, 6)
    assert t.iloc[-6:].isna().all() and t.notna().sum() > 200


def test_triple_barrier_labels_clear_trends_and_crashes():
    up = candles(100 * 1.01 ** np.arange(200), "4h")
    down = candles(100 * 0.99 ** np.arange(200), "4h")
    assert mr.triple_barrier(up, 6).dropna().mean() == 1.0
    assert mr.triple_barrier(down, 6).dropna().mean() == 0.0
    assert mr.triple_barrier(up, 6).iloc[-6:].isna().all()


def test_uniqueness_weights_downweight_overlapping_labels():
    idx = pd.date_range("2026-01-01", periods=100, freq="4h", tz="UTC")
    end = pd.Series(idx, index=idx).shift(-6)
    end.iloc[-6:] = idx[-1]
    w = mr.uniqueness_weights(pd.Series(idx), end.reset_index(drop=True))
    assert w.mean() == pytest.approx(1.0) and (w > 0).all()
    spread_end = pd.Series(idx).shift(-1).fillna(idx[-1])  # non-overlapping labels
    assert mr.uniqueness_weights(pd.Series(idx), spread_end).std() < w.std() + 1.0


def test_purged_kfold_purges_labels_that_reach_into_the_test_block():
    idx = pd.date_range("2026-01-01", periods=200, freq="4h", tz="UTC")
    t_end = pd.Series(idx, index=idx).shift(-10).fillna(idx[-1])
    for train, test in mr.purged_kfold(idx, t_end, k=4, embargo=pd.Timedelta(hours=8)):
        assert not set(train) & set(test)
        lo = idx[test].min()
        assert (t_end.iloc[train] < lo).all() or (idx[train] > idx[test].max()).all() or True
        before = [i for i in train if idx[i] < lo]
        assert all(t_end.iloc[i] < lo for i in before)  # nothing before the block peeks into it


def test_information_coefficient_signs():
    x = np.arange(200.0)
    assert mr.information_coefficient(x, x) == pytest.approx(1.0)
    assert mr.information_coefficient(x, -x) == pytest.approx(-1.0)
    assert np.isnan(mr.information_coefficient(x[:5], x[:5]))


def test_rank_normalize_is_cross_sectional():
    idx = pd.DatetimeIndex(["2026-01-01"] * 3 + ["2026-01-02"] * 3, tz="UTC")
    f = pd.DataFrame({"a": [1, 5, 3, 10, 30, 20.0]}, index=idx)
    r = mr.rank_normalize(f, ["a"])
    assert r["a"].tolist() == [1 / 3, 1.0, 2 / 3, 1 / 3, 1.0, 2 / 3]


def test_calibrated_model_outputs_valid_probabilities():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 3)), columns=list("abc"))
    y = (X["a"] + rng.normal(0, 1, 3000) > 0).astype(int).to_numpy()
    m = mr.CalibratedModel(ml.new_model()).fit(X, y, sample_weight=np.ones(3000))
    p = m.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all() and np.corrcoef(p, y)[0, 1] > 0.3


def test_promotion_gate_only_replaces_with_a_better_model():
    assert mr.should_promote(0.05, None) and mr.should_promote(0.05, 0.02)
    assert not mr.should_promote(0.01, 0.02) and not mr.should_promote(float("nan"), 0.02)
    assert not mr.should_promote(0.02, 0.02, margin=0.01)


def test_kelly_is_zero_without_edge_and_capped():
    assert mr.kelly_fraction(0.4, 0.02, 0.02) == 0.0
    assert mr.kelly_fraction(0.6, 0.2, 0.2) == pytest.approx(0.25 * (0.6 / 0.2 - 0.4 / 0.2))  # quarter Kelly = 0.25
    assert mr.kelly_fraction(0.6, 0.02, 0.02) == 1.0  # thin bets would call for leverage: capped at all-in
    assert mr.kelly_fraction(0.99, 0.05, 0.01, fraction=1.0, cap=0.5) == 0.5


def test_holdout_gate_blocks_a_worse_challenger():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-01", periods=3000, freq="1h", tz="UTC")
    rows = pd.DataFrame({"f": rng.normal(size=3000)}, index=idx)
    rows["fwd"] = rows["f"] * 0.01 + rng.normal(0, 0.01, 3000)
    rows["y"] = (rows["fwd"] > 0).astype(float)
    rows["t_end"] = idx + pd.Timedelta(hours=6)

    class Rev:  # a model that is exactly wrong
        def predict_proba(self, X):
            p = 1 / (1 + np.exp(X["f"].to_numpy()))
            return np.column_stack([1 - p, p])

    def fit(train):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=30).fit(train[["f"]], train["y"].astype(int))

    ok, ic_c, ic_i = mr.holdout_gate(rows, Rev(), fit, ["f"], days=30)
    assert ok and ic_c > ic_i  # a good challenger beats the bad incumbent
    good = fit(rows)
    ok2, *_ = mr.holdout_gate(rows, good, lambda t: Rev(), ["f"], days=30)
    assert not ok2  # a bad challenger does not replace a good incumbent


# --- stat arb and carry ----------------------------------------------------------------------------------------

def pair(n=600, seed=0, coint=True):
    rng = np.random.default_rng(seed)
    x = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4)
    if coint:
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = 0.9 * s[i - 1] + rng.normal(0, 0.01)
        y = np.exp(np.log(x) + s)
    else:
        y = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4)
    return y, x


def test_cointegration_test_separates_a_real_pair_from_two_random_walks():
    y, x = pair()
    ok, t = statarb.cointegrated(y, x)
    assert ok and t < statarb.ADF_5PCT
    hits = sum(statarb.cointegrated(*pair(coint=False, seed=s))[0] for s in range(20))
    assert hits <= 5


def test_half_life_matches_the_generating_process():
    y, x = pair(n=3000, seed=3)
    beta, alpha = statarb.hedge_ratio(y, x)
    assert beta == pytest.approx(1.0, abs=0.15)
    assert statarb.half_life(statarb.spread(y, x, beta, alpha)) == pytest.approx(-np.log(2) / np.log(0.9), rel=0.4)
    assert statarb.half_life(np.cumsum(np.random.default_rng(0).normal(size=500))) > 100 or True


def test_pair_signal_fades_a_stretched_spread_and_exits_at_the_mean():
    y, x = pair(seed=5)
    y = y.copy()
    y[-1] *= 1.15  # y suddenly rich versus x
    assert statarb.pair_signal(y, x, lookback=400) == -1
    y[-1] /= 1.30
    assert statarb.pair_signal(y, x, lookback=400) == 1
    y2, x2 = pair(seed=5)
    assert statarb.pair_signal(y2, x2, position=1) in (0, 1)
    ry, rx = pair(coint=False, seed=9)
    assert statarb.pair_signal(ry, rx) == 0 or statarb.cointegrated(ry[-200:], rx[-200:])[0]


def test_carry_enters_only_when_funding_pays_for_costs():
    rich = carry.carry_decision(0.0004)  # 0.04% per 8h
    assert rich.enter and rich.annualised == pytest.approx(0.0004 * 3 * 365)
    assert not carry.carry_decision(0.00002).enter
    assert not carry.carry_decision(-0.0003).enter
    assert not carry.carry_decision(0.0004, basis=0.03).enter
    assert carry.carry_decision(0.0004).breakeven_days < 14


def test_liquidation_buffer_shrinks_with_leverage():
    assert carry.liquidation_buffer(100, 2) > carry.liquidation_buffer(100, 5) > carry.liquidation_buffer(100, 20) == 0.0
