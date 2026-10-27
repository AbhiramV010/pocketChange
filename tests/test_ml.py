import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot import db, ml
from bot.config import load_settings
from bot.run import Context, SimBroker, run_cycle
from bot.strategy import Params, Signal

SETUP = "2026-01-01T00:00:00+00:00"  # setup trades predate every candle in these tests

CFG = replace(load_settings({}), ml_horizon=6, ml_target_pct=0.4, ml_enter=0.6, ml_exit=0.45)


def walk(n=400, seed=0, persistence=0.0, vol=0.004, start="2026-01-01"):
    """Hourly OHLCV from a random walk; persistence > 0 makes returns trend (something learnable)."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = persistence * ret[i - 1] + rng.normal(0, vol)
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="timestamp")
    c = pd.Series(close, index=idx)
    spread = c * rng.uniform(0.001, 0.004, n)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c + spread, "low": c - spread,
                         "close": c, "volume": rng.uniform(50, 150, n)})


def with_forming_candle(f):
    last = f.iloc[[-1]].copy()
    last.index = last.index + pd.Timedelta(hours=1)
    return pd.concat([f, last])


def market(n=400, persistence=0.0, assets=("SOL", "BTC", "ETH")):
    return {a: walk(n, seed=i, persistence=persistence) for i, a in enumerate(assets)}


# --- features and labels -------------------------------------------------------------------------------------

def test_features_never_look_ahead():
    c, btc = walk(300, seed=1), walk(300, seed=2)
    before = ml.features(c, btc)
    c2, btc2 = c.copy(), btc.copy()
    c2.iloc[-20:] *= 1.5  # rewrite the future of row 279
    btc2.iloc[-20:] *= 0.5
    after = ml.features(c2, btc2)
    pd.testing.assert_frame_equal(before.iloc[:-20], after.iloc[:-20])


def test_feature_rows_are_complete_after_warmup_and_names_are_stable():
    c, btc = walk(300, seed=1), walk(300, seed=2)
    f = ml.features(c, btc)
    assert list(f.columns) == ml.FEATURE_NAMES
    assert f.iloc[:10].isna().any(axis=1).all() and not f.iloc[ml.MIN_CANDLES:].isna().any().any()
    assert not np.isinf(f.to_numpy(float)).any()
    no_bench = ml.features(c, None)
    assert no_bench["btc_ret1"].isna().all()  # missing market context shows up as NaN, never as a made-up value


def test_labels_mark_moves_bigger_than_the_target():
    idx = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")
    c = pd.DataFrame({"close": [100.0, 100.0, 102.0, 100.0, 100.0, 100.0]}, index=idx)
    y = ml.labels(c, ml.MLParams(horizon=2, target_pct=1.0))
    assert list(y.iloc[:4]) == [1.0, 0.0, 0.0, 0.0] and y.iloc[4:].isna().all()  # t=0 sees 102; the last 2 are unknown


@pytest.mark.parametrize(
    "prob, holding, bars, expected",
    [
        (0.7, False, 0, Signal.BUY), (0.59, False, 0, Signal.HOLD), (None, False, 0, Signal.HOLD),
        (0.5, True, 2, Signal.HOLD), (0.45, True, 2, Signal.SELL), (0.3, True, 0, Signal.SELL),
        (0.55, True, 6, Signal.SELL),  # the horizon passed and the AI no longer loves it
        (0.7, True, 9, Signal.HOLD),  # it still loves it: keep holding
        (None, True, 9, Signal.HOLD),
    ],
)
def test_ml_signal_rules(prob, holding, bars, expected):
    assert ml.ml_signal(prob, holding, bars, ml.MLParams(horizon=6, enter=0.6, exit=0.45)) is expected


# --- the model ----------------------------------------------------------------------------------------------

def test_model_learns_a_planted_pattern_and_bundle_handles_missing_features():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(rng.normal(size=(4000, len(ml.FEATURE_NAMES))), columns=ml.FEATURE_NAMES)
    rows["y"] = (rows["ret3"] + 0.5 * rows["rsi14"] > 0).astype(float)
    bundle = ml.Bundle(ml.fit(rows), ml.MLParams(), "1h", "2026-01-01T00:00:00+00:00", len(rows))
    probs = bundle.proba_series(rows.iloc[:500])
    assert ((probs > 0.5) == (rows["y"].iloc[:500] == 1)).mean() > 0.9
    row = rows.iloc[0].copy()
    assert 0.0 <= bundle.proba(row) <= 1.0
    row["ret3"] = np.nan
    assert bundle.proba(row) is None


def test_model_roundtrips_through_disk_and_rejects_a_stale_feature_set(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "MODEL_DIR", tmp_path)
    frames = market(500, persistence=0.3)
    bundle = ml.train(frames, ml.MLParams(), "1h", oos_auc=0.53)
    ml.save(bundle)
    again = ml.load("1h")
    assert again is not None and again.rows == bundle.rows and again.oos_auc == 0.53 and again.params == bundle.params
    f = ml.features(frames["SOL"], frames["BTC"]).iloc[-1]
    assert again.proba(f) == pytest.approx(bundle.proba(f))
    assert ml.load("5m") is None
    monkeypatch.setattr(ml, "FEATURE_NAMES", ml.FEATURE_NAMES[:-1])
    assert ml.load("1h") is None  # a model trained on other features is never used


def test_walk_forward_never_uses_the_future():
    p = ml.MLParams(horizon=6)
    frames = market(3200, persistence=0.3)
    full, summary = ml.walk_forward(frames, p, "1h", min_train_days=100, test_days=15)
    block_end = summary["start"] + pd.Timedelta(days=15)
    cutoff = block_end + pd.Timedelta(hours=p.horizon)
    cut = {a: df[df.index <= cutoff] for a, df in frames.items()}
    truncated, _ = ml.walk_forward(cut, p, "1h", min_train_days=100, test_days=15)
    for asset in frames:
        a = full[asset][(full[asset].index >= summary["start"]) & (full[asset].index < block_end)]
        b = truncated[asset].reindex(a.index)
        assert a.notna().sum() > 100
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)  # same answers without the later data


def test_walk_forward_finds_an_edge_that_is_really_there_and_not_one_that_is_not():
    p = ml.MLParams(horizon=6)
    _, trending = ml.walk_forward(market(3200, persistence=0.6), p, "1h", min_train_days=100, test_days=15)
    _, random_walk = ml.walk_forward(market(3200, persistence=0.0), p, "1h", min_train_days=100, test_days=15)
    assert trending["folds"] >= 2 and trending["auc"] > 0.58  # strongly trending prices: clearly learnable
    assert 0.46 < random_walk["auc"] < 0.54  # pure noise: no edge to find


# --- keeping the model fresh ----------------------------------------------------------------------------------

def wait_idle(manager, seconds=60):
    deadline = time.time() + seconds
    while manager._training and time.time() < deadline:
        time.sleep(0.05)
    assert not manager._training


def test_model_manager_trains_in_the_background_and_only_when_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "MODEL_DIR", tmp_path)
    calls = []

    def history(timeframe):
        calls.append(timeframe)
        return market(600, persistence=0.3)

    manager = ml.ModelManager(CFG, history=history)
    assert manager.get("1h") is None
    now = datetime.now(timezone.utc)
    manager.maybe_retrain("1h", now)
    wait_idle(manager)
    bundle = manager.get("1h")
    assert bundle is not None and calls == ["1h"] and ml.model_path("1h").exists()

    manager.maybe_retrain("1h", now + timedelta(hours=1))
    wait_idle(manager)
    assert calls == ["1h"]  # fresh: nothing to do
    manager.maybe_retrain("1h", now + timedelta(hours=CFG.ml_retrain_hours + 1))
    wait_idle(manager)
    assert calls == ["1h", "1h"]  # a day old: retrained
    manager._cfg = replace(CFG, ml_horizon=12)
    manager.maybe_retrain("1h", now + timedelta(hours=2))
    wait_idle(manager)
    assert calls == ["1h", "1h", "1h"] and manager.get("1h").params.horizon == 12  # settings changed: retrained


def test_a_failed_retrain_keeps_the_old_model(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "MODEL_DIR", tmp_path)
    old = ml.train(market(500, persistence=0.3), ml.MLParams(), "1h")
    ml.save(old)

    def broken(timeframe):
        raise ConnectionError("exchange down")

    manager = ml.ModelManager(CFG, history=broken)
    manager.maybe_retrain("1h", datetime.now(timezone.utc) + timedelta(days=3))
    wait_idle(manager)
    assert manager.get("1h").trained_at == old.trained_at


# --- inside the trading loop ------------------------------------------------------------------------------------

class Scripted:
    """A stand-in model: returns the next scripted probability for each call, in asset order."""

    def __init__(self, probs):
        self.probs, self.calls = list(probs), 0

    def predict_proba(self, X):
        p = self.probs[self.calls % len(self.probs)]
        self.calls += 1
        return np.array([[1 - p, p]])


class Models:
    def __init__(self, probs, horizon=6):
        self.bundle = None if probs is None else ml.Bundle(
            Scripted(probs), ml.MLParams(horizon=horizon), "1h", "2026-01-01T00:00:00+00:00", 1000, 0.53
        )

    def get(self, timeframe):
        return self.bundle

    def maybe_retrain(self, timeframe, now):
        pass


class Rig:
    def __init__(self, conn, probs, cfg=None, assets=("SOL", "BTC", "ETH")):
        self.conn = conn
        self.cfg = cfg or replace(CFG, strategy="ml", assets=assets, max_positions=3, min_equity_usd=1.0)
        self.frames = {}
        for i, a in enumerate(assets):
            self.frames[a] = with_forming_candle(walk(260, seed=i))
        self.now = self.frames["SOL"].index[-1].to_pydatetime() + timedelta(minutes=5)
        self.swaps = []
        broker = SimBroker(conn, self.cfg)
        swaps = self.swaps
        inner_swap = broker.swap

        def spy(side, amount_in, ref_price, asset="SOL"):
            fill = inner_swap(side, amount_in, ref_price, asset)
            swaps.append((side, asset))
            return fill

        broker.swap = spy
        self.models = Models(probs)
        feeds = {a: (lambda tf, a=a: self.frames[a]) for a in assets if a != "SOL"}
        params = Params(fast=9, slow=21, stop_atr=0.0, hard_stop_pct=0.0, regime_len=0)
        self.ctx = Context(self.cfg, conn, broker, feed=lambda tf: self.frames["SOL"], clock=lambda: self.now,
                           extra_feeds=feeds, params=params, models=self.models)

    def cycle(self):
        run_cycle(self.ctx)


def test_the_ai_buys_what_it_likes_best_first_within_the_position_limit(conn):
    rig = Rig(conn, [0.2, 0.7, 0.65])  # SOL no, BTC 0.70, ETH 0.65
    rig.cycle()
    assert rig.swaps == [("buy", "BTC"), ("buy", "ETH")]
    scores, info = db.get_ml(conn)
    assert scores == {"SOL": 0.2, "BTC": 0.7, "ETH": 0.65}
    assert info["oos_auc"] == 0.53 and info["enter"] == CFG.ml_enter and info["horizon"] == 6


def test_only_the_single_best_idea_is_taken_when_there_is_one_slot(conn):
    cfg = replace(CFG, strategy="ml", assets=("SOL", "BTC", "ETH"), max_positions=1, min_equity_usd=1.0)
    rig = Rig(conn, [0.2, 0.7, 0.9], cfg=cfg)
    rig.cycle()
    assert rig.swaps == [("buy", "ETH")]


def test_thresholds_come_from_the_settings_not_the_saved_model(conn):
    cfg = replace(CFG, strategy="ml", assets=("SOL", "BTC", "ETH"), ml_enter=0.8, min_equity_usd=1.0)
    rig = Rig(conn, [0.7, 0.7, 0.7], cfg=cfg)
    rig.cycle()
    assert rig.swaps == []


def test_the_ai_sells_when_its_confidence_fades(conn):
    rig = Rig(conn, [0.2, 0.7, 0.2])
    rig.cycle()
    assert rig.swaps == [("buy", "BTC")]
    rig.models.bundle.model.probs = [0.2, 0.3, 0.2]  # BTC drops below the exit threshold
    rig.now += timedelta(hours=1)
    for frame in rig.frames.values():
        frame.index = frame.index + pd.Timedelta(hours=1)  # a new candle, so a new decision
    rig.cycle()
    assert rig.swaps == [("buy", "BTC"), ("sell", "BTC")]


def test_without_a_model_nothing_trades_and_nothing_breaks(conn):
    rig = Rig(conn, None)
    rig.cycle()
    assert rig.swaps == [] and not db.is_halted(conn)
    assert db.get_ml(conn) == ({}, {})


def test_a_stop_loss_overrides_a_confident_ai(conn):
    db.insert_trade(conn, "buy", db.Fill(5.0, 0.0166, 300.0, None), True, asset="BTC", timestamp=SETUP)  # far above today's price
    rig = Rig(conn, [0.2, 0.9, 0.2])
    rig.ctx.params = Params(fast=9, slow=21, hard_stop_pct=10.0)
    rig.cycle()
    assert rig.swaps == [("sell", "BTC")]


def test_the_btc_filter_still_gates_ai_entries(conn):
    rig = Rig(conn, [0.9, 0.9, 0.9])
    rig.ctx.params = Params(fast=9, slow=21, regime_len=100)
    falling = walk(260, seed=7)
    falling["close"] = np.linspace(200, 100, 260)  # BTC sliding: below its average
    rig.frames["BTC"] = with_forming_candle(falling)
    rig.cycle()
    assert rig.swaps == []


def test_training_history_is_capped_for_short_timeframes(monkeypatch):
    from bot import data

    seen = []
    monkeypatch.setattr(data, "fetch_history", lambda market, days, tf, ex: seen.append((tf, days)) or walk(60))
    ml.fetch_frames(replace(CFG, ml_train_days=400), "1m")
    ml.fetch_frames(replace(CFG, ml_train_days=400), "1h")
    one_minute = {d for tf, d in seen if tf == "1m"}
    one_hour = {d for tf, d in seen if tf == "1h"}
    assert one_minute == {17} and one_hour == {400}  # 25,000 one-minute candles is about 17 days


def test_the_missing_model_notice_is_logged_once_per_cycle(conn, caplog):
    rig = Rig(conn, None)
    with caplog.at_level("INFO", logger="bot"):
        rig.cycle()
    assert sum("no trained model" in r.message for r in caplog.records) == 1


SIX_HOURS = pd.Timedelta(hours=6)  # horizon 6 on 1h candles


def graded(p, returns, every=SIX_HOURS):
    """Graded calls made `every` apart from the same start, so equal positions share a window."""
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    return [(p, r, t0 + every * i) for i, r in enumerate(returns)]


def test_best_threshold_chases_proven_reward_not_hit_rate():
    big = graded(0.9, [0.08, -0.02] * 30)  # 50% hits, +3% average
    small = graded(0.6, [-0.01] * 60)  # same windows, a loss every time
    # any threshold above 0.6 keeps only the high-reward calls; the losing 0.6 band is avoided
    assert ml.best_threshold(big + small, 0.6, SIX_HOURS) > 0.6


def test_best_threshold_falls_back_without_enough_data():
    assert ml.best_threshold(graded(0.9, [0.5] * 5), 0.6, SIX_HOURS) == 0.6


def test_overlapping_calls_are_not_counted_as_separate_evidence():
    hourly = graded(0.9, [0.01] * 100, every=pd.Timedelta(hours=1))  # 100 calls, but only 17 six-hour windows
    assert len(ml.independent(hourly, SIX_HOURS)) == 17
    assert ml.best_threshold(hourly, 0.6, SIX_HOURS) == 0.6


def test_a_noisy_edge_that_could_be_luck_is_not_adopted():
    rng = np.random.default_rng(3)
    noisy = graded(0.9, list(0.002 + rng.normal(0, 0.05, 60)))  # tiny mean, big spread: indistinguishable from zero
    assert ml.best_threshold(noisy, 0.6, SIX_HOURS) == 0.6


def test_a_move_that_only_pays_the_fees_is_labelled_a_loss():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    c = pd.DataFrame({"close": [100.0, 100.5, 101.5, 101.5]}, index=idx)
    y = ml.labels(c, ml.MLParams(horizon=1, target_pct=0.0))
    assert list(y.iloc[:3]) == [0.0, 1.0, 0.0]  # +0.5% is eaten by the 0.8% round trip; +1% is not


def test_ml_decide_keeps_the_stops_without_a_model_and_gates_entries():
    closes = np.concatenate([np.linspace(100, 130, 40), [110.0]])
    h, lo = closes + 0.5, closes - 0.5
    mp, stops = ml.MLParams(horizon=6, enter=0.6, exit=0.45), Params(fast=9, slow=21, hard_stop_pct=10.0)
    assert ml.ml_decide(None, closes, h, lo, mp, stops, holding=True, entry_price=128.0, bars_held=3) is Signal.SELL
    assert ml.ml_decide(None, closes, h, lo, mp, stops, holding=True, entry_price=100.0, bars_held=3) is Signal.HOLD
    assert ml.ml_decide(0.9, closes, h, lo, mp, stops, holding=False) is Signal.BUY
    assert ml.ml_decide(0.9, closes, h, lo, mp, stops, holding=False, risk_on=False) is Signal.HOLD
    wait = Params(fast=9, slow=21, reentry_candles=4)
    assert ml.ml_decide(0.9, closes, h, lo, mp, wait, holding=False, bars_since_sell=2) is Signal.HOLD
    assert ml.ml_decide(0.5, closes, h, lo, mp, stops, holding=True, entry_price=100.0, exiting=True) is Signal.SELL


def test_predictions_are_graded_once_the_horizon_has_passed(conn):
    from bot import db as dbm
    from bot.run import _grade_predictions

    idx = pd.date_range("2026-01-01", periods=12, freq="1h", tz="UTC")
    closed = pd.DataFrame({"close": np.linspace(100, 110, 12)}, index=idx)
    dbm.record_prediction(conn, "BTC", "1h", idx[0].isoformat(), 0.7, 100.0, 6, 0.4)
    dbm.record_prediction(conn, "BTC", "1h", idx[10].isoformat(), 0.7, 100.0, 6, 0.4)  # horizon not over
    _grade_predictions(conn, "BTC", "1h", closed)
    rows = dbm.resolved_predictions(conn, "1h")
    assert len(rows) == 1 and rows[0]["candle"] == idx[0].isoformat()
    assert rows[0]["win"] == 1 and rows[0]["fwd_return"] == pytest.approx((1 + closed["close"].iloc[6] / 100 - 1) * (1 - 2 * (0.003 + 0.001)) - 1)
