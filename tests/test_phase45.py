from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot import attribution, db, execution, lifecycle, monitor, reconcile, research, store
from bot.lifecycle import Evidence, Stage
from tests.test_portfolio_backtest import candles
from tests.test_run import CFG


# --- execution -------------------------------------------------------------------------------------------------

def test_shortfall_sign_convention():
    assert execution.implementation_shortfall_bps("buy", 100, 100.5) == pytest.approx(50)
    assert execution.implementation_shortfall_bps("sell", 100, 99.5) == pytest.approx(50)
    assert execution.implementation_shortfall_bps("buy", 100, 99.9) < 0
    assert np.isnan(execution.implementation_shortfall_bps("buy", 0, 1))


def test_priority_level_rises_only_on_a_fee_spike():
    calm = [100] * 100
    assert execution.choose_priority_level(calm) == "medium"
    busy = [100] * 70 + [10_000] * 30  # a quarter or more of slots paying up
    assert execution.choose_priority_level(busy) == "high"
    assert execution.choose_priority_level([5]) == "medium"


def test_dynamic_slippage_is_bounded():
    assert execution.dynamic_slippage_bps(0.0) == 30
    assert execution.dynamic_slippage_bps(0.5, cap=100) == 100
    assert execution.dynamic_slippage_bps(0.01) > execution.dynamic_slippage_bps(0.001)


def test_price_sanity_check():
    assert execution.price_sane(100, 100.5) and not execution.price_sane(100, 110) and execution.price_sane(100, None)


def test_impact_fit_recovers_a_line():
    n = np.linspace(1, 100, 40)
    fixed, slope = execution.fit_impact(n, 5 + 0.2 * n)
    assert fixed == pytest.approx(5, abs=1e-6) and slope == pytest.approx(0.2, abs=1e-6)


def test_pyth_price_parses_and_degrades():
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"parsed": [{"price": {"price": "15000000000", "expo": -8}}]}

    class Http:
        def get(self, *a, **k): return Resp()

    assert execution.pyth_price(Http(), "SOL") == pytest.approx(150.0)
    assert execution.pyth_price(Http(), "NOPE") is None

    class Broken:
        def get(self, *a, **k): raise execution.requests.ConnectionError()

    assert execution.pyth_price(Broken(), "SOL") is None


def test_tca_rows_are_recorded_and_reported(conn):
    db.record_tca(conn, "SOL", "buy", 100.0, 100.4, 5.0, 0.03, True)
    db.record_tca(conn, "SOL", "sell", 110.0, 109.6, 5.5, 0.03, True)
    rows = db.read_tca(conn)
    assert len(rows) == 2 and rows[0]["shortfall_bps"] == pytest.approx(40)
    text = execution.report_text(rows)
    assert "SOL" in text and "impact model" in text


def test_the_live_loop_records_cost_analysis(conn):
    from tests.test_run import Rig, buy_signal_series
    Rig(conn, buy_signal_series()).cycle()
    rows = db.read_tca(conn)
    assert len(rows) == 1 and rows[0]["side"] == "buy" and rows[0]["shortfall_bps"] > 0


def test_rpc_falls_back_to_the_next_endpoint():
    import requests
    from dataclasses import replace
    from bot.chain import ChainClient

    calls = []

    class Http:
        def post(self, url, json=None, timeout=None):
            calls.append(url)
            if "dead" in url:
                raise requests.ConnectionError("down")

            class R:
                def raise_for_status(self): pass
                def json(self): return {"result": 123}
            return R()

    cfg = replace(CFG, solana_rpc_url="https://dead.example", solana_rpc_fallbacks=("https://alive.example",))
    client = ChainClient(cfg, address="11111111111111111111111111111111", http=Http())
    assert client._rpc("getSlot") == 123 and calls == ["https://dead.example", "https://alive.example"]


# --- store -----------------------------------------------------------------------------------------------------

def test_store_roundtrip_merge_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    a = candles(np.linspace(100, 100.5, 50), "4h")
    store.save_frame("SOL", "4h", a.iloc[:30])
    merged = store.save_frame("SOL", "4h", a.iloc[20:])
    assert len(merged) == 50 and store.load_frame("SOL", "4h").equals(merged)
    assert store.first_listed("SOL", "4h") == a.index[0]
    assert store.validate(merged, "4h") == {"rows": 50, "gaps": 0, "duplicates": 0, "unsorted": 0, "bad_ohlc": 0, "outliers": 0}


def test_validation_flags_gaps_bad_candles_and_outliers():
    df = candles(np.linspace(100, 120, 50), "4h")
    df = df.drop(df.index[10:13])
    df.iloc[5, df.columns.get_loc("high")] = 1.0  # high below close
    df.iloc[30, df.columns.get_loc("close")] = 400.0
    v = store.validate(df, "4h")
    assert v["gaps"] == 3 and v["bad_ohlc"] >= 1 and v["outliers"] >= 1


# --- monitor ---------------------------------------------------------------------------------------------------

def test_psi_is_small_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert monitor.psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert monitor.psi(a, rng.normal(2, 1, 5000)) > 0.25
    drift = monitor.feature_drift(pd.DataFrame({"x": a, "y": a}), pd.DataFrame({"x": a + 3, "y": a}), ["x", "y"])
    assert list(drift) == ["x"]


def test_drawdown_and_decay_helpers():
    assert monitor.drawdown(pd.Series([100, 120, 90])) == pytest.approx(0.25)
    assert monitor.model_decay([(0.7, True)] * 10, 0.6) is None
    assert monitor.model_decay([(0.7, False)] * 30, 0.6) == 0.0


def test_check_flags_a_stale_heartbeat_and_a_halt(conn):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.set_state(conn, db.HEARTBEAT, (now - timedelta(hours=2)).isoformat())
    db.set_halted(conn, "test halt")
    msgs = [a.message for a in monitor.check(conn, CFG, now)]
    assert any("heartbeat" in m for m in msgs) and any("HALTED" in m for m in msgs)
    db.set_state(conn, db.HEARTBEAT, now.isoformat())
    db.clear_halt(conn)
    assert monitor.check(conn, CFG, now) == []


# --- attribution -----------------------------------------------------------------------------------------------

def test_attribution_reconciles_gross_costs_and_net():
    t = pd.DataFrame([
        {"timestamp": "2026-01-01T00:00:00", "side": "buy", "amount_in": 10.0, "amount_out": 0.1, "asset": "SOL"},
        {"timestamp": "2026-01-02T00:00:00", "side": "sell", "amount_in": 0.1, "amount_out": 11.0, "asset": "SOL"},
        {"timestamp": "2026-01-01T00:00:00", "side": "buy", "amount_in": 5.0, "amount_out": 5.0, "asset": "BTC"},
    ])
    out = attribution.attribute(t).set_index("asset")
    assert out.loc["SOL", "net"] == pytest.approx(1.0) and out.loc["SOL", "trades"] == 2
    assert out.loc["BTC", "net"] == 0.0
    row = out.loc["SOL"]
    assert row["gross"] == pytest.approx(row["net"] + row["fees"] + row["slippage"] + row["fixed"])
    assert "net" in attribution.summary(out.reset_index())


# --- lifecycle -------------------------------------------------------------------------------------------------

GOOD = Evidence(days=120, deflated_sharpe=0.97, sharpe=1.2, max_drawdown=0.08, tracking_error_bps=15)


def test_lifecycle_promotes_only_with_time_and_evidence():
    assert lifecycle.decide(Stage.PAPER, GOOD)[0] == "promote"
    assert lifecycle.decide(Stage.PAPER, Evidence(30, 0.97, 1.2, 0.08, 15))[0] == "hold"
    assert lifecycle.decide(Stage.PAPER, Evidence(120, 0.60, 1.2, 0.08, 15))[0] == "hold"
    assert lifecycle.decide(Stage.RESEARCH, Evidence(0, 0.99, 2.0, 0.05, 0, stress_2x_positive=False))[0] == "hold"


def test_lifecycle_kills_on_drawdown_or_sustained_losses():
    assert lifecycle.decide(Stage.SMALL_LIVE, Evidence(10, 0.9, 0.5, 0.30, 10))[0] == "kill"
    assert lifecycle.decide(Stage.SMALL_LIVE, Evidence(120, 0.5, -1.0, 0.10, 10))[0] == "kill"
    assert lifecycle.decide(Stage.PAPER, Evidence(120, 0.99, 1.0, 0.30, 10))[0] != "kill"  # paper money cannot be 'killed'


# --- reconciliation --------------------------------------------------------------------------------------------

def test_reconcile_matches_trades_and_measures_slippage():
    t0 = pd.Timestamp("2026-01-01 04:00", tz="UTC")
    fills = [(t0, "buy", "SOL", 10.0, 0.1), (t0 + pd.Timedelta(hours=8), "sell", "SOL", 0.1, 11.0), (t0 + pd.Timedelta(hours=40), "buy", "ETH", 5.0, 0.05)]
    live = pd.DataFrame([
        {"timestamp": (t0 + pd.Timedelta(minutes=30)).isoformat(), "side": "buy", "amount_in": 10.0, "amount_out": 0.0995, "asset": "SOL"},
        {"timestamp": (t0 + pd.Timedelta(hours=9)).isoformat(), "side": "sell", "amount_in": 0.1, "amount_out": 10.945, "asset": "SOL"},
        {"timestamp": (t0 + pd.Timedelta(hours=90)).isoformat(), "side": "buy", "amount_in": 5.0, "amount_out": 5.0, "asset": "DOGE"},
    ])
    r = reconcile.reconcile(live, fills, "4h")
    assert r.matched == 2 and len(r.missing_live) == 1 and len(r.extra_live) == 1
    assert r.slippage_bps == pytest.approx(50, abs=1)
    assert r.alerts()  # a third of the trades disagree and fills are 50 bps worse


# --- research: trial hashes and universe -----------------------------------------------------------------------

def test_trials_record_code_and_data_hashes(tmp_path):
    from bot.strategy import Params
    path = tmp_path / "t.jsonl"
    d = {"SOL": candles(np.linspace(100, 120, 50), "4h")}
    research.log_trial("x", Params(), 1.0, 0.1, path, data=research.data_hash(d))
    row = __import__("json").loads(path.read_text())
    assert row["data"] == research.data_hash(d) and row["code"]
    assert research.data_hash(d) != research.data_hash({"SOL": candles(np.linspace(100, 121, 50), "4h")})


def test_point_in_time_universe_excludes_coins_not_yet_listed():
    old = candles(np.linspace(100, 120, 300), "4h")
    new = old.iloc[200:]
    u = research.listed_before({"BTC": old, "TRUMP": new}, old.index[100])
    assert list(u) == ["BTC"]
    assert set(research.listed_before({"BTC": old, "TRUMP": new}, old.index[250])) == {"BTC", "TRUMP"}
    assert list(research.listed_before({"BTC": old, "TRUMP": new}, old.index[250], min_bars=100)) == ["BTC"]


def test_the_live_loop_logs_new_alerts_once(conn, caplog):
    from tests.test_run import Rig, buy_signal_series
    rig = Rig(conn, buy_signal_series())
    db.set_halted(conn, "test halt")
    with caplog.at_level("WARNING", logger="bot"):
        rig.cycle()
        rig.ctx.alerts_at = None  # let the monitor run again straight away
        rig.cycle()
    assert sum("ALERT: HALTED" in r.getMessage() for r in caplog.records) == 1
