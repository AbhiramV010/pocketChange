import numpy as np
import pandas as pd
import pytest

from dataclasses import replace

from bot import ml
from bot.portfolio_backtest import align, buy_and_hold, ml_decider, report, simulate
from bot.strategy import Params

TREND = Params(fast=5, slow=20, entry="trend", edge_bps=0.0, confirm=False, stop_atr=0.0, hard_stop_pct=0.0)


def candles(closes, freq="1h"):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq=freq, tz="UTC", name="timestamp")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0})


def market(rate, n=400):
    return candles(100 * (1 + rate) ** np.arange(n))


def test_trend_following_profits_on_a_clean_uptrend_without_costs():
    r = simulate({"SOL": market(0.002), "BTC": market(0.002)}, TREND, fee_rate=0.0, slippage_rate=0.0)
    assert r.final > r.start and r.trades >= 2 and r.exposure > 0.9


def test_costs_are_charged_on_every_fill():
    data = {"SOL": market(0.002), "BTC": market(0.002)}
    free = simulate(data, TREND, fee_rate=0.0, slippage_rate=0.0)
    costly = simulate(data, TREND, fee_rate=0.01, slippage_rate=0.005)
    assert costly.final < free.final


def test_the_btc_regime_filter_keeps_it_out_while_btc_is_falling():
    data = {"SOL": market(0.002), "BTC": market(-0.002)}
    filtered = simulate(data, Params(**{**TREND.__dict__, "regime_len": 50}), fee_rate=0.0, slippage_rate=0.0)
    assert filtered.trades == 0 and filtered.final == pytest.approx(filtered.start)


def test_max_positions_limits_open_positions():
    data = {a: market(0.002) for a in ("SOL", "BTC", "ETH", "BNB")}
    r = simulate(data, TREND, max_positions=2, fee_rate=0.0, slippage_rate=0.0)
    assert r.trades == 2  # two slots, held to the end
    assert sum(1 for v in r.per_asset.values() if v != 0) <= 2 or r.trades == 2


def test_series_are_aligned_and_buy_and_hold_splits_cash_equally():
    a, b = market(0.001, 300), market(0.001, 300).iloc[10:]
    aligned = align({"SOL": a, "BTC": b})
    assert len(aligned["SOL"]) == len(aligned["BTC"]) == 290
    bh = buy_and_hold({"SOL": market(0.001), "BTC": market(0.001)}, fee_rate=0.0, slippage_rate=0.0)
    assert bh.final > bh.start and bh.trades == 2


def test_report_has_one_line_per_result():
    r = simulate({"SOL": market(0.002), "BTC": market(0.002)}, TREND, label="trend")
    text = report([buy_and_hold({"SOL": market(0.002), "BTC": market(0.002)}), r])
    assert "trend" in text and "buy & hold" in text and len(text.splitlines()) == 3


# --- the backtest is the live loop ---------------------------------------------------------------------------------

def walk(n, seed, drift, vol=0.006):
    """Hourly candles with trending returns, real gaps between close and next open, and wicks."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = 0.7 * ret[i - 1] + rng.normal(drift, vol)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


FULL = Params(
    fast=5, slow=20, entry="trend", edge_bps=5.0, confirm=False, stop_atr=2.0, hard_stop_pct=4.0,
    reentry_candles=3, regime_len=50, risk_pct=1.0,
)


def live_fills(conn, candles, p, max_positions=2, max_trade=5.0, models=None, **cfg_kw):
    """Run the real live loop several cycles per candle, as it runs every minute. During candle t the feed shows
    the closed candles before it plus t still forming, priced at its open, so a live fill lands where the
    backtest's next-open fill does, and later cycles in the same candle must not trade back and forth."""
    from dataclasses import replace
    from datetime import timedelta

    from bot.config import load_settings
    from bot.run import Context, SimBroker, run_cycle

    cfg = replace(load_settings({}), assets=tuple(candles), max_positions=max_positions, max_trade_usdc=max_trade,
                  min_equity_usd=1.0, sim_start_usdc=20.0, **cfg_kw)
    at = {"t": 0, "minute": 5}

    def view(a):
        df = candles[a].iloc[: at["t"] + 1].copy()
        df.iloc[-1, df.columns.get_loc("close")] = df["open"].iloc[-1]
        return df

    ctx = Context(cfg, conn, SimBroker(conn, cfg), feed=lambda tf: view("SOL"),
                  clock=lambda: candles["SOL"].index[at["t"]].to_pydatetime() + timedelta(minutes=at["minute"]),
                  extra_feeds={a: (lambda tf, a=a: view(a)) for a in candles if a != "SOL"}, params=p, models=models)
    first = max(p.slow + 3, ml.MIN_CANDLES + 1) if models is not None else p.slow + 3  # the backtest's first decision
    for t in range(first, len(candles["SOL"])):
        at["t"] = t
        for minute in (5, 6, 7, 30):
            at["minute"] = minute
            run_cycle(ctx)
    rows = conn.execute("SELECT timestamp, side, asset, amount_in, amount_out FROM trades ORDER BY id").fetchall()
    return [(pd.Timestamp(r["timestamp"]).floor("1h"), r["side"], r["asset"], r["amount_in"], r["amount_out"]) for r in rows]


def same_fills(a, b):
    assert [(t, s, x) for t, s, x, _, _ in a] == [(t, s, x) for t, s, x, _, _ in b]
    for (_, _, _, ia, oa), (_, _, _, ib, ob) in zip(a, b):
        assert ia == pytest.approx(ib, rel=1e-9) and oa == pytest.approx(ob, rel=1e-9)


def test_the_backtest_makes_exactly_the_trades_the_live_loop_makes(conn):
    data = {"SOL": walk(500, 1, 0.0006), "BTC": walk(500, 2, 0.0004), "ETH": walk(500, 3, -0.0001, vol=0.012)}
    live = live_fills(conn, data, FULL)
    bt = simulate(data, FULL, max_positions=2, trade_usdc=5.0, start_cash=20.0)
    assert len(live) >= 10 and {s for _, s, _, _, _ in live} == {"buy", "sell"}  # a real test: plenty of both
    same_fills(live, bt.fills)


def test_parity_holds_for_the_plain_crossover_too(conn):
    data = {"SOL": walk(400, 4, 0.0003), "BTC": walk(400, 5, 0.0002)}
    plain = Params(fast=9, slow=21, entry="cross", edge_bps=0.0, confirm=False, stop_atr=0.0, hard_stop_pct=0.0,
                   reentry_candles=0, regime_len=0, risk_pct=0.0)
    live = live_fills(conn, data, plain, max_positions=3)
    assert len(live) >= 4
    same_fills(live, simulate(data, plain, max_positions=3).fills)


def test_decisions_never_see_the_future():
    data = {"SOL": walk(300, 6, 0.0005), "BTC": walk(300, 7, 0.0003)}
    base = simulate(data, FULL)
    cut = data["SOL"].index[200]
    rewritten = {a: df.copy() for a, df in data.items()}
    for df in rewritten.values():
        df.loc[df.index > cut, ["open", "high", "low", "close"]] *= 3.0  # a wildly different future
    other = simulate(rewritten, FULL)
    before = lambda fills: [f for f in fills if f[0] <= cut]  # noqa: E731
    assert before(base.fills) == before(other.fills) and base.equity[:cut].equals(other.equity[:cut])


def test_a_capped_sell_sells_every_chunk_in_the_same_candle_until_flat():
    up = np.concatenate([np.full(30, 100.0), np.linspace(100, 300, 40), np.linspace(300, 200, 30)])
    data = {"SOL": candles(up)}
    p = Params(fast=5, slow=20, entry="trend", edge_bps=0.0, confirm=False, stop_atr=0.0, hard_stop_pct=0.0, risk_pct=0.0)
    r = simulate(data, p, trade_usdc=5.0, fee_rate=0.0, slippage_rate=0.0)
    sells = [f for f in r.fills if f[1] == "sell"]
    assert len(sells) >= 2 and all(s[4] <= 5.0 + 1e-9 for s in sells)  # ~$12 of coin, at most $5 per sell
    assert len({t for t, *_ in sells}) == 1  # back to back at one open, as the live loop's cycles in one candle
    assert len(r.trade_pnl) == 1 and r.win_rate == 1.0  # one winning trade, however many chunks it took
    assert r.invested.iloc[-1] < 1.0


def test_the_live_loop_does_not_trade_back_and_forth_inside_one_candle(conn):
    """A buy filled 12% above the decision close must not be stopped out against that stale close a minute later
    (and then bought again): positions are judged on candles that closed after the fill, as in the backtest."""
    from tests.test_run import Rig, buy_signal_series

    closes = buy_signal_series()[:-1]
    from dataclasses import replace as rep
    from bot.config import load_settings
    cfg = rep(load_settings({}), min_equity_usd=1.0)
    rig = Rig(conn, closes + [closes[-1] * 1.12], cfg=cfg)
    rig.ctx.params = Params(fast=20, slow=50, entry="cross", edge_bps=0.0, confirm=False, stop_atr=3.0,
                            hard_stop_pct=10.0, reentry_candles=0, regime_len=0, risk_pct=0.0)
    for minute in range(5, 25):
        rig.now = rig.frame.index[-1].to_pydatetime() + pd.Timedelta(minutes=minute)
        rig.cycle()
    sides = [r["side"] for r in conn.execute("SELECT side FROM trades ORDER BY id")]
    assert sides == ["buy"]


def test_risk_sizing_buys_less_of_a_volatile_coin():
    calm, wild = walk(200, 8, 0.002, vol=0.002), walk(200, 8, 0.002, vol=0.002)
    wild[["high", "low"]] = np.column_stack([wild["close"] * 1.05, wild["close"] * 0.95])  # 10% candle ranges
    p = replace(FULL, regime_len=0, hard_stop_pct=0.0)  # the 2-ATR trailing stop sets the risk distance
    spend = lambda df: simulate({"SOL": df}, p).fills[0][3]  # noqa: E731
    assert spend(calm) == pytest.approx(5.0) and spend(wild) < 2.5
    tight = replace(p, hard_stop_pct=4.0)  # a nearer hard stop caps the loss at 4%, whatever the volatility
    assert simulate({"SOL": wild}, tight).fills[0][3] == pytest.approx(5.0)
    assert simulate({"SOL": wild}, replace(p, risk_pct=0.0)).fills[0][3] == pytest.approx(5.0)


def test_a_market_filter_without_btc_candles_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError, match="BTC"):
        simulate({"SOL": market(0.002)}, replace(TREND, regime_len=50))


def test_metrics_describe_the_run():
    r = simulate({"SOL": market(0.002), "BTC": market(0.002)}, TREND, fee_rate=0.0, slippage_rate=0.0)
    assert r.sharpe > 0 and r.equity is not None and len(r.equity) > 300
    wave = {"SOL": candles(100 + 30 * np.sin(np.arange(600) / 12))}
    w = simulate(wave, TREND)
    assert 0 < len(w.trade_pnl) <= sum(f[1] == "sell" for f in w.fills) and 0 <= w.win_rate <= 1 and w.profit_factor >= 0


class Momentum:
    """A stand-in model whose P(win) is a fixed function of the features, so live and backtest see the same calls."""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X["ret6"], float) * 150 - 1.0)))
        return np.column_stack([1 - p, p])


class Models:
    def __init__(self, bundle):
        self.bundle = bundle

    def get(self, timeframe):
        return self.bundle

    def maybe_retrain(self, timeframe, now):
        pass


def test_the_ai_backtest_makes_exactly_the_trades_the_live_ai_makes(conn):
    data = {"SOL": walk(500, 11, 0.0004, vol=0.01), "BTC": walk(500, 12, 0.0003), "ETH": walk(500, 13, 0.0, vol=0.012)}
    mp = ml.MLParams(horizon=6, target_pct=0.4, enter=0.6, exit=0.45)
    p = replace(FULL, regime_len=0)
    bundle = ml.Bundle(Momentum(), mp, "1h", "2026-01-01T00:00:00+00:00", 1000)
    live = live_fills(conn, data, p, models=Models(bundle), strategy="ml", ml_enter=0.6, ml_exit=0.45, ml_horizon=6,
                      ml_target_pct=0.4, ml_adapt=False)
    proba = {a: bundle.proba_series(ml.features(df, data["BTC"])).to_numpy(float) for a, df in data.items()}
    proba = {a: np.where(np.arange(len(v)) >= ml.MIN_CANDLES, v, np.nan) for a, v in proba.items()}  # live warm-up
    bt = simulate(data, p, max_positions=2, decider=lambda c, h, lo: ml_decider(proba, c, h, lo, mp, p))
    assert len(live) >= 6 and {s for _, s, _, _, _ in live} == {"buy", "sell"}
    same_fills(live, bt.fills)
