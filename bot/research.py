"""Research-integrity tools: how much of a backtest result could be luck?

    python -m bot.portfolio_backtest --days 1500 --timeframe 4h --research

Every backtest run through `portfolio_backtest` is appended to data/trials.jsonl; the Deflated Sharpe Ratio uses that
count, so trying many settings makes a good-looking result harder to believe, as it should.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

from bot.config import DATA_DIR
from bot.strategy import Params

TRIALS_PATH = DATA_DIR / "trials.jsonl"
EULER = 0.5772156649015329
_N = NormalDist()


# --- trial log ---------------------------------------------------------------------------------------------

_code_hash: str | None = None


def code_hash() -> str:
    """Git commit the code was run from ('+dirty' with uncommitted changes), or 'unknown'."""
    global _code_hash
    if _code_hash is None:
        import subprocess
        try:
            root = DATA_DIR.parent
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
            dirty = subprocess.run(["git", "status", "--porcelain", "--", "bot"], cwd=root, capture_output=True, text=True, timeout=5)
            _code_hash = (head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")) if head.returncode == 0 else "unknown"
        except Exception:
            _code_hash = "unknown"
    return _code_hash


def data_hash(frames: dict[str, pd.DataFrame]) -> str:
    """Short fingerprint of the candles a run used: coins, date range, row counts and a price checksum."""
    import hashlib
    h = hashlib.sha1()
    for a in sorted(frames):
        df = frames[a]
        h.update(f"{a}|{len(df)}|{df.index[0] if len(df) else ''}|{df.index[-1] if len(df) else ''}|{float(df['close'].sum()):.6f}".encode())
    return h.hexdigest()[:10]


def log_trial(label: str, params: Params, sharpe: float, ret: float, path=None, data: str | None = None) -> None:
    path = path or TRIALS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label, "params": vars(params),
           "sharpe": None if math.isnan(sharpe) else sharpe, "ret": ret, "code": code_hash(), "data": data}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def trial_stats(path=None) -> tuple[int, float]:
    """(number of distinct configurations tried, variance of their Sharpe ratios)."""
    path = path or TRIALS_PATH
    if not path.exists():
        return 1, 0.0
    seen: dict[str, float] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("sharpe") is not None:
            seen[json.dumps([row["label"], row["params"]], sort_keys=True)] = row["sharpe"]
    sharpes = list(seen.values())
    return max(1, len(seen)), float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0


# --- return statistics -------------------------------------------------------------------------------------

def periods_per_year(equity: pd.Series) -> float:
    return pd.Timedelta(days=365) / pd.Series(equity.index).diff().median()


def returns_of(equity: pd.Series) -> np.ndarray:
    return equity.pct_change().dropna().to_numpy(float)


def sharpe_ratio(r: np.ndarray, ann: float = 1.0) -> float:
    if len(r) < 3 or not np.std(r, ddof=1) > 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(ann))


def max_drawdown(r: np.ndarray) -> float:
    curve = np.cumprod(1 + r)
    return float(1 - (curve / np.maximum.accumulate(curve)).min()) if len(curve) else 0.0


def cagr(r: np.ndarray, ann: float) -> float:
    if not len(r):
        return float("nan")
    total = float(np.prod(1 + r))
    return total ** (ann / len(r)) - 1 if total > 0 else -1.0


def expected_max_sharpe(n_trials: int, sharpe_var: float) -> float:
    """Per-period Sharpe the best of `n_trials` unskilled strategies is expected to show (Bailey & Lopez de Prado)."""
    if n_trials <= 1 or sharpe_var <= 0:
        return 0.0
    sd = math.sqrt(sharpe_var)
    return sd * ((1 - EULER) * _N.inv_cdf(1 - 1 / n_trials) + EULER * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(r: np.ndarray, n_trials: int, sharpe_var: float) -> float:
    """Probability the true Sharpe is above what luck across `n_trials` tries would produce. Per-period units."""
    n = len(r)
    if n < 10 or not np.std(r, ddof=1) > 0:
        return float("nan")
    sr = float(np.mean(r) / np.std(r, ddof=1))
    centred = (r - r.mean()) / r.std()
    skew, kurt = float((centred**3).mean()), float((centred**4).mean())
    bench = expected_max_sharpe(n_trials, sharpe_var)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return _N.cdf((sr - bench) * math.sqrt(n - 1) / math.sqrt(denom))


def sharpe_stderr(r: np.ndarray, ann: float) -> float:
    n = len(r)
    sr = sharpe_ratio(r)
    return float("nan") if math.isnan(sr) else math.sqrt((1 + 0.5 * sr**2) / n) * math.sqrt(ann)


def block_bootstrap(r: np.ndarray, ann: float, n: int = 1000, block: int | None = None, seed: int = 0) -> dict:
    """Stationary-style block bootstrap (fixed mean block length) of Sharpe, max drawdown and CAGR: 5th/50th/95th pct."""
    rng = np.random.default_rng(seed)
    size = len(r)
    block = block or max(2, int(round(size ** (1 / 3))))
    out = {"sharpe": [], "max_dd": [], "cagr": []}
    for _ in range(n):
        idx = []
        while len(idx) < size:
            start = int(rng.integers(0, size))
            length = int(rng.geometric(1 / block))
            idx.extend((start + k) % size for k in range(length))
        s = r[np.array(idx[:size])]
        out["sharpe"].append(sharpe_ratio(s, ann))
        out["max_dd"].append(max_drawdown(s))
        out["cagr"].append(cagr(s, ann))
    return {k: tuple(float(x) for x in np.nanpercentile(v, [5, 50, 95])) for k, v in out.items()}


def pbo(returns: np.ndarray, slices: int = 8) -> float:
    """Probability of Backtest Overfitting by combinatorially symmetric CV. `returns` is (periods, configs).
    Share of train/test splits where the config that was best in-sample ranks below the median out-of-sample."""
    t, k = returns.shape
    if k < 2 or t < slices * 2:
        return float("nan")
    parts = np.array_split(np.arange(t), slices)
    below = total = 0
    for combo in itertools.combinations(range(slices), slices // 2):
        train = np.concatenate([parts[i] for i in combo])
        test = np.concatenate([parts[i] for i in range(slices) if i not in combo])
        in_s = [sharpe_ratio(returns[train, j]) for j in range(k)]
        out_s = np.array([sharpe_ratio(returns[test, j]) for j in range(k)])
        if np.isnan(in_s).all() or np.isnan(out_s).all():
            continue
        best = int(np.nanargmax(in_s))
        rank = (np.nan_to_num(out_s, nan=-1e9) < out_s[best]).mean()  # relative OOS rank of the in-sample winner
        below += rank <= 0.5
        total += 1
    return below / total if total else float("nan")


# --- point-in-time universe --------------------------------------------------------------------------------

def listed_before(candles: dict[str, pd.DataFrame], when: pd.Timestamp, min_bars: int = 0) -> dict[str, pd.DataFrame]:
    """Coins that already traded at `when` (with at least `min_bars` of history by then). Using today's winners for
    the whole period is survivorship bias; pick the universe as it stood at the start of each test instead."""
    when = pd.Timestamp(when)
    return {a: df for a, df in candles.items() if len(df) and df.index[0] <= when and (df.index <= when).sum() >= min_bars}


# --- benchmarks --------------------------------------------------------------------------------------------

def vol_matched_hold(hold: pd.Series, target: pd.Series) -> pd.Series:
    """Buy and hold scaled with cash so its volatility equals `target`'s: the fair yardstick for a lower-risk rule."""
    h, t = hold.pct_change().dropna(), target.pct_change().dropna()
    w = min(1.0, float(t.std() / h.std())) if h.std() > 0 else 1.0
    return (1 + w * h).cumprod() * float(hold.iloc[0])


def btc_filter_only(candles: dict[str, pd.DataFrame], bench: pd.DataFrame, regime_len: int = 400) -> pd.Series:
    """Equal-weight coins, held only while BTC is above its SMA, else cash. Free of fees: an upper bound on how much
    the market filter alone explains."""
    from bot.portfolio_backtest import align
    frames = align({**candles, "__b__": bench})
    b = frames.pop("__b__")["close"]
    on = (b > b.rolling(regime_len).mean()).shift(1, fill_value=False)  # decided on the previous close
    basket = sum(f["close"].pct_change().fillna(0.0) for f in frames.values()) / len(frames)
    return (1 + basket.where(on, 0.0)).cumprod()


# --- walk-forward, cost stress, parameter surface ----------------------------------------------------------

def _slice(frames: dict[str, pd.DataFrame], lo: int, hi: int) -> dict[str, pd.DataFrame]:
    return {a: df.iloc[lo:hi] for a, df in frames.items()}


def rolling_walk_forward(
    candles: dict[str, pd.DataFrame], bench: pd.DataFrame, grid: list[Params], *, train_bars: int, test_bars: int,
    **kw,
) -> tuple[pd.Series, list[Params]]:
    """Pick the best-Sharpe setting on the trailing `train_bars`, trade the next `test_bars` with it, roll forward.
    Returns the stitched out-of-sample equity curve (start 1.0) and the setting chosen for each block."""
    from bot.portfolio_backtest import align, simulate
    frames = align({**candles, "__b__": bench})
    b = frames.pop("__b__")
    n = len(b)
    pieces, chosen = [], []
    cursor = train_bars
    while cursor < n:
        end = min(cursor + test_bars, n)
        train, bt = _slice(frames, cursor - train_bars, cursor), b.iloc[cursor - train_bars:cursor]
        scores = [simulate(train, p, bench=bt, **kw).sharpe for p in grid]
        best = grid[int(np.nanargmax(np.nan_to_num(scores, nan=-1e9)))]
        warm = best.slow + 2  # indicator history before the block; its own returns are discarded
        lo = max(0, cursor - max(warm, best.regime_len + 2))
        res = simulate(_slice(frames, lo, end), best, bench=b.iloc[lo:end], **kw)
        block = res.equity.iloc[res.equity.index >= frames[next(iter(frames))].index[cursor]]
        if len(block) > 1:
            pieces.append(block.pct_change().dropna())
        chosen.append(best)
        cursor = end
    if not pieces:
        return pd.Series(dtype=float), chosen
    r = pd.concat(pieces)
    return (1 + r).cumprod(), chosen


def breakeven_cost(candles, bench, p: Params, *, hi: float = 8.0, **kw) -> float:
    """Multiple of the modelled costs (fee, slippage, fixed) at which the strategy's return falls to zero."""
    from bot.config import FEE_RATE, FIXED_COST_USD, SLIPPAGE_RATE
    from bot.portfolio_backtest import simulate

    def ret(m: float) -> float:
        return simulate(candles, p, bench=bench, fee_rate=FEE_RATE * m, slippage_rate=SLIPPAGE_RATE * m,
                        fixed_cost=FIXED_COST_USD * m, **kw).ret

    if ret(0.0) <= 0:
        return 0.0
    if ret(hi) > 0:
        return math.inf
    lo, up = 0.0, hi
    for _ in range(14):
        mid = (lo + up) / 2
        lo, up = (mid, up) if ret(mid) > 0 else (lo, mid)
    return (lo + up) / 2


def parameter_surface(candles, bench, base: Params, fasts=(10, 20, 30), slows=(30, 50, 80), stops=(2.0, 3.0, 4.0), **kw):
    """Sharpe over fast x slow x stop width, to see whether the chosen cell sits on a plateau or a spike."""
    from bot.portfolio_backtest import simulate
    rows = []
    for f, s, a in itertools.product(fasts, slows, stops):
        if f >= s:
            continue
        res = simulate(candles, replace(base, fast=f, slow=s, stop_atr=a), bench=bench, **kw)
        rows.append({"fast": f, "slow": s, "stop_atr": a, "sharpe": res.sharpe, "ret": res.ret, "equity": res.equity})
    return rows


def surface_text(rows: list[dict]) -> str:
    lines = [f"{'fast':>5}{'slow':>6}{'stop':>6}{'sharpe':>8}{'return':>9}"]
    lines += [f"{r['fast']:>5}{r['slow']:>6}{r['stop_atr']:>6g}{r['sharpe']:>8.2f}{r['ret']:>+9.1%}" for r in rows]
    sh = np.array([r["sharpe"] for r in rows], float)
    lines.append(f"cells positive: {(sh > 0).sum()}/{len(sh)}; median Sharpe {np.nanmedian(sh):.2f}; "
                 f"worst {np.nanmin(sh):.2f}; best {np.nanmax(sh):.2f}")
    return "\n".join(lines)


def research_report(candles, bench, base: Params, **kw) -> str:
    """Everything above for the configured rules, as text."""
    from bot.portfolio_backtest import buy_and_hold, simulate

    res = simulate(candles, base, bench=bench, label="configured rules", **kw)
    ann = periods_per_year(res.equity)
    r = returns_of(res.equity)
    dh = data_hash(candles)
    rows = parameter_surface(candles, bench, base, **kw)
    for row in rows:
        log_trial(f"surface {row['fast']}/{row['slow']}/{row['stop_atr']}", replace(base, fast=row["fast"], slow=row["slow"], stop_atr=row["stop_atr"]), row["sharpe"], row["ret"], data=dh)
    log_trial("configured rules", base, res.sharpe, res.ret, data=dh)
    n_trials, var = trial_stats()
    per_period_var = var / ann if ann else 0.0  # trial Sharpes are annualised; the DSR works per period
    dsr = deflated_sharpe(r, n_trials, per_period_var)
    ci = block_bootstrap(r, ann)
    hold = buy_and_hold(candles).equity
    vm = vol_matched_hold(hold, res.equity)
    btc = btc_filter_only(candles, bench, base.regime_len)
    cols = [returns_of(row["equity"]) for row in rows]
    short = min((len(c) for c in cols), default=0)
    mat = np.column_stack([c[-short:] for c in cols]) if short else np.empty((0, 0))  # same candles for every cell
    lines = [
        f"configured rules: return {res.ret:+.1%}, Sharpe {res.sharpe:.2f} (stderr ~{sharpe_stderr(r, ann):.2f}), max DD {res.max_drawdown:.1%}",
        f"configurations tried so far: {n_trials}; Deflated Sharpe probability {dsr:.1%} (want > 95%)",
        "bootstrap 5/50/95 pct: Sharpe {:.2f}/{:.2f}/{:.2f}, max DD {:.0%}/{:.0%}/{:.0%}, CAGR {:+.0%}/{:+.0%}/{:+.0%}".format(
            *ci["sharpe"], *ci["max_dd"], *ci["cagr"]),
        f"PBO over the surface ({len(rows)} cells): {pbo(mat):.0%} (want < 50%)" if mat.size else "PBO: not enough cells",
        f"benchmarks: buy&hold Sharpe {sharpe_ratio(returns_of(hold), ann):.2f}; vol-matched hold Sharpe "
        f"{sharpe_ratio(returns_of(vm), ann):.2f}, return {vm.iloc[-1] / vm.iloc[0] - 1:+.1%}; BTC-filter-only (no fees) "
        f"Sharpe {sharpe_ratio(returns_of(btc), ann):.2f}, return {btc.iloc[-1] / btc.iloc[0] - 1:+.1%}",
        f"break-even cost: {breakeven_cost(candles, bench, base, **kw):.1f}x the modelled costs (want > 2x)",
        "", "parameter surface:", surface_text(rows),
    ]
    return "\n".join(lines)
