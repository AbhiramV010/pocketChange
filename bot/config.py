from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from bot.assets import ASSETS, CORE, PRIMARY, RISKY

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Real environment variables win over .env.
load_dotenv(ROOT / ".env", override=False)

# Market and strategy constants (not env-tunable).
SYMBOL = "SOL/USDC"
TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h")
# Defaults are the settings that held up best in the backtests (README "Results"): 4h candles, the 20/50
# crossover with the trend filter, ATR and hard stops, the BTC market filter and 1% risk sizing.
# Set a refinement to 0 (or TREND_CONFIRM=false) to switch it off.
TIMEFRAME = (os.environ.get("TIMEFRAME") or "4h").strip().lower()  # candle size for strategies and backtests
SMA_FAST = int(os.environ.get("SMA_FAST") or 20)
SMA_SLOW = int(os.environ.get("SMA_SLOW") or 50)
# cross: buy/sell only on the candle where the averages cross (patient, few trades).
# trend: be long whenever fast > slow, flat whenever fast < slow (aggressive, enters right away).
ENTRY_MODES = ("cross", "trend")
ENTRY = (os.environ.get("ENTRY") or "cross").strip().lower()
EDGE_BPS = float(os.environ.get("EDGE_BPS") or 0)  # enter only once fast is this far above slow (cross: crosses that line)
TREND_CONFIRM = (os.environ.get("TREND_CONFIRM") or "true").strip().lower() in ("1", "true", "yes", "on")
STOP_ATR = float(os.environ.get("STOP_ATR") or 3)  # trailing stop this many ATRs below the highest close since entry
HARD_STOP_PCT = float(os.environ.get("HARD_STOP_PCT") or 10)  # also exit if price falls this % below entry
REENTRY_CANDLES = int(os.environ.get("REENTRY_CANDLES") or 0)  # wait this many candles after a sell before re-buying
REGIME_LEN = int(os.environ.get("REGIME_LEN") or 400)  # only open positions while BTC is above its SMA of this length
RISK_PCT = float(os.environ.get("RISK_PCT") or 1)  # size buys so a stop-out loses this % of equity (0 = flat size)
MIN_TRADE_USDC = 1.0  # below this a trade is not worth the fees

# Cost model shared by the backtest, the dry-run simulator and the AI's labels.
FEE_RATE = 0.003
SLIPPAGE_RATE = 0.001
ROUND_TRIP_COST = 2 * (FEE_RATE + SLIPPAGE_RATE)  # buy and sell, each with fee and slippage

STRATEGIES = ("sma", "dca", "ml")


class ConfigError(ValueError):
    pass


if not 2 <= SMA_FAST < SMA_SLOW <= 500:
    raise ConfigError(f"need 2 <= SMA_FAST < SMA_SLOW <= 500, got {SMA_FAST}/{SMA_SLOW}")
if ENTRY not in ENTRY_MODES:
    raise ConfigError(f"ENTRY must be one of {ENTRY_MODES}, got {ENTRY!r}")
if EDGE_BPS < 0 or STOP_ATR < 0 or not 0 <= HARD_STOP_PCT < 100:
    raise ConfigError("EDGE_BPS and STOP_ATR must be >= 0, HARD_STOP_PCT in [0, 100)")
if REENTRY_CANDLES < 0 or not 0 <= REGIME_LEN <= 1000:
    raise ConfigError("REENTRY_CANDLES must be >= 0 and REGIME_LEN in [0, 1000]")
if not 0 <= RISK_PCT <= 10:
    raise ConfigError("RISK_PCT must be in [0, 10]")
if TIMEFRAME not in TIMEFRAMES:
    raise ConfigError(f"TIMEFRAME must be one of {TIMEFRAMES}, got {TIMEFRAME!r}")
DASHBOARD_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h")  # choices on the dashboard switch


def candle_freq(timeframe: str) -> str:
    return timeframe.replace("m", "min") if timeframe.endswith("m") else timeframe


CANDLE_FREQ = candle_freq(TIMEFRAME)


def _raw(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = _raw(env, name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false, got {value!r}")


def _number(env: Mapping[str, str], name: str, default: float, cast=float):
    value = _raw(env, name)
    if value is None:
        return default
    try:
        return cast(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {value!r}") from None


def _path(env: Mapping[str, str], name: str, default: Path) -> Path:
    value = _raw(env, name)
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    dry_run: bool = True
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    jupiter_api_url: str = "https://lite-api.jup.ag/swap/v1"
    max_trade_usdc: float = 5.0
    max_slippage_bps: int = 100
    min_equity_usd: float = 15.0
    gas_reserve_sol: float = 0.01
    strategy: str = "sma"
    dca_amount_usdc: float = 1.0
    dca_interval_hours: float = 24.0
    loop_interval_seconds: float = 60.0
    sim_start_usdc: float = 20.0
    sim_start_sol: float = 0.05
    max_quote_deviation_bps: int = 300
    max_fee_usd: float = 0.5
    candle_exchange: str = "binance"
    dashboard_public: bool = False  # read-only public view: no pause control, no transaction links
    assets: tuple[str, ...] = (PRIMARY,)  # core symbols traded, each with its own signal and position
    risky_assets: tuple[str, ...] = ()  # memecoins traded only while the dashboard RISKY switch is on
    max_positions: int = 3  # open positions across all assets at once
    risky: bool = False  # initial state of the RISKY switch (the dashboard can flip it later)
    sniper: bool = True  # RISKY mode also snipes brand-new tokens (simulation only)
    sniper_bankroll_usdc: float = 10.0
    sniper_trade_usdc: float = 2.0
    sniper_max_open: int = 3
    sniper_min_liquidity_usd: float = 3000.0
    sniper_min_holders: int = 5
    sniper_max_age_min: float = 15.0
    sniper_take_profit: float = 0.5  # exit at +50%
    sniper_stop_loss: float = 0.2  # exit at -20%
    sniper_max_hold_min: float = 30.0
    sniper_interval_seconds: float = 30.0
    sniper_latency_slippage: float = 0.02  # assumed extra cost of being late on both legs
    jupiter_tokens_url: str = "https://lite-api.jup.ag/tokens/v2"
    ml_horizon: int = 6  # candles ahead the AI looks
    ml_target_pct: float = 0.4  # a move must beat this many percent to count as a win (costs are inside it)
    ml_enter: float = 0.6  # buy when the AI's probability of a win is at least this
    ml_exit: float = 0.45  # sell when it falls to this or below
    ml_retrain_hours: float = 24.0  # retrain on fresh candles this often
    ml_train_days: int = 400  # history used for training
    ml_adapt: bool = True  # learn the buy threshold from the AI's own graded predictions (reward-based)
    db_path: Path = DATA_DIR / "pocketchange.db"
    candles_path: Path = DATA_DIR / ("candles.csv" if TIMEFRAME == "1h" else f"candles_{TIMEFRAME}.csv")
    log_path: Path = DATA_DIR / "bot.log"

    def __post_init__(self) -> None:
        def need(ok: bool, message: str) -> None:
            if not ok:
                raise ConfigError(message)

        need(self.max_trade_usdc > 0, "MAX_TRADE_USDC must be > 0")
        need(0 < self.max_slippage_bps <= 1000, "MAX_SLIPPAGE_BPS must be in 1..1000")
        need(self.min_equity_usd >= 0, "MIN_EQUITY_USD must be >= 0")
        need(self.gas_reserve_sol >= 0, "GAS_RESERVE_SOL must be >= 0")
        need(self.strategy in STRATEGIES, f"STRATEGY must be one of {STRATEGIES}")
        need(self.dca_amount_usdc > 0, "DCA_AMOUNT_USDC must be > 0")
        need(self.dca_interval_hours > 0, "DCA_INTERVAL_HOURS must be > 0")
        need(self.loop_interval_seconds >= 1, "LOOP_INTERVAL_SECONDS must be >= 1")
        need(self.sim_start_usdc >= 0 and self.sim_start_sol >= 0, "SIM_START_* must be >= 0")
        need(0 < self.max_quote_deviation_bps <= 2000, "MAX_QUOTE_DEVIATION_BPS must be in 1..2000")
        need(self.max_fee_usd > 0, "MAX_FEE_USD must be > 0")
        need(self.solana_rpc_url.startswith(("http://", "https://")), "SOLANA_RPC_URL must be an http(s) URL")
        need(self.jupiter_api_url.startswith("https://"), "JUPITER_API_URL must be an https URL")
        need(bool(self.assets) and PRIMARY in self.assets, f"ASSETS must include {PRIMARY}")
        need(len(set(self.assets + self.risky_assets)) == len(self.assets + self.risky_assets), "ASSETS/RISKY_ASSETS has a duplicate")
        need(all(a in ASSETS and ASSETS[a].tier == "core" for a in self.assets), f"ASSETS must be core symbols: {','.join(CORE)}")
        need(
            all(a in ASSETS and ASSETS[a].tier == "risky" for a in self.risky_assets),
            f"RISKY_ASSETS must be risky symbols: {','.join(RISKY)}",
        )
        need(self.max_positions >= 1, "MAX_POSITIONS must be >= 1")
        need(0 < self.sniper_trade_usdc <= self.max_trade_usdc, "SNIPER_TRADE_USDC must be in (0, MAX_TRADE_USDC]")
        need(self.sniper_bankroll_usdc >= self.sniper_trade_usdc, "SNIPER_BANKROLL_USDC must cover one trade")
        need(self.sniper_max_open >= 1, "SNIPER_MAX_OPEN must be >= 1")
        need(0 < self.sniper_take_profit and 0 < self.sniper_stop_loss < 1, "SNIPER_TAKE_PROFIT > 0 and 0 < SNIPER_STOP_LOSS < 1")
        need(self.sniper_interval_seconds >= 5, "SNIPER_INTERVAL_SECONDS must be >= 5")
        need(0 <= self.sniper_latency_slippage < 0.5, "SNIPER_LATENCY_SLIPPAGE must be in [0, 0.5)")
        need(self.jupiter_tokens_url.startswith("https://"), "JUPITER_TOKENS_URL must be an https URL")
        need(1 <= self.ml_horizon <= 200, "ML_HORIZON must be in 1..200")
        need(0 <= self.ml_target_pct < 20, "ML_TARGET_PCT must be in [0, 20)")
        need(0.5 <= self.ml_enter < 1 and 0 < self.ml_exit < self.ml_enter, "need 0 < ML_EXIT < ML_ENTER < 1 and ML_ENTER >= 0.5")
        need(self.ml_retrain_hours >= 1 and self.ml_train_days >= 60, "ML_RETRAIN_HOURS >= 1 and ML_TRAIN_DAYS >= 60")


def _symbols(env: Mapping[str, str], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _raw(env, name)
    if value is None:
        return default
    symbols = [part.strip().upper() for part in value.split(",") if part.strip()]
    return tuple(dict.fromkeys(symbols))  # keep order, drop duplicates


def _core_assets(env: Mapping[str, str]) -> tuple[str, ...]:
    symbols = _symbols(env, "ASSETS", (PRIMARY,))
    return tuple([PRIMARY] + [s for s in symbols if s != PRIMARY])  # SOL always first


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    defaults = Settings()
    strategy = (_raw(env, "STRATEGY") or defaults.strategy).lower()
    return Settings(
        dry_run=_bool(env, "DRY_RUN", True),
        solana_rpc_url=_raw(env, "SOLANA_RPC_URL") or defaults.solana_rpc_url,
        jupiter_api_url=(_raw(env, "JUPITER_API_URL") or defaults.jupiter_api_url).rstrip("/"),
        max_trade_usdc=_number(env, "MAX_TRADE_USDC", defaults.max_trade_usdc),
        max_slippage_bps=_number(env, "MAX_SLIPPAGE_BPS", defaults.max_slippage_bps, int),
        min_equity_usd=_number(env, "MIN_EQUITY_USD", defaults.min_equity_usd),
        gas_reserve_sol=_number(env, "GAS_RESERVE_SOL", defaults.gas_reserve_sol),
        strategy=strategy,
        dca_amount_usdc=_number(env, "DCA_AMOUNT_USDC", defaults.dca_amount_usdc),
        dca_interval_hours=_number(env, "DCA_INTERVAL_HOURS", defaults.dca_interval_hours),
        loop_interval_seconds=_number(env, "LOOP_INTERVAL_SECONDS", defaults.loop_interval_seconds),
        sim_start_usdc=_number(env, "SIM_START_USDC", defaults.sim_start_usdc),
        sim_start_sol=_number(env, "SIM_START_SOL", defaults.sim_start_sol),
        max_quote_deviation_bps=_number(
            env, "MAX_QUOTE_DEVIATION_BPS", defaults.max_quote_deviation_bps, int
        ),
        max_fee_usd=_number(env, "MAX_FEE_USD", defaults.max_fee_usd),
        candle_exchange=(_raw(env, "CANDLE_EXCHANGE") or defaults.candle_exchange).lower(),
        dashboard_public=_bool(env, "DASHBOARD_PUBLIC", False),
        db_path=_path(env, "DB_PATH", defaults.db_path),
        assets=_core_assets(env),
        risky_assets=_symbols(env, "RISKY_ASSETS", ()),
        max_positions=_number(env, "MAX_POSITIONS", defaults.max_positions, int),
        risky=_bool(env, "RISKY", False),
        sniper=_bool(env, "SNIPER", True),
        sniper_bankroll_usdc=_number(env, "SNIPER_BANKROLL_USDC", defaults.sniper_bankroll_usdc),
        sniper_trade_usdc=_number(env, "SNIPER_TRADE_USDC", defaults.sniper_trade_usdc),
        sniper_max_open=_number(env, "SNIPER_MAX_OPEN", defaults.sniper_max_open, int),
        sniper_min_liquidity_usd=_number(env, "SNIPER_MIN_LIQUIDITY_USD", defaults.sniper_min_liquidity_usd),
        sniper_min_holders=_number(env, "SNIPER_MIN_HOLDERS", defaults.sniper_min_holders, int),
        sniper_max_age_min=_number(env, "SNIPER_MAX_AGE_MIN", defaults.sniper_max_age_min),
        sniper_take_profit=_number(env, "SNIPER_TAKE_PROFIT", defaults.sniper_take_profit),
        sniper_stop_loss=_number(env, "SNIPER_STOP_LOSS", defaults.sniper_stop_loss),
        sniper_max_hold_min=_number(env, "SNIPER_MAX_HOLD_MIN", defaults.sniper_max_hold_min),
        sniper_interval_seconds=_number(env, "SNIPER_INTERVAL_SECONDS", defaults.sniper_interval_seconds),
        sniper_latency_slippage=_number(env, "SNIPER_LATENCY_SLIPPAGE", defaults.sniper_latency_slippage),
        jupiter_tokens_url=(_raw(env, "JUPITER_TOKENS_URL") or defaults.jupiter_tokens_url).rstrip("/"),
        ml_horizon=_number(env, "ML_HORIZON", defaults.ml_horizon, int),
        ml_target_pct=_number(env, "ML_TARGET_PCT", defaults.ml_target_pct),
        ml_enter=_number(env, "ML_ENTER", defaults.ml_enter),
        ml_exit=_number(env, "ML_EXIT", defaults.ml_exit),
        ml_retrain_hours=_number(env, "ML_RETRAIN_HOURS", defaults.ml_retrain_hours),
        ml_train_days=_number(env, "ML_TRAIN_DAYS", defaults.ml_train_days, int),
        ml_adapt=_bool(env, "ML_ADAPT", True),
    )


settings = load_settings()
