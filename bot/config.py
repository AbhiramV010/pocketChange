"""Settings and hard limits. Loads .env; every tunable limit lives here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Real environment variables win over .env.
load_dotenv(ROOT / ".env", override=False)

# Market and strategy constants (not env-tunable).
SYMBOL = "SOL/USDC"
TIMEFRAME = "1h"
SMA_FAST = 20
SMA_SLOW = 50
MIN_TRADE_USDC = 1.0  # below this a trade is not worth the fees

# Cost model shared by the backtest and the dry-run simulator.
FEE_RATE = 0.003
SLIPPAGE_RATE = 0.001

STRATEGIES = ("sma", "dca")


class ConfigError(ValueError):
    """Raised for an invalid setting. Messages name the variable, never secrets."""


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
    db_path: Path = DATA_DIR / "pocketchange.db"
    candles_path: Path = DATA_DIR / "candles.csv"
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


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from the environment. DRY_RUN defaults to true."""
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
    )


settings = load_settings()
