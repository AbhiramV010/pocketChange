import pytest

from bot.config import ConfigError, load_settings


def test_defaults_are_safe():
    s = load_settings({})
    assert s.dry_run is True
    assert s.max_trade_usdc == 5
    assert s.max_slippage_bps == 100
    assert s.min_equity_usd == 15


def test_env_overrides():
    s = load_settings({"DRY_RUN": "false", "MAX_TRADE_USDC": "2.5", "STRATEGY": "DCA", "MAX_SLIPPAGE_BPS": "50"})
    assert s.dry_run is False
    assert s.max_trade_usdc == 2.5
    assert s.strategy == "dca"
    assert s.max_slippage_bps == 50


def test_blank_values_use_defaults():
    assert load_settings({"DRY_RUN": "", "MAX_TRADE_USDC": "  "}).dry_run is True


@pytest.mark.parametrize(
    "env",
    [
        {"DRY_RUN": "maybe"},
        {"MAX_TRADE_USDC": "abc"},
        {"MAX_TRADE_USDC": "0"},
        {"MAX_SLIPPAGE_BPS": "0"},
        {"MAX_SLIPPAGE_BPS": "5000"},
        {"STRATEGY": "yolo"},
        {"SOLANA_RPC_URL": "ftp://x"},
    ],
)
def test_invalid_values_raise(env):
    with pytest.raises(ConfigError):
        load_settings(env)


@pytest.mark.parametrize("tf, freq, minutes", [("5m", "5min", 5), ("15m", "15min", 15), ("1h", "1h", 60), ("4h", "4h", 240)])
def test_timeframe_maps_to_pandas_minutes_not_months(tf, freq, minutes):
    import subprocess, sys

    code = (
        "import sys; sys.path.insert(0, '.'); from bot import config, data; "
        "print(config.CANDLE_FREQ, int(data.CANDLE.total_seconds() // 60))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "TIMEFRAME": tf}).stdout.split()
    assert out == [freq, str(minutes)]
