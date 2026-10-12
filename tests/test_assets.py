from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest
from solders.pubkey import Pubkey

from bot import data
from bot.assets import ASSETS, CORE, PRIMARY, RISKY, label
from bot.config import ConfigError, load_settings


def test_registry_is_consistent():
    mints = [a.mint for a in ASSETS.values()]
    assert len(set(mints)) == len(mints)
    assert PRIMARY == "SOL" and CORE[0] == "SOL"
    assert {"BTC", "ETH", "BNB", "XRP"} <= set(CORE) and {"BONK", "WIF"} <= set(RISKY)
    assert set(CORE).isdisjoint(RISKY)
    for asset in ASSETS.values():
        assert asset.market == f"{asset.symbol}/USDC" and 0 < asset.decimals <= 9
        Pubkey.from_string(asset.mint)  # a valid address


def test_label_falls_back_to_the_symbol():
    assert label("BTC") == "wBTC" and label("XRP") == "wXRP" and label("MYSTERY") == "MYSTERY"


def test_assets_env_keeps_sol_first_and_dedupes():
    cfg = load_settings({"ASSETS": "btc, eth,BTC", "RISKY_ASSETS": "bonk"})
    assert cfg.assets == ("SOL", "BTC", "ETH") and cfg.risky_assets == ("BONK",)
    assert load_settings({}).assets == ("SOL",) and load_settings({}).risky_assets == ()


@pytest.mark.parametrize(
    "env, message",
    [
        ({"ASSETS": "BONK"}, "ASSETS must be core"),  # a memecoin cannot sneak into the always-on list
        ({"RISKY_ASSETS": "BTC"}, "RISKY_ASSETS must be risky"),
        ({"ASSETS": "DOGECOIN9000"}, "ASSETS must be core"),
        ({"MAX_POSITIONS": "0"}, "MAX_POSITIONS"),
    ],
)
def test_bad_asset_settings_are_rejected(env, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(env)


def test_sniper_settings_are_validated():
    with pytest.raises(ConfigError, match="SNIPER_TRADE_USDC"):
        load_settings({"SNIPER_TRADE_USDC": "6"})  # above MAX_TRADE_USDC
    with pytest.raises(ConfigError, match="SNIPER_BANKROLL_USDC"):
        load_settings({"SNIPER_BANKROLL_USDC": "1"})
    assert replace(load_settings({}), risky=True).risky


def test_fetch_recent_passes_the_symbol_through(monkeypatch):
    seen = []

    def fake_download(since, exchange_id, until=None, timeframe="1h", symbol="SOL/USDC"):
        seen.append((exchange_id, symbol))
        idx = pd.date_range(datetime.now(timezone.utc) - pd.Timedelta(hours=3), periods=3, freq="1h", name="timestamp")
        return pd.DataFrame({c: 1.0 for c in data.COLUMNS}, index=idx)

    monkeypatch.setattr(data, "download", fake_download)
    data.fetch_recent(3, "binance", symbol="BTC/USDC")
    assert seen == [("binance", "BTC/USDC")]


@pytest.mark.parametrize(
    "env, message",
    [
        ({"ML_ENTER": "0.4"}, "ML_ENTER"),
        ({"ML_ENTER": "0.6", "ML_EXIT": "0.7"}, "ML_EXIT"),
        ({"ML_HORIZON": "0"}, "ML_HORIZON"),
        ({"ML_TARGET_PCT": "-1"}, "ML_TARGET_PCT"),
        ({"ML_TRAIN_DAYS": "10"}, "ML_TRAIN_DAYS"),
    ],
)
def test_bad_ai_settings_are_rejected(env, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(env)


def test_ai_strategy_is_selectable_and_has_sane_defaults():
    cfg = load_settings({"STRATEGY": "ml"})
    assert cfg.strategy == "ml" and cfg.ml_exit < cfg.ml_enter and cfg.ml_horizon == 6
