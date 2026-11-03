import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from bot import data

SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def feed(monkeypatch):
    """Exchanges that fail on demand, a clock the test moves, and a log of which source was asked."""
    state = SimpleNamespace(calls=[], down=set(), now=1_000.0)

    def download(since, exchange_id, until=None, timeframe="1h", symbol="SOL/USDC"):
        state.calls.append((exchange_id, symbol))
        if exchange_id in state.down or (exchange_id, symbol) in state.down:
            raise RuntimeError("451 Service unavailable from a restricted location")
        return data.to_frame([[1_767_225_600_000, 1.0, 1.0, 1.0, 1.0, 1.0]])

    monkeypatch.setattr(data, "download", download)
    monkeypatch.setattr(data, "_failed_at", {})
    monkeypatch.setattr(data, "time", SimpleNamespace(monotonic=lambda: state.now, time=time.time))
    return state


def sources(feed):
    asked = [exchange for exchange, _ in feed.calls]
    feed.calls.clear()
    return asked


def test_a_blocked_source_is_tried_last_until_its_cooldown_ends(feed):
    feed.down = {"binance", "bybit"}  # both refuse US connections
    data._download_with_fallback(SINCE, "binance")
    assert sources(feed) == ["binance", "bybit", "okx"]
    data._download_with_fallback(SINCE, "binance")
    assert sources(feed) == ["okx"]  # no doomed requests every cycle
    feed.now += data.FAIL_COOLDOWN_S + 1
    data._download_with_fallback(SINCE, "binance")
    assert sources(feed) == ["binance", "bybit", "okx"]  # checked again once the cooldown is over


def test_a_source_that_recovers_goes_back_first(feed):
    feed.down = {"binance"}
    data._download_with_fallback(SINCE, "binance")
    feed.down = set()
    feed.now += data.FAIL_COOLDOWN_S + 1
    data._download_with_fallback(SINCE, "binance")
    sources(feed)
    data._download_with_fallback(SINCE, "binance")
    assert sources(feed) == ["binance"]


def test_a_market_one_source_lacks_does_not_demote_it_for_other_markets(feed):
    feed.down = {("binance", "BONK/USDC")}
    data._download_with_fallback(SINCE, "binance", symbol="BONK/USDC")
    sources(feed)
    data._download_with_fallback(SINCE, "binance", symbol="SOL/USDC")
    assert sources(feed) == ["binance"]


def test_when_every_source_fails_the_cooling_ones_are_still_tried(feed):
    feed.down = set(data.FALLBACK_EXCHANGES)
    with pytest.raises(data.DataError):
        data._download_with_fallback(SINCE, "binance")
    sources(feed)
    with pytest.raises(data.DataError):
        data._download_with_fallback(SINCE, "binance")
    assert sources(feed) == list(data.FALLBACK_EXCHANGES)
