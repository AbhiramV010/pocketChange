import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from bot import db
from bot.chain import TOKEN_PROGRAM, USDC_MINT
from bot.config import load_settings
from bot.sniper import MAX_ENTRIES_PER_SCAN, TOKEN_2022, Sniper, clean_symbol

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
CFG = replace(
    load_settings({}),
    sniper_interval_seconds=5.0, sniper_trade_usdc=2.0, sniper_bankroll_usdc=10.0, sniper_max_open=3,
    sniper_min_liquidity_usd=3000.0, sniper_min_holders=5, sniper_max_age_min=15.0, sniper_take_profit=0.5,
    sniper_stop_loss=0.2, sniper_max_hold_min=30.0, sniper_latency_slippage=0.02,
)
USDC = str(USDC_MINT)


class Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.text = status, body, json.dumps(body)
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeHttp:
    """The two Jupiter endpoints the sniper uses."""

    def __init__(self):
        self.tokens, self.recent_status = [], 200
        self.buy, self.sell = {}, {}  # mint -> (out_raw, impact) / out_usdc_raw; None means no route
        self.quote_calls = 0
        self.window, self.used = None, 0  # requests allowed until the rate-limit window resets (None = unlimited)

    def get(self, url, params=None, timeout=None):
        if self.window is not None:
            if self.used >= self.window:
                return Resp(429, {"error": "rate limited"}, {"x-ratelimit-remaining": "0"})
            self.used += 1
        if url.endswith("/recent"):
            return Resp(self.recent_status, self.tokens)
        assert url.endswith("/quote")
        self.quote_calls += 1
        if params["inputMint"] == USDC:
            hit = self.buy.get(params["outputMint"])
            return Resp(400, {"error": "no route"}) if hit is None else Resp(
                200, {"outAmount": str(hit[0]), "priceImpactPct": str(hit[1])}
            )
        out = self.sell.get(params["inputMint"])
        return Resp(400, {"error": "no route"}) if out is None else Resp(200, {"outAmount": str(out)})


def token(mint="TokenA", *, age_min=1.0, liquidity=5000.0, holders=10, dev_mints=1, program=TOKEN_2022,
          mint_ok=True, freeze_ok=True, top=10.0, symbol="AAA"):
    return {
        "id": mint, "symbol": symbol, "decimals": 6, "tokenProgram": program, "liquidity": liquidity,
        "holderCount": holders, "createdAt": (NOW - timedelta(minutes=age_min)).isoformat().replace("+00:00", "Z"),
        "audit": {"mintAuthorityDisabled": mint_ok, "freezeAuthorityDisabled": freeze_ok,
                  "topHoldersPercentage": top, "devMints": dev_mints},
    }


class Rig:
    def __init__(self, conn, cfg=CFG, risky=True):
        self.conn, self.http, self.now, self.waits = conn, FakeHttp(), NOW, []
        self.sniper = Sniper(cfg, conn, http=self.http, clock=lambda: self.now, sleep=self._sleep)
        db.set_risky(conn, risky)

    def list(self, *tokens, out_raw=1_000_000_000, impact=0.01, sells=2_000_000):
        self.http.tokens = list(tokens)
        for t in tokens:
            self.http.buy[t["id"]] = (out_raw, impact)
            self.http.sell[t["id"]] = sells

    def _sleep(self, seconds):
        self.waits.append(seconds)
        self.http.used = 0  # the rate-limit window resets while the sniper waits

    def step(self, minutes=1.0):
        self.now += timedelta(minutes=minutes)
        self.sniper.step()

    def snipes(self):
        return db.read_snipes(self.conn)


def test_eligible_token_is_bought_with_a_latency_haircut(conn):
    rig = Rig(conn)
    rig.list(token())
    rig.step()
    row = rig.snipes().iloc[0]
    assert row["status"] == "open" and row["entry_usdc"] == 2.0 and row["symbol"] == "AAA"
    assert row["qty_raw"] == int(1_000_000_000 * 0.98)
    assert db.sniper_free_usdc(conn, 10.0) == 8.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"liquidity": 2999.0}, {"holders": 4}, {"mint_ok": False}, {"freeze_ok": False}, {"age_min": 16.0},
        {"dev_mints": 11}, {"top": 41.0}, {"program": "SomeOtherProgram111111111111111111111111111"},
    ],
)
def test_filters_keep_out_risky_launches(conn, overrides):
    rig = Rig(conn)
    rig.list(token(**overrides))
    rig.step()
    assert rig.snipes().empty


def test_classic_token_program_is_accepted_too(conn):
    rig = Rig(conn)
    rig.list(token(program=str(TOKEN_PROGRAM)))
    rig.step()
    assert len(rig.snipes()) == 1


def test_thin_pools_and_missing_sell_routes_are_rejected_and_not_requoted(conn):
    rig = Rig(conn)
    rig.list(token("Impact"), token("Trap"), token("Thin"))
    rig.http.buy["Impact"] = (10**9, 0.05)  # moves the price 5%
    rig.http.sell["Trap"] = None  # you can buy it but cannot sell it
    rig.http.sell["Thin"] = 1_000_000  # an immediate sell returns 50% of the $2
    rig.step()
    assert rig.snipes().empty
    calls = rig.http.quote_calls
    rig.step()
    assert rig.http.quote_calls == calls  # rejected tokens are not re-quoted for a while


def test_entries_need_the_risky_switch_and_no_pause_or_halt(conn):
    rig = Rig(conn, risky=False)
    rig.list(token())
    rig.step()
    assert rig.snipes().empty
    db.set_risky(conn, True)
    db.set_paused(conn, True)
    rig.step()
    assert rig.snipes().empty
    db.set_paused(conn, False)
    db.set_halted(conn, "test")
    rig.step()
    assert rig.snipes().empty
    db.clear_halt(conn)
    rig.step()
    assert len(rig.snipes()) == 1


def test_open_positions_are_managed_even_with_the_switch_off(conn):
    rig = Rig(conn)
    rig.list(token())
    rig.step()
    db.set_risky(conn, False)
    rig.http.sell["TokenA"] = 3_300_000  # +60% after the haircut
    rig.step()
    assert rig.snipes().iloc[0]["status"] == "closed"


@pytest.mark.parametrize(
    "sell_out, minutes, reason",
    [
        (3_300_000, 1, "take profit"),
        (1_500_000, 1, "stop loss"),
        (2_100_000, 31, "time stop"),
        (60_000, 1, "rugged"),
    ],
)
def test_exit_rules(conn, sell_out, minutes, reason):
    rig = Rig(conn)
    rig.list(token())
    rig.step()
    rig.http.sell["TokenA"] = sell_out
    rig.step(minutes)
    row = rig.snipes().iloc[0]
    assert row["status"] == "closed" and row["reason"].startswith(reason)
    assert row["exit_usdc"] == pytest.approx(sell_out / 1e6 * 0.98)


def test_a_position_with_no_sell_route_is_written_off_after_three_checks(conn):
    rig = Rig(conn)
    rig.list(token())
    rig.step()
    rig.http.sell["TokenA"] = None
    rig.step()
    rig.step()
    assert rig.snipes().iloc[0]["status"] == "open"
    rig.step()
    row = rig.snipes().iloc[0]
    assert row["status"] == "closed" and row["exit_usdc"] == 0.0 and "no exit route" in row["reason"]
    assert db.sniper_free_usdc(conn, 10.0) == 8.0  # the $2 is gone


def test_bankroll_and_slot_limits(conn):
    rig = Rig(conn)
    rig.list(*[token(f"Token{i}", symbol=f"T{i}") for i in range(6)])
    rig.step()
    assert len(rig.snipes()) == MAX_ENTRIES_PER_SCAN  # a cap per scan
    rig.step()
    rig.step()
    assert len(rig.snipes()) == CFG.sniper_max_open  # then the slot limit
    assert db.sniper_free_usdc(conn, 10.0) == 4.0


def test_steps_are_throttled_and_network_trouble_is_swallowed(conn):
    rig = Rig(conn)
    rig.list(token())
    rig.sniper.step()  # first call runs
    calls = rig.http.quote_calls
    rig.sniper.step()  # same instant: skipped
    assert rig.http.quote_calls == calls
    rig.http.recent_status = 429
    rig.http.tokens = [token("Other")]
    rig.step()  # a rate limit never raises
    assert len(rig.snipes()) == 1


def test_a_scan_that_needs_more_requests_than_the_rate_limit_allows_waits_for_the_window(conn):
    rig = Rig(conn)
    rig.http.window = 3  # Jupiter's keyless tier allows a handful of requests per window; a scan needs more
    rig.list(token("A"), token("B"))
    rig.step()
    assert len(rig.snipes()) == 2 and rig.waits  # it waited for the window instead of dropping the step
    assert all(0 < w <= 12 for w in rig.waits)


def test_symbols_are_sanitised_before_they_reach_logs_and_the_database():
    assert clean_symbol("\U0001f525PEPE\U0001f525") == "PEPE"
    assert clean_symbol("a" * 40) == "a" * 12 and clean_symbol(None) == "?" and clean_symbol("\U0001f4a9") == "?"
