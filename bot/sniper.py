"""Simulated new-token sniper.

Runs only in dry runs and only while the dashboard's RISKY switch is on. It never touches the wallet: it has its
own bankroll (SNIPER_BANKROLL_USDC) in the `snipes` table. Entries and exits are priced from real Jupiter quotes
(so thin liquidity, price impact and missing sell routes show up as real losses), then a latency haircut is
applied to both legs because a 30-second poll is always late against bots that act within a slot.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Callable

import requests

from bot import db
from bot.chain import TOKEN_PROGRAM, USDC_MINT
from bot.config import Settings

log = logging.getLogger("bot.sniper")

USDC_DECIMALS = 6
TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"  # most new launchpad tokens use this program
HTTP_TIMEOUT_S = 15
MAX_PRICE_IMPACT = 0.03  # refuse entries that move the price more than 3%
MIN_ROUND_TRIP = 0.85  # a $ in must come back at >= 85% on an immediate sell quote (catches thin pools)
MAX_TOP_HOLDERS_PCT = 40.0
MAX_DEV_MINTS = 10  # serial launchers are token farms
MAX_QUOTES_PER_SCAN = 8
MAX_ENTRIES_PER_SCAN = 2
EXIT_ROUTE_FAILURES = 3
RUG_VALUE_FRACTION = 0.05  # a position worth under 5% of its cost is treated as rugged
REJECT_TTL_S = 600


class RateLimited(Exception):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clean_symbol(symbol: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9$_.-]", "", symbol or "")[:12] or "?"


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class Sniper:

    def __init__(
        self, cfg: Settings, conn: sqlite3.Connection, http: requests.Session | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ):
        self._cfg, self._conn, self._clock = cfg, conn, clock
        self._http = http or requests.Session()
        self._last_step: datetime | None = None
        self._rejected: dict[str, datetime] = {}

    # --- network -------------------------------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None):
        resp = self._http.get(url, params=params, timeout=HTTP_TIMEOUT_S)
        if resp.status_code == 429:
            raise RateLimited(url)
        return resp

    def _quote(self, mint_in: str, mint_out: str, amount_raw: int) -> dict | None:
        """A Jupiter quote, or None when there is no route (which is itself information)."""
        resp = self._get(
            f"{self._cfg.jupiter_api_url}/quote",
            {
                "inputMint": mint_in, "outputMint": mint_out, "amount": str(amount_raw),
                "slippageBps": str(self._cfg.max_slippage_bps * 3), "restrictIntermediateTokens": "true",
            },
        )
        if resp.status_code != 200:
            return None
        body = resp.json()
        return body if int(body.get("outAmount", 0)) > 0 else None

    # --- the loop ------------------------------------------------------------------------------------------

    def step(self) -> None:
        cfg, conn = self._cfg, self._conn
        now = self._clock()
        if self._last_step is not None and (now - self._last_step).total_seconds() < cfg.sniper_interval_seconds:
            return
        self._last_step = now
        try:
            self._manage_open(now)
            entries_allowed = db.is_risky(conn, cfg.risky) and not db.is_paused(conn) and not db.is_halted(conn)
            if entries_allowed:
                self._scan(now)
        except RateLimited:
            log.warning("sniper: Jupiter rate limit hit; skipping until the next step")
        except requests.RequestException as exc:
            log.warning("sniper: network error (%s); skipping this step", type(exc).__name__)

    def _manage_open(self, now: datetime) -> None:
        cfg, conn = self._cfg, self._conn
        stamp = now.isoformat(timespec="seconds")
        for row in conn.execute("SELECT * FROM snipes WHERE status = 'open' ORDER BY id").fetchall():
            quote = self._quote(row["mint"], str(USDC_MINT), row["qty_raw"])
            if quote is None:
                fails = row["fail_count"] + 1
                db.mark_snipe(conn, row["id"], None, fails, stamp)
                if fails >= EXIT_ROUTE_FAILURES:
                    db.close_snipe(conn, row["id"], 0.0, "no exit route (rugged or delisted)", stamp)
                    log.info("sniper: %s closed at 0: no sell route for %d checks", row["symbol"], fails)
                continue
            value = int(quote["outAmount"]) / 10**USDC_DECIMALS * (1 - cfg.sniper_latency_slippage)
            db.mark_snipe(conn, row["id"], value, 0, stamp)
            pnl = value / row["entry_usdc"] - 1
            opened = _parse_time(row["opened_at"]) or now
            age_min = (now - opened).total_seconds() / 60
            reason = None
            if value < row["entry_usdc"] * RUG_VALUE_FRACTION:
                reason = "rugged (value collapsed)"
            elif pnl >= cfg.sniper_take_profit:
                reason = "take profit"
            elif pnl <= -cfg.sniper_stop_loss:
                reason = "stop loss"
            elif age_min >= cfg.sniper_max_hold_min:
                reason = "time stop"
            if reason:
                db.close_snipe(conn, row["id"], value, reason, stamp)
                log.info("sniper: %s closed (%s): $%.2f -> $%.2f (%+.1f%%)", row["symbol"], reason, row["entry_usdc"], value, pnl * 100)

    def _eligible(self, token: dict, now: datetime) -> bool:
        cfg = self._cfg
        mint = token.get("id", "")
        rejected_at = self._rejected.get(mint)
        if rejected_at is not None and (now - rejected_at).total_seconds() < REJECT_TTL_S:
            return False
        created = _parse_time(token.get("createdAt"))
        if created is None or (now - created).total_seconds() / 60 > cfg.sniper_max_age_min:
            return False
        if token.get("tokenProgram") not in (str(TOKEN_PROGRAM), TOKEN_2022):
            return False
        audit = token.get("audit") or {}
        if not (audit.get("mintAuthorityDisabled") and audit.get("freezeAuthorityDisabled")):
            return False
        if (audit.get("topHoldersPercentage") or 0) > MAX_TOP_HOLDERS_PCT or (audit.get("devMints") or 0) > MAX_DEV_MINTS:
            return False
        return (token.get("liquidity") or 0) >= cfg.sniper_min_liquidity_usd and (
            token.get("holderCount") or 0
        ) >= cfg.sniper_min_holders

    def _scan(self, now: datetime) -> None:
        cfg, conn = self._cfg, self._conn
        free = db.sniper_free_usdc(conn, cfg.sniper_bankroll_usdc)
        open_count = conn.execute("SELECT COUNT(*) FROM snipes WHERE status = 'open'").fetchone()[0]
        if open_count >= cfg.sniper_max_open or free < cfg.sniper_trade_usdc:
            return
        resp = self._get(f"{cfg.jupiter_tokens_url}/recent")
        if resp.status_code != 200:
            log.warning("sniper: token feed returned HTTP %s", resp.status_code)
            return
        known = {row["mint"] for row in conn.execute("SELECT mint FROM snipes")}
        quotes = entries = 0
        for token in resp.json():
            if token.get("id") in known or not self._eligible(token, now):
                continue
            if quotes >= MAX_QUOTES_PER_SCAN or entries >= MAX_ENTRIES_PER_SCAN:
                break
            quotes += 2
            if self._try_enter(token, now):
                entries += 1
                open_count += 1
                free -= cfg.sniper_trade_usdc
                if open_count >= cfg.sniper_max_open or free < cfg.sniper_trade_usdc:
                    break

    def _try_enter(self, token: dict, now: datetime) -> bool:
        cfg, conn = self._cfg, self._conn
        mint, symbol = token["id"], clean_symbol(token.get("symbol"))
        decimals = int(token.get("decimals", 6))
        spend_raw = int(cfg.sniper_trade_usdc * 10**USDC_DECIMALS)

        def reject(why: str) -> bool:
            self._rejected[mint] = now
            log.info("sniper: skipped %s: %s", symbol, why)
            return False

        buy = self._quote(str(USDC_MINT), mint, spend_raw)
        if buy is None:
            return reject("no buy route")
        impact = abs(float(buy.get("priceImpactPct") or 0))
        if impact > MAX_PRICE_IMPACT:
            return reject(f"price impact {impact:.1%}")
        out_raw = int(buy["outAmount"])
        back = self._quote(mint, str(USDC_MINT), out_raw)
        if back is None:
            return reject("no sell route")
        round_trip = int(back["outAmount"]) / 10**USDC_DECIMALS / cfg.sniper_trade_usdc
        if round_trip < MIN_ROUND_TRIP:
            return reject(f"round trip only {round_trip:.0%}")
        qty_raw = int(out_raw * (1 - cfg.sniper_latency_slippage))  # arriving late costs us tokens on entry
        db.open_snipe(
            conn, mint=mint, symbol=symbol, decimals=decimals, entry_usdc=cfg.sniper_trade_usdc, qty_raw=qty_raw,
            liquidity_usd=token.get("liquidity"), timestamp=now.isoformat(timespec="seconds"),
        )
        log.info(
            "sniper: bought $%.2f of %s (liquidity $%.0f, impact %.2f%%, round trip %.0f%%)",
            cfg.sniper_trade_usdc, symbol, token.get("liquidity") or 0, impact * 100, round_trip * 100,
        )
        return True
