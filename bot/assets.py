from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    symbol: str
    label: str  # what the dashboard calls the on-chain token
    mint: str
    decimals: int
    market: str  # ccxt symbol used as the reference price
    tier: str = "core"  # core: always traded. risky: only while the RISKY switch is on


# Every mint was checked on-chain (classic SPL Token program, decimals) and quoted through Jupiter
# both ways for $5 (round trip within ~0.4%). BTC/ETH/BNB/XRP are Wormhole-wrapped.
# Note: wXRP has a freeze authority (the bridge could freeze balances). It is a core asset on request.
_ASSETS = (
    Asset("SOL", "wSOL", "So11111111111111111111111111111111111111112", 9, "SOL/USDC"),
    Asset("BTC", "wBTC", "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh", 8, "BTC/USDC"),
    Asset("ETH", "wETH", "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs", 8, "ETH/USDC"),
    Asset("BNB", "wBNB", "9gP2kCy3wA1ctvYWQk75guqXuHfrEomqydHLtcTCqiLa", 8, "BNB/USDC"),
    Asset("XRP", "wXRP", "6UpQcMAb5xMzxc7ZfPaVMgx3KqsvKZdT5U718BzD5We2", 6, "XRP/USDC"),
    Asset("DOGE", "DOGE", "DoGEV7LASBkQbibMc5k5vKnTZoMg423GpJ5QtJEGfm7R", 8, "DOGE/USDC"),
    Asset("AVAX", "AVAX", "avaxGHCq3T7hoxd73oY2KY9hJSTaeMibXvHy5KNzh5D", 9, "AVAX/USDC"),
    Asset("BONK", "BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", 5, "BONK/USDC", "risky"),
    Asset("WIF", "WIF", "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm", 6, "WIF/USDC", "risky"),
    Asset("TRUMP", "TRUMP", "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN", 6, "TRUMP/USDC", "risky"),
    Asset("PENGU", "PENGU", "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv", 6, "PENGU/USDC", "risky"),
)
ASSETS: dict[str, Asset] = {a.symbol: a for a in _ASSETS}
PRIMARY = "SOL"  # always traded: native SOL pays the fees and wSOL is the original position
CORE = tuple(a.symbol for a in _ASSETS if a.tier == "core")
RISKY = tuple(a.symbol for a in _ASSETS if a.tier == "risky")


def label(symbol: str) -> str:
    return ASSETS[symbol].label if symbol in ASSETS else symbol
