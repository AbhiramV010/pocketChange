"""Funding-carry and perp-versus-spot decisions (pure functions; no venue adapter is wired in yet).

Long spot + short perpetual earns the funding rate while staying market-neutral. It pays only if the funding
collected over the expected holding period beats the cost of opening and closing both legs, with a safety margin.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarryDecision:
    enter: bool
    annualised: float  # funding earned per year at the current rate, as a fraction of notional
    breakeven_days: float  # days of funding needed to repay the round-trip cost
    reason: str


def annualised_funding(rate: float, interval_hours: float = 8.0) -> float:
    return rate * (24 / interval_hours) * 365


def carry_decision(
    funding_rate: float, *, interval_hours: float = 8.0, round_trip_cost: float = 0.004, hold_days: float = 14.0,
    min_edge: float = 0.01, max_basis: float = 0.01, basis: float = 0.0,
) -> CarryDecision:
    """`funding_rate` per interval (positive: longs pay shorts). `round_trip_cost`: all four fills as a fraction of
    notional. `min_edge`: net return over the holding period required to bother. `basis`: perp/spot - 1 now."""
    ann = annualised_funding(funding_rate, interval_hours)
    per_day = ann / 365
    be = round_trip_cost / per_day if per_day > 0 else float("inf")
    net = per_day * hold_days - round_trip_cost
    if abs(basis) > max_basis:
        return CarryDecision(False, ann, be, f"basis {basis:.2%} too wide")
    if net < min_edge:
        return CarryDecision(False, ann, be, f"net {net:.2%} over {hold_days:g}d below the {min_edge:.1%} minimum")
    return CarryDecision(True, ann, be, "funding pays for costs with margin")


def liquidation_buffer(entry: float, leverage: float, side: str = "short", maintenance: float = 0.05) -> float:
    """Adverse price move (fraction of entry) that liquidates a perp leg; size the hedge so this is not reachable."""
    move = 1 / leverage - maintenance
    return max(move, 0.0)
