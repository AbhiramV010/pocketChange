"""Strategy lifecycle: research -> paper -> small live -> scaled live, with kill criteria written down in advance.

A strategy moves up only when it meets the gate for its stage, and is stopped the moment it breaks a kill rule.
Nothing here trades; it turns evidence into a decision so the call is not made in the heat of a drawdown.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    RESEARCH = "research"
    PAPER = "paper"  # at least 90 days in a dry run
    SMALL_LIVE = "small_live"  # real money at the minimum size
    SCALED = "scaled"


@dataclass(frozen=True)
class Evidence:
    days: float  # time spent in the current stage
    deflated_sharpe: float  # probability true Sharpe beats luck (bot.research.deflated_sharpe)
    sharpe: float
    max_drawdown: float  # of this stage's equity curve
    tracking_error_bps: float  # average gap between live fills and the backtest's, per trade
    stress_2x_positive: bool = True  # still profitable with doubled costs


@dataclass(frozen=True)
class Rules:
    min_days: dict[Stage, float] = None  # type: ignore[assignment]
    kill_drawdown: float = 0.25
    kill_sharpe: float = -0.5  # after min_days in a live stage
    max_tracking_error_bps: float = 50.0
    min_dsr: float = 0.95

    def __post_init__(self):
        if self.min_days is None:
            object.__setattr__(self, "min_days", {Stage.RESEARCH: 0, Stage.PAPER: 90, Stage.SMALL_LIVE: 90, Stage.SCALED: 0})


NEXT = {Stage.RESEARCH: Stage.PAPER, Stage.PAPER: Stage.SMALL_LIVE, Stage.SMALL_LIVE: Stage.SCALED}


def decide(stage: Stage, ev: Evidence, rules: Rules = Rules()) -> tuple[str, str]:
    """('kill' | 'promote' | 'hold', reason)."""
    if stage in (Stage.SMALL_LIVE, Stage.SCALED) and ev.max_drawdown >= rules.kill_drawdown:
        return "kill", f"drawdown {ev.max_drawdown:.0%} hit the {rules.kill_drawdown:.0%} kill line"
    if stage in (Stage.SMALL_LIVE, Stage.SCALED) and ev.days >= rules.min_days[stage] and ev.sharpe < rules.kill_sharpe:
        return "kill", f"Sharpe {ev.sharpe:.2f} after {ev.days:.0f} days"
    if stage in (Stage.PAPER, Stage.SMALL_LIVE, Stage.SCALED) and ev.tracking_error_bps > rules.max_tracking_error_bps:
        return "kill" if stage is Stage.SCALED else "hold", f"live fills drift {ev.tracking_error_bps:.0f} bps from the backtest"
    if stage is Stage.SCALED:
        return "hold", "fully scaled"
    if ev.days < rules.min_days[stage]:
        return "hold", f"{ev.days:.0f} of {rules.min_days[stage]:.0f} days"
    if ev.deflated_sharpe < rules.min_dsr:
        return "hold", f"deflated Sharpe {ev.deflated_sharpe:.0%} is under {rules.min_dsr:.0%}"
    if not ev.stress_2x_positive:
        return "hold", "does not survive doubled costs"
    return "promote", f"{stage.value} -> {NEXT[stage].value}"
