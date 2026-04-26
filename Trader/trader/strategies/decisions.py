from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalVote:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class RuleDecision:
    passed: bool
    reason: str
    votes: tuple[SignalVote, ...] = ()


@dataclass(frozen=True)
class TradeDecision:
    entry: RuleDecision
    exit: RuleDecision
