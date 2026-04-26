from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trader.strategies.decisions import RuleDecision, SignalVote, TradeDecision


def test_signal_vote_is_immutable() -> None:
    vote = SignalVote("rsi_below", True, "RSI 28.4 < 30.0")

    assert vote.name == "rsi_below"
    assert vote.passed is True
    assert vote.detail == "RSI 28.4 < 30.0"
    with pytest.raises(FrozenInstanceError):
        vote.passed = False  # type: ignore[misc]


def test_rule_decision_is_immutable_and_keeps_votes() -> None:
    vote = SignalVote("rsi_below", True)
    decision = RuleDecision(True, "all passed", (vote,))

    assert decision.passed is True
    assert decision.reason == "all passed"
    assert decision.votes == (vote,)
    with pytest.raises(FrozenInstanceError):
        decision.reason = "changed"  # type: ignore[misc]


def test_trade_decision_is_immutable_and_keeps_entry_exit_decisions() -> None:
    entry = RuleDecision(True, "entry passed")
    exit = RuleDecision(False, "exit blocked")
    decision = TradeDecision(entry, exit)

    assert decision.entry == entry
    assert decision.exit == exit
    with pytest.raises(FrozenInstanceError):
        decision.exit = RuleDecision(True, "changed")  # type: ignore[misc]
