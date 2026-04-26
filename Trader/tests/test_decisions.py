from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trader.strategies.decisions import (
    RuleDecision,
    SignalVote,
    TradeDecision,
    signal_vote_from_payload,
    trade_decision_from_json,
    trade_decision_from_payload,
    trade_decision_to_json,
    trade_decision_to_payload,
)


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


def test_trade_decision_payload_and_json_round_trip() -> None:
    decision = TradeDecision(
        entry=RuleDecision(
            True,
            "k_of_n passed: 2/3 signals",
            (
                SignalVote("rsi_below", True, "RSI 28.4 < 30.0"),
                SignalVote("vwap_distance", False, "VWAP distance 8.0 < 25.0 bps"),
            ),
        ),
        exit=RuleDecision(
            False,
            "any failed: 0/2 signals",
            (SignalVote("rsi_above", False),),
        ),
    )

    payload = trade_decision_to_payload(decision)
    encoded = trade_decision_to_json(decision)

    assert trade_decision_from_payload(payload) == decision
    assert trade_decision_from_json(encoded) == decision
    assert '": ' not in encoded
    assert ', "' not in encoded


def test_decision_payload_readers_accept_legacy_missing_optional_fields() -> None:
    vote = signal_vote_from_payload({"name": "legacy_signal", "passed": True})
    payload = {
        "entry": {"passed": True, "reason": "signal_on"},
        "exit": {"passed": False, "reason": "signal_hold"},
    }

    assert vote == SignalVote("legacy_signal", True)
    assert trade_decision_from_payload(payload) == TradeDecision(
        entry=RuleDecision(True, "signal_on"),
        exit=RuleDecision(False, "signal_hold"),
    )
