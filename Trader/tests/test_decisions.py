from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trader.strategies.decisions import SignalVote


def test_signal_vote_is_immutable() -> None:
    vote = SignalVote("rsi_below", True, "RSI 28.4 < 30.0")

    assert vote.name == "rsi_below"
    assert vote.passed is True
    assert vote.detail == "RSI 28.4 < 30.0"
    with pytest.raises(FrozenInstanceError):
        vote.passed = False  # type: ignore[misc]
