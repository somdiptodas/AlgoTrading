from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


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


def signal_vote_to_payload(vote: SignalVote) -> dict[str, object]:
    return {
        "name": vote.name,
        "passed": vote.passed,
        "detail": vote.detail,
    }


def signal_vote_from_payload(payload: Mapping[str, Any]) -> SignalVote:
    return SignalVote(
        name=str(payload["name"]),
        passed=_payload_bool(payload, "passed"),
        detail=str(payload.get("detail", "")),
    )


def rule_decision_to_payload(decision: RuleDecision) -> dict[str, object]:
    return {
        "passed": decision.passed,
        "reason": decision.reason,
        "votes": [signal_vote_to_payload(vote) for vote in decision.votes],
    }


def rule_decision_from_payload(payload: Mapping[str, Any]) -> RuleDecision:
    return RuleDecision(
        passed=_payload_bool(payload, "passed"),
        reason=str(payload["reason"]),
        votes=tuple(
            signal_vote_from_payload(dict(vote))
            for vote in payload.get("votes", ())
        ),
    )


def trade_decision_to_payload(decision: TradeDecision) -> dict[str, object]:
    return {
        "entry": rule_decision_to_payload(decision.entry),
        "exit": rule_decision_to_payload(decision.exit),
    }


def trade_decision_from_payload(payload: Mapping[str, Any]) -> TradeDecision:
    return TradeDecision(
        entry=rule_decision_from_payload(dict(payload["entry"])),
        exit=rule_decision_from_payload(dict(payload["exit"])),
    )


def trade_decision_to_json(decision: TradeDecision, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(
            trade_decision_to_payload(decision),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return json.dumps(
        trade_decision_to_payload(decision),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def trade_decision_from_json(payload: str | bytes | bytearray) -> TradeDecision:
    return trade_decision_from_payload(dict(json.loads(payload)))


def _payload_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
