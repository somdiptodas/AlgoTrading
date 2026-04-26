from __future__ import annotations

import json
from typing import Sequence

from trader.data.models import MarketBar
from trader.strategies.decisions import RuleDecision, SignalVote, TradeDecision
from trader.strategies.predicates import PREDICATES


_COMBINERS = {"all", "any", "k_of_n"}


def normalize_params(params: dict[str, object]) -> dict[str, object]:
    return {
        "entry_rule": _normalize_rule("entry_rule", params.get("entry_rule")),
        "exit_rule": _normalize_rule("exit_rule", params.get("exit_rule")),
    }


def required_history(params: dict[str, object]) -> int:
    return 0


def generate_decisions(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, object],
) -> list[TradeDecision]:
    history_tuple = tuple(history_bars)
    test_tuple = tuple(test_bars)
    entry_decisions = _evaluate_rule(params["entry_rule"], history_tuple, test_tuple)
    exit_decisions = _evaluate_rule(params["exit_rule"], history_tuple, test_tuple)
    return [
        TradeDecision(entry=entry, exit=exit)
        for entry, exit in zip(entry_decisions, exit_decisions)
    ]


def generate_regime(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, object],
) -> list[bool]:
    raise ValueError("multi_signal requires decision-based execution")


def parameter_grid() -> tuple[dict[str, object], ...]:
    return tuple()


def neighbors(params: dict[str, object]) -> tuple[dict[str, object], ...]:
    return tuple()


def _normalize_rule(scope: str, raw_rule: object) -> dict[str, object]:
    if not isinstance(raw_rule, dict):
        raise ValueError(f"multi_signal.{scope} must be a rule payload")
    combiner = str(raw_rule.get("combiner", "all"))
    if combiner not in _COMBINERS:
        raise ValueError("multi_signal.combiner must be all, any, or k_of_n")
    raw_signals = raw_rule.get("signals")
    if not isinstance(raw_signals, (list, tuple)):
        raise ValueError(f"multi_signal.{scope}.signals must be a list of predicate payloads")
    if scope == "entry_rule" and len(raw_signals) < 3:
        raise ValueError("multi_signal.entry_rule.signals must contain at least 3 predicates")
    if scope == "exit_rule" and len(raw_signals) < 3:
        raise ValueError("multi_signal.exit_rule.signals must contain at least 3 predicates")

    signals: list[dict[str, object]] = []
    for index, raw_signal in enumerate(raw_signals):
        if not isinstance(raw_signal, dict):
            raise ValueError(f"multi_signal.{scope}.signals[{index}] must be a predicate payload")
        name = str(raw_signal.get("name", ""))
        raw_params = raw_signal.get("params", {})
        if not isinstance(raw_params, dict):
            raise ValueError(f"multi_signal.{scope}.signals[{index}].params must be a dict")
        signals.append({"name": name, "params": _canonical_params(PREDICATES.validate_params(name, raw_params))})
    signals = sorted(signals, key=_signal_sort_key)

    rule: dict[str, object] = {"combiner": combiner, "signals": signals}
    if combiner == "k_of_n":
        k = int(raw_rule.get("k", len(signals)))
        if not 1 <= k <= len(signals):
            raise ValueError("multi_signal.k must be between 1 and signal count")
        rule["k"] = k
    return rule


def _evaluate_rule(
    rule: object,
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
) -> list[RuleDecision]:
    if not isinstance(rule, dict):
        raise ValueError("multi_signal rules must be normalized before evaluation")
    signals = _signals(rule)
    child_votes = [
        PREDICATES.generate_votes(str(signal["name"]), history_bars, test_bars, signal["params"])
        for signal in signals
    ]
    return [_combine_votes(str(rule["combiner"]), tuple(votes), rule) for votes in zip(*child_votes)]


def _combine_votes(combiner: str, votes: tuple[SignalVote, ...], rule: dict[str, object]) -> RuleDecision:
    passed_count = sum(1 for vote in votes if vote.passed)
    total = len(votes)
    if combiner == "all":
        passed = passed_count == total
        return RuleDecision(passed, _reason("all", passed, passed_count, total), votes)
    if combiner == "any":
        passed = passed_count > 0
        return RuleDecision(passed, _reason("any", passed, passed_count, total), votes)

    k = int(rule["k"])
    passed = passed_count >= k
    return RuleDecision(passed, f"k_of_n {_status(passed)}: {passed_count}/{total} signals, k={k}", votes)


def _signals(rule: dict[str, object]) -> list[dict[str, object]]:
    signals = rule.get("signals")
    if not isinstance(signals, list):
        raise ValueError("multi_signal rule signals must be normalized before evaluation")
    return signals


def _reason(combiner: str, passed: bool, passed_count: int, total: int) -> str:
    return f"{combiner} {_status(passed)}: {passed_count}/{total} signals"


def _status(passed: bool) -> str:
    return "passed" if passed else "failed"


def _canonical_params(params: dict[str, object]) -> dict[str, object]:
    return {key: params[key] for key in sorted(params)}


def _signal_sort_key(signal: dict[str, object]) -> str:
    return json.dumps(signal, sort_keys=True, separators=(",", ":"), allow_nan=False)
