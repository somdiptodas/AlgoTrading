from __future__ import annotations

from typing import Sequence

from trader.data.models import MarketBar
from trader.strategies.decisions import TradeDecision
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
    raise NotImplementedError("multi_signal decision evaluation is not implemented yet")


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

    signals: list[dict[str, object]] = []
    for index, raw_signal in enumerate(raw_signals):
        if not isinstance(raw_signal, dict):
            raise ValueError(f"multi_signal.{scope}.signals[{index}] must be a predicate payload")
        name = str(raw_signal.get("name", ""))
        raw_params = raw_signal.get("params", {})
        if not isinstance(raw_params, dict):
            raise ValueError(f"multi_signal.{scope}.signals[{index}].params must be a dict")
        signals.append({"name": name, "params": PREDICATES.validate_params(name, raw_params)})

    rule: dict[str, object] = {"combiner": combiner, "signals": signals}
    if combiner == "k_of_n":
        rule["k"] = int(raw_rule.get("k", len(signals)))
    return rule
