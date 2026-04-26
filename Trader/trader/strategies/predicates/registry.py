from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

from trader.data.models import MarketBar
from trader.strategies.decisions import SignalVote

PredicateParams = dict[str, object]
PredicateNormalizer = Callable[[PredicateParams], PredicateParams]
PredicateHistory = Callable[[PredicateParams], int]
PredicateGenerator = Callable[[tuple[MarketBar, ...], tuple[MarketBar, ...], PredicateParams], list[SignalVote]]


@dataclass(frozen=True)
class PredicateHandler:
    normalize_params: PredicateNormalizer
    required_history: PredicateHistory
    generate_votes: PredicateGenerator


class PredicateRegistry:
    def __init__(self, handlers: Mapping[str, PredicateHandler] | None = None) -> None:
        self.handlers = dict(handlers or {})

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.handlers))

    def register(self, name: str, handler: PredicateHandler) -> None:
        if name in self.handlers:
            raise ValueError(f"Duplicate atomic predicate: {name}")
        self.handlers[name] = handler

    def validate_params(self, name: str, params: PredicateParams) -> PredicateParams:
        handler = self._handler(name)
        normalized = handler.normalize_params(dict(params))
        _validate_finite_params(name, normalized)
        return normalized

    def required_history(self, name: str, params: PredicateParams) -> int:
        handler = self._handler(name)
        normalized = self.validate_params(name, params)
        return int(handler.required_history(normalized))

    def generate_votes(
        self,
        name: str,
        history_bars: tuple[MarketBar, ...],
        test_bars: tuple[MarketBar, ...],
        params: PredicateParams,
    ) -> list[SignalVote]:
        handler = self._handler(name)
        normalized = self.validate_params(name, params)
        votes = handler.generate_votes(history_bars, test_bars, normalized)
        if len(votes) != len(test_bars):
            raise ValueError(f"{name} produced {len(votes)} votes for {len(test_bars)} test bars")
        if any(vote.name != name for vote in votes):
            raise ValueError(f"{name} produced a vote with the wrong predicate name")
        return votes

    def _handler(self, name: str) -> PredicateHandler:
        if name not in self.handlers:
            raise ValueError(f"Unknown atomic predicate: {name}")
        return self.handlers[name]


def _validate_finite_params(scope: str, params: PredicateParams) -> None:
    for name, value in params.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{scope}.{name} must be finite")


PREDICATES = PredicateRegistry()
