from trader.strategies.predicates.registry import (
    PREDICATES,
    PredicateHandler,
    PredicateRegistry,
)
from trader.strategies.predicates import ema as _ema
from trader.strategies.predicates import rsi as _rsi

_ema.register(PREDICATES)
_rsi.register(PREDICATES)

__all__ = ["PREDICATES", "PredicateHandler", "PredicateRegistry"]
