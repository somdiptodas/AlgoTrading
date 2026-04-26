from trader.strategies.predicates.registry import (
    PREDICATES,
    PredicateHandler,
    PredicateRegistry,
)
from trader.strategies.predicates import breakout as _breakout
from trader.strategies.predicates import ema as _ema
from trader.strategies.predicates import regime as _regime
from trader.strategies.predicates import rsi as _rsi
from trader.strategies.predicates import vwap as _vwap

_breakout.register(PREDICATES)
_ema.register(PREDICATES)
_regime.register(PREDICATES)
_rsi.register(PREDICATES)
_vwap.register(PREDICATES)

__all__ = ["PREDICATES", "PredicateHandler", "PredicateRegistry"]
