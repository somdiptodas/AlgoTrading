from __future__ import annotations

import math
from typing import Sequence

from trader.data.models import MarketBar
from trader.strategies.signals import breakout, ema_cross, rsi_reversion, vwap_deviation


_SIGNAL_HANDLERS = {
    "ema_cross": {
        "normalize_params": ema_cross.normalize_params,
        "required_history": ema_cross.required_history,
        "generate_regime": ema_cross.generate_regime,
        "neighbors": ema_cross.neighbors,
    },
    "breakout": {
        "normalize_params": breakout.normalize_params,
        "required_history": breakout.required_history,
        "generate_regime": breakout.generate_regime,
        "neighbors": breakout.neighbors,
    },
    "rsi_reversion": {
        "normalize_params": rsi_reversion.normalize_params,
        "required_history": rsi_reversion.required_history,
        "generate_regime": rsi_reversion.generate_regime,
        "neighbors": rsi_reversion.neighbors,
    },
    "vwap_deviation": {
        "normalize_params": vwap_deviation.normalize_params,
        "required_history": vwap_deviation.required_history,
        "generate_regime": vwap_deviation.generate_regime,
        "neighbors": vwap_deviation.neighbors,
    },
}
_COMBINERS = {"all", "any", "vote_k_of_n", "primary_plus_confirmations"}


def normalize_params(params: dict[str, object]) -> dict[str, object]:
    combiner = str(params.get("combiner", "all"))
    if combiner not in _COMBINERS:
        raise ValueError("composite.combiner must be all, any, vote_k_of_n, or primary_plus_confirmations")

    raw_children = params.get("children")
    if not isinstance(raw_children, (list, tuple)):
        raise ValueError("composite.children must be a list of signal payloads")
    if not 2 <= len(raw_children) <= 4:
        raise ValueError("composite.children must contain 2 to 4 signals")

    children: list[dict[str, object]] = []
    for index, raw_child in enumerate(raw_children):
        if not isinstance(raw_child, dict):
            raise ValueError(f"composite.children[{index}] must be a signal payload")
        name = str(raw_child.get("name", ""))
        if name not in _SIGNAL_HANDLERS:
            raise ValueError(f"Unknown composite child signal handler: {name}")
        raw_params = raw_child.get("params", {})
        if not isinstance(raw_params, dict):
            raise ValueError(f"composite.children[{index}].params must be a dict")
        normalized_params = _SIGNAL_HANDLERS[name]["normalize_params"](raw_params)
        _validate_finite_params(f"composite.children[{index}].params", normalized_params)
        children.append({"name": name, "params": normalized_params})

    normalized: dict[str, object] = {
        "combiner": combiner,
        "children": children,
    }
    if combiner == "vote_k_of_n":
        min_agreeing = int(params.get("min_agreeing", (len(children) + 1) // 2))
        if not 1 <= min_agreeing <= len(children):
            raise ValueError("composite.min_agreeing must be between 1 and child count")
        normalized["min_agreeing"] = min_agreeing
    if combiner == "primary_plus_confirmations":
        primary_index = int(params.get("primary_index", 0))
        if not 0 <= primary_index < len(children):
            raise ValueError("composite.primary_index must select a child")
        normalized["primary_index"] = primary_index
    return normalized


def required_history(params: dict[str, object]) -> int:
    children = _children(params)
    return max(
        int(_SIGNAL_HANDLERS[str(child["name"])]["required_history"](child["params"]))
        for child in children
    )


def generate_regime(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, object],
) -> list[bool]:
    children = _children(params)
    child_regimes = [
        _SIGNAL_HANDLERS[str(child["name"])]["generate_regime"](history_bars, test_bars, child["params"])
        for child in children
    ]
    combiner = str(params["combiner"])
    if combiner == "all":
        return [all(values) for values in zip(*child_regimes)]
    if combiner == "any":
        return [any(values) for values in zip(*child_regimes)]
    if combiner == "vote_k_of_n":
        min_agreeing = int(params["min_agreeing"])
        return [sum(values) >= min_agreeing for values in zip(*child_regimes)]

    primary_index = int(params["primary_index"])
    return [
        values[primary_index] and all(values[index] for index in range(len(values)) if index != primary_index)
        for values in zip(*child_regimes)
    ]


def parameter_grid() -> tuple[dict[str, object], ...]:
    return tuple()


def neighbors(params: dict[str, object]) -> tuple[dict[str, object], ...]:
    children = _children(params)
    candidates: list[dict[str, object]] = []
    for child_index, child in enumerate(children):
        handler = _SIGNAL_HANDLERS[str(child["name"])]
        for neighbor_params in handler["neighbors"](child["params"]):
            neighbor_children = [
                {"name": str(item["name"]), "params": dict(item["params"])}
                for item in children
            ]
            neighbor_children[child_index] = {"name": str(child["name"]), "params": neighbor_params}
            candidate = dict(params)
            candidate["children"] = neighbor_children
            candidates.append(normalize_params(candidate))
            if len(candidates) >= 6:
                return tuple(candidates)
    return tuple(candidates)


def _children(params: dict[str, object]) -> list[dict[str, object]]:
    children = params.get("children")
    if not isinstance(children, list):
        raise ValueError("composite.children must be normalized before use")
    return children


def _validate_finite_params(scope: str, params: dict[str, object]) -> None:
    for name, value in params.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{scope}.{name} must be finite")
