from __future__ import annotations

from trader.data.models import MarketBar
from trader.strategies.decisions import SignalVote
from trader.strategies.predicates.registry import PredicateHandler, PredicateRegistry


def normalize_vwap_distance_params(params: dict[str, object]) -> dict[str, object]:
    merged = {"side": "below", "min_bps": 0.0, "max_bps": 100_000.0, **params}
    side = str(merged["side"])
    min_bps = float(merged["min_bps"])
    max_bps = float(merged["max_bps"])
    if side not in {"below", "above"}:
        raise ValueError("vwap_distance.side must be below or above")
    if min_bps < 0:
        raise ValueError("vwap_distance.min_bps must be >= 0")
    if max_bps < min_bps:
        raise ValueError("vwap_distance.max_bps must be >= min_bps")
    return {"side": side, "min_bps": min_bps, "max_bps": max_bps}


def normalize_vwap_reclaimed_params(params: dict[str, object]) -> dict[str, object]:
    merged = {"min_bps": 0.0, **params}
    min_bps = float(merged["min_bps"])
    if min_bps < 0:
        raise ValueError("vwap_reclaimed.min_bps must be >= 0")
    return {"min_bps": min_bps}


def required_history(params: dict[str, object]) -> int:
    return 0


def generate_vwap_distance_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    side = str(params["side"])
    min_bps = float(params["min_bps"])
    max_bps = float(params["max_bps"])
    votes: list[SignalVote] = []
    for bar in test_bars:
        distance = _vwap_distance_bps(bar, side)
        passed = distance is not None and min_bps <= distance <= max_bps
        detail = (
            "VWAP unavailable"
            if distance is None
            else f"VWAP distance {distance:.2f} bps {side} in [{min_bps:.2f}, {max_bps:.2f}]"
        )
        votes.append(SignalVote("vwap_distance", passed, detail))
    return votes


def generate_vwap_reclaimed_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    min_bps = float(params["min_bps"])
    votes: list[SignalVote] = []
    for bar in test_bars:
        reclaim_level = None if bar.vwap is None or bar.vwap <= 0 else bar.vwap * (1.0 + min_bps / 10_000.0)
        passed = reclaim_level is not None and bar.close >= reclaim_level
        detail = (
            "VWAP unavailable"
            if reclaim_level is None
            else f"close {bar.close:.2f} >= VWAP reclaim {reclaim_level:.2f}"
        )
        votes.append(SignalVote("vwap_reclaimed", passed, detail))
    return votes


def _vwap_distance_bps(bar: MarketBar, side: str) -> float | None:
    if bar.vwap is None or bar.vwap <= 0:
        return None
    if side == "below":
        return ((bar.vwap - bar.close) / bar.vwap) * 10_000.0
    return ((bar.close - bar.vwap) / bar.vwap) * 10_000.0


def register(registry: PredicateRegistry) -> None:
    registry.register(
        "vwap_distance",
        PredicateHandler(
            normalize_params=normalize_vwap_distance_params,
            required_history=required_history,
            generate_votes=generate_vwap_distance_votes,
        ),
    )
    registry.register(
        "vwap_reclaimed",
        PredicateHandler(
            normalize_params=normalize_vwap_reclaimed_params,
            required_history=required_history,
            generate_votes=generate_vwap_reclaimed_votes,
        ),
    )
