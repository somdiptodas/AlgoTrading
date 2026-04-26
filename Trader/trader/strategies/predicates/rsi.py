from __future__ import annotations

from trader.data.models import MarketBar
from trader.features.pipeline import FeaturePipeline
from trader.strategies.decisions import SignalVote
from trader.strategies.predicates.registry import PredicateHandler, PredicateRegistry


def normalize_rsi_params(params: dict[str, object]) -> dict[str, object]:
    merged = {"length": 14, "threshold": 30.0, **params}
    length = int(merged["length"])
    threshold = float(merged["threshold"])
    if length < 2:
        raise ValueError("rsi.length must be >= 2")
    if not 0.0 < threshold < 100.0:
        raise ValueError("rsi.threshold must be in (0, 100)")
    return {"length": length, "threshold": threshold}


def required_history(params: dict[str, object]) -> int:
    return int(params["length"]) + 1


def generate_rsi_below_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    pipeline = FeaturePipeline.from_segments(history_bars, test_bars)
    threshold = float(params["threshold"])
    votes: list[SignalVote] = []
    for value in pipeline.rsi_for_test(int(params["length"])):
        passed = value is not None and value < threshold
        detail = "RSI unavailable" if value is None else f"RSI {value:.2f} < {threshold:.2f}"
        votes.append(SignalVote("rsi_below", passed, detail))
    return votes


def generate_rsi_above_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    pipeline = FeaturePipeline.from_segments(history_bars, test_bars)
    threshold = float(params["threshold"])
    votes: list[SignalVote] = []
    for value in pipeline.rsi_for_test(int(params["length"])):
        passed = value is not None and value > threshold
        detail = "RSI unavailable" if value is None else f"RSI {value:.2f} > {threshold:.2f}"
        votes.append(SignalVote("rsi_above", passed, detail))
    return votes


def register(registry: PredicateRegistry) -> None:
    registry.register(
        "rsi_below",
        PredicateHandler(
            normalize_params=normalize_rsi_params,
            required_history=required_history,
            generate_votes=generate_rsi_below_votes,
        ),
    )
    registry.register(
        "rsi_above",
        PredicateHandler(
            normalize_params=normalize_rsi_params,
            required_history=required_history,
            generate_votes=generate_rsi_above_votes,
        ),
    )
