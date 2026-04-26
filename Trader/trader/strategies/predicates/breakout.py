from __future__ import annotations

from trader.data.models import MarketBar
from trader.features.pipeline import FeaturePipeline
from trader.strategies.decisions import SignalVote
from trader.strategies.predicates.registry import PredicateHandler, PredicateRegistry


def normalize_breakout_params(params: dict[str, object]) -> dict[str, object]:
    merged = {"window": 20, "buffer_bps": 0.0, **params}
    window = int(merged["window"])
    buffer_bps = float(merged["buffer_bps"])
    if window < 2:
        raise ValueError("breakout.window must be >= 2")
    if buffer_bps < 0:
        raise ValueError("breakout.buffer_bps must be >= 0")
    return {"window": window, "buffer_bps": buffer_bps}


def required_history(params: dict[str, object]) -> int:
    return int(params["window"])


def generate_breakout_up_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    pipeline = FeaturePipeline.from_segments(history_bars, test_bars)
    highs = pipeline.rolling_high_for_test(int(params["window"]))
    buffer = 1.0 + float(params["buffer_bps"]) / 10_000.0
    votes: list[SignalVote] = []
    for bar, prior_high in zip(test_bars, highs):
        passed = prior_high is not None and bar.close > prior_high * buffer
        detail = (
            "prior high unavailable"
            if prior_high is None
            else f"close {bar.close:.2f} > prior high {prior_high * buffer:.2f}"
        )
        votes.append(SignalVote("breakout_up", passed, detail))
    return votes


def register(registry: PredicateRegistry) -> None:
    registry.register(
        "breakout_up",
        PredicateHandler(
            normalize_params=normalize_breakout_params,
            required_history=required_history,
            generate_votes=generate_breakout_up_votes,
        ),
    )
