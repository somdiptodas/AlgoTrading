from __future__ import annotations

from trader.data.models import MarketBar
from trader.features.pipeline import FeaturePipeline
from trader.strategies.decisions import SignalVote
from trader.strategies.predicates.registry import PredicateHandler, PredicateRegistry


def normalize_ema_params(params: dict[str, object]) -> dict[str, object]:
    merged = {"fast": 20, "slow": 80, "buffer_bps": 0.0, **params}
    fast = int(merged["fast"])
    slow = int(merged["slow"])
    buffer_bps = float(merged["buffer_bps"])
    if fast < 2:
        raise ValueError("ema.fast must be >= 2")
    if slow <= fast:
        raise ValueError("ema.slow must be greater than fast")
    if buffer_bps < 0:
        raise ValueError("ema.buffer_bps must be >= 0")
    return {"fast": fast, "slow": slow, "buffer_bps": buffer_bps}


def required_history(params: dict[str, object]) -> int:
    return int(params["slow"])


def generate_ema_trend_up_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    pipeline = FeaturePipeline.from_segments(history_bars, test_bars)
    fast_ema = pipeline.ema_for_test(int(params["fast"]))
    slow_ema = pipeline.ema_for_test(int(params["slow"]))
    buffer = 1.0 + float(params["buffer_bps"]) / 10_000.0
    votes: list[SignalVote] = []
    for index in range(len(test_bars)):
        enough_history = pipeline.combined_index_for_test(index) + 1 >= int(params["slow"])
        passed = enough_history and fast_ema[index] > slow_ema[index] * buffer
        detail = f"fast EMA {fast_ema[index]:.2f} > slow EMA {slow_ema[index] * buffer:.2f}"
        votes.append(SignalVote("ema_trend_up", passed, detail if enough_history else "EMA unavailable"))
    return votes


def generate_ema_trend_down_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    pipeline = FeaturePipeline.from_segments(history_bars, test_bars)
    fast_ema = pipeline.ema_for_test(int(params["fast"]))
    slow_ema = pipeline.ema_for_test(int(params["slow"]))
    buffer = 1.0 - float(params["buffer_bps"]) / 10_000.0
    votes: list[SignalVote] = []
    for index in range(len(test_bars)):
        enough_history = pipeline.combined_index_for_test(index) + 1 >= int(params["slow"])
        passed = enough_history and fast_ema[index] < slow_ema[index] * buffer
        detail = f"fast EMA {fast_ema[index]:.2f} < slow EMA {slow_ema[index] * buffer:.2f}"
        votes.append(SignalVote("ema_trend_down", passed, detail if enough_history else "EMA unavailable"))
    return votes


def register(registry: PredicateRegistry) -> None:
    registry.register(
        "ema_trend_up",
        PredicateHandler(
            normalize_params=normalize_ema_params,
            required_history=required_history,
            generate_votes=generate_ema_trend_up_votes,
        ),
    )
    registry.register(
        "ema_trend_down",
        PredicateHandler(
            normalize_params=normalize_ema_params,
            required_history=required_history,
            generate_votes=generate_ema_trend_down_votes,
        ),
    )
