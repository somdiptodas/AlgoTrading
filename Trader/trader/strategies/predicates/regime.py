from __future__ import annotations

from trader.data.models import MarketBar
from trader.strategies.decisions import SignalVote
from trader.strategies.predicates.registry import PredicateHandler, PredicateRegistry


def normalize_relative_volume_params(params: dict[str, object]) -> dict[str, object]:
    merged = {"lookback": 20, "min_ratio": 1.25, "max_ratio": 100.0, **params}
    lookback = int(merged["lookback"])
    min_ratio = float(merged["min_ratio"])
    max_ratio = float(merged["max_ratio"])
    if lookback < 1:
        raise ValueError("relative_volume.lookback must be >= 1")
    if min_ratio < 0:
        raise ValueError("relative_volume.min_ratio must be >= 0")
    if max_ratio < min_ratio:
        raise ValueError("relative_volume.max_ratio must be >= min_ratio")
    return {"lookback": lookback, "min_ratio": min_ratio, "max_ratio": max_ratio}


def relative_volume_required_history(params: dict[str, object]) -> int:
    return int(params["lookback"])


def generate_relative_volume_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    bars = history_bars + test_bars
    history_count = len(history_bars)
    lookback = int(params["lookback"])
    min_ratio = float(params["min_ratio"])
    max_ratio = float(params["max_ratio"])
    votes: list[SignalVote] = []
    for index in range(history_count, len(bars)):
        previous = [
            bar.volume
            for bar in bars[max(0, index - lookback) : index]
            if bar.session_date == bars[index].session_date
        ]
        average = sum(previous) / len(previous) if previous else None
        ratio = bars[index].volume / average if average and average > 0 else None
        passed = ratio is not None and min_ratio <= ratio <= max_ratio
        detail = (
            "relative volume unavailable"
            if ratio is None
            else f"relative volume {ratio:.2f} in [{min_ratio:.2f}, {max_ratio:.2f}]"
        )
        votes.append(SignalVote("relative_volume", passed, detail))
    return votes


def register(registry: PredicateRegistry) -> None:
    registry.register(
        "relative_volume",
        PredicateHandler(
            normalize_params=normalize_relative_volume_params,
            required_history=relative_volume_required_history,
            generate_votes=generate_relative_volume_votes,
        ),
    )
