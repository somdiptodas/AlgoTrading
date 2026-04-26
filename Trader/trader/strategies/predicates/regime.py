from __future__ import annotations

from trader.data.models import MarketBar
from trader.strategies.filters.regime import (
    _intraday_realized_volatility_bps,
    _percentile_rank,
    _session_progress_stats,
)
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


def normalize_intraday_volatility_params(params: dict[str, object]) -> dict[str, object]:
    merged = {"lookback": 20, "percentile_window": 120, "min_percentile": 50.0, "max_percentile": 100.0, **params}
    lookback = int(merged["lookback"])
    percentile_window = int(merged["percentile_window"])
    min_percentile = float(merged["min_percentile"])
    max_percentile = float(merged["max_percentile"])
    if lookback < 2:
        raise ValueError("intraday_volatility.lookback must be >= 2")
    if percentile_window < 1:
        raise ValueError("intraday_volatility.percentile_window must be >= 1")
    if not 0.0 <= min_percentile <= max_percentile <= 100.0:
        raise ValueError("intraday_volatility percentile bounds must satisfy 0 <= min_percentile <= max_percentile <= 100")
    return {
        "lookback": lookback,
        "percentile_window": percentile_window,
        "min_percentile": min_percentile,
        "max_percentile": max_percentile,
    }


def intraday_volatility_required_history(params: dict[str, object]) -> int:
    return int(params["lookback"]) + int(params["percentile_window"])


def normalize_day_type_params(params: dict[str, object]) -> dict[str, object]:
    merged = {
        "mode": "trend",
        "min_bars": 30,
        "trend_bps": 50.0,
        "min_efficiency": 0.60,
        "max_efficiency": 0.35,
        **params,
    }
    mode = str(merged["mode"])
    min_bars = int(merged["min_bars"])
    trend_bps = float(merged["trend_bps"])
    min_efficiency = float(merged["min_efficiency"])
    max_efficiency = float(merged["max_efficiency"])
    if mode not in {"trend", "mean_reversion"}:
        raise ValueError("day_type.mode must be trend or mean_reversion")
    if min_bars < 1:
        raise ValueError("day_type.min_bars must be >= 1")
    if trend_bps < 0:
        raise ValueError("day_type.trend_bps must be >= 0")
    if not 0.0 <= max_efficiency <= min_efficiency <= 1.0:
        raise ValueError("day_type efficiency bounds must satisfy 0 <= max_efficiency <= min_efficiency <= 1")
    return {
        "mode": mode,
        "min_bars": min_bars,
        "trend_bps": trend_bps,
        "min_efficiency": min_efficiency,
        "max_efficiency": max_efficiency,
    }


def day_type_required_history(params: dict[str, object]) -> int:
    return int(params["min_bars"])


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


def generate_intraday_volatility_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    bars = history_bars + test_bars
    history_count = len(history_bars)
    realized = _intraday_realized_volatility_bps(bars, int(params["lookback"]))
    window = int(params["percentile_window"])
    min_percentile = float(params["min_percentile"])
    max_percentile = float(params["max_percentile"])
    votes: list[SignalVote] = []
    for index in range(history_count, len(bars)):
        sample = [value for value in realized[max(0, index - window) : index] if value is not None]
        percentile = _percentile_rank(realized[index], sample)
        passed = percentile is not None and min_percentile <= percentile <= max_percentile
        detail = (
            "intraday volatility unavailable"
            if percentile is None
            else f"intraday volatility percentile {percentile:.2f} in [{min_percentile:.2f}, {max_percentile:.2f}]"
        )
        votes.append(SignalVote("intraday_volatility", passed, detail))
    return votes


def generate_day_type_votes(
    history_bars: tuple[MarketBar, ...],
    test_bars: tuple[MarketBar, ...],
    params: dict[str, object],
) -> list[SignalVote]:
    bars = history_bars + test_bars
    history_count = len(history_bars)
    stats = _session_progress_stats(bars)
    votes: list[SignalVote] = []
    for index in range(history_count, len(bars)):
        count, move_bps, range_bps, efficiency = stats[index]
        if count < int(params["min_bars"]):
            votes.append(SignalVote("day_type", False, "day type unavailable"))
        elif params["mode"] == "trend":
            passed = move_bps >= float(params["trend_bps"]) and efficiency >= float(params["min_efficiency"])
            detail = f"trend day move {move_bps:.2f} bps efficiency {efficiency:.2f}"
            votes.append(SignalVote("day_type", passed, detail))
        else:
            passed = range_bps >= float(params["trend_bps"]) and efficiency <= float(params["max_efficiency"])
            detail = f"mean reversion day range {range_bps:.2f} bps efficiency {efficiency:.2f}"
            votes.append(SignalVote("day_type", passed, detail))
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
    registry.register(
        "intraday_volatility",
        PredicateHandler(
            normalize_params=normalize_intraday_volatility_params,
            required_history=intraday_volatility_required_history,
            generate_votes=generate_intraday_volatility_votes,
        ),
    )
    registry.register(
        "day_type",
        PredicateHandler(
            normalize_params=normalize_day_type_params,
            required_history=day_type_required_history,
            generate_votes=generate_day_type_votes,
        ),
    )
