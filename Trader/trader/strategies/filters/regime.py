from __future__ import annotations

import math
from collections import OrderedDict
from typing import Sequence

from trader.data.models import MarketBar


def normalize_intraday_volatility_params(params: dict[str, object]) -> dict[str, int | float]:
    merged = {"lookback_bars": 20, "percentile_window": 120, "min_percentile": 50.0, "max_percentile": 100.0, **params}
    lookback_bars = int(merged["lookback_bars"])
    percentile_window = int(merged["percentile_window"])
    min_percentile = float(merged["min_percentile"])
    max_percentile = float(merged["max_percentile"])
    if lookback_bars < 2:
        raise ValueError("intraday_volatility.lookback_bars must be >= 2")
    if percentile_window < 1:
        raise ValueError("intraday_volatility.percentile_window must be >= 1")
    _validate_percentile_bounds("intraday_volatility", min_percentile, max_percentile)
    return {
        "lookback_bars": lookback_bars,
        "percentile_window": percentile_window,
        "min_percentile": min_percentile,
        "max_percentile": max_percentile,
    }


def generate_intraday_volatility_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float],
) -> list[bool]:
    bars = tuple(history_bars) + tuple(test_bars)
    history_count = len(history_bars)
    lookback_bars = int(params["lookback_bars"])
    percentile_window = int(params["percentile_window"])
    realized = _intraday_realized_volatility_bps(bars, lookback_bars)
    output: list[bool] = []
    for index in range(history_count, len(bars)):
        value = realized[index]
        sample = [
            item for item in realized[max(0, index - percentile_window) : index]
            if item is not None
        ]
        percentile = _percentile_rank(value, sample)
        output.append(
            percentile is not None
            and float(params["min_percentile"]) <= percentile <= float(params["max_percentile"])
        )
    return output


def normalize_prior_day_range_params(params: dict[str, object]) -> dict[str, float]:
    merged = {"min_range_bps": 50.0, "max_range_bps": 100_000.0, **params}
    min_range_bps = float(merged["min_range_bps"])
    max_range_bps = float(merged["max_range_bps"])
    if min_range_bps < 0:
        raise ValueError("prior_day_range.min_range_bps must be >= 0")
    if max_range_bps < min_range_bps:
        raise ValueError("prior_day_range.max_range_bps must be >= min_range_bps")
    return {"min_range_bps": min_range_bps, "max_range_bps": max_range_bps}


def generate_prior_day_range_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, float],
) -> list[bool]:
    bars = tuple(history_bars) + tuple(test_bars)
    ranges = _prior_session_range_bps(bars)
    return [
        (range_bps := ranges.get(bar.session_date)) is not None
        and params["min_range_bps"] <= range_bps <= params["max_range_bps"]
        for bar in test_bars
    ]


def normalize_relative_volume_params(params: dict[str, object]) -> dict[str, int | float]:
    merged = {"lookback_bars": 20, "min_ratio": 1.25, "max_ratio": 100.0, **params}
    lookback_bars = int(merged["lookback_bars"])
    min_ratio = float(merged["min_ratio"])
    max_ratio = float(merged["max_ratio"])
    if lookback_bars < 1:
        raise ValueError("relative_volume.lookback_bars must be >= 1")
    if min_ratio < 0:
        raise ValueError("relative_volume.min_ratio must be >= 0")
    if max_ratio < min_ratio:
        raise ValueError("relative_volume.max_ratio must be >= min_ratio")
    return {"lookback_bars": lookback_bars, "min_ratio": min_ratio, "max_ratio": max_ratio}


def generate_relative_volume_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float],
) -> list[bool]:
    bars = tuple(history_bars) + tuple(test_bars)
    history_count = len(history_bars)
    lookback_bars = int(params["lookback_bars"])
    output: list[bool] = []
    for index in range(history_count, len(bars)):
        previous = [
            bar.volume for bar in bars[max(0, index - lookback_bars) : index]
            if bar.session_date == bars[index].session_date
        ]
        average_volume = sum(previous) / len(previous) if previous else None
        ratio = bars[index].volume / average_volume if average_volume and average_volume > 0 else None
        output.append(ratio is not None and float(params["min_ratio"]) <= ratio <= float(params["max_ratio"]))
    return output


def normalize_day_type_params(params: dict[str, object]) -> dict[str, int | float | str]:
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


def generate_day_type_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float | str],
) -> list[bool]:
    bars = tuple(history_bars) + tuple(test_bars)
    history_count = len(history_bars)
    stats = _session_progress_stats(bars)
    output: list[bool] = []
    for index in range(history_count, len(bars)):
        count, move_bps, range_bps, efficiency = stats[index]
        if count < int(params["min_bars"]):
            output.append(False)
        elif params["mode"] == "trend":
            output.append(move_bps >= float(params["trend_bps"]) and efficiency >= float(params["min_efficiency"]))
        else:
            output.append(range_bps >= float(params["trend_bps"]) and efficiency <= float(params["max_efficiency"]))
    return output


def _intraday_realized_volatility_bps(bars: Sequence[MarketBar], lookback_bars: int) -> list[float | None]:
    returns: list[float | None] = [None]
    for previous, current in zip(bars, bars[1:]):
        if previous.session_date != current.session_date or previous.close <= 0:
            returns.append(None)
        else:
            returns.append(abs((current.close / previous.close) - 1.0) * 10_000.0)
    realized: list[float | None] = []
    for index in range(len(bars)):
        window = returns[index - lookback_bars + 1 : index + 1]
        if len(window) != lookback_bars or any(value is None for value in window):
            realized.append(None)
        else:
            realized.append(math.sqrt(sum(float(value) ** 2 for value in window) / lookback_bars))
    return realized


def _prior_session_range_bps(bars: Sequence[MarketBar]) -> dict[str, float]:
    sessions: OrderedDict[str, list[MarketBar]] = OrderedDict()
    for bar in bars:
        sessions.setdefault(bar.session_date, []).append(bar)
    prior_by_session: dict[str, float] = {}
    previous_range: float | None = None
    for session_date, session_bars in sessions.items():
        prior_by_session[session_date] = previous_range if previous_range is not None else math.nan
        last_close = session_bars[-1].close
        previous_range = (
            ((max(bar.high for bar in session_bars) - min(bar.low for bar in session_bars)) / last_close) * 10_000.0
            if last_close > 0
            else None
        )
    return {key: value for key, value in prior_by_session.items() if value is not None and math.isfinite(value)}


def _session_progress_stats(bars: Sequence[MarketBar]) -> list[tuple[int, float, float, float]]:
    stats: list[tuple[int, float, float, float]] = []
    session_date: str | None = None
    session_open = 0.0
    session_high = 0.0
    session_low = 0.0
    count = 0
    for bar in bars:
        if bar.session_date != session_date:
            session_date = bar.session_date
            session_open = bar.open
            session_high = bar.high
            session_low = bar.low
            count = 0
        session_high = max(session_high, bar.high)
        session_low = min(session_low, bar.low)
        count += 1
        move = abs(bar.close - session_open)
        range_ = session_high - session_low
        move_bps = (move / session_open) * 10_000.0 if session_open > 0 else 0.0
        range_bps = (range_ / session_open) * 10_000.0 if session_open > 0 else 0.0
        efficiency = move / range_ if range_ > 0 else 0.0
        stats.append((count, move_bps, range_bps, efficiency))
    return stats


def _percentile_rank(value: float | None, sample: Sequence[float]) -> float | None:
    if value is None or not sample:
        return None
    return 100.0 * sum(item <= value for item in sample) / len(sample)


def _validate_percentile_bounds(scope: str, minimum: float, maximum: float) -> None:
    if not 0.0 <= minimum <= maximum <= 100.0:
        raise ValueError(f"{scope} percentile bounds must satisfy 0 <= min_percentile <= max_percentile <= 100")
