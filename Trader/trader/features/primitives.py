from __future__ import annotations

from typing import Sequence


def ema(values: Sequence[float], length: int) -> list[float]:
    if length < 1:
        raise ValueError("length must be >= 1")
    if not values:
        return []
    alpha = 2.0 / (length + 1)
    result: list[float] = []
    current: float | None = None
    for value in values:
        current = value if current is None else (value * alpha) + (current * (1.0 - alpha))
        result.append(current)
    return result


def rolling_max_exclusive(values: Sequence[float], window: int) -> list[float | None]:
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float | None] = []
    for index in range(len(values)):
        if index < window:
            result.append(None)
        else:
            result.append(max(values[index - window : index]))
    return result


def rolling_min_exclusive(values: Sequence[float], window: int) -> list[float | None]:
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float | None] = []
    for index in range(len(values)):
        if index < window:
            result.append(None)
        else:
            result.append(min(values[index - window : index]))
    return result
