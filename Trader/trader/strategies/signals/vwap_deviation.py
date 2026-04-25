from __future__ import annotations

from typing import Sequence

from trader.data.models import MarketBar


DEFAULT_PARAMS = {
    "entry_deviation_bps": 25.0,
    "exit_deviation_bps": 0.0,
    "max_hold_bars": 30,
}


def normalize_params(params: dict[str, object]) -> dict[str, int | float]:
    merged = {**DEFAULT_PARAMS, **params}
    entry_deviation_bps = float(merged["entry_deviation_bps"])
    exit_deviation_bps = float(merged["exit_deviation_bps"])
    max_hold_bars = int(merged["max_hold_bars"])
    if entry_deviation_bps <= 0:
        raise ValueError("vwap_deviation.entry_deviation_bps must be > 0")
    if exit_deviation_bps < 0:
        raise ValueError("vwap_deviation.exit_deviation_bps must be >= 0")
    if exit_deviation_bps >= entry_deviation_bps:
        raise ValueError("vwap_deviation.exit_deviation_bps must be less than entry_deviation_bps")
    if max_hold_bars < 1:
        raise ValueError("vwap_deviation.max_hold_bars must be >= 1")
    return {
        "entry_deviation_bps": entry_deviation_bps,
        "exit_deviation_bps": exit_deviation_bps,
        "max_hold_bars": max_hold_bars,
    }


def required_history(params: dict[str, int | float]) -> int:
    return 0


def generate_regime(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float],
) -> list[bool]:
    entry_deviation_bps = float(params["entry_deviation_bps"])
    exit_deviation_bps = float(params["exit_deviation_bps"])
    max_hold_bars = int(params["max_hold_bars"])
    regime = False
    held_bars = 0
    output: list[bool] = []
    for bar in test_bars:
        vwap = bar.vwap
        if vwap is None or vwap <= 0:
            regime = False
            held_bars = 0
        elif not regime:
            if bar.close <= vwap * (1.0 - entry_deviation_bps / 10_000.0):
                regime = True
                held_bars = 1
        else:
            held_bars += 1
            reverted = bar.close >= vwap * (1.0 - exit_deviation_bps / 10_000.0)
            if reverted or held_bars > max_hold_bars:
                regime = False
                held_bars = 0
        output.append(regime)
    return output


def parameter_grid() -> tuple[dict[str, int | float], ...]:
    grid: list[dict[str, int | float]] = []
    for entry_deviation_bps in (10.0, 25.0, 50.0, 100.0):
        for exit_deviation_bps in (0.0, 5.0, 10.0):
            if exit_deviation_bps >= entry_deviation_bps:
                continue
            for max_hold_bars in (10, 30, 60):
                grid.append(
                    {
                        "entry_deviation_bps": entry_deviation_bps,
                        "exit_deviation_bps": exit_deviation_bps,
                        "max_hold_bars": max_hold_bars,
                    }
                )
    return tuple(grid)


def neighbors(params: dict[str, int | float]) -> tuple[dict[str, int | float], ...]:
    entry_deviation_bps = float(params["entry_deviation_bps"])
    exit_deviation_bps = float(params["exit_deviation_bps"])
    max_hold_bars = int(params["max_hold_bars"])
    candidates = {
        (entry_deviation_bps - 5.0, exit_deviation_bps, max_hold_bars),
        (entry_deviation_bps + 5.0, exit_deviation_bps, max_hold_bars),
        (entry_deviation_bps, max(0.0, exit_deviation_bps - 5.0), max_hold_bars),
        (entry_deviation_bps, exit_deviation_bps + 5.0, max_hold_bars),
        (entry_deviation_bps, exit_deviation_bps, max_hold_bars - 5),
        (entry_deviation_bps, exit_deviation_bps, max_hold_bars + 5),
    }
    normalized: list[dict[str, int | float]] = []
    for candidate_entry, candidate_exit, candidate_hold in sorted(candidates):
        try:
            normalized.append(
                normalize_params(
                    {
                        "entry_deviation_bps": candidate_entry,
                        "exit_deviation_bps": candidate_exit,
                        "max_hold_bars": candidate_hold,
                    }
                )
            )
        except ValueError:
            continue
    return tuple(normalized)
