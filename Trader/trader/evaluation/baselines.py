from __future__ import annotations

import math
import random
from dataclasses import replace
from datetime import datetime
from hashlib import sha256

from trader.data.models import NEW_YORK, MarketBar
from trader.evaluation.metrics import annualized_sharpe, calculate_metrics, max_drawdown_pct, sharpe_like
from trader.execution.engine import BacktestResult, run_long_only_engine
from trader.execution.fills import Trade, enter_long, exit_long
from trader.strategies.spec import ExecConfig


def evaluate_baselines(
    bars: tuple[MarketBar, ...],
    exec_config: ExecConfig,
    *,
    strategy_trades: tuple[Trade, ...],
    seed_material: str,
) -> dict[str, dict[str, float]]:
    return {
        "always_flat": always_flat(exec_config.initial_cash),
        "buy_and_hold": buy_and_hold(bars, exec_config.initial_cash),
        "regular_session_open_to_close_long": regular_session_open_to_close_long(bars, exec_config),
        "session_long_flat_at_close": session_long_flat_at_close(bars, exec_config),
        "randomized_entry_same_exposure": randomized_entry_same_exposure(
            bars,
            exec_config,
            strategy_trades,
            seed_material=seed_material,
        ),
    }


def always_flat(initial_cash: float) -> dict[str, float]:
    return {
        "return_pct": 0.0,
        "annualized_sharpe": 0.0,
        "sharpe_like": 0.0,
        "max_drawdown_pct": 0.0,
        "exposure_pct": 0.0,
        "trade_count": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "avg_trade_pct": 0.0,
        "final_cash": initial_cash,
    }


def buy_and_hold(bars: tuple[MarketBar, ...], initial_cash: float) -> dict[str, float]:
    if not bars:
        return always_flat(initial_cash)
    starting_close = bars[0].close
    equity_curve = tuple(initial_cash * (bar.close / starting_close) for bar in bars)
    total_return_pct = ((bars[-1].close / starting_close) - 1.0) * 100.0
    return {
        "return_pct": total_return_pct,
        "annualized_sharpe": annualized_sharpe(equity_curve, starting_equity=initial_cash),
        "sharpe_like": sharpe_like(equity_curve),
        "max_drawdown_pct": max_drawdown_pct(equity_curve),
        "exposure_pct": 100.0,
        "trade_count": 1.0,
        "win_rate_pct": 100.0 if total_return_pct > 0 else 0.0,
        "profit_factor": float("inf") if total_return_pct > 0 else 0.0,
        "avg_trade_pct": total_return_pct,
        "final_cash": equity_curve[-1],
    }


def regular_session_open_to_close_long(bars: tuple[MarketBar, ...], exec_config: ExecConfig) -> dict[str, float]:
    if not bars:
        return always_flat(exec_config.initial_cash)
    cash = exec_config.initial_cash
    equity_curve: list[float] = []
    trades: list[Trade] = []
    for start, end in _session_ranges(bars):
        entry_bar = bars[start]
        exit_bar = bars[end - 1]
        cash, position = enter_long(cash, entry_bar, exec_config, sizing_fraction=1.0)
        if position is None:
            equity_curve.extend(cash for _ in bars[start:end])
            continue
        position.bars_held = end - start
        for bar in bars[start:end]:
            equity_curve.append(cash + (position.shares * bar.close))
        cash, trade = exit_long(cash, position, exit_bar, exec_config, "session_close", fill_at_close=True)
        trades.append(trade)
        equity_curve[-1] = cash
    return _result_metrics(
        BacktestResult(
            bars=bars,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve),
            initial_cash=exec_config.initial_cash,
            final_cash=cash,
        )
    )


def session_long_flat_at_close(bars: tuple[MarketBar, ...], exec_config: ExecConfig) -> dict[str, float]:
    if not bars:
        return always_flat(exec_config.initial_cash)
    result = run_long_only_engine(
        bars,
        tuple(True for _ in bars),
        replace(exec_config, flat_at_close=True),
        sizing_fraction=1.0,
    )
    return _result_metrics(result)


def randomized_entry_same_exposure(
    bars: tuple[MarketBar, ...],
    exec_config: ExecConfig,
    strategy_trades: tuple[Trade, ...],
    *,
    seed_material: str,
) -> dict[str, float]:
    trade_targets = tuple(
        (_trade_session_date(trade.entry_timestamp_utc), max(1, trade.bars_held))
        for trade in strategy_trades
        if trade.bars_held > 0
    )
    if not bars or not trade_targets:
        return always_flat(exec_config.initial_cash)

    rng = random.Random(_random_seed(bars, trade_targets, seed_material))
    expected_trade_count = float(len(trade_targets))
    expected_exposure_pct = (sum(hold for _, hold in trade_targets) / len(bars)) * 100.0
    for attempt in range(65):
        randomize = attempt < 64
        windows = _randomized_windows(bars, trade_targets, rng, randomize=randomize)
        if len(windows) != len(trade_targets):
            continue
        metrics = _simulate_windows(bars, exec_config, sorted(windows))
        if metrics["trade_count"] == expected_trade_count and math.isclose(
            metrics["exposure_pct"],
            expected_exposure_pct,
        ):
            return metrics
    return always_flat(exec_config.initial_cash)


def baseline_deltas(metrics: dict[str, float], baselines: dict[str, dict[str, float]]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for baseline_name, baseline_metrics in baselines.items():
        deltas[f"delta_{baseline_name}_return_pct"] = metrics["return_pct"] - baseline_metrics["return_pct"]
        deltas[f"delta_{baseline_name}_annualized_sharpe"] = (
            metrics["annualized_sharpe"] - baseline_metrics["annualized_sharpe"]
        )
        deltas[f"delta_{baseline_name}_sharpe_like"] = metrics["sharpe_like"] - baseline_metrics["sharpe_like"]
    if "buy_and_hold" in baselines:
        deltas["delta_exposure_adjusted_buy_and_hold_pct"] = metrics["return_pct"] - (
            (metrics.get("exposure_pct", 0.0) / 100.0) * baselines["buy_and_hold"]["return_pct"]
        )
    return deltas


def _result_metrics(result: BacktestResult) -> dict[str, float]:
    metrics = calculate_metrics(result)
    metrics["final_cash"] = result.final_cash
    return metrics


def _session_ranges(bars: tuple[MarketBar, ...]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(bars)):
        if bars[index].session_date != bars[start].session_date:
            ranges.append((start, index))
            start = index
    if bars:
        ranges.append((start, len(bars)))
    return tuple(ranges)


def _holds_by_session(trade_targets: tuple[tuple[str, int], ...]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for session_date, hold in trade_targets:
        grouped.setdefault(session_date, []).append(hold)
    return grouped


def _randomized_windows(
    bars: tuple[MarketBar, ...],
    trade_targets: tuple[tuple[str, int], ...],
    rng: random.Random,
    *,
    randomize: bool,
) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    for session_date, holds in _holds_by_session(trade_targets).items():
        session_range = _range_for_session(bars, session_date)
        if session_range is None:
            continue
        session_start, session_end = session_range
        session_windows = _random_session_windows(session_start, session_end, holds, rng, randomize=randomize)
        if not session_windows:
            return []
        windows.extend(session_windows)
    return windows


def _range_for_session(bars: tuple[MarketBar, ...], session_date: str) -> tuple[int, int] | None:
    for start, end in _session_ranges(bars):
        if bars[start].session_date == session_date:
            return start, end
    return None


def _random_session_windows(
    session_start: int,
    session_end: int,
    holds: list[int],
    rng: random.Random,
    *,
    randomize: bool,
) -> list[tuple[int, int, int]]:
    total_hold = sum(holds)
    session_length = session_end - session_start
    if total_hold > session_length:
        return []
    shuffled_holds = list(holds)
    if randomize:
        rng.shuffle(shuffled_holds)
    gaps = [0 for _ in range(len(shuffled_holds) + 1)]
    if randomize:
        for _ in range(session_length - total_hold):
            gaps[rng.randrange(len(gaps))] += 1
    else:
        gaps[-1] = session_length - total_hold

    windows: list[tuple[int, int, int]] = []
    cursor = session_start + gaps[0]
    for index, hold in enumerate(shuffled_holds):
        exit_idx = cursor + hold - 1
        windows.append((cursor, exit_idx, hold))
        cursor = exit_idx + 1 + gaps[index + 1]
    return windows


def _simulate_windows(
    bars: tuple[MarketBar, ...],
    exec_config: ExecConfig,
    windows: list[tuple[int, int, int]],
) -> dict[str, float]:
    cash = exec_config.initial_cash
    equity_curve: list[float] = []
    trades: list[Trade] = []
    active = None
    by_entry = {entry_idx: (exit_idx, hold) for entry_idx, exit_idx, hold in windows}

    for index, bar in enumerate(bars):
        if active is None and index in by_entry:
            exit_idx, hold = by_entry[index]
            cash, position = enter_long(cash, bar, exec_config, sizing_fraction=1.0)
            if position is not None:
                position.bars_held = hold
                active = (index, exit_idx, position)
        if active is None:
            equity_curve.append(cash)
            continue

        entry_idx, exit_idx, position = active
        equity_curve.append(cash + (position.shares * bar.close))
        if index == exit_idx:
            cash, trade = exit_long(cash, position, bar, exec_config, "randomized_exit", fill_at_close=True)
            trades.append(trade)
            equity_curve[-1] = cash
            active = None

    return _result_metrics(
        BacktestResult(
            bars=bars,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve),
            initial_cash=exec_config.initial_cash,
            final_cash=cash,
        )
    )


def _random_seed(
    bars: tuple[MarketBar, ...],
    trade_targets: tuple[tuple[str, int], ...],
    seed_material: str,
) -> int:
    payload = "|".join(
        (
            seed_material,
            bars[0].timestamp_utc,
            bars[-1].timestamp_utc,
            ",".join(f"{session_date}:{hold}" for session_date, hold in trade_targets),
        )
    )
    return int(sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _trade_session_date(timestamp_utc: str) -> str:
    timestamp = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.astimezone(NEW_YORK).date().isoformat()
