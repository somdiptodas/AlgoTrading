from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from trader.config import load_settings
from trader.data.view import DataView
from trader.evaluation.metrics import exposure_pct, max_drawdown_pct, safe_pct
from trader.evaluation.runner import EvaluationRunner
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import ExecConfig, SignalSpec, StrategySpec
from trader.data.models import NEW_YORK


@dataclass(frozen=True)
class BacktestRequest:
    ticker: str
    multiplier: int
    timespan: str
    database: str | None
    strategy: str
    start: datetime | None
    end: datetime | None
    initial_cash: float
    commission_per_order: float
    commission_per_share: float
    slippage_bps: float
    spread_bps: float
    max_position_notional: float | None
    stop_loss_bps: float | None
    entry_session_window: str
    no_new_entry_minutes_before_close: int | None
    fast_length: int
    slow_length: int
    signal_buffer_bps: float
    regular_session_only: bool
    flat_at_close: bool
    trades_output: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest a single-position strategy without forward-looking on local SQLite bar data"
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--timespan", default="minute")
    parser.add_argument("--database")
    parser.add_argument("--strategy", default="ema_cross", choices=["ema_cross"])
    parser.add_argument("--start", help="Start timestamp in ISO format")
    parser.add_argument("--end", help="End timestamp in ISO format")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--commission-per-order", type=float, default=0.0)
    parser.add_argument("--commission-per-share", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--max-position-notional", type=float)
    parser.add_argument("--stop-loss-bps", type=float)
    parser.add_argument(
        "--entry-session-window",
        choices=("all", "first_30m", "last_30m", "avoid_midday"),
        default="all",
    )
    parser.add_argument("--no-new-entry-minutes-before-close", type=int)
    parser.add_argument("--fast-length", type=int, default=20)
    parser.add_argument("--slow-length", type=int, default=80)
    parser.add_argument("--signal-buffer-bps", type=float, default=0.0)
    parser.add_argument("--extended-hours", action="store_true", help="Include bars outside regular session")
    parser.add_argument("--hold-overnight", action="store_true", help="Keep positions open across sessions")
    parser.add_argument("--trades-output", help="Optional CSV path for the trade log")
    return parser


def parse_args(argv: list[str] | None = None) -> BacktestRequest:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fast_length < 2:
        raise ValueError("--fast-length must be >= 2")
    if args.slow_length <= args.fast_length:
        raise ValueError("--slow-length must be greater than --fast-length")
    if args.initial_cash <= 0:
        raise ValueError("--initial-cash must be > 0")
    if args.commission_per_order < 0:
        raise ValueError("--commission-per-order must be >= 0")
    if args.commission_per_share < 0:
        raise ValueError("--commission-per-share must be >= 0")
    if args.slippage_bps < 0:
        raise ValueError("--slippage-bps must be >= 0")
    if args.spread_bps < 0:
        raise ValueError("--spread-bps must be >= 0")
    if args.max_position_notional is not None and args.max_position_notional <= 0:
        raise ValueError("--max-position-notional must be > 0 when set")
    if args.stop_loss_bps is not None and args.stop_loss_bps <= 0:
        raise ValueError("--stop-loss-bps must be > 0 when set")
    if args.no_new_entry_minutes_before_close is not None and args.no_new_entry_minutes_before_close < 0:
        raise ValueError("--no-new-entry-minutes-before-close must be >= 0 when set")
    if args.no_new_entry_minutes_before_close is not None and args.no_new_entry_minutes_before_close > 390:
        raise ValueError("--no-new-entry-minutes-before-close must be <= 390 when set")
    if args.signal_buffer_bps < 0:
        raise ValueError("--signal-buffer-bps must be >= 0")
    start = _parse_datetime(args.start) if args.start else None
    end = _parse_datetime(args.end) if args.end else None
    if start and end and start > end:
        raise ValueError("--start must be on or before --end")
    return BacktestRequest(
        ticker=args.ticker.upper(),
        multiplier=args.multiplier,
        timespan=args.timespan,
        database=args.database,
        strategy=args.strategy,
        start=start,
        end=end,
        initial_cash=args.initial_cash,
        commission_per_order=args.commission_per_order,
        commission_per_share=args.commission_per_share,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        max_position_notional=args.max_position_notional,
        stop_loss_bps=args.stop_loss_bps,
        entry_session_window=args.entry_session_window,
        no_new_entry_minutes_before_close=args.no_new_entry_minutes_before_close,
        fast_length=args.fast_length,
        slow_length=args.slow_length,
        signal_buffer_bps=args.signal_buffer_bps,
        regular_session_only=not args.extended_hours,
        flat_at_close=not args.hold_overnight,
        trades_output=Path(args.trades_output).expanduser() if args.trades_output else None,
    )


def run_backtest(request: BacktestRequest) -> BacktestResult:
    settings = load_settings(database_path=request.database)
    runner = EvaluationRunner(DataView(settings.database_path), REGISTRY)
    spec = StrategySpec(
        name="legacy_ema_cross",
        instrument=request.ticker,
        multiplier=request.multiplier,
        timespan=request.timespan,
        signal=SignalSpec(
            "ema_cross",
            {
                "fast_length": request.fast_length,
                "slow_length": request.slow_length,
                "signal_buffer_bps": request.signal_buffer_bps,
            },
        ),
        exec_config=ExecConfig(
            initial_cash=request.initial_cash,
            commission_per_order=request.commission_per_order,
            commission_per_share=request.commission_per_share,
            slippage_bps=request.slippage_bps,
            spread_bps=request.spread_bps,
            max_position_notional=request.max_position_notional,
            stop_loss_bps=request.stop_loss_bps,
            entry_session_window=request.entry_session_window,
            no_new_entry_minutes_before_close=request.no_new_entry_minutes_before_close,
            regular_session_only=request.regular_session_only,
            flat_at_close=request.flat_at_close,
        ),
    )
    start_ms = int(request.start.timestamp() * 1000) if request.start else None
    end_ms = int(request.end.timestamp() * 1000) if request.end else None
    result = runner.evaluate_single_window(spec, start_ms=start_ms, end_ms=end_ms)
    _print_summary(result, request)
    if request.trades_output is not None:
        _write_trade_log(request.trades_output, result.trades)
    return result


def _write_trade_log(path: Path, trades: tuple[Trade, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "entry_timestamp_utc",
                "exit_timestamp_utc",
                "entry_price",
                "exit_price",
                "shares",
                "bars_held",
                "pnl_cash",
                "pnl_pct",
                "exit_reason",
            ]
        )
        for trade in trades:
            writer.writerow(
                [
                    trade.entry_timestamp_utc,
                    trade.exit_timestamp_utc,
                    f"{trade.entry_price:.4f}",
                    f"{trade.exit_price:.4f}",
                    trade.shares,
                    trade.bars_held,
                    f"{trade.pnl_cash:.2f}",
                    f"{trade.pnl_pct:.4f}",
                    trade.exit_reason,
                ]
            )
    print(f"Wrote trade log to {path}")


def _print_summary(result: BacktestResult, request: BacktestRequest) -> None:
    total_return_pct = ((result.final_cash / result.initial_cash) - 1.0) * 100.0
    buy_hold_pct = ((result.bars[-1].close / result.bars[0].close) - 1.0) * 100.0
    winners = [trade for trade in result.trades if trade.pnl_cash > 0]
    losers = [trade for trade in result.trades if trade.pnl_cash <= 0]
    gross_profit = sum(trade.pnl_cash for trade in winners)
    gross_loss = abs(sum(trade.pnl_cash for trade in losers))
    print(f"Strategy: {request.strategy}")
    print(
        f"Universe: {request.ticker} {request.multiplier}-{request.timespan} "
        f"bars={len(result.bars)} regular_session_only={request.regular_session_only} flat_at_close={request.flat_at_close}"
    )
    print(
        f"Parameters: fast={request.fast_length} slow={request.slow_length} "
        f"buffer_bps={request.signal_buffer_bps:.2f} slippage_bps={request.slippage_bps:.2f} "
        f"spread_bps={request.spread_bps:.2f} stop_loss_bps={request.stop_loss_bps} "
        f"entry_session_window={request.entry_session_window} "
        f"no_new_entry_minutes_before_close={request.no_new_entry_minutes_before_close}"
    )
    print(
        f"Equity: start=${result.initial_cash:,.2f} end=${result.final_cash:,.2f} "
        f"return={total_return_pct:+.2f}% buy_hold={buy_hold_pct:+.2f}% max_drawdown={max_drawdown_pct(result.equity_curve):.2f}%"
    )
    print(
        f"Trades: count={len(result.trades)} wins={len(winners)} losses={len(losers)} "
        f"win_rate={safe_pct(len(winners), max(len(result.trades), 1)):.2f}% "
        f"profit_factor={gross_profit / gross_loss if gross_loss else float('inf'):.2f} "
        f"avg_trade={mean([trade.pnl_pct for trade in result.trades]) if result.trades else 0.0:+.3f}% "
        f"exposure={exposure_pct(result.trades, len(result.bars)):.2f}%"
    )


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=NEW_YORK).astimezone(timezone.utc)
    return parsed.astimezone(timezone.utc)


def main(argv: list[str] | None = None) -> None:
    request = parse_args(argv)
    run_backtest(request)
