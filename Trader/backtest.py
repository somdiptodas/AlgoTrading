from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

try:
    from Trader.config import load_settings
    from Trader.storage import SQLiteBarStore
except ModuleNotFoundError:
    from config import load_settings
    from storage import SQLiteBarStore


NEW_YORK = ZoneInfo("America/New_York")
REGULAR_SESSION_START = time(9, 30)
REGULAR_SESSION_END = time(16, 0)


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
    slippage_bps: float
    fast_length: int
    slow_length: int
    signal_buffer_bps: float
    regular_session_only: bool
    flat_at_close: bool
    trades_output: Path | None


@dataclass(frozen=True)
class MarketBar:
    timestamp_ms: int
    timestamp_utc: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def dt_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc)

    @property
    def dt_local(self) -> datetime:
        return self.dt_utc.astimezone(NEW_YORK)

    @property
    def session_date(self) -> str:
        return self.dt_local.date().isoformat()


@dataclass
class Position:
    entry_timestamp_ms: int
    entry_timestamp_utc: str
    entry_price: float
    shares: int
    entry_commission: float
    bars_held: int = 0


@dataclass(frozen=True)
class Trade:
    entry_timestamp_utc: str
    exit_timestamp_utc: str
    entry_price: float
    exit_price: float
    shares: int
    bars_held: int
    pnl_cash: float
    pnl_pct: float
    exit_reason: str


@dataclass(frozen=True)
class BacktestResult:
    bars: list[MarketBar]
    trades: list[Trade]
    equity_curve: list[float]
    initial_cash: float
    final_cash: float


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
    parser.add_argument("--slippage-bps", type=float, default=1.0)
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
    if args.slippage_bps < 0:
        raise ValueError("--slippage-bps must be >= 0")
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
        slippage_bps=args.slippage_bps,
        fast_length=args.fast_length,
        slow_length=args.slow_length,
        signal_buffer_bps=args.signal_buffer_bps,
        regular_session_only=not args.extended_hours,
        flat_at_close=not args.hold_overnight,
        trades_output=Path(args.trades_output).expanduser() if args.trades_output else None,
    )


def run_backtest(request: BacktestRequest) -> BacktestResult:
    settings = load_settings(database_path=request.database)
    store = SQLiteBarStore(settings.database_path)
    summary = store.fetch_summary(request.ticker, request.multiplier, request.timespan)
    if summary is None or summary["row_count"] == 0:
        raise RuntimeError("No data available for the requested instrument")

    start_ms = int(request.start.timestamp() * 1000) if request.start else None
    end_ms = int(request.end.timestamp() * 1000) if request.end else None
    rows = store.fetch_bars(
        ticker=request.ticker,
        multiplier=request.multiplier,
        timespan=request.timespan,
        start_timestamp_ms=start_ms,
        end_timestamp_ms=end_ms,
    )
    bars = _load_bars(rows, regular_session_only=request.regular_session_only)
    if len(bars) <= request.slow_length + 1:
        raise RuntimeError("Not enough bars for the selected lookback lengths")

    result = _run_ema_cross_backtest(bars, request)
    _print_summary(result, request)
    if request.trades_output is not None:
        _write_trade_log(request.trades_output, result.trades)
    return result


def _run_ema_cross_backtest(bars: list[MarketBar], request: BacktestRequest) -> BacktestResult:
    cash = request.initial_cash
    equity_curve: list[float] = []
    trades: list[Trade] = []
    position: Position | None = None
    pending_action: tuple[str, str] | None = None

    fast_ema: float | None = None
    slow_ema: float | None = None
    fast_alpha = 2.0 / (request.fast_length + 1)
    slow_alpha = 2.0 / (request.slow_length + 1)

    for index, bar in enumerate(bars):
        if pending_action is not None:
            action, reason = pending_action
            if action == "enter":
                cash, position = _enter_long(cash, bar, request)
            elif action == "exit" and position is not None:
                cash, trade = _exit_long(cash, position, bar, request, reason, fill_at_close=False)
                trades.append(trade)
                position = None
            pending_action = None

        if position is not None:
            position.bars_held += 1

        fast_ema = bar.close if fast_ema is None else (bar.close * fast_alpha) + (fast_ema * (1.0 - fast_alpha))
        slow_ema = bar.close if slow_ema is None else (bar.close * slow_alpha) + (slow_ema * (1.0 - slow_alpha))
        regime_is_long = fast_ema > slow_ema * (1.0 + request.signal_buffer_bps / 10_000.0)

        last_bar = index == len(bars) - 1
        next_session_changes = (not last_bar) and bars[index + 1].session_date != bar.session_date

        if position is not None and request.flat_at_close and (next_session_changes or last_bar):
            cash, trade = _exit_long(cash, position, bar, request, "session_close", fill_at_close=True)
            trades.append(trade)
            position = None
        elif position is not None and not regime_is_long and not last_bar:
            pending_action = ("exit", "signal_flip")
        elif position is None and index + 1 >= request.slow_length and regime_is_long and not last_bar:
            if request.flat_at_close and next_session_changes:
                pass
            else:
                pending_action = ("enter", "signal_on")

        equity_curve.append(_mark_to_market(cash, position, bar.close))

    if position is not None:
        cash, trade = _exit_long(cash, position, bars[-1], request, "final_bar", fill_at_close=True)
        trades.append(trade)
        position = None
        equity_curve[-1] = cash

    return BacktestResult(
        bars=bars,
        trades=trades,
        equity_curve=equity_curve,
        initial_cash=request.initial_cash,
        final_cash=cash,
    )


def _load_bars(rows: list[object], regular_session_only: bool) -> list[MarketBar]:
    bars: list[MarketBar] = []
    for row in rows:
        if None in (row["open"], row["high"], row["low"], row["close"]):
            continue

        bar = MarketBar(
            timestamp_ms=int(row["timestamp_ms"]),
            timestamp_utc=str(row["timestamp_utc"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"] or 0.0),
        )
        if regular_session_only and not _is_regular_session(bar):
            continue
        bars.append(bar)
    return bars


def _is_regular_session(bar: MarketBar) -> bool:
    local_time = bar.dt_local.timetz().replace(tzinfo=None)
    return REGULAR_SESSION_START <= local_time < REGULAR_SESSION_END


def _enter_long(cash: float, bar: MarketBar, request: BacktestRequest) -> tuple[float, Position | None]:
    fill_price = bar.open * (1.0 + request.slippage_bps / 10_000.0)
    max_shares = int((cash - request.commission_per_order) // fill_price)
    if max_shares < 1:
        return cash, None

    cost = (max_shares * fill_price) + request.commission_per_order
    new_cash = cash - cost
    position = Position(
        entry_timestamp_ms=bar.timestamp_ms,
        entry_timestamp_utc=bar.timestamp_utc,
        entry_price=fill_price,
        shares=max_shares,
        entry_commission=request.commission_per_order,
    )
    return new_cash, position


def _exit_long(
    cash: float,
    position: Position,
    bar: MarketBar,
    request: BacktestRequest,
    reason: str,
    *,
    fill_at_close: bool,
) -> tuple[float, Trade]:
    raw_price = bar.close if fill_at_close else bar.open
    fill_price = raw_price * (1.0 - request.slippage_bps / 10_000.0)
    proceeds = (position.shares * fill_price) - request.commission_per_order
    new_cash = cash + proceeds
    pnl_cash = (
        (fill_price - position.entry_price) * position.shares
        - position.entry_commission
        - request.commission_per_order
    )
    invested = (position.entry_price * position.shares) + position.entry_commission
    trade = Trade(
        entry_timestamp_utc=position.entry_timestamp_utc,
        exit_timestamp_utc=bar.timestamp_utc,
        entry_price=position.entry_price,
        exit_price=fill_price,
        shares=position.shares,
        bars_held=position.bars_held,
        pnl_cash=pnl_cash,
        pnl_pct=(pnl_cash / invested) * 100 if invested else 0.0,
        exit_reason=reason,
    )
    return new_cash, trade


def _mark_to_market(cash: float, position: Position | None, close_price: float) -> float:
    if position is None:
        return cash
    return cash + (position.shares * close_price)


def _write_trade_log(path: Path, trades: list[Trade]) -> None:
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
    max_drawdown_pct = _max_drawdown_pct(result.equity_curve)
    winners = [trade for trade in result.trades if trade.pnl_cash > 0]
    losers = [trade for trade in result.trades if trade.pnl_cash <= 0]
    gross_profit = sum(trade.pnl_cash for trade in winners)
    gross_loss = abs(sum(trade.pnl_cash for trade in losers))
    exposure_pct = _exposure_pct(result.trades, len(result.bars))

    print(f"Strategy: {request.strategy}")
    print(
        f"Universe: {request.ticker} {request.multiplier}-{request.timespan} "
        f"bars={len(result.bars)} regular_session_only={request.regular_session_only} flat_at_close={request.flat_at_close}"
    )
    print(
        f"Parameters: fast={request.fast_length} slow={request.slow_length} "
        f"buffer_bps={request.signal_buffer_bps:.2f} slippage_bps={request.slippage_bps:.2f}"
    )
    print(
        f"Equity: start=${result.initial_cash:,.2f} end=${result.final_cash:,.2f} "
        f"return={total_return_pct:+.2f}% buy_hold={buy_hold_pct:+.2f}% max_drawdown={max_drawdown_pct:.2f}%"
    )
    print(
        f"Trades: count={len(result.trades)} wins={len(winners)} losses={len(losers)} "
        f"win_rate={_safe_pct(len(winners), max(len(result.trades), 1)):.2f}% "
        f"profit_factor={gross_profit / gross_loss if gross_loss else float('inf'):.2f} "
        f"avg_trade={mean([trade.pnl_pct for trade in result.trades]) if result.trades else 0.0:+.3f}% "
        f"exposure={exposure_pct:.2f}%"
    )


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = 0.0 if peak == 0 else ((peak - equity) / peak) * 100.0
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _exposure_pct(trades: list[Trade], total_bars: int) -> float:
    if total_bars == 0:
        return 0.0
    return (sum(trade.bars_held for trade in trades) / total_bars) * 100.0


def _safe_pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator) * 100.0 if denominator else 0.0


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=NEW_YORK).astimezone(timezone.utc)
    return parsed.astimezone(timezone.utc)
