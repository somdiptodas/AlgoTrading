from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trader.backtest import BacktestRequest, run_backtest


def test_legacy_default_ema_matches_characterized_fixed_window() -> None:
    database_path = Path(__file__).resolve().parents[1] / "data" / "market_data.db"
    if not database_path.exists():
        pytest.skip("Repo market data DB not available")
    result = run_backtest(
        BacktestRequest(
            ticker="SPY",
            multiplier=1,
            timespan="minute",
            database=str(database_path),
            strategy="ema_cross",
            start=datetime(2025, 10, 21, 13, 30, tzinfo=timezone.utc),
            end=datetime(2026, 4, 20, 19, 59, tzinfo=timezone.utc),
            initial_cash=100_000.0,
            commission_per_order=0.0,
            commission_per_share=0.0,
            slippage_bps=1.0,
            spread_bps=0.0,
            max_position_notional=None,
            fast_length=20,
            slow_length=80,
            signal_buffer_bps=0.0,
            regular_session_only=True,
            flat_at_close=True,
            trades_output=None,
        )
    )
    assert len(result.bars) == 48002
    assert result.bars[0].timestamp_utc == "2025-10-21T13:30:00+00:00"
    assert result.bars[-1].timestamp_utc == "2026-04-20T19:59:00+00:00"
    assert len(result.trades) == 391
    assert result.final_cash == pytest.approx(94_959.18, abs=0.02)
