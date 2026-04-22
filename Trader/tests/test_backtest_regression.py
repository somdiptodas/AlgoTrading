from __future__ import annotations

from pathlib import Path

import pytest

from trader.backtest import BacktestRequest, run_backtest


def test_legacy_default_ema_matches_characterized_repo_result() -> None:
    database_path = Path("/Users/sdas/Code/AlgoTrading/Trader/data/market_data.db")
    if not database_path.exists():
        pytest.skip("Repo market data DB not available")
    result = run_backtest(
        BacktestRequest(
            ticker="SPY",
            multiplier=1,
            timespan="minute",
            database=str(database_path),
            strategy="ema_cross",
            start=None,
            end=None,
            initial_cash=100_000.0,
            commission_per_order=0.0,
            slippage_bps=1.0,
            fast_length=20,
            slow_length=80,
            signal_buffer_bps=0.0,
            regular_session_only=True,
            flat_at_close=True,
            trades_output=None,
        )
    )
    assert len(result.bars) == 48002
    assert len(result.trades) == 391
    assert result.final_cash == pytest.approx(94_959.18, abs=0.02)
