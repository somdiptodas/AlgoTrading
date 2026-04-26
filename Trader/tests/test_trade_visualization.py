from __future__ import annotations

import json
from pathlib import Path

from trader.execution.fills import Trade
from trader.ledger.entry import trade_from_payload, trade_to_payload
from trader.reporting.trade_visualization import write_trade_visualization


def test_trade_payload_round_trips_entry_reason_and_defaults_legacy_payload() -> None:
    trade = Trade(
        "2026-01-05T14:30:00+00:00",
        "2026-01-05T14:35:00+00:00",
        100.0,
        101.0,
        10,
        5,
        10.0,
        1.0,
        "signal_flip",
        entry_reason="signal_on",
    )

    payload = trade_to_payload(trade)

    assert payload["entry_reason"] == "signal_on"
    assert trade_from_payload(payload).entry_reason == "signal_on"
    legacy_payload = dict(payload)
    legacy_payload.pop("entry_reason")
    assert trade_from_payload(legacy_payload).entry_reason == "signal_on"


def test_trade_visualization_renders_equity_and_reason_text(tmp_path: Path) -> None:
    experiment_dir = _write_experiment_artifacts(
        tmp_path,
        {
            "entry_timestamp_utc": "2026-01-05T14:30:00+00:00",
            "exit_timestamp_utc": "2026-01-05T14:35:00+00:00",
            "entry_price": 100.0,
            "exit_price": 101.0,
            "shares": 10,
            "bars_held": 5,
            "pnl_cash": 10.0,
            "pnl_pct": 1.0,
            "entry_reason": "signal_on",
            "exit_reason": "signal_flip",
            "cost_cash": 0.0,
        },
    )

    output = write_trade_visualization(experiment_dir, tmp_path / "trade_review.html")
    html = output.read_text(encoding="utf-8")

    assert "Trade Review: exp_1" in html
    assert "equity-line" in html
    assert "Strategy signal was long on the prior bar" in html
    assert "Strategy signal turned off" in html


def test_trade_visualization_renders_rule_reasons(tmp_path: Path) -> None:
    experiment_dir = _write_experiment_artifacts(
        tmp_path,
        {
            "entry_timestamp_utc": "2026-01-05T14:30:00+00:00",
            "exit_timestamp_utc": "2026-01-05T14:35:00+00:00",
            "entry_price": 100.0,
            "exit_price": 101.0,
            "shares": 10,
            "bars_held": 5,
            "pnl_cash": 10.0,
            "pnl_pct": 1.0,
            "entry_reason": "signal_on",
            "exit_reason": "signal_flip",
            "entry_rule": {"passed": True, "reason": "entry k_of_n passed: 3/5 signals"},
            "exit_rule": {"passed": False, "reason": "exit any blocked: 0/3 signals"},
            "cost_cash": 0.0,
        },
    )

    output = write_trade_visualization(experiment_dir, tmp_path / "trade_review.html")
    html = output.read_text(encoding="utf-8")

    assert "Entry rule passed" in html
    assert "entry k_of_n passed: 3/5 signals" in html
    assert "Exit rule failed" in html
    assert "exit any blocked: 0/3 signals" in html


def test_trade_visualization_renders_expandable_vote_details(tmp_path: Path) -> None:
    experiment_dir = _write_experiment_artifacts(
        tmp_path,
        {
            "entry_timestamp_utc": "2026-01-05T14:30:00+00:00",
            "exit_timestamp_utc": "2026-01-05T14:35:00+00:00",
            "entry_price": 100.0,
            "exit_price": 101.0,
            "shares": 10,
            "bars_held": 5,
            "pnl_cash": 10.0,
            "pnl_pct": 1.0,
            "entry_reason": "signal_on",
            "exit_reason": "signal_flip",
            "entry_rule": {"passed": True, "reason": "entry k_of_n passed: 3/5 signals"},
            "exit_rule": {"passed": True, "reason": "exit any passed: 1/3 signals"},
            "entry_votes": [
                {"name": "rsi_below", "passed": True, "detail": "RSI 28.4 < 30.0"},
            ],
            "exit_votes": [
                {"name": "rsi_above", "passed": False, "detail": "RSI 45.0 < 70.0"},
            ],
            "cost_cash": 0.0,
        },
    )

    output = write_trade_visualization(experiment_dir, tmp_path / "trade_review.html")
    html = output.read_text(encoding="utf-8")

    assert '<details class="vote-details">' in html
    assert "Vote details" in html
    assert "Entry votes" in html
    assert "Exit votes" in html
    assert "rsi_below" in html
    assert "RSI 28.4 &lt; 30.0" in html


def _write_experiment_artifacts(tmp_path: Path, trade: dict[str, object]) -> Path:
    experiment_dir = tmp_path / "artifacts" / "exp_1"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "result.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_1",
                "promotion_stage": "exploratory",
                "aggregate_metrics": {
                    "return_pct": 0.5,
                    "sharpe_like": 1.25,
                    "max_drawdown_pct": 0.1,
                },
                "spec": {
                    "signal": {"name": "rsi_reversion", "params": {"rsi_length": 7}},
                    "sizing": {"name": "fixed_fraction", "params": {"fraction": 0.5}},
                },
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "trades.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_1",
                "folds": [
                    {
                        "fold_id": "fold_1",
                        "trades": [trade],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "equity.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_1",
                "folds": [
                    {
                        "fold_id": "fold_1",
                        "timestamps_utc": [
                            "2026-01-05T14:30:00+00:00",
                            "2026-01-05T14:35:00+00:00",
                        ],
                        "equity_curve": [100000.0, 100010.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return experiment_dir
