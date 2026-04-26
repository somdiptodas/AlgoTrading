from __future__ import annotations

import json
from pathlib import Path

from trader.cli.reports_cmd import ReportsRequest, parse_args, run_reports
from trader.reporting.run_dashboard import ReportPathConventions


def test_parse_reports_rebuild_args_with_directory_overrides(tmp_path: Path) -> None:
    request = parse_args(
        [
            "rebuild",
            "--reports-dir",
            str(tmp_path / "reports"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    assert request.reports_dir == tmp_path / "reports"
    assert request.artifacts_dir == tmp_path / "artifacts"


def test_run_reports_rebuilds_trade_run_and_dashboard_html(tmp_path: Path, capsys) -> None:
    paths = ReportPathConventions(
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )
    experiment_dir = paths.experiment_artifact_dir("exp_1")
    experiment_dir.mkdir(parents=True)
    paths.run_reports_dir.mkdir(parents=True)
    (experiment_dir / "result.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_1",
                "spec": {"name": "multi_signal_test", "signal": {"name": "multi_signal", "params": {}}},
                "aggregate_metrics": {"return_pct": 2.5, "trade_count": 0},
                "promotion_stage": "candidate",
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "trades.json").write_text(
        json.dumps({"experiment_id": "exp_1", "folds": []}),
        encoding="utf-8",
    )
    (experiment_dir / "equity.json").write_text(
        json.dumps({"experiment_id": "exp_1", "folds": []}),
        encoding="utf-8",
    )
    (experiment_dir / "manifest.json").write_text(
        json.dumps({"experiment_id": "exp_1", "promotion_stage": "candidate"}),
        encoding="utf-8",
    )
    (experiment_dir / "spec.json").write_text(
        json.dumps({"name": "multi_signal_test", "signal": {"name": "multi_signal", "params": {}}}),
        encoding="utf-8",
    )
    paths.experiment_markdown_path("exp_1").write_text("# Experiment exp_1\n", encoding="utf-8")
    paths.loop_json_path("run_1").write_text(
        json.dumps(
            {
                "loop_run_id": "run_1",
                "completed_at_utc": "2026-04-26T10:00:00+00:00",
                "planned": 1,
                "completed": 1,
                "experiments": [{"experiment_id": "exp_1"}],
            }
        ),
        encoding="utf-8",
    )

    result = run_reports(ReportsRequest(reports_dir=paths.reports_dir, artifacts_dir=paths.artifacts_dir))

    assert result.trade_reports == (paths.experiment_trade_html_path("exp_1"),)
    assert result.run_reports == (paths.run_html_path("run_1"),)
    assert result.dashboard_path == paths.dashboard_path
    assert paths.experiment_trade_html_path("exp_1").exists()
    assert paths.run_html_path("run_1").exists()
    assert paths.dashboard_path.exists()
    output = json.loads(capsys.readouterr().out)
    assert output["dashboard"] == str(paths.dashboard_path.resolve())
