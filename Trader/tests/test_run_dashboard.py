from __future__ import annotations

from pathlib import Path

import pytest

from trader.reporting.run_dashboard import ReportPathConventions


def test_report_path_conventions_are_stable(tmp_path: Path) -> None:
    paths = ReportPathConventions(
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )

    assert paths.dashboard_path == tmp_path / "reports" / "index.html"
    assert paths.run_reports_dir == tmp_path / "reports" / "loop_runs"
    assert paths.loop_json_path("run_1") == tmp_path / "reports" / "loop_runs" / "run_1.json"
    assert paths.run_html_path("run_1") == tmp_path / "reports" / "loop_runs" / "run_1.html"
    assert paths.experiment_markdown_path("exp_1") == tmp_path / "reports" / "exp_1.md"
    assert paths.experiment_trade_html_path("exp_1") == tmp_path / "reports" / "exp_1_trades.html"
    assert paths.experiment_artifact_dir("exp_1") == tmp_path / "artifacts" / "exp_1"


@pytest.mark.parametrize("unsafe_id", ["", "../run", "nested/run", "nested\\run", ".", ".."])
def test_report_path_conventions_reject_path_traversal(tmp_path: Path, unsafe_id: str) -> None:
    paths = ReportPathConventions(
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )

    with pytest.raises(ValueError):
        paths.run_html_path(unsafe_id)
