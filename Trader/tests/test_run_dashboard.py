from __future__ import annotations

import json
from pathlib import Path

import pytest

from trader.reporting.run_dashboard import ReportPathConventions, write_run_report_from_json


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


def test_write_run_report_from_json_renders_loop_and_experiment_links(tmp_path: Path) -> None:
    paths = ReportPathConventions(
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )
    experiment_dir = paths.experiment_artifact_dir("exp_1")
    experiment_dir.mkdir(parents=True)
    paths.run_reports_dir.mkdir(parents=True)
    paths.experiment_markdown_path("exp_1").write_text("# Experiment exp_1\n", encoding="utf-8")
    paths.experiment_trade_html_path("exp_1").write_text("<html>trades</html>\n", encoding="utf-8")
    for name in ("spec.json", "trades.json", "equity.json"):
        (experiment_dir / name).write_text("{}", encoding="utf-8")
    (experiment_dir / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_1",
                "promotion_stage": "candidate",
                "generator_kind": "frontier_neighborhood",
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "result.json").write_text(
        json.dumps(
            {
                "spec": {"signal": {"name": "multi_signal"}},
                "aggregate_metrics": {
                    "return_pct": 4.25,
                    "sharpe_like": 1.1,
                    "max_drawdown_pct": 0.8,
                    "trade_count": 9,
                },
            }
        ),
        encoding="utf-8",
    )
    loop_json_path = paths.loop_json_path("run_1")
    loop_json_path.write_text(
        json.dumps(
            {
                "loop_run_id": "run_1",
                "signal_families": ["multi_signal"],
                "planned": 10,
                "accepted": 4,
                "completed": 1,
                "counts": {"previewed": 4, "selected": 4, "duplicate": 2, "suppressed": 1},
                "experiments": [{"experiment_id": "exp_1"}],
                "frontier": [
                    {
                        "experiment_id": "exp_1",
                        "family": "multi_signal",
                        "promotion_stage": "candidate",
                        "score_vector": {"return_pct": 4.25},
                    }
                ],
                "rejected": ["bad_spec"],
                "timings_sec": {"planning": 0.125},
            }
        ),
        encoding="utf-8",
    )

    output = write_run_report_from_json(loop_json_path, paths)

    assert output == paths.run_html_path("run_1")
    html = output.read_text(encoding="utf-8")
    assert "Loop Run run_1" in html
    assert "Promoted" in html
    assert "exp_1" in html
    assert 'href="run_1.json"' in html
    assert 'href="../exp_1_trades.html"' in html
    assert 'href="../exp_1.md"' in html
    assert 'href="../../artifacts/exp_1/result.json"' in html
    assert "Generator Mix" in html
    assert "Family Mix" in html
    assert "Top Candidates" in html
    assert "Failed Candidates" in html
    assert "planning" in html
