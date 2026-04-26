from __future__ import annotations

import json
from pathlib import Path

import pytest

from trader.reporting.run_dashboard import (
    ReportPathConventions,
    write_dashboard,
    write_loop_run_outputs,
    write_run_report_from_json,
)


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
    assert 'href="../index.html"' in html
    assert 'href="run_1.json"' in html
    assert 'href="../exp_1_trades.html"' in html
    assert 'href="../exp_1.md"' in html
    assert 'href="../../artifacts/exp_1/result.json"' in html
    assert "Generator Mix" in html
    assert "Family Mix" in html
    assert "Top Candidates" in html
    assert "Failed Candidates" in html
    assert "planning" in html


def test_write_dashboard_lists_runs_newest_first_with_report_and_artifact_links(tmp_path: Path) -> None:
    paths = ReportPathConventions(
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )
    paths.run_reports_dir.mkdir(parents=True)
    for run_id, experiment_id, completed_at in (
        ("run_old", "exp_old", "2026-04-25T10:00:00+00:00"),
        ("run_new", "exp_new", "2026-04-26T10:00:00+00:00"),
    ):
        experiment_dir = paths.experiment_artifact_dir(experiment_id)
        experiment_dir.mkdir(parents=True)
        paths.run_html_path(run_id).write_text(f"<html>{run_id}</html>\n", encoding="utf-8")
        paths.experiment_markdown_path(experiment_id).write_text(f"# {experiment_id}\n", encoding="utf-8")
        paths.experiment_trade_html_path(experiment_id).write_text("<html>trades</html>\n", encoding="utf-8")
        for name in ("manifest.json", "spec.json", "trades.json", "equity.json"):
            (experiment_dir / name).write_text("{}", encoding="utf-8")
        (experiment_dir / "result.json").write_text(
            json.dumps(
                {
                    "spec": {"signal": {"name": "multi_signal"}},
                    "promotion_stage": "candidate",
                    "aggregate_metrics": {"return_pct": 1.0, "trade_count": 3},
                }
            ),
            encoding="utf-8",
        )
        paths.loop_json_path(run_id).write_text(
            json.dumps(
                {
                    "loop_run_id": run_id,
                    "completed_at_utc": completed_at,
                    "planned": 4,
                    "completed": 1,
                    "counts": {"previewed": 2, "selected": 1, "duplicate": 0, "suppressed": 0},
                    "experiments": [{"experiment_id": experiment_id}],
                }
            ),
            encoding="utf-8",
        )

    output = write_dashboard(paths)

    assert output == paths.dashboard_path
    html = output.read_text(encoding="utf-8")
    assert "Research Loop Dashboard" in html
    assert html.index("Loop Run run_new") < html.index("Loop Run run_old")
    assert 'href="loop_runs/run_new.html"' in html
    assert 'href="loop_runs/run_new.json"' in html
    assert 'href="exp_new_trades.html"' in html
    assert 'href="exp_new.md"' in html
    assert 'href="../artifacts/exp_new/result.json"' in html


def test_write_loop_run_outputs_persists_json_and_refreshes_html(tmp_path: Path) -> None:
    paths = ReportPathConventions(
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )
    experiment_dir = paths.experiment_artifact_dir("exp_1")
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "result.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_1",
                "spec": {"name": "multi_signal_test", "signal": {"name": "multi_signal", "params": {}}},
                "aggregate_metrics": {"return_pct": 1.0, "trade_count": 0},
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
    paths.experiment_markdown_path("exp_1").parent.mkdir(parents=True)
    paths.experiment_markdown_path("exp_1").write_text("# Experiment exp_1\n", encoding="utf-8")

    outputs = write_loop_run_outputs(
        {
            "loop_run_id": "run_1",
            "completed_at_utc": "2026-04-26T10:00:00+00:00",
            "planned": 1,
            "completed": 1,
            "experiments": [{"experiment_id": "exp_1"}],
        },
        paths,
    )

    assert outputs.loop_json_path == paths.loop_json_path("run_1")
    assert outputs.run_report_path == paths.run_html_path("run_1")
    assert outputs.dashboard_path == paths.dashboard_path
    assert outputs.trade_reports == (paths.experiment_trade_html_path("exp_1"),)
    assert paths.loop_json_path("run_1").exists()
    assert paths.run_html_path("run_1").exists()
    assert paths.dashboard_path.exists()
    assert paths.experiment_trade_html_path("exp_1").exists()
    assert 'href="../index.html"' in paths.run_html_path("run_1").read_text(encoding="utf-8")
