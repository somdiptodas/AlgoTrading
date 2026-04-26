from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RUN_REPORTS_DIRNAME = "loop_runs"
DASHBOARD_FILENAME = "index.html"


@dataclass(frozen=True)
class ReportPathConventions:
    reports_dir: Path
    artifacts_dir: Path

    @property
    def dashboard_path(self) -> Path:
        return self.reports_dir / DASHBOARD_FILENAME

    @property
    def run_reports_dir(self) -> Path:
        return self.reports_dir / RUN_REPORTS_DIRNAME

    def loop_json_path(self, loop_run_id: str) -> Path:
        return self.run_reports_dir / f"{_safe_report_id(loop_run_id)}.json"

    def run_html_path(self, loop_run_id: str) -> Path:
        return self.run_reports_dir / f"{_safe_report_id(loop_run_id)}.html"

    def experiment_markdown_path(self, experiment_id: str) -> Path:
        return self.reports_dir / f"{_safe_report_id(experiment_id)}.md"

    def experiment_trade_html_path(self, experiment_id: str) -> Path:
        return self.reports_dir / f"{_safe_report_id(experiment_id)}_trades.html"

    def experiment_artifact_dir(self, experiment_id: str) -> Path:
        return self.artifacts_dir / _safe_report_id(experiment_id)


def _safe_report_id(value: str) -> str:
    candidate = str(value).strip()
    if not candidate or "/" in candidate or "\\" in candidate or candidate in {".", ".."}:
        raise ValueError(f"Report id is not a safe filename: {value!r}")
    return candidate
