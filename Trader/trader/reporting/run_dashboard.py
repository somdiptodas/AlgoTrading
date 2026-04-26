from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from trader.ledger.entry import json_dumps
from trader.reporting.trade_visualization import write_trade_visualization


RUN_REPORTS_DIRNAME = "loop_runs"
DASHBOARD_FILENAME = "index.html"
PROMOTED_STAGES = frozenset({"candidate", "research_frontier"})


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


@dataclass(frozen=True)
class RebuildReportsResult:
    trade_reports: tuple[Path, ...]
    run_reports: tuple[Path, ...]
    dashboard_path: Path


@dataclass(frozen=True)
class LoopRunOutputs:
    loop_json_path: Path
    run_report_path: Path
    dashboard_path: Path
    trade_reports: tuple[Path, ...]


def _safe_report_id(value: str) -> str:
    candidate = str(value).strip()
    if not candidate or "/" in candidate or "\\" in candidate or candidate in {".", ".."}:
        raise ValueError(f"Report id is not a safe filename: {value!r}")
    return candidate


def rebuild_reports(paths: ReportPathConventions) -> RebuildReportsResult:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    trade_reports = tuple(
        write_trade_visualization(experiment_dir, paths.experiment_trade_html_path(experiment_dir.name))
        for experiment_dir in _experiment_artifact_dirs(paths)
    )
    run_reports = tuple(write_run_report_from_json(json_path, paths) for json_path in _loop_json_paths(paths))
    dashboard_path = write_dashboard(paths)
    return RebuildReportsResult(
        trade_reports=trade_reports,
        run_reports=run_reports,
        dashboard_path=dashboard_path,
    )


def write_loop_run_outputs(loop_payload: Mapping[str, Any], paths: ReportPathConventions) -> LoopRunOutputs:
    loop_run_id = _loop_run_id(loop_payload)
    paths.run_reports_dir.mkdir(parents=True, exist_ok=True)
    loop_json_path = paths.loop_json_path(loop_run_id)
    loop_json_path.write_text(json_dumps(dict(loop_payload), pretty=True), encoding="utf-8")
    trade_reports = _write_trade_reports_for_payload(loop_payload, paths)
    run_report_path = write_run_report_from_json(loop_json_path, paths)
    dashboard_path = write_dashboard(paths)
    return LoopRunOutputs(
        loop_json_path=loop_json_path,
        run_report_path=run_report_path,
        dashboard_path=dashboard_path,
        trade_reports=trade_reports,
    )


def write_run_report_from_json(loop_json_path: Path, paths: ReportPathConventions) -> Path:
    loop_payload = _read_json(loop_json_path)
    loop_run_id = _loop_run_id(loop_payload)
    output_path = paths.run_html_path(loop_run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_run_report(loop_payload, paths, output_path=output_path, loop_json_path=loop_json_path),
        encoding="utf-8",
    )
    return output_path


def write_dashboard(paths: ReportPathConventions) -> Path:
    output_path = paths.dashboard_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard(paths, output_path=output_path), encoding="utf-8")
    return output_path


def render_dashboard(paths: ReportPathConventions, *, output_path: Path | None = None) -> str:
    target_path = output_path or paths.dashboard_path
    runs = _loop_run_records(paths)
    body = "\n".join(
        [
            "<h1>Research Loop Dashboard</h1>",
            f'<p class="muted">{len(runs)} loop runs found under <code>{html.escape(str(paths.run_reports_dir))}</code>.</p>',
            _render_dashboard_runs(runs, paths, target_path),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Research Loop Dashboard</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172026; background: #f7f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 48px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; margin-top: 28px; }}
    .muted {{ color: #62717d; }}
    section {{ background: #ffffff; border: 1px solid #dce3e8; border-radius: 8px; padding: 18px; margin-top: 16px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e8edf1; padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #42515c; font-weight: 650; background: #fbfcfd; }}
    a {{ color: #0f6cbd; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .missing {{ color: #8a5a00; }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def render_run_report(
    loop_payload: Mapping[str, Any],
    paths: ReportPathConventions,
    *,
    output_path: Path | None = None,
    loop_json_path: Path | None = None,
) -> str:
    loop_run_id = _loop_run_id(loop_payload)
    target_path = output_path or paths.run_html_path(loop_run_id)
    experiments = _experiment_summaries(loop_payload, paths)
    counts = dict(loop_payload.get("counts") or {})
    promoted = sum(1 for item in experiments if item.get("promotion_stage") in PROMOTED_STAGES)

    body = "\n".join(
        [
            _render_summary(loop_payload, counts, promoted),
            _render_artifact_links(loop_json_path, target_path),
            _render_experiment_table(experiments, paths, target_path),
            _render_frontier(loop_payload.get("frontier") or ()),
            _render_mix("Generator Mix", _count_by_key(experiments, "generator_kind")),
            _render_mix("Family Mix", _count_by_key(experiments, "family")),
            _render_mix("Shape Mix", _count_by_key(experiments, "shape_key")),
            _render_rejections(loop_payload.get("rejected") or ()),
            _render_timing(loop_payload.get("timings_sec") or {}),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Loop Run {html.escape(loop_run_id)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172026; background: #f7f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 48px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; margin-top: 28px; }}
    .muted {{ color: #62717d; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 10px; margin: 18px 0 8px; }}
    .card, section {{ background: #ffffff; border: 1px solid #dce3e8; border-radius: 8px; }}
    .card {{ padding: 14px; }}
    .card strong {{ display: block; font-size: 22px; margin-bottom: 3px; }}
    section {{ padding: 18px; margin-top: 16px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e8edf1; padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #42515c; font-weight: 650; background: #fbfcfd; }}
    a {{ color: #0f6cbd; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .missing {{ color: #8a5a00; }}
    .pill {{ display: inline-block; padding: 2px 7px; border-radius: 999px; background: #eaf2f8; color: #24485f; font-size: 12px; }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def _loop_run_id(loop_payload: Mapping[str, Any]) -> str:
    return _safe_report_id(str(loop_payload.get("loop_run_id") or "unknown_run"))


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _experiment_artifact_dirs(paths: ReportPathConventions) -> tuple[Path, ...]:
    if not paths.artifacts_dir.exists():
        return tuple()
    return tuple(
        path
        for path in sorted(paths.artifacts_dir.iterdir())
        if path.is_dir()
        and (path / "result.json").exists()
        and (path / "trades.json").exists()
        and (path / "equity.json").exists()
    )


def _write_trade_reports_for_payload(loop_payload: Mapping[str, Any], paths: ReportPathConventions) -> tuple[Path, ...]:
    written = []
    for item in loop_payload.get("experiments") or loop_payload.get("completed_experiments") or ():
        experiment_id = _safe_report_id(str(dict(item).get("experiment_id") or ""))
        experiment_dir = paths.experiment_artifact_dir(experiment_id)
        if not (
            (experiment_dir / "result.json").exists()
            and (experiment_dir / "trades.json").exists()
            and (experiment_dir / "equity.json").exists()
        ):
            continue
        written.append(write_trade_visualization(experiment_dir, paths.experiment_trade_html_path(experiment_id)))
    return tuple(written)


def _loop_json_paths(paths: ReportPathConventions) -> tuple[Path, ...]:
    if not paths.run_reports_dir.exists():
        return tuple()
    return tuple(sorted(paths.run_reports_dir.glob("*.json")))


def _loop_run_records(paths: ReportPathConventions) -> list[dict[str, Any]]:
    if not paths.run_reports_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for json_path in sorted(paths.run_reports_dir.glob("*.json")):
        payload = _read_json(json_path)
        records.append(
            {
                "payload": payload,
                "json_path": json_path,
                "sort_key": _run_sort_key(payload, json_path),
            }
        )
    return sorted(records, key=lambda item: item["sort_key"], reverse=True)


def _run_sort_key(loop_payload: Mapping[str, Any], json_path: Path) -> str:
    for key in ("completed_at_utc", "run_completed_at_utc", "generated_at_utc", "started_at_utc"):
        value = loop_payload.get(key)
        if value:
            return str(value)
    return str(json_path.stat().st_mtime_ns)


def _experiment_summaries(
    loop_payload: Mapping[str, Any],
    paths: ReportPathConventions,
) -> list[dict[str, Any]]:
    experiments = loop_payload.get("experiments") or loop_payload.get("completed_experiments") or ()
    summaries: list[dict[str, Any]] = []
    for item in experiments:
        raw = dict(item)
        experiment_id = _safe_report_id(str(raw.get("experiment_id") or ""))
        artifact_dir = paths.experiment_artifact_dir(experiment_id)
        manifest = _read_json_if_exists(artifact_dir / "manifest.json")
        result = _read_json_if_exists(artifact_dir / "result.json")
        spec = dict(result.get("spec") or {})
        signal = dict(spec.get("signal") or {})
        metrics = dict(raw.get("aggregate_metrics") or manifest.get("aggregate_metrics") or result.get("aggregate_metrics") or {})
        summaries.append(
            {
                **raw,
                "experiment_id": experiment_id,
                "family": raw.get("family") or signal.get("name") or "unknown",
                "promotion_stage": raw.get("promotion_stage") or manifest.get("promotion_stage") or result.get("promotion_stage") or "unknown",
                "generator_kind": raw.get("generator_kind") or manifest.get("generator_kind") or "unknown",
                "shape_key": raw.get("shape_key") or raw.get("strategy_shape") or "unknown",
                "aggregate_metrics": metrics,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            _metric_sort_value(item, "return_pct"),
            _metric_sort_value(item, "sharpe_like"),
            str(item.get("experiment_id")),
        ),
        reverse=True,
    )


def _render_summary(loop_payload: Mapping[str, Any], counts: Mapping[str, Any], promoted: int) -> str:
    loop_run_id = _loop_run_id(loop_payload)
    signal_families = ", ".join(str(item) for item in loop_payload.get("signal_families") or ()) or "not recorded"
    cards = (
        ("Planned", loop_payload.get("planned", counts.get("planned", 0))),
        ("Previewed", counts.get("previewed", 0)),
        ("Selected", counts.get("selected", loop_payload.get("accepted", 0))),
        ("Completed", loop_payload.get("completed", counts.get("evaluated", 0))),
        ("Duplicate/Reused", counts.get("duplicate", loop_payload.get("reused", 0))),
        ("Suppressed", counts.get("suppressed", 0)),
        ("Promoted", promoted),
    )
    return "\n".join(
        [
            f"<h1>Loop Run {html.escape(loop_run_id)}</h1>",
            f'<p class="muted">Signal families: {html.escape(signal_families)}</p>',
            '<div class="cards">',
            *[
                f'<div class="card"><strong>{html.escape(str(value))}</strong><span>{html.escape(label)}</span></div>'
                for label, value in cards
            ],
            "</div>",
        ]
    )


def _render_artifact_links(loop_json_path: Path | None, output_path: Path) -> str:
    if loop_json_path is None:
        return ""
    return (
        "<section><h2>Run Files</h2><p>"
        f"{_link_or_unavailable('Loop JSON', loop_json_path, output_path)}"
        "</p></section>"
    )


def _render_dashboard_runs(
    runs: Sequence[Mapping[str, Any]],
    paths: ReportPathConventions,
    output_path: Path,
) -> str:
    if not runs:
        return '<section><h2>Loop Runs</h2><p>No loop JSON files found.</p></section>'
    sections = []
    for record in runs:
        payload = dict(record["payload"])
        loop_run_id = _loop_run_id(payload)
        counts = dict(payload.get("counts") or {})
        experiments = _experiment_summaries(payload, paths)
        promoted = sum(1 for item in experiments if item.get("promotion_stage") in PROMOTED_STAGES)
        sections.append(
            "<section>"
            f"<h2>Loop Run {html.escape(loop_run_id)}</h2>"
            "<p>"
            f"{_link_or_unavailable('Run HTML', paths.run_html_path(loop_run_id), output_path)} | "
            f"{_link_or_unavailable('Loop JSON', Path(record['json_path']), output_path)}"
            "</p>"
            "<table>"
            "<thead><tr><th>Planned</th><th>Previewed</th><th>Selected</th><th>Completed</th><th>Duplicate/Reused</th><th>Suppressed</th><th>Promoted</th></tr></thead>"
            "<tbody><tr>"
            f"<td>{html.escape(str(payload.get('planned', counts.get('planned', 0))))}</td>"
            f"<td>{html.escape(str(counts.get('previewed', 0)))}</td>"
            f"<td>{html.escape(str(counts.get('selected', payload.get('accepted', 0))))}</td>"
            f"<td>{html.escape(str(payload.get('completed', counts.get('evaluated', 0))))}</td>"
            f"<td>{html.escape(str(counts.get('duplicate', payload.get('reused', 0))))}</td>"
            f"<td>{html.escape(str(counts.get('suppressed', 0)))}</td>"
            f"<td>{promoted}</td>"
            "</tr></tbody></table>"
            f"{_render_experiment_table(experiments, paths, output_path, wrap=False)}"
            "</section>"
        )
    return "\n".join(sections)


def _render_experiment_table(
    experiments: Sequence[Mapping[str, Any]],
    paths: ReportPathConventions,
    output_path: Path,
    *,
    wrap: bool = True,
) -> str:
    rows = []
    for item in experiments:
        experiment_id = str(item["experiment_id"])
        artifact_dir = paths.experiment_artifact_dir(experiment_id)
        rows.append(
            "<tr>"
            f"<td>{html.escape(experiment_id)}</td>"
            f"<td>{html.escape(str(item.get('family', 'unknown')))}</td>"
            f"<td>{html.escape(str(item.get('promotion_stage', 'unknown')))}</td>"
            f"<td>{_format_metric(_float_metric(item, 'return_pct'), suffix='%')}</td>"
            f"<td>{_format_metric(_float_metric(item, 'sharpe_like'))}</td>"
            f"<td>{_format_metric(_float_metric(item, 'max_drawdown_pct'), suffix='%')}</td>"
            f"<td>{_format_metric(_float_metric(item, 'trade_count'), decimals=0)}</td>"
            "<td>"
            f"{_link_or_unavailable('Trade HTML', paths.experiment_trade_html_path(experiment_id), output_path)}<br>"
            f"{_link_or_unavailable('Markdown', paths.experiment_markdown_path(experiment_id), output_path)}<br>"
            f"{_link_or_unavailable('Result JSON', artifact_dir / 'result.json', output_path)}<br>"
            f"{_link_or_unavailable('Trades JSON', artifact_dir / 'trades.json', output_path)}<br>"
            f"{_link_or_unavailable('Equity JSON', artifact_dir / 'equity.json', output_path)}<br>"
            f"{_link_or_unavailable('Manifest', artifact_dir / 'manifest.json', output_path)}"
            "</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else '<tr><td colspan="8">No completed experiments recorded for this run.</td></tr>'
    table = f"""<h2>Completed Experiments</h2>
<table>
  <thead><tr><th>Experiment</th><th>Family</th><th>Stage</th><th>Return</th><th>Sharpe-like</th><th>Drawdown</th><th>Trades</th><th>Files</th></tr></thead>
  <tbody>{body}</tbody>
</table>"""
    if not wrap:
        return table.replace("<h2>", "<h3>", 1).replace("</h2>", "</h3>", 1)
    return f"<section>{table}</section>"


def _render_frontier(frontier: object) -> str:
    rows = []
    frontier_items = frontier if isinstance(frontier, Sequence) and not isinstance(frontier, (str, bytes)) else ()
    for item in frontier_items:
        row = dict(item)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('experiment_id', 'unknown')))}</td>"
            f"<td>{html.escape(str(row.get('family', 'unknown')))}</td>"
            f"<td>{html.escape(str(row.get('promotion_stage', 'unknown')))}</td>"
            f"<td><code>{html.escape(json.dumps(row.get('score_vector', {}), sort_keys=True))}</code></td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<section><h2>Top Candidates</h2><table>"
        "<thead><tr><th>Experiment</th><th>Family</th><th>Stage</th><th>Score Vector</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _render_mix(title: str, counts: Mapping[str, int]) -> str:
    if not counts:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>"
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return f"<section><h2>{html.escape(title)}</h2><table><tbody>{rows}</tbody></table></section>"


def _render_rejections(rejected: object) -> str:
    if not rejected:
        return ""
    rows = "".join(f"<tr><td><code>{html.escape(str(item))}</code></td></tr>" for item in rejected)
    return f"<section><h2>Failed Candidates</h2><table><tbody>{rows}</tbody></table></section>"


def _render_timing(timings: object) -> str:
    if not isinstance(timings, Mapping) or not timings:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{_format_metric(_to_float(value), decimals=3)} sec</td></tr>"
        for name, value in timings.items()
    )
    return f"<section><h2>Timing</h2><table><tbody>{rows}</tbody></table></section>"


def _count_by_key(experiments: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in experiments:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _float_metric(item: Mapping[str, Any], metric_name: str) -> float | None:
    return _to_float(dict(item.get("aggregate_metrics") or {}).get(metric_name))


def _metric_sort_value(item: Mapping[str, Any], metric_name: str) -> float:
    value = _float_metric(item, metric_name)
    return value if value is not None else -1_000_000_000.0


def _to_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: float | None, *, suffix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def _link_or_unavailable(label: str, target_path: Path, output_path: Path) -> str:
    escaped_label = html.escape(label)
    if not target_path.exists():
        return f'<span class="missing">{escaped_label} unavailable</span>'
    return f'<a href="{html.escape(_relative_href(output_path, target_path))}">{escaped_label}</a>'


def _relative_href(source_path: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path.resolve(), start=source_path.resolve().parent)).as_posix()
