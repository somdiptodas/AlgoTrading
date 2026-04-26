from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ENTRY_REASON_DETAILS = {
    "signal_on": "Strategy signal was long on the prior bar; entry filled at the next allowed open.",
}

EXIT_REASON_DETAILS = {
    "signal_flip": "Strategy signal turned off; exit filled at the next open.",
    "session_close": "Flat-at-close rule closed the position at the session close.",
    "final_bar": "Backtest ended while the position was open; exit filled on the final bar.",
    "stop_loss": "Configured stop loss was reached inside the bar.",
}


def write_trade_visualization(experiment_dir: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_trade_visualization(experiment_dir), encoding="utf-8")
    return output_path


def render_trade_visualization(experiment_dir: Path) -> str:
    result = _load_json(experiment_dir / "result.json")
    trades_payload = _load_json(experiment_dir / "trades.json")
    equity_payload = _load_json(experiment_dir / "equity.json")
    manifest = _load_json(experiment_dir / "manifest.json", default={})
    spec_payload = _load_json(experiment_dir / "spec.json", default=result.get("spec", {}))

    experiment_id = str(
        result.get("experiment_id")
        or trades_payload.get("experiment_id")
        or manifest.get("experiment_id")
        or experiment_dir.name
    )
    aggregate = dict(result.get("aggregate_metrics") or manifest.get("aggregate_metrics") or {})
    spec = dict(result.get("spec") or spec_payload or {})
    trade_folds = _trade_fold_map(trades_payload.get("folds", ()))
    equity_folds = _equity_fold_map(equity_payload.get("folds", ()))
    fold_ids = _ordered_fold_ids(trade_folds, equity_folds)
    total_trades = sum(len(trade_folds.get(fold_id, ())) for fold_id in fold_ids)
    total_pnl = sum(
        float(trade.get("pnl_cash", 0.0) or 0.0)
        for fold_id in fold_ids
        for trade in trade_folds.get(fold_id, ())
    )

    fold_sections = "\n".join(
        _render_fold_section(
            fold_id=fold_id,
            trades=trade_folds.get(fold_id, ()),
            equity=equity_folds.get(fold_id, {}),
        )
        for fold_id in fold_ids
    )
    if not fold_sections:
        fold_sections = '<section class="panel"><h2>No Folds</h2><p>No trade or equity artifacts were found.</p></section>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(experiment_id)} Trade Review</title>
  <style>
    :root {{
      --bg: #f6f7f8;
      --ink: #17202a;
      --muted: #5c6670;
      --panel: #ffffff;
      --line: #d7dde3;
      --accent: #2f80ed;
      --entry: #138a5b;
      --exit: #c74634;
      --warn: #9b6a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 18px;
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 28px; line-height: 1.15; }}
    h2 {{ font-size: 18px; margin-bottom: 12px; }}
    h3 {{ font-size: 15px; margin: 18px 0 8px; }}
    .sub {{ color: var(--muted); margin-top: 8px; font-size: 14px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .metric strong {{ font-size: 18px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
      overflow-x: auto;
    }}
    .chart {{
      width: 100%;
      min-width: 760px;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .equity-line {{ fill: none; stroke: var(--accent); stroke-width: 2.5; }}
    .grid-line {{ stroke: #e5e9ed; stroke-width: 1; }}
    .axis-label {{ fill: var(--muted); font-size: 11px; }}
    .entry-marker {{ fill: var(--entry); }}
    .exit-marker {{ fill: var(--exit); }}
    .trade-path {{ stroke: rgba(23, 32, 42, 0.22); stroke-width: 1.5; stroke-dasharray: 4 4; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 980px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #f9fafb;
    }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .good {{ color: var(--entry); }}
    .bad {{ color: var(--exit); }}
    .reason-code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      color: var(--ink);
    }}
    .reason-detail {{ color: var(--muted); font-size: 12px; line-height: 1.35; margin-top: 2px; }}
    .rule-status {{ font-size: 12px; font-weight: 700; margin-bottom: 3px; }}
    .vote-details summary {{ cursor: pointer; color: var(--accent); font-weight: 700; }}
    .vote-group {{ margin-top: 10px; }}
    .vote-group-title {{ color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 4px; }}
    .vote-table {{ min-width: 440px; font-size: 12px; }}
    .vote-table th, .vote-table td {{ padding: 5px 6px; }}
    .vote-status {{ font-weight: 700; }}
    .vote-status.passed {{ color: var(--entry); }}
    .vote-status.failed {{ color: var(--exit); }}
    .vote-row.vote-passed .reason-code {{ color: var(--entry); }}
    .vote-row.vote-failed .reason-code {{ color: var(--exit); }}
    .empty {{ color: var(--muted); margin: 8px 0 0; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Trade Review: {html.escape(experiment_id)}</h1>
      <div class="sub">{html.escape(_strategy_label(spec))}</div>
    </header>
    <section class="metrics">
      {_metric_card("Return", _format_pct(aggregate.get("return_pct")))}
      {_metric_card("Sharpe-like", _format_number(aggregate.get("sharpe_like")))}
      {_metric_card("Max Drawdown", _format_pct(aggregate.get("max_drawdown_pct")))}
      {_metric_card("Trades", str(total_trades))}
      {_metric_card("Trade PnL", _format_cash(total_pnl))}
      {_metric_card("Stage", str(result.get("promotion_stage") or manifest.get("promotion_stage") or "unknown"))}
    </section>
    {fold_sections}
  </main>
</body>
</html>
"""


def _render_fold_section(
    *,
    fold_id: str,
    trades: Sequence[Mapping[str, Any]],
    equity: Mapping[str, Any],
) -> str:
    chart = _render_equity_chart(equity, trades)
    rows = "\n".join(_render_trade_row(index, trade) for index, trade in enumerate(trades, start=1))
    if rows:
        table = f"""<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Entry</th>
      <th>Exit</th>
      <th class="num">Shares</th>
      <th class="num">Bars</th>
      <th class="num">Entry</th>
      <th class="num">Exit</th>
      <th class="num">PnL</th>
      <th>Entry Reason</th>
      <th>Entry Rule</th>
      <th>Exit Reason</th>
      <th>Exit Rule</th>
      <th>Votes</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""
    else:
        table = '<p class="empty">No trades were taken in this fold.</p>'
    return f"""<section class="panel">
  <h2>{html.escape(fold_id)}</h2>
  {chart}
  <h3>Trades</h3>
  {table}
</section>"""


def _render_equity_chart(equity: Mapping[str, Any], trades: Sequence[Mapping[str, Any]]) -> str:
    timestamps = [str(value) for value in equity.get("timestamps_utc", ())]
    values = [float(value) for value in equity.get("equity_curve", ())]
    points = [(timestamp, value) for timestamp, value in zip(timestamps, values)]
    if len(points) < 2:
        return '<p class="empty">No equity curve points were available for this fold.</p>'

    width = 1160
    height = 340
    left = 64
    right = 24
    top = 24
    bottom = 44
    chart_width = width - left - right
    chart_height = height - top - bottom
    datetimes = [_parse_time(timestamp) for timestamp, _ in points]
    min_time = min(datetimes)
    max_time = max(datetimes)
    min_equity = min(values)
    max_equity = max(values)
    if math.isclose(min_equity, max_equity):
        min_equity -= 1.0
        max_equity += 1.0
    pad = (max_equity - min_equity) * 0.08
    min_equity -= pad
    max_equity += pad

    def x_at(dt: datetime) -> float:
        span = max((max_time - min_time).total_seconds(), 1.0)
        return left + ((dt - min_time).total_seconds() / span) * chart_width

    def y_at(value: float) -> float:
        return top + ((max_equity - value) / (max_equity - min_equity)) * chart_height

    path = " ".join(
        f"{'M' if index == 0 else 'L'}{x_at(datetimes[index]):.2f},{y_at(value):.2f}"
        for index, (_, value) in enumerate(points)
    )
    grid = _chart_grid(left, right, top, bottom, width, height, min_equity, max_equity)
    x_labels = _x_axis_labels(left, chart_width, height, min_time, max_time)
    markers = "\n".join(
        _trade_markers(index, trade, points, x_at, y_at)
        for index, trade in enumerate(trades, start=1)
    )
    return f"""<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve with trade entry and exit markers">
  {grid}
  <path class="equity-line" d="{path}" />
  {markers}
  {x_labels}
</svg>"""


def _trade_markers(
    index: int,
    trade: Mapping[str, Any],
    points: Sequence[tuple[str, float]],
    x_at,
    y_at,
) -> str:
    entry_time = _parse_time(str(trade.get("entry_timestamp_utc")))
    exit_time = _parse_time(str(trade.get("exit_timestamp_utc")))
    entry_equity = _nearest_equity(points, entry_time)
    exit_equity = _nearest_equity(points, exit_time)
    entry_x = x_at(entry_time)
    exit_x = x_at(exit_time)
    entry_y = y_at(entry_equity)
    exit_y = y_at(exit_equity)
    title = html.escape(
        f"Trade {index}\n"
        f"Entry: {_format_time(trade.get('entry_timestamp_utc'))} ({_entry_reason(trade)})\n"
        f"Exit: {_format_time(trade.get('exit_timestamp_utc'))} ({_exit_reason(trade)})\n"
        f"PnL: {_format_cash(trade.get('pnl_cash'))}"
    )
    return f"""<g>
    <title>{title}</title>
    <line class="trade-path" x1="{entry_x:.2f}" y1="{entry_y:.2f}" x2="{exit_x:.2f}" y2="{exit_y:.2f}" />
    <circle class="entry-marker" cx="{entry_x:.2f}" cy="{entry_y:.2f}" r="6" />
    <rect class="exit-marker" x="{exit_x - 5:.2f}" y="{exit_y - 5:.2f}" width="10" height="10" />
    <text class="axis-label" x="{entry_x + 8:.2f}" y="{entry_y - 8:.2f}">{index}</text>
  </g>"""


def _render_trade_row(index: int, trade: Mapping[str, Any]) -> str:
    pnl = float(trade.get("pnl_cash", 0.0) or 0.0)
    pnl_class = "good" if pnl >= 0 else "bad"
    return f"""<tr>
  <td class="num">{index}</td>
  <td>{html.escape(_format_time(trade.get("entry_timestamp_utc")))}</td>
  <td>{html.escape(_format_time(trade.get("exit_timestamp_utc")))}</td>
  <td class="num">{int(trade.get("shares", 0) or 0):,}</td>
  <td class="num">{int(trade.get("bars_held", 0) or 0):,}</td>
  <td class="num">{_format_price(trade.get("entry_price"))}</td>
  <td class="num">{_format_price(trade.get("exit_price"))}</td>
  <td class="num {pnl_class}">{_format_cash(pnl)}</td>
  <td>{_reason_cell(_entry_reason(trade), ENTRY_REASON_DETAILS)}</td>
  <td>{_rule_cell(trade, "entry")}</td>
  <td>{_reason_cell(_exit_reason(trade), EXIT_REASON_DETAILS)}</td>
  <td>{_rule_cell(trade, "exit")}</td>
  <td>{_vote_details_cell(trade)}</td>
</tr>"""


def _reason_cell(reason: str, details: Mapping[str, str]) -> str:
    detail = details.get(reason, "Recorded by the execution engine.")
    return (
        f'<div class="reason-code">{html.escape(reason)}</div>'
        f'<div class="reason-detail">{html.escape(detail)}</div>'
    )


def _rule_cell(trade: Mapping[str, Any], prefix: str) -> str:
    rule = _rule_decision(trade, prefix)
    if rule is None:
        return (
            '<div class="reason-code">n/a</div>'
            '<div class="reason-detail">No structured rule data was recorded.</div>'
        )

    passed = rule.get("passed")
    if passed is True:
        status = "passed"
        status_class = " good"
    elif passed is False:
        status = "failed"
        status_class = " bad"
    else:
        status = "unknown"
        status_class = ""
    reason = str(rule.get("reason") or "n/a")
    return (
        f'<div class="rule-status{status_class}">{html.escape(prefix.title())} rule {status}</div>'
        f'<div class="reason-code">{html.escape(reason)}</div>'
    )


def _rule_decision(trade: Mapping[str, Any], prefix: str) -> Mapping[str, Any] | None:
    rule = trade.get(f"{prefix}_rule")
    return rule if isinstance(rule, Mapping) else None


def _vote_details_cell(trade: Mapping[str, Any]) -> str:
    entry_votes = _rule_votes(trade, "entry")
    exit_votes = _rule_votes(trade, "exit")
    if not entry_votes and not exit_votes:
        return '<div class="reason-detail">No vote details were recorded.</div>'

    groups = "\n".join(
        group
        for group in (
            _vote_group("Entry votes", entry_votes),
            _vote_group("Exit votes", exit_votes),
        )
        if group
    )
    return f"""<details class="vote-details">
  <summary>Vote details</summary>
  {groups}
</details>"""


def _vote_group(label: str, votes: Sequence[Mapping[str, Any]]) -> str:
    if not votes:
        return ""
    rows = "\n".join(_vote_row(vote) for vote in votes)
    return f"""<div class="vote-group">
  <div class="vote-group-title">{html.escape(label)}</div>
  <table class="vote-table">
    <thead>
      <tr><th>Signal</th><th>Passed</th><th>Detail</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _vote_row(vote: Mapping[str, Any]) -> str:
    passed = vote.get("passed")
    if passed is True:
        status = "passed"
        status_class = "passed"
        row_class = "vote-passed"
    elif passed is False:
        status = "failed"
        status_class = "failed"
        row_class = "vote-failed"
    else:
        status = "unknown"
        status_class = "unknown"
        row_class = "vote-unknown"
    return (
        f'<tr class="vote-row {row_class}">'
        f'<td class="reason-code">{html.escape(str(vote.get("name") or "unknown"))}</td>'
        f'<td><span class="vote-status {status_class}">{html.escape(status)}</span></td>'
        f"<td>{html.escape(str(vote.get('detail') or ''))}</td>"
        "</tr>"
    )


def _rule_votes(trade: Mapping[str, Any], prefix: str) -> Sequence[Mapping[str, Any]]:
    raw_votes = trade.get(f"{prefix}_votes")
    if raw_votes is None:
        rule = _rule_decision(trade, prefix)
        raw_votes = rule.get("votes") if rule is not None else None
    if not isinstance(raw_votes, list):
        return ()
    return tuple(vote for vote in raw_votes if isinstance(vote, Mapping))


def _chart_grid(
    left: int,
    right: int,
    top: int,
    bottom: int,
    width: int,
    height: int,
    min_equity: float,
    max_equity: float,
) -> str:
    lines: list[str] = []
    ticks = 5
    chart_bottom = height - bottom
    for index in range(ticks):
        ratio = index / (ticks - 1)
        y = top + ratio * (chart_bottom - top)
        value = max_equity - ratio * (max_equity - min_equity)
        lines.append(
            f'<line class="grid-line" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" />'
        )
        lines.append(
            f'<text class="axis-label" x="8" y="{y + 4:.2f}">{html.escape(_format_cash(value))}</text>'
        )
    return "\n  ".join(lines)


def _x_axis_labels(left: int, chart_width: int, height: int, min_time: datetime, max_time: datetime) -> str:
    labels = []
    for ratio in (0.0, 0.5, 1.0):
        dt = min_time + (max_time - min_time) * ratio
        x = left + chart_width * ratio
        labels.append(
            f'<text class="axis-label" x="{x:.2f}" y="{height - 14}" text-anchor="middle">{html.escape(_short_time(dt))}</text>'
        )
    return "\n  ".join(labels)


def _nearest_equity(points: Sequence[tuple[str, float]], target: datetime) -> float:
    _, value = min(
        ((_parse_time(timestamp), value) for timestamp, value in points),
        key=lambda item: abs((item[0] - target).total_seconds()),
    )
    return float(value)


def _trade_fold_map(folds: object) -> dict[str, Sequence[Mapping[str, Any]]]:
    mapped: dict[str, Any] = {}
    for item in folds if isinstance(folds, list) else ():
        if not isinstance(item, dict):
            continue
        fold_id = str(item.get("fold_id", "unknown"))
        trades = item.get("trades", ())
        mapped[fold_id] = trades if isinstance(trades, list) else []
    return mapped


def _equity_fold_map(folds: object) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    for item in folds if isinstance(folds, list) else ():
        if not isinstance(item, dict):
            continue
        fold_id = str(item.get("fold_id", "unknown"))
        mapped[fold_id] = item
    return mapped


def _ordered_fold_ids(
    trade_folds: Mapping[str, object],
    equity_folds: Mapping[str, object],
) -> list[str]:
    return sorted(
        set(trade_folds) | set(equity_folds),
        key=lambda value: (0, int(value.split("_")[-1])) if value.split("_")[-1].isdigit() else (1, value),
    )


def _load_json(path: Path, *, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_label(spec: Mapping[str, Any]) -> str:
    signal = dict(spec.get("signal", {}))
    sizing = dict(spec.get("sizing", {}))
    params = ", ".join(f"{key}={value}" for key, value in sorted(dict(signal.get("params", {})).items()))
    signal_name = signal.get("name", "unknown")
    sizing_name = sizing.get("name", "unknown")
    return f"{signal_name}({params}) | sizing={sizing_name}"


def _metric_card(label: str, value: str) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'


def _entry_reason(trade: Mapping[str, Any]) -> str:
    return str(trade.get("entry_reason") or "signal_on")


def _exit_reason(trade: Mapping[str, Any]) -> str:
    return str(trade.get("exit_reason") or "unknown")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_time(value: object) -> str:
    if value in (None, ""):
        return "n/a"
    return _parse_time(str(value)).strftime("%Y-%m-%d %H:%M UTC")


def _short_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _format_number(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.4f}"


def _format_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}%"


def _format_cash(value: object) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):,.2f}"


def _format_price(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.4f}"
