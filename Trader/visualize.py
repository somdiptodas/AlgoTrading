from __future__ import annotations

import argparse
import html
import math
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

try:
    from Trader.config import load_settings
    from Trader.storage import SQLiteBarStore
except ModuleNotFoundError:
    from config import load_settings
    from storage import SQLiteBarStore


MARKET_TZ = timezone(timedelta(hours=-5))


@dataclass(frozen=True)
class VisualizeRequest:
    ticker: str
    multiplier: int
    timespan: str
    database: str | None
    start: datetime | None
    end: datetime | None
    days: int | None
    max_points: int
    output_path: Path | None
    open_browser: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize aggregate bars from the local SQLite database")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--timespan", default="minute")
    parser.add_argument("--database")
    parser.add_argument("--start", help="Start timestamp in ISO format, e.g. 2026-04-14T08:30:00-05:00")
    parser.add_argument("--end", help="End timestamp in ISO format")
    parser.add_argument("--days", type=int, default=5, help="Trailing market days to plot when --start is omitted")
    parser.add_argument("--max-points", type=int, default=1500)
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--no-open", action="store_true", help="Write the chart without opening a browser")
    return parser


def parse_args(argv: list[str] | None = None) -> VisualizeRequest:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = _parse_datetime(args.start) if args.start else None
    end = _parse_datetime(args.end) if args.end else None
    days = args.days

    if start and end and start > end:
        raise ValueError("--start must be on or before --end")
    if days is not None and days < 1:
        raise ValueError("--days must be >= 1")
    if args.max_points < 10:
        raise ValueError("--max-points must be >= 10")

    return VisualizeRequest(
        ticker=args.ticker.upper(),
        multiplier=args.multiplier,
        timespan=args.timespan,
        database=args.database,
        start=start,
        end=end,
        days=days,
        max_points=args.max_points,
        output_path=Path(args.output).expanduser() if args.output else None,
        open_browser=not args.no_open,
    )


def run_visualize(request: VisualizeRequest) -> Path:
    settings = load_settings(database_path=request.database)
    store = SQLiteBarStore(settings.database_path)

    end_dt = request.end or _latest_bar_timestamp(store, request)
    start_dt = request.start or _default_start(end_dt, request.days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    raw_rows = store.fetch_bars(
        ticker=request.ticker,
        multiplier=request.multiplier,
        timespan=request.timespan,
        start_timestamp_ms=start_ms,
        end_timestamp_ms=end_ms,
    )
    if not raw_rows:
        raise RuntimeError("No bars found for the requested range")

    rows = raw_rows
    if len(raw_rows) > request.max_points:
        bucket_ms = _bucket_size_ms(start_ms, end_ms, request.max_points)
        rows = store.fetch_bucketed_bars(
            ticker=request.ticker,
            multiplier=request.multiplier,
            timespan=request.timespan,
            bucket_ms=bucket_ms,
            start_timestamp_ms=start_ms,
            end_timestamp_ms=end_ms,
        )

    output_path = request.output_path or (
        settings.database_path.parent
        / f"{request.ticker.lower()}_{request.multiplier}_{request.timespan}_chart.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(
            request=request,
            database_path=settings.database_path,
            raw_count=len(raw_rows),
            plotted_rows=rows,
            start_dt=start_dt,
            end_dt=end_dt,
        ),
        encoding="utf-8",
    )

    print(f"Wrote chart to {output_path}")
    print(f"Rows in range: {len(raw_rows)}")
    print(f"Points plotted: {len(rows)}")

    if request.open_browser:
        webbrowser.open(output_path.resolve().as_uri())

    return output_path


def _latest_bar_timestamp(store: SQLiteBarStore, request: VisualizeRequest) -> datetime:
    summary = store.fetch_summary(request.ticker, request.multiplier, request.timespan)
    if summary is None or summary["last_bar_utc"] is None:
        raise RuntimeError("No bars available to visualize")
    return _parse_datetime(summary["last_bar_utc"])


def _default_start(end_dt: datetime, days: int | None) -> datetime:
    trailing_days = days or 5
    return end_dt - timedelta(days=trailing_days)


def _bucket_size_ms(start_ms: int, end_ms: int, max_points: int) -> int:
    span_ms = max(end_ms - start_ms, 60_000)
    raw_bucket_ms = max(math.ceil(span_ms / max_points), 60_000)
    minute_ms = 60_000
    return math.ceil(raw_bucket_ms / minute_ms) * minute_ms


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=MARKET_TZ)
    return parsed.astimezone(timezone.utc)


def _format_axis_label(timestamp_ms: int) -> str:
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).astimezone(MARKET_TZ)
    return dt.strftime("%Y-%m-%d %H:%M")


def _render_html(
    request: VisualizeRequest,
    database_path: Path,
    raw_count: int,
    plotted_rows: Sequence[object],
    start_dt: datetime,
    end_dt: datetime,
) -> str:
    width = 1400
    height = 820
    left = 80
    right = 40
    top = 60
    price_height = 470
    volume_top = 580
    volume_height = 140
    chart_width = width - left - right
    price_bottom = top + price_height
    volume_bottom = volume_top + volume_height

    closes = [float(row["close"]) for row in plotted_rows if row["close"] is not None]
    highs = [float(row["high"]) for row in plotted_rows if row["high"] is not None]
    lows = [float(row["low"]) for row in plotted_rows if row["low"] is not None]
    volumes = [float(row["volume"]) for row in plotted_rows if row["volume"] is not None]

    min_price = min(lows or closes)
    max_price = max(highs or closes)
    if math.isclose(min_price, max_price):
        min_price -= 1.0
        max_price += 1.0
    pad = (max_price - min_price) * 0.06
    min_price -= pad
    max_price += pad
    max_volume = max(volumes) if volumes else 1.0

    def x_at(index: int) -> float:
        if len(plotted_rows) == 1:
            return left + chart_width / 2
        return left + (index / (len(plotted_rows) - 1)) * chart_width

    def price_y(value: float) -> float:
        return top + (max_price - value) / (max_price - min_price) * price_height

    def volume_y(value: float) -> float:
        return volume_bottom - (value / max_volume) * volume_height

    price_path: list[str] = []
    volume_rects: list[str] = []
    markers: list[str] = []
    for index, row in enumerate(plotted_rows):
        close = row["close"]
        high = row["high"]
        low = row["low"]
        open_price = row["open"]
        volume = row["volume"] or 0
        timestamp_ms = int(row["timestamp_ms"])
        x = x_at(index)

        if close is not None:
            cmd = "M" if not price_path else "L"
            price_path.append(f"{cmd}{x:.2f},{price_y(float(close)):.2f}")

        bar_width = max(chart_width / max(len(plotted_rows), 1) * 0.75, 1.0)
        bar_top = volume_y(float(volume))
        volume_rects.append(
            f'<rect x="{x - bar_width / 2:.2f}" y="{bar_top:.2f}" width="{bar_width:.2f}" '
            f'height="{volume_bottom - bar_top:.2f}" fill="rgba(212, 175, 55, 0.35)" />'
        )

        price_text = (
            f"O {float(open_price):.2f}  H {float(high):.2f}  "
            f"L {float(low):.2f}  C {float(close):.2f}"
            if None not in (open_price, high, low, close)
            else "Price data incomplete"
        )
        tooltip = html.escape(
            f"{_format_axis_label(timestamp_ms)}\n{price_text}\nVolume {float(volume):,.0f}"
        )
        markers.append(
            f'<circle cx="{x:.2f}" cy="{price_y(float(close)):.2f}" r="4" fill="#f8efe1" '
            f'class="point"><title>{tooltip}</title></circle>'
            if close is not None
            else ""
        )
    price_grid = _render_horizontal_grid(
        ticks=6,
        minimum=min_price,
        maximum=max_price,
        top=top,
        bottom=price_bottom,
        left=left,
        right=width - right,
        formatter=lambda value: f"{value:,.2f}",
    )
    volume_grid = _render_horizontal_grid(
        ticks=3,
        minimum=0.0,
        maximum=max_volume,
        top=volume_top,
        bottom=volume_bottom,
        left=left,
        right=width - right,
        formatter=lambda value: f"{value:,.0f}",
    )
    x_axis = _render_x_axis(plotted_rows, left, width - right, volume_bottom)

    stats = _summary_stats(raw_count, plotted_rows)
    title = f"{request.ticker} {request.multiplier}-{request.timespan} bars"
    range_text = (
        f"{start_dt.astimezone(MARKET_TZ).strftime('%Y-%m-%d %H:%M %Z')} to "
        f"{end_dt.astimezone(MARKET_TZ).strftime('%Y-%m-%d %H:%M %Z')}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5efe6;
      --panel: #1e2a36;
      --panel-alt: #243342;
      --ink: #f5efe6;
      --muted: #b8c3cf;
      --accent: #d4af37;
      --accent-soft: rgba(212, 175, 55, 0.16);
      --grid: rgba(245, 239, 230, 0.18);
      --line: #ffd166;
      --volume: rgba(212, 175, 55, 0.35);
      --good: #84dcc6;
      --bad: #ff7a59;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(212, 175, 55, 0.22), transparent 28%),
        linear-gradient(135deg, #f5efe6, #eadfce 55%, #e3d6c3);
      color: #11161c;
    }}
    main {{
      max-width: 1520px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 12px;
      color: #5c6670;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 0.94;
      max-width: 11ch;
    }}
    .sub {{
      max-width: 70ch;
      color: #2f3740;
      font-size: 15px;
      line-height: 1.5;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 8px 0 26px;
    }}
    .meta-card {{
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.5);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(30, 42, 54, 0.08);
    }}
    .meta-card strong {{
      display: block;
      margin-top: 6px;
      font-size: 20px;
      color: #121922;
    }}
    .chart-shell {{
      border-radius: 28px;
      overflow: hidden;
      background: linear-gradient(180deg, var(--panel), var(--panel-alt));
      box-shadow: 0 24px 80px rgba(26, 27, 31, 0.18);
      border: 1px solid rgba(30, 42, 54, 0.16);
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .grid {{
      stroke: var(--grid);
      stroke-width: 1;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .section-label {{
      fill: var(--ink);
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .price-line {{
      fill: none;
      stroke: var(--line);
      stroke-width: 2.5;
      stroke-linejoin: round;
      stroke-linecap: round;
      filter: drop-shadow(0 0 12px rgba(255, 209, 102, 0.28));
    }}
    .point {{
      opacity: 0;
      transition: opacity 160ms ease;
    }}
    .chart-shell:hover .point {{
      opacity: 1;
    }}
    .footer {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 0 4px;
      margin-top: 14px;
      color: #4a525a;
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 20px 14px 28px;
      }}
      .footer {{
        display: grid;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">Local Market Data</div>
      <h1>{html.escape(title)}</h1>
      <div class="sub">
        Minute-level aggregate bars pulled from <code>{html.escape(str(database_path))}</code>. The chart
        automatically buckets the selected range when needed so the browser stays responsive.
      </div>
    </section>

    <section class="meta">
      <div class="meta-card">Range<strong>{html.escape(range_text)}</strong></div>
      <div class="meta-card">Rows In Range<strong>{raw_count:,}</strong></div>
      <div class="meta-card">Points Plotted<strong>{len(plotted_rows):,}</strong></div>
      <div class="meta-card">Last Close<strong>{stats["last_close"]}</strong></div>
    </section>

    <section class="chart-shell">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <text x="{left}" y="34" class="section-label">Price</text>
        <text x="{left}" y="{volume_top - 18}" class="section-label">Volume</text>
        {price_grid}
        {volume_grid}
        {x_axis}
        <path class="price-line" d="{' '.join(price_path)}" />
        {''.join(volume_rects)}
        {''.join(markers)}
      </svg>
    </section>

    <section class="footer">
      <div>{html.escape(stats["change_text"])}</div>
      <div>{html.escape(stats["high_low_text"])}</div>
      <div>{html.escape(stats["volume_text"])}</div>
    </section>
  </main>
</body>
</html>
"""


def _render_horizontal_grid(
    *,
    ticks: int,
    minimum: float,
    maximum: float,
    top: float,
    bottom: float,
    left: float,
    right: float,
    formatter,
) -> str:
    parts: list[str] = []
    span = max(maximum - minimum, 1e-9)
    for index in range(ticks + 1):
        fraction = index / ticks
        y = top + fraction * (bottom - top)
        value = maximum - fraction * span
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" class="grid" />')
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">'
            f"{html.escape(formatter(value))}</text>"
        )
    return "".join(parts)


def _render_x_axis(rows: Sequence[object], left: float, right: float, baseline_y: float) -> str:
    if not rows:
        return ""

    ticks = min(8, len(rows) - 1) if len(rows) > 1 else 1
    parts: list[str] = []
    span = right - left

    for index in range(ticks + 1):
        row_index = round((index / ticks) * (len(rows) - 1)) if ticks else 0
        x = left + (row_index / max(len(rows) - 1, 1)) * span
        label = _format_axis_label(int(rows[row_index]["timestamp_ms"]))
        parts.append(f'<line x1="{x:.2f}" y1="60" x2="{x:.2f}" y2="{baseline_y}" class="grid" />')
        parts.append(
            f'<text x="{x:.2f}" y="{baseline_y + 24:.2f}" text-anchor="middle" class="axis-label">'
            f"{html.escape(label)}</text>"
        )
    return "".join(parts)


def _summary_stats(raw_count: int, rows: Sequence[object]) -> dict[str, str]:
    closes = [float(row["close"]) for row in rows if row["close"] is not None]
    highs = [float(row["high"]) for row in rows if row["high"] is not None]
    lows = [float(row["low"]) for row in rows if row["low"] is not None]
    volumes = [float(row["volume"]) for row in rows if row["volume"] is not None]

    first_close = closes[0]
    last_close = closes[-1]
    change = last_close - first_close
    change_pct = (change / first_close) * 100 if first_close else 0.0

    return {
        "last_close": f"{last_close:,.2f}",
        "change_text": f"Move across plotted window: {change:+,.2f} ({change_pct:+.2f}%)",
        "high_low_text": f"Window high/low: {max(highs):,.2f} / {min(lows):,.2f}",
        "volume_text": f"Volume shown: {sum(volumes):,.0f} across {raw_count:,} source rows",
    }
