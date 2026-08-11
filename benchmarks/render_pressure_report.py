"""Render a multi-arrival-rate FCFS/PALS pressure report without plot dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

REPORT_METRICS = (
    "interactive.e2e_ms_p95",
    "interactive.slo_attainment",
    "overall.slo_goodput_rps",
    "gpu.utilization_gpu_percent.mean",
    "gpu.memory_used_mib.max",
)


def build_pressure_summary(comparisons):
    points = []
    for comparison in comparisons:
        baseline = comparison["baseline"]
        candidate = comparison["candidate"]
        baseline_config = baseline["configuration"]
        candidate_config = candidate["configuration"]
        rate = baseline_config.get("interactive_arrival_rate_rps")
        if rate != candidate_config.get("interactive_arrival_rate_rps"):
            raise ValueError("pressure comparison contains different arrival rates")
        if rate is None:
            raise ValueError("pressure comparison is missing an arrival rate")
        metrics = {}
        for field in REPORT_METRICS:
            if field not in baseline["metrics"] or field not in candidate["metrics"]:
                raise ValueError(f"pressure comparison is missing metric: {field}")
            metrics[field] = {
                "baseline": baseline["metrics"][field],
                "candidate": candidate["metrics"][field],
            }
        points.append({
            "label": baseline_config.get("workload_label") or f"{rate:g} req/s",
            "arrival_rate_rps": float(rate),
            "baseline_name": baseline["name"],
            "candidate_name": candidate["name"],
            "runs": min(baseline["runs"], candidate["runs"]),
            "metrics": metrics,
        })
    points.sort(key=lambda item: item["arrival_rate_rps"])
    if not points:
        raise ValueError("at least one pressure comparison is required")
    names = {(point["baseline_name"], point["candidate_name"]) for point in points}
    if len(names) != 1:
        raise ValueError("all pressure comparisons must use the same policy names")
    return {
        "title": "PALS Multi-Arrival-Rate GPU Pressure Test",
        "baseline_name": points[0]["baseline_name"],
        "candidate_name": points[0]["candidate_name"],
        "points": points,
    }


def _format_ci(summary, scale=1.0, suffix=""):
    mean = summary["mean"] * scale
    half_width = summary.get("ci95_half_width")
    if half_width is None:
        return f"{mean:.2f}{suffix}"
    return f"{mean:.2f} +/- {half_width * scale:.2f}{suffix}"


def render_markdown(summary):
    lines = [
        "# PALS Multi-Arrival-Rate GPU Pressure Test",
        "",
        "Values are mean +/- 95% Student-t confidence-interval half-width.",
        "",
        (
            "| Load | Rate | Policy | Interactive E2E p95 | Interactive SLO | "
            "SLO Goodput | GPU util mean | Peak GPU memory |"
        ),
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in summary["points"]:
        for key, policy in (
            ("baseline", point["baseline_name"]),
            ("candidate", point["candidate_name"]),
        ):
            metrics = point["metrics"]
            lines.append(
                f"| {point['label']} | {point['arrival_rate_rps']:.1f} req/s | "
                f"{policy} | "
                f"{_format_ci(metrics['interactive.e2e_ms_p95'][key], suffix=' ms')} | "
                f"{_format_ci(metrics['interactive.slo_attainment'][key], 100, '%')} | "
                f"{_format_ci(metrics['overall.slo_goodput_rps'][key], suffix=' req/s')} | "
                f"{_format_ci(metrics['gpu.utilization_gpu_percent.mean'][key], suffix='%')} | "
                f"{_format_ci(metrics['gpu.memory_used_mib.max'][key], suffix=' MiB')} |"
            )
    lines.extend([
        "",
        "The chart reports engine-side latency and goodput. Browser rendering time is excluded.",
        "",
    ])
    return "\n".join(lines)


def _polyline(points, color):
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
    elements = [
        (
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" />'
        )
    ]
    for x, y, half_height in points:
        elements.extend([
            (
                f'<line x1="{x:.1f}" y1="{y - half_height:.1f}" x2="{x:.1f}" '
                f'y2="{y + half_height:.1f}" stroke="{color}" stroke-width="1.5" />'
            ),
            (
                f'<line x1="{x - 5:.1f}" y1="{y - half_height:.1f}" '
                f'x2="{x + 5:.1f}" y2="{y - half_height:.1f}" stroke="{color}" />'
            ),
            (
                f'<line x1="{x - 5:.1f}" y1="{y + half_height:.1f}" '
                f'x2="{x + 5:.1f}" y2="{y + half_height:.1f}" stroke="{color}" />'
            ),
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" />',
        ])
    return elements


def _chart(summary, metric, title, unit, left, top, width, height):
    baseline_color = "#525866"
    candidate_color = "#00856a"
    points = summary["points"]
    values = []
    for point in points:
        for key in ("baseline", "candidate"):
            sample = point["metrics"][metric][key]
            values.append(sample["mean"] + (sample.get("ci95_half_width") or 0))
    y_max = max(values) * 1.1 if max(values) > 0 else 1.0
    x_step = width / max(1, len(points) - 1)
    elements = [
        (
            f'<text x="{left}" y="{top - 18}" font-size="18" font-weight="600">'
            f"{escape(title)}</text>"
        )
    ]
    for index in range(5):
        value = y_max * index / 4
        y = top + height - height * index / 4
        elements.extend([
            (
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" '
                'stroke="#d9dde3" stroke-width="1" />'
            ),
            (
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-size="12" fill="#606772">{value:.1f}</text>'
            ),
        ])
    series = {"baseline": [], "candidate": []}
    for index, point in enumerate(points):
        x = left + index * x_step
        label = f"{point['label']} ({point['arrival_rate_rps']:.1f})"
        elements.append(
            f'<text x="{x:.1f}" y="{top + height + 24}" text-anchor="middle" '
            f'font-size="12" fill="#3f454d">{escape(label)}</text>'
        )
        for key, series_points in series.items():
            sample = point["metrics"][metric][key]
            y = top + height - sample["mean"] / y_max * height
            half_height = (sample.get("ci95_half_width") or 0) / y_max * height
            series_points.append((x, y, half_height))
    elements.extend(_polyline(series["baseline"], baseline_color))
    elements.extend(_polyline(series["candidate"], candidate_color))
    elements.append(
        f'<text x="{left - 58}" y="{top + height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 {left - 58} {top + height / 2})" '
        f'font-size="12" fill="#606772">{escape(unit)}</text>'
    )
    return elements


def render_svg(summary):
    width, height = 1280, 700
    baseline_name = escape(summary["baseline_name"])
    candidate_name = escape(summary["candidate_name"])
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<g font-family="Arial, Microsoft YaHei, sans-serif">',
        (
            '<text x="60" y="46" font-size="26" font-weight="700" fill="#171a1f">'
            'PALS GPU Pressure Test</text>'
        ),
        '<text x="60" y="70" font-size="13" fill="#606772">Mean with 95% Student-t CI</text>',
        '<line x1="910" y1="47" x2="940" y2="47" stroke="#525866" stroke-width="3" />',
        f'<text x="948" y="52" font-size="13">{baseline_name}</text>',
        '<line x1="1060" y1="47" x2="1090" y2="47" stroke="#00856a" stroke-width="3" />',
        f'<text x="1098" y="52" font-size="13">{candidate_name}</text>',
    ]
    elements.extend(
        _chart(
            summary,
            "interactive.e2e_ms_p95",
            "Interactive E2E p95",
            "milliseconds",
            105,
            135,
            470,
            430,
        )
    )
    elements.extend(
        _chart(
            summary,
            "overall.slo_goodput_rps",
            "SLO Goodput",
            "requests / second",
            745,
            135,
            470,
            430,
        )
    )
    elements.extend([
        (
            '<text x="640" y="660" text-anchor="middle" font-size="12" fill="#606772">'
            'Load label (interactive arrival rate in req/s)</text>'
        ),
        "</g>",
        "</svg>",
    ])
    return "\n".join(elements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    comparisons = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.comparisons
    ]
    summary = build_pressure_summary(comparisons)
    markdown = render_markdown(summary)
    for path in (args.output_json, args.output_markdown, args.output_svg):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.output_markdown.write_text(markdown, encoding="utf-8")
    args.output_svg.write_text(render_svg(summary), encoding="utf-8")
    print(markdown)
    print(f"SVG: {args.output_svg}")


if __name__ == "__main__":
    main()
