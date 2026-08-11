from xml.etree import ElementTree

from benchmarks.render_pressure_report import (
    REPORT_METRICS,
    build_pressure_summary,
    render_markdown,
    render_svg,
)


def make_group(name, rate, label, scale):
    return {
        "name": name,
        "runs": 3,
        "configuration": {
            "workload_label": label,
            "interactive_arrival_rate_rps": rate,
        },
        "metrics": {
            field: {
                "samples": 3,
                "mean": scale * (index + 1),
                "stdev": 0.1,
                "min": 0.9,
                "max": 1.1,
                "ci95_low": scale * (index + 1) - 0.2,
                "ci95_high": scale * (index + 1) + 0.2,
                "ci95_half_width": 0.2,
            }
            for index, field in enumerate(REPORT_METRICS)
        },
    }


def make_comparison(rate, label):
    return {
        "baseline": make_group("FCFS", rate, label, 2),
        "candidate": make_group("PALS", rate, label, 1),
        "metrics": {},
    }


def test_pressure_report_orders_rates_and_renders_markdown_and_svg():
    summary = build_pressure_summary([
        make_comparison(20, "high"),
        make_comparison(2, "low"),
    ])

    assert [point["label"] for point in summary["points"]] == ["low", "high"]
    markdown = render_markdown(summary)
    assert "Interactive E2E p95" in markdown
    assert "PALS" in markdown
    svg = render_svg(summary)
    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "SLO Goodput" in svg
