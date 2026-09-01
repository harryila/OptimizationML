#!/usr/bin/env python3
"""Render dependency-free SVG figures from committed result manifests."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

AUDIT_PATH = Path("results/summaries/deficit_audit.json")
QUADRATIC_PATH = Path("results/summaries/quadratic_lr_sweep.json")
OUTPUT_DIR = Path("results/figures")

INK = "#182230"
MUTED = "#667085"
GRID = "#DDE3EA"
BLUE = "#2F6BFF"
ORANGE = "#E46C2A"
GOLD = "#B58105"
PALE = "#F7F9FC"
WHITE = "#FFFFFF"

DISPLAY_NAMES = {
    "jordan_quintic": "Jordan quintic",
    "classical_cubic": "Classical NS",
    "taylor_quintic": "Taylor NS5",
    "polar_express_repo_71cc_steps5": "Polar Express 71cc",
    "cans_degree5_iter4_delta0.3": "CANS 5\u00d74, δ=0.3",
}


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    fill: str = INK,
    anchor: str = "start",
    weight: int = 400,
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="Inter, ui-sans-serif, system-ui, sans-serif"{transform}>'
        f"{html.escape(value)}</text>"
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 1.5,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def _circle(
    x: float,
    y: float,
    *,
    color: str,
    filled: bool,
    radius: float = 4.0,
) -> str:
    fill = color if filled else WHITE
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" '
        f'stroke="{color}" stroke-width="1.8"/>'
    )


def _polyline(points: list[tuple[float, float]], *, color: str, dash: str | None = None) -> str:
    if len(points) < 2:
        return ""
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linejoin="round"{dash_attr}/>'
    )


def _svg_document(
    width: int,
    height: int,
    body: list[str],
    *,
    title: str,
    metadata: dict[str, Any],
) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                f'aria-labelledby="title desc">'
            ),
            f'<title id="title">{html.escape(title)}</title>',
            '<desc id="desc">Reproducible research figure generated from committed JSON.</desc>',
            f"<metadata>{html.escape(json.dumps(metadata, sort_keys=True))}</metadata>",
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def _source_metadata(paths: list[Path]) -> dict[str, Any]:
    paths = [Path("scripts/make_figures.py"), *paths]
    return {
        "generator": "scripts/make_figures.py",
        "sources": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
    }


def make_deficit_figure(audit: dict[str, Any]) -> Path:
    width, height = 1380, 650
    left, right, top, bottom = 85, 30, 150, 95
    panel_gap = 20
    families = list(DISPLAY_NAMES)
    panel_width = (width - left - right - panel_gap * (len(families) - 1)) / len(families)
    plot_height = height - top - bottom
    log_min, log_max = -3.0, 4.0

    def y_position(value: float) -> float:
        clipped = max(10**log_min, min(10**log_max, value))
        fraction = (np_log10(clipped) - log_min) / (log_max - log_min)
        return top + plot_height * (1 - fraction)

    body = [
        _text(left, 42, "Unit-spectrum local passivity deficits", size=25, weight=650),
        _text(
            left,
            70,
            "Exact-current normalization, fixed scale, and the normalization-clean "
            "subset; 2\u00d72 angular grid",
            size=14,
            fill=MUTED,
        ),
    ]
    legend = [
        (BLUE, True, None, "Exact current norm"),
        (ORANGE, False, None, "Fixed scale"),
        (GOLD, True, "5 4", "Current norm where fixed-scale Jacobian ≥ 0"),
    ]
    legend_x = left
    for color, filled, dash, label in legend:
        body.append(_line(legend_x, 108, legend_x + 28, 108, stroke=color, width=2, dash=dash))
        body.append(_circle(legend_x + 14, 108, color=color, filled=filled, radius=3.5))
        body.append(_text(legend_x + 36, 113, label, size=12, fill=MUTED))
        legend_x += 250 if label != legend[-1][3] else 0

    for tick in range(-3, 5):
        y = y_position(10.0**tick)
        body.append(_line(left, y, width - right, y, stroke=GRID, width=0.8))
        body.append(_text(left - 10, y + 4, f"10^{tick}", size=11, fill=MUTED, anchor="end"))

    rows = audit["rows"]
    series = (
        ("unit_exact_local_deficit", BLUE, True, None),
        ("unit_fixed_scale_deficit_grid", ORANGE, False, None),
        ("clean_normalization_deficit", GOLD, True, "5 4"),
    )
    for panel_index, family in enumerate(families):
        x0 = left + panel_index * (panel_width + panel_gap)
        family_rows = sorted(
            (row for row in rows if row["baseline"] == family),
            key=lambda row: row["steps"],
        )
        x_positions = {
            row["steps"]: x0
            + (row["steps"] - 1) * panel_width / max(1, family_rows[-1]["steps"] - 1)
            for row in family_rows
        }
        body.append(
            _text(
                x0 + panel_width / 2,
                top - 14,
                DISPLAY_NAMES[family],
                size=13,
                anchor="middle",
                weight=600,
            )
        )
        body.append(
            _line(x0, top + plot_height, x0 + panel_width, top + plot_height, stroke=INK, width=1)
        )
        for row in family_rows:
            x = x_positions[row["steps"]]
            body.append(_line(x, top + plot_height, x, top + plot_height + 5, stroke=INK, width=1))
            body.append(
                _text(
                    x,
                    top + plot_height + 22,
                    str(row["steps"]),
                    size=11,
                    fill=MUTED,
                    anchor="middle",
                )
            )
        for field, color, filled, dash in series:
            points = []
            zero_points = []
            for row in family_rows:
                value = row[field]
                if value is None:
                    continue
                if float(value) <= 0:
                    zero_points.append((x_positions[row["steps"]], top + plot_height))
                else:
                    points.append((x_positions[row["steps"]], y_position(float(value))))
            body.append(_polyline(points, color=color, dash=dash))
            for x, y in points:
                body.append(_circle(x, y, color=color, filled=filled))
            for x, y in zero_points:
                body.append(_circle(x, y, color=color, filled=False, radius=3.5))
        if panel_index == len(families) // 2:
            body.append(
                _text(
                    x0 + panel_width / 2,
                    height - 24,
                    "Polynomial stage",
                    size=13,
                    anchor="middle",
                    weight=550,
                )
            )
    body.append(
        _text(
            18,
            top + plot_height / 2,
            "Deficit (log scale)",
            size=13,
            anchor="middle",
            weight=550,
            rotate=-90,
        )
    )
    body.append(
        _text(
            left,
            height - 50,
            "Open markers on the floor denote an observed value of zero; missing "
            "clean-subset markers mean no sampled fixed-scale-monotone point.",
            size=11,
            fill=MUTED,
        )
    )
    path = OUTPUT_DIR / "unit_spectrum_deficits.svg"
    path.write_text(
        _svg_document(
            width,
            height,
            body,
            title="Unit-spectrum local passivity deficits",
            metadata=_source_metadata([AUDIT_PATH]),
        ),
        encoding="utf-8",
    )
    return path


def np_log10(value: float) -> float:
    import math

    return math.log10(value)


def make_quadratic_figure(quadratic: dict[str, Any]) -> Path:
    width, height = 1380, 650
    left, right, top, bottom = 85, 30, 150, 95
    panel_gap = 20
    families = list(DISPLAY_NAMES)
    panel_width = (width - left - right - panel_gap * (len(families) - 1)) / len(families)
    plot_height = height - top - bottom
    y_min, y_max = 0.0, 7.0

    def y_position(value: float) -> float:
        return top + plot_height * (1 - (value - y_min) / (y_max - y_min))

    body = [
        _text(left, 42, "Normalized upper endpoint of the finite target band", size=25, weight=650),
        _text(
            left,
            70,
            "32 diagonal quadratics, eps=1, 750 steps, final-50 hold, objective "
            "ratio ≤10⁻⁴ for ≥90% of problems",
            size=14,
            fill=MUTED,
        ),
    ]
    legend = (
        (BLUE, False, "Unrepaired"),
        (BLUE, True, "Gain-matched repair"),
        (ORANGE, False, "Gain-only control"),
        (ORANGE, True, "Raw repair"),
    )
    legend_x = left
    for color, filled, label in legend:
        body.append(_line(legend_x, 108, legend_x + 28, 108, stroke=color, width=2))
        body.append(_circle(legend_x + 14, 108, color=color, filled=filled, radius=3.5))
        body.append(_text(legend_x + 36, 113, label, size=12, fill=MUTED))
        legend_x += 205

    for tick in range(0, 8):
        y = y_position(float(tick))
        body.append(_line(left, y, width - right, y, stroke=GRID, width=0.8))
        body.append(_text(left - 10, y + 4, str(tick), size=11, fill=MUTED, anchor="end"))
    ceiling_y = y_position(2.0)
    body.append(_line(left, ceiling_y, width - right, ceiling_y, stroke=INK, width=1.2, dash="7 5"))
    body.append(
        _text(
            width - right,
            ceiling_y - 7,
            "local linear ceiling = 2",
            size=11,
            fill=MUTED,
            anchor="end",
        )
    )

    rows = quadratic["rows"]
    series = (
        ("unrepaired", BLUE, False, -6),
        ("gain_matched_grid_repair_1.02x", BLUE, True, -2),
        ("gain_only_unrepaired_control", ORANGE, False, 2),
        ("raw_grid_repair_1.02x", ORANGE, True, 6),
    )
    for panel_index, family in enumerate(families):
        x0 = left + panel_index * (panel_width + panel_gap)
        family_rows = [
            row
            for row in rows
            if row["baseline"] == family and row["normalization"] == "current_plus_eps"
        ]
        steps = sorted({row["steps"] for row in family_rows})
        x_positions = {
            step: x0 + (step - 1) * panel_width / max(1, steps[-1] - 1) for step in steps
        }
        body.append(
            _text(
                x0 + panel_width / 2,
                top - 14,
                DISPLAY_NAMES[family],
                size=13,
                anchor="middle",
                weight=600,
            )
        )
        body.append(
            _line(x0, top + plot_height, x0 + panel_width, top + plot_height, stroke=INK, width=1)
        )
        for step in steps:
            x = x_positions[step]
            body.append(_line(x, top + plot_height, x, top + plot_height + 5, stroke=INK, width=1))
            body.append(
                _text(x, top + plot_height + 22, str(step), size=11, fill=MUTED, anchor="middle")
            )
        for intervention, color, filled, offset in series:
            points = []
            missing = []
            for step in steps:
                row = next(
                    row
                    for row in family_rows
                    if row["steps"] == step and row["intervention"] == intervention
                )
                x = x_positions[step] + offset
                if row["target_upper_pass"] is None:
                    missing.append((x, top + plot_height - 5))
                else:
                    normalized = row["target_upper_pass"] * row["effective_zero_slope_gain"]
                    points.append((x, y_position(float(normalized))))
            body.append(_polyline(points, color=color))
            for x, y in points:
                body.append(_circle(x, y, color=color, filled=filled))
            for x, y in missing:
                body.append(_line(x - 3, y - 3, x + 3, y + 3, stroke=color, width=1.6))
                body.append(_line(x - 3, y + 3, x + 3, y - 3, stroke=color, width=1.6))
        if panel_index == len(families) // 2:
            body.append(
                _text(
                    x0 + panel_width / 2,
                    height - 24,
                    "Polynomial stage",
                    size=13,
                    anchor="middle",
                    weight=550,
                )
            )
    body.append(
        _text(
            18,
            top + plot_height / 2,
            "η upper \u00d7 effective zero-slope gain",
            size=13,
            anchor="middle",
            weight=550,
            rotate=-90,
        )
    )
    body.append(
        _text(
            left,
            height - 50,
            "\u00d7 denotes no passing target-and-hold band on the sampled learning-rate "
            "grid. Values above 2 are finite-time target endpoints, not local stability.",
            size=11,
            fill=MUTED,
        )
    )
    path = OUTPUT_DIR / "quadratic_normalized_upper_endpoints.svg"
    path.write_text(
        _svg_document(
            width,
            height,
            body,
            title="Normalized upper endpoint of the finite target band",
            metadata=_source_metadata([QUADRATIC_PATH]),
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    quadratic = json.loads(QUADRATIC_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [make_deficit_figure(audit), make_quadratic_figure(quadratic)]
    print(json.dumps({"figures": [str(path) for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
