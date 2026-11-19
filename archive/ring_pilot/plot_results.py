#!/usr/bin/env python3
"""Render a small PSD replication summary as a dependency-free SVG."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from html import escape
from pathlib import Path


COLORS = {
    "joint": "#0072B2",
    "frozen": "#D55E00",
    "fixed_random": "#009E73",
}
LABELS = {
    "joint": "Joint training",
    "frozen": "Oracle encoder",
    "fixed_random": "Random encoder",
}
FALLBACK_COLORS = ("#CC79A7", "#E69F00", "#56B4E9", "#6B6B6B")
WIDTH = 1120
HEIGHT = 1080
FONT = "Arial, Helvetica, sans-serif"


def _text(x: float, y: float, value: object, *, size: int = 13,
          fill: str = "#222", anchor: str = "start", weight: str = "normal",
          rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate:g} {x:g} {y:g})"' if rotate is not None else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" '
            f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{transform}>{escape(str(value))}</text>')


def _line(x1: float, y1: float, x2: float, y2: float, *, color: str = "#bbb",
          width: float = 1, dash: str = "") -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width:g}"{dashed}/>')


def _circle(x: float, y: float, radius: float, color: str, *, opacity: float = 0.82,
            stroke: str = "white", stroke_width: float = 1) -> str:
    return (f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:g}" fill="{color}" '
            f'fill-opacity="{opacity:g}" stroke="{stroke}" stroke-width="{stroke_width:g}"/>')


def _diamond(x: float, y: float, radius: float, color: str, *, opacity: float = 0.82,
             stroke: str = "white", stroke_width: float = 1) -> str:
    points = f"{x:.2f},{y-radius:.2f} {x+radius:.2f},{y:.2f} {x:.2f},{y+radius:.2f} {x-radius:.2f},{y:.2f}"
    return (f'<polygon points="{points}" fill="{color}" fill-opacity="{opacity:g}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:g}"/>')


def _marker(x: float, y: float, kind: str, color: str, *, radius: float = 5.5,
            opacity: float = 0.82, stroke: str = "white", stroke_width: float = 1) -> str:
    if kind == "diamond":
        return _diamond(x, y, radius, color, opacity=opacity, stroke=stroke, stroke_width=stroke_width)
    if kind == "square":
        return (f'<rect x="{x-radius:.2f}" y="{y-radius:.2f}" width="{2*radius:.2f}" '
                f'height="{2*radius:.2f}" fill="{color}" fill-opacity="{opacity:g}" '
                f'stroke="{stroke}" stroke-width="{stroke_width:g}"/>')
    return _circle(x, y, radius, color, opacity=opacity, stroke=stroke, stroke_width=stroke_width)


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return [low]
    raw = (high - low) / max(1, count - 1)
    power = 10 ** math.floor(math.log10(raw))
    fraction = raw / power
    step = (1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10) * power
    first = math.ceil(low / step) * step
    values = []
    value = first
    while value <= high + step * 1e-8:
        values.append(0.0 if abs(value) < step * 1e-8 else value)
        value += step
    return values or [low, high]


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) < 100 else f"{value:.0f}"


def _label(experiment: str) -> str:
    return LABELS.get(experiment, experiment)


def _validate_rows(data: object) -> tuple[list[dict], list[str]]:
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise ValueError('input JSON must have a top-level "rows" array')
    rows = []
    for idx, raw in enumerate(data["rows"]):
        if not isinstance(raw, dict):
            raise ValueError(f"rows[{idx}] must be an object")
        try:
            row = {
                "experiment": str(raw["experiment"]),
                "seed": int(raw["seed"]),
                "L": int(raw["L"]),
                "dominant_period": None if raw.get("dominant_period") is None else float(raw["dominant_period"]),
                "cycle_match": float(raw["cycle_match"]),
                "intrinsic_reward": float(raw["intrinsic_reward"]),
                "step_ratio": float(raw["step_ratio"]),
                "half_ratio": float(raw["half_ratio"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"rows[{idx}] is missing a valid required field: {exc}") from exc
        for field in ("cycle_match", "intrinsic_reward", "step_ratio", "half_ratio"):
            if not math.isfinite(row[field]):
                raise ValueError(f"rows[{idx}].{field} must be finite")
        if not 0 <= row["cycle_match"] <= 1:
            raise ValueError(f"rows[{idx}].cycle_match must be between 0 and 1")
        if not 0 <= row["intrinsic_reward"] <= 1:
            raise ValueError(f"rows[{idx}].intrinsic_reward must be between 0 and 1")
        if row["dominant_period"] is not None and not math.isfinite(row["dominant_period"]):
            raise ValueError(f"rows[{idx}].dominant_period must be finite or null")
        rows.append(row)
    exp_order = list(dict.fromkeys(row["experiment"] for row in rows))
    return rows, exp_order


def render_svg(rows: list[dict], experiments: list[str]) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">']
    out.append(f'<rect width="100%" height="100%" fill="white"/>')
    out.append(_text(42, 36, "Periodic Skill Discovery: compact replication diagnostics", size=20, weight="bold"))
    out.append(_text(42, 58, "Each mark is one run at one L; hover for seed and values. Conditions use consistent colors.", size=12, fill="#555"))

    colors = dict(COLORS)
    for experiment in experiments:
        if experiment not in colors:
            colors[experiment] = FALLBACK_COLORS[len(colors) % len(FALLBACK_COLORS)]

    # Period plot: square axes preserve the meaning of the expected=measured diagonal.
    period_rows = [r for r in rows if r["dominant_period"] is not None]
    panel_top, panel_bottom = 86, 436
    out.append(_text(42, 82, "1. Period control", size=16, weight="bold"))
    x0, y0, plot = 190, 110, 278
    if period_rows:
        xvals = [2 * r["L"] for r in period_rows]
        yvals = [r["dominant_period"] for r in period_rows]
        lo = min(0.0, min(xvals), min(yvals))
        hi = max(max(xvals), max(yvals))
        span = max(1.0, hi - lo)
        pad = 0.08 * span
        lo -= pad
        hi += pad
        ticks = _nice_ticks(lo, hi, 5)
        def px(v: float) -> float: return x0 + (v - lo) / (hi - lo) * plot
        def py(v: float) -> float: return y0 + plot - (v - lo) / (hi - lo) * plot
        for t in ticks:
            out.append(_line(px(t), y0, px(t), y0 + plot, color="#e4e7eb"))
            out.append(_line(x0, py(t), x0 + plot, py(t), color="#e4e7eb"))
            out.append(_text(px(t), y0 + plot + 18, _fmt(t), size=11, anchor="middle", fill="#555"))
            out.append(_text(x0 - 9, py(t) + 4, _fmt(t), size=11, anchor="end", fill="#555"))
        out.append(_line(x0, y0 + plot, x0 + plot, y0 + plot, color="#333"))
        out.append(_line(x0, y0, x0, y0 + plot, color="#333"))
        out.append(_line(px(lo), py(lo), px(hi), py(hi), color="#555", width=1.7, dash="6 4"))
        out.append(_text(x0 + plot / 2, y0 + plot + 40, "Expected period (2L steps)", size=12, anchor="middle"))
        out.append(_text(x0 - 53, y0 + plot / 2, "Measured dominant period (steps)", size=12, anchor="middle", rotate=-90))
        # A very small deterministic horizontal offset keeps repeated runs visible.
        by_expected: dict[int, list[dict]] = defaultdict(list)
        for row in period_rows:
            by_expected[2 * row["L"]].append(row)
        for expected, group in by_expected.items():
            group.sort(key=lambda r: (experiments.index(r["experiment"]), r["seed"]))
            for i, row in enumerate(group):
                offset = 0 if len(group) == 1 else (i - (len(group) - 1) / 2) * min(3.5, plot * 0.008)
                cx, cy = px(expected) + offset, py(row["dominant_period"])
                title = (f'{row["experiment"]}; seed={row["seed"]}; L={row["L"]}; '
                         f'expected={expected}; measured={_fmt(row["dominant_period"])}; '
                         f'cycle_match={row["cycle_match"]:.3f}')
                out.append(f'<g><title>{escape(title)}</title>')
                out.append(_marker(cx, cy, "circle", colors[row["experiment"]], radius=6.2))
                out.append("</g>")
    else:
        out.append(_text(330, 240, "No measured periods in input", size=15, anchor="middle", fill="#666"))

    # Color legend lives beside the square period panel and includes only conditions with data here.
    legend_exps = [e for e in experiments if any(r["experiment"] == e for r in period_rows)]
    lx, ly = 565, 126
    out.append(_text(lx, ly, "Condition", size=13, weight="bold"))
    for i, exp in enumerate(legend_exps):
        yy = ly + 25 + i * 24
        out.append(_marker(lx + 8, yy - 4, "circle", colors[exp], radius=5))
        out.append(_text(lx + 22, yy, _label(exp), size=12))
    out.append(_line(lx, ly + 111, lx + 24, ly + 111, color="#555", width=1.7, dash="6 4"))
    out.append(_text(lx + 32, ly + 115, "ideal period = 2L", size=12))
    out.append(_text(lx, ly + 190, "Period x offsets are tiny and only separate", size=11, fill="#555"))
    out.append(_text(lx, ly + 206, "runs with the same expected 2L value.", size=11, fill="#555"))

    # Latent distance ratios grouped by condition, with both ratios shown and target=1.
    top, bottom = 474, 785
    out.append(_line(42, panel_bottom + 10, WIDTH - 42, panel_bottom + 10, color="#ddd"))
    out.append(_text(42, 468, "2. Latent geometry", size=16, weight="bold"))
    rx0, ry0, rw, rh = 95, 500, 960, 220
    ratio_values = [r[key] for r in rows for key in ("step_ratio", "half_ratio")]
    rlo = min(0.0, min(ratio_values, default=0.0))
    rhi = max(1.4, max(ratio_values, default=1.0))
    if rhi - rlo < 0.2:
        rhi = rlo + 0.2
    rpad = 0.05 * (rhi - rlo)
    rlo -= rpad
    rhi += rpad
    def rpy(v: float) -> float: return ry0 + rh - (v - rlo) / (rhi - rlo) * rh
    for t in _nice_ticks(rlo, rhi, 5):
        out.append(_line(rx0, rpy(t), rx0 + rw, rpy(t), color="#e4e7eb"))
        out.append(_text(rx0 - 10, rpy(t) + 4, _fmt(t), size=11, anchor="end", fill="#555"))
    out.append(_line(rx0, rpy(1), rx0 + rw, rpy(1), color="#333", width=1.7, dash="6 4"))
    out.append(_line(rx0, ry0, rx0, ry0 + rh, color="#333"))
    out.append(_text(rx0 + rw / 2, ry0 + rh + 39, "Condition and latent distance", size=12, anchor="middle"))
    out.append(_text(rx0 - 54, ry0 + rh / 2, "Measured / ideal distance", size=12, anchor="middle", rotate=-90))
    out.append(_text(rx0 + rw - 3, rpy(1) - 7, "ideal = 1", size=11, anchor="end", fill="#444"))

    categories = [(exp, metric) for exp in experiments for metric in ("step", "L-step")]
    category_x = {key: rx0 + (i + 0.5) * rw / max(1, len(categories)) for i, key in enumerate(categories)}
    metric_specs = (("step_ratio", "step", "circle", "1-step"), ("half_ratio", "L-step", "diamond", "L-step"))
    for exp, metric in categories:
        label = "1-step" if metric == "step" else "L-step"
        x = category_x[(exp, metric)]
        out.append(_text(x, ry0 + rh + 19, label, size=10, anchor="middle", fill="#555"))
    # show condition names beneath the paired categories
    for i, exp in enumerate(experiments):
        mid = (category_x[(exp, "step")] + category_x[(exp, "L-step")]) / 2
        out.append(_text(mid, ry0 + rh + 56, _label(exp), size=12, anchor="middle", fill=colors[exp], weight="bold"))
    seeds_by_group: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        seeds_by_group[(row["experiment"], row["L"])].append(row)
    for row in rows:
        for field, metric, shape, label in metric_specs:
            cx = category_x.get((row["experiment"], metric))
            if cx is None:
                continue
            siblings = sorted(seeds_by_group[(row["experiment"], row["L"])], key=lambda r: r["seed"])
            idx = next((i for i, other in enumerate(siblings) if other["seed"] == row["seed"]), 0)
            jx = (idx - (len(siblings) - 1) / 2) * min(5, rw / (len(categories) * 10))
            cy = rpy(row[field])
            title = (f'{row["experiment"]}; seed={row["seed"]}; L={row["L"]}; '
                     f'{label} ratio={row[field]:.4f}; cycle_match={row["cycle_match"]:.3f}')
            out.append(f'<g><title>{escape(title)}</title>')
            bad = abs(row[field] - 1.0) > 0.2
            out.append(_marker(cx + jx, cy, shape, colors[row["experiment"]], radius=5.6,
                               stroke="#9b1c1c" if bad else "white", stroke_width=2 if bad else 1))
            out.append("</g>")
    # symbol legend
    sx, sy = 760, 490
    out.append(_marker(sx, sy, "circle", "#555", radius=4.5))
    out.append(_text(sx + 10, sy + 4, "1-step ratio", size=11))
    out.append(_marker(sx + 107, sy, "diamond", "#555", radius=4.5))
    out.append(_text(sx + 117, sy + 4, "L-step ratio", size=11))
    out.append(_line(sx + 222, sy, sx + 244, sy, color="#333", width=1.7, dash="6 4"))
    out.append(_text(sx + 251, sy + 4, "target", size=11))
    out.append(_text(42, 773, "Red outlines mark ratios more than 20% from the target; hover a point for seed and L.", size=11, fill="#555"))

    # Show the intrinsic reward and exact cycle-match fraction as separate measures.
    out.append(_line(42, bottom + 10, WIDTH - 42, bottom + 10, color="#ddd"))
    out.append(_text(42, 817, "3. Intrinsic reward and exact cycle match", size=16, weight="bold"))
    cx0, cy0, cw, ch = 95, 858, 960, 120
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        yy = cy0 + ch - t * ch
        out.append(_line(cx0, yy, cx0 + cw, yy, color="#e4e7eb"))
        out.append(_text(cx0 - 10, yy + 4, _fmt(t), size=11, anchor="end", fill="#555"))
    out.append(_line(cx0, cy0, cx0, cy0 + ch, color="#333"))
    out.append(_line(cx0, cy0 + ch, cx0 + cw, cy0 + ch, color="#333"))
    out.append(_text(cx0 - 57, cy0 + ch / 2, "Score (0–1)", size=12, anchor="middle", rotate=-90))
    score_x = {exp: cx0 + (i + 0.5) * cw / max(1, len(experiments)) for i, exp in enumerate(experiments)}
    for i, exp in enumerate(experiments):
        out.append(_text(score_x[exp], cy0 + ch + 22, _label(exp), size=12, anchor="middle", fill=colors[exp], weight="bold"))
    score_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        score_groups[row["experiment"]].append(row)
    for exp, group in score_groups.items():
        ordered = sorted(group, key=lambda r: (r["L"], r["seed"]))
        group_width = cw / max(1, len(experiments))
        for i, row in enumerate(ordered):
            spread = min(group_width * 0.3, 120)
            run_x = score_x[exp] + (i - (len(ordered) - 1) / 2) * spread / max(1, len(ordered) - 1)
            reward_y = cy0 + ch - row["intrinsic_reward"] * ch
            match_y = cy0 + ch - row["cycle_match"] * ch
            out.append(_line(run_x - 4, reward_y, run_x + 4, match_y, color="#777", width=0.8))
            for metric, value, dx, shape in (("intrinsic reward", row["intrinsic_reward"], -4, "circle"),
                                             ("exact cycle match", row["cycle_match"], 4, "diamond")):
                title = (f'{exp}; seed={row["seed"]}; L={row["L"]}; {metric}={value:.4f}; '
                         f'expected period={2*row["L"]}; measured period={row["dominant_period"]}')
                out.append(f'<g><title>{escape(title)}</title>')
                out.append(_marker(run_x + dx, cy0 + ch - value * ch, shape, colors[exp], radius=5.2))
                out.append("</g>")
    out.append(_marker(cx0 + 4, 836, "circle", "#555", radius=4.5))
    out.append(_text(cx0 + 14, 840, "Intrinsic reward", size=11))
    out.append(_marker(cx0 + 125, 836, "diamond", "#555", radius=4.5))
    out.append(_text(cx0 + 135, 840, "Exact cycle match", size=11))
    out.append(_text(cx0 + 272, 840, "Thin lines pair values from the same run", size=11, fill="#555"))
    out.append(_text(cx0 + cw / 2, 1024, "Condition (horizontal separation within groups separates L/seed runs for display only)", size=11, anchor="middle", fill="#555"))
    out.append(_text(42, HEIGHT - 24, "Diagnostics support a feasibility check; they do not establish a full benchmark replication.", size=11, fill="#666"))
    out.append("</svg>")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python plot_results.py results/summary.json results/periods.svg", file=sys.stderr)
        return 2
    source, target = map(Path, argv[1:])
    try:
        with source.open("r", encoding="utf-8") as stream:
            rows, experiments = _validate_rows(json.load(stream))
        svg = render_svg(rows, experiments)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(svg, encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {target} ({len(rows)} rows; {len(experiments)} conditions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
