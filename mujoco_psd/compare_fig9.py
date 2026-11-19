"""Compare a saved HalfCheetah adaptive-sampling trace with Figure 9."""

import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape


PAPER_EPISODES = 320_000
# Approximate breakpoints read from the paper's plot. Exact episode numbers are
# not reported. The paper's plotted upper bound is treated as exclusive.
PAPER_LOW = ((0, 10), (52_000, 9), (65_000, 8), (70_000, 7),
             (230_000, 6), (252_000, 5))
PAPER_UPPER = ((0, 11), (52_000, 12), (65_000, 13), (97_000, 14),
               (206_000, 15), (236_000, 16))
MILESTONES = (0, 50_000, 65_000, 70_000, 100_000, 205_000,
              230_000, 240_000, 255_000, PAPER_EPISODES)


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return value


def load_summary(path):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("summary JSON must be an object")
    episodes = int(number(data.get("episodes"), "episodes", 1))
    elapsed = number(data.get("elapsed_seconds"), "elapsed_seconds")
    raw_history = data.get("bound_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise ValueError("bound_history must be a nonempty array")

    history = []
    for index, row in enumerate(raw_history):
        if not isinstance(row, dict):
            raise ValueError(f"bound_history[{index}] must be an object")
        at = int(number(row.get("episodes"), f"bound_history[{index}].episodes"))
        low = int(number(row.get("low"), f"bound_history[{index}].low", 1))
        high = int(number(row.get("high"), f"bound_history[{index}].high", 1))
        if low > high or at > episodes or (history and at <= history[-1][0]):
            raise ValueError(f"Invalid bound_history entry at index {index}")
        history.append((at, low, high))
    if history[0][0] != 0:
        history.insert(0, (0, history[0][1], history[0][2]))
    if history[-1][0] < episodes:
        history.append((episodes, history[-1][1], history[-1][2]))
    return data, episodes, elapsed, history


def step_value(points, at):
    value = points[0][1]
    for episode, next_value in points[1:]:
        if episode > at:
            break
        value = next_value
    return value


def measured_value(history, at):
    selected = history[0]
    for row in history:
        if row[0] > at:
            break
        selected = row
    return selected[1], selected[2] + 1


def svg_text(x, y, text, size=12, anchor="start", color="#344054", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{color}">{escape(str(text))}</text>')


def path_for(points, x_map, y_map):
    commands = []
    for index, (episode, value) in enumerate(points):
        x, y = x_map(episode), y_map(value)
        commands.append(f"M {x:.1f} {y:.1f}" if index == 0 else f"H {x:.1f} V {y:.1f}")
    return " ".join(commands)


def make_svg(episodes, history):
    left, top, width, height = 92, 66, 890, 438
    x = lambda at: left + width * at / PAPER_EPISODES
    y = lambda value: top + height * (20 - value) / 20
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="635" viewBox="0 0 1080 635">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(52, 32, "HalfCheetah adaptive period bounds", 20, weight="bold"),
        svg_text(52, 51, "Figure 9 breakpoints are visual estimates; exact transition episodes are unreported.", 12, color="#667085"),
    ]
    x_ticks = list(range(0, PAPER_EPISODES + 1, 50_000))
    if x_ticks[-1] != PAPER_EPISODES:
        x_ticks.append(PAPER_EPISODES)
    for at in x_ticks:
        out.append(f'<line x1="{x(at):.1f}" y1="{top}" x2="{x(at):.1f}" y2="{top+height}" stroke="#e4e7ec"/>')
        out.append(svg_text(x(at), top + height + 20, f"{at//1000}k", 11, "middle", "#667085"))
    for value in range(0, 21, 2):
        out.append(f'<line x1="{left}" y1="{y(value):.1f}" x2="{left+width}" y2="{y(value):.1f}" stroke="#e4e7ec"/>')
        out.append(svg_text(left - 10, y(value) + 4, value, 11, "end", "#667085"))
    out += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#667085"/>',
        f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#667085"/>',
        svg_text(left + width/2, top + height + 44, "Training episodes", 13, "middle"),
        svg_text(24, top + height/2, "Period bound", 13, "middle"),
    ]

    traces = (
        (list(PAPER_LOW), "#238443", "7 5", "Figure 9 approximate Lmin"),
        (list(PAPER_UPPER), "#b54735", "7 5", "Figure 9 approximate upper bound"),
        ([(at, low) for at, low, _ in history], "#1261a0", "", "Run Lmin"),
        ([(at, high + 1) for at, _, high in history], "#a52a2a", "", "Run upper bound (high + 1)"),
    )
    for points, color, dash, _ in traces:
        stroke_dash = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<path d="{path_for(points,x,y)}" fill="none" stroke="{color}" '
                   f'stroke-width="3"{stroke_dash}/>')
    end_x = x(episodes)
    out.append(f'<line x1="{end_x:.1f}" y1="{top}" x2="{end_x:.1f}" y2="{top+height}" stroke="#344054" stroke-dasharray="3 4"/>')
    out.append(svg_text(min(end_x + 5, left + width - 5), top + 15,
                        f"run ends at {episodes:,}", 10, color="#344054"))

    legend_y = 553
    for index, (points, color, dash, label) in enumerate(traces):
        lx = 80 + (index % 2) * 490
        ly = legend_y + (index // 2) * 24
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+36}" y2="{ly}" stroke="{color}" stroke-width="3"{d}/>')
        out.append(svg_text(lx+45, ly+4, label, 12))
    low, upper = measured_value(history, episodes)
    out.append(svg_text(52, 612, f"Run end: Lmin={low}, plotted upper bound={upper}. Paper upper bound is interpreted as exclusive.",
                        11, color="#667085"))
    out += ['</svg>']
    return "\n".join(out)


def make_report(data, episodes, elapsed, history):
    low, upper = measured_value(history, episodes)
    reference_at = min(episodes, PAPER_EPISODES)
    reference_low = step_value(PAPER_LOW, reference_at)
    reference_upper = step_value(PAPER_UPPER, reference_at)
    estimated_seconds = elapsed / episodes * PAPER_EPISODES
    remaining_seconds = elapsed / episodes * max(0, PAPER_EPISODES - episodes)
    device = data.get("device", data.get("config", {}).get("device_type", "unknown"))
    rows = [
        "# HalfCheetah comparison with Figure 9",
        "",
        f"The run reached **{episodes:,} episodes** on **{device}** in **{elapsed/60:.2f} minutes** "
        f"({elapsed/3600:.2f} hours). The measured bounds are Lmin={low} and plotted upper "
        f"bound={upper}.",
        "",
        f"Figure 9 runs to about **{PAPER_EPISODES:,} episodes**. Linear extrapolation from "
        f"this run gives **{estimated_seconds/3600:.2f} hours** "
        f"({estimated_seconds/86400:.2f} days) from scratch, or **{remaining_seconds/3600:.2f} "
        "additional hours** from this checkpoint. This assumes constant throughput. A short "
        "pilot may not capture the added overhead of adaptive evaluations and checkpoints.",
        "",
        "The reference transitions below are approximate readings from the paper's plot. "
        "The paper does not report exact transition episodes. The plot uses its upper bound "
        "labels directly and interprets the run's stored inclusive high value as high + 1.",
        "",
        "| Episode | Run Lmin | Figure 9 Lmin | Run upper | Figure 9 upper |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for at in (e for e in (0, 50_000, 65_000, 70_000, 100_000, 205_000, 230_000, 240_000, 255_000, PAPER_EPISODES) if e <= episodes):
        run_low, run_upper = measured_value(history, at)
        rows.append(f"| {at:,} | {run_low} | {step_value(PAPER_LOW,at)} | {run_upper} | {step_value(PAPER_UPPER,at)} |")
    rows += [
        "",
        f"At episode {reference_at:,}, the paper trace is approximately Lmin={reference_low} "
        f"and upper={reference_upper}. Run-minus-reference differences are "
        f"{low-reference_low:+d} and {upper-reference_upper:+d}.",
        "",
    ]
    if episodes < PAPER_EPISODES:
        rows.append(f"This run covers {episodes/PAPER_EPISODES:.3%} of the plotted episode horizon, so the final Figure 9 bounds have not yet been tested.")
    else:
        rows.append("This run covers the full plotted episode horizon.")
    return "\n".join(rows) + "\n"


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) != 3:
        print("Usage: python compare_fig9.py <summary.json> <output_dir>", file=sys.stderr)
        return 2
    try:
        data, episodes, elapsed, history = load_summary(Path(argv[1]))
        output = Path(argv[2])
        output.mkdir(parents=True, exist_ok=True)
        (output / "figure9_bounds.svg").write_text(make_svg(episodes, history), encoding="utf-8")
        (output / "figure9_comparison.md").write_text(make_report(data, episodes, elapsed, history), encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote Figure 9 comparison for {episodes:,} episodes to {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
