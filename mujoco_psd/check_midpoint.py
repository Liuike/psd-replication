"""Check the HalfCheetah run at its configured midpoint."""

import json
import math
import sys
from pathlib import Path

import compare_fig9


DEFAULT_MIDPOINT_EPISODES = 160_000


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def inspect(progress):
    if not isinstance(progress, dict):
        raise ValueError("Progress JSON must contain an object.")

    episodes = int(finite_number(progress.get("episodes"), "episodes"))
    midpoint = int(finite_number(
        progress.get("midpoint_episodes", DEFAULT_MIDPOINT_EPISODES),
        "midpoint_episodes",
    ))
    if episodes != midpoint:
        raise ValueError(f"Expected the {midpoint:,}-episode checkpoint, got {episodes:,}.")

    config = progress.get("config")
    if not isinstance(config, dict):
        raise ValueError("Progress JSON is missing its config object.")
    interval = int(finite_number(config.get("adaptive_interval"), "adaptive interval"))
    if interval <= 0 or episodes % interval:
        raise ValueError("The midpoint must fall on an adaptive evaluation boundary.")

    elapsed = finite_number(progress.get("elapsed_seconds"), "elapsed_seconds")
    raw_history = progress.get("bound_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise ValueError("Progress JSON needs a nonempty bound_history array.")
    history = []
    for index, row in enumerate(raw_history):
        if not isinstance(row, dict):
            raise ValueError(f"bound_history[{index}] must be an object.")
        at = int(finite_number(row.get("episodes"), f"bound_history[{index}].episodes"))
        low = int(finite_number(row.get("low"), f"bound_history[{index}].low"))
        high = int(finite_number(row.get("high"), f"bound_history[{index}].high"))
        if at > episodes or low > high or (history and at <= history[-1][0]):
            raise ValueError(f"Invalid ordered bounds history at index {index}.")
        history.append((at, low, high))
    if history[0][0] != 0 or history[-1][0] != episodes:
        raise ValueError("Bounds history must cover episode 0 through the midpoint.")
    summary = dict(progress)
    summary["device"] = progress.get("device", config.get("device_type", "unknown"))

    expected_checks = episodes // interval
    observed_checks = len(history) - 1
    adaptive_scores = []
    for index, row in enumerate(raw_history[1:], start=1):
        if row.get("episodes") != index * interval:
            raise ValueError(f"Adaptive evaluation {index} is missing or off schedule.")
        for name in ("low_score", "high_score"):
            adaptive_scores.append(
                finite_number(row.get(name), f"bound_history[{index}].{name}")
            )

    metrics = progress.get("last_batch")
    if not isinstance(metrics, dict):
        raise ValueError("Progress JSON is missing last_batch metrics.")
    finite_metrics = {
        name: finite_number(metrics.get(name), f"last_batch.{name}")
        for name in ("encoder_loss", "critic_loss", "actor_loss", "alpha_loss", "alpha")
    }

    low, upper = compare_fig9.measured_value(history, episodes)
    reference_low = compare_fig9.step_value(compare_fig9.PAPER_LOW, episodes)
    reference_upper = compare_fig9.step_value(compare_fig9.PAPER_UPPER, episodes)
    return {
        "episodes": episodes,
        "midpoint": midpoint,
        "interval": interval,
        "expected_checks": expected_checks,
        "observed_checks": observed_checks,
        "last_bounds": (low, upper),
        "reference_bounds": (reference_low, reference_upper),
        "bound_difference": (low - reference_low, upper - reference_upper),
        "finite_metrics": finite_metrics,
        "scores_finite": all(math.isfinite(score) for score in adaptive_scores),
        "summary": summary,
        "elapsed": elapsed,
        "history": history,
    }


def render(progress, result, output):
    elapsed = result["elapsed"]
    episodes = result["episodes"]
    total_estimate = elapsed * compare_fig9.PAPER_EPISODES / episodes
    remaining_estimate = elapsed * (compare_fig9.PAPER_EPISODES - episodes) / episodes

    comparison_dir = output / "figure9_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "figure9_bounds.svg").write_text(
        compare_fig9.make_svg(episodes, result["history"]),
        encoding="utf-8",
    )
    (comparison_dir / "figure9_comparison.md").write_text(
        compare_fig9.make_report(
            result["summary"], episodes, elapsed, result["history"]
        ),
        encoding="utf-8",
    )

    metrics = result["finite_metrics"]
    delta_low, delta_upper = result["bound_difference"]
    bounds_within_one = abs(delta_low) <= 1 and abs(delta_upper) <= 1
    alpha = metrics["alpha"]
    alpha_review = alpha > 100 or alpha < 1e-6
    checks_complete = result["observed_checks"] == result["expected_checks"]

    lines = [
        "# HalfCheetah midpoint check",
        "",
        f"Progress reached **{episodes:,} of {compare_fig9.PAPER_EPISODES:,} episodes** "
        f"({episodes / compare_fig9.PAPER_EPISODES:.0%}) in **{elapsed / 3600:.2f} hours**.",
        f"At the observed throughput, the full run is estimated at **{total_estimate / 3600:.2f} hours** "
        f"total with **{remaining_estimate / 3600:.2f} hours** remaining. This linear estimate "
        "does not account for later throughput changes.",
        "",
        "## Training health",
        "",
        f"- Latest encoder, critic, actor, and temperature losses are finite: "
        f"{metrics['encoder_loss']:.4g}, {metrics['critic_loss']:.4g}, "
        f"{metrics['actor_loss']:.4g}, and {metrics['alpha_loss']:.4g}.",
        f"- Entropy temperature alpha is {alpha:.4g}.",
        f"- Adaptive evaluations recorded **{result['observed_checks']} of "
        f"{result['expected_checks']} expected** at {result['interval']:,}-episode intervals. "
        f"All recorded boundary scores are finite: **{result['scores_finite']}**.",
        f"- Approximate Figure 9 bounds at this episode are "
        f"{result['reference_bounds'][0]} / {result['reference_bounds'][1]}. The run has "
        f"{result['last_bounds'][0]} / {result['last_bounds'][1]}, a difference of "
        f"{delta_low:+d} / {delta_upper:+d}. The Figure 9 breakpoints are visual estimates.",
        "",
        "## Review flags",
        "",
    ]
    flags = []
    if not checks_complete:
        flags.append("The expected adaptive evaluation count does not match the recorded history.")
    if alpha_review:
        flags.append(
            f"Alpha {alpha:.4g} crossed the monitoring threshold of 100 or 1e-6. "
            "This is a review heuristic, not a paper acceptance criterion."
        )
    if not bounds_within_one:
        flags.append(
            "At least one period bound differs from the approximate Figure 9 midpoint by more than one."
        )
    if not flags:
        flags.append(
            "No midpoint monitoring flags fired. Finite training values and matching approximate "
            "bounds do not establish that training will remain healthy in the second half."
        )
    lines.extend(f"- {flag}" for flag in flags)
    lines.extend([
        "",
        "Bounds overlay and detailed comparison: "
        "[Figure 9 comparison](figure9_comparison/figure9_comparison.md).",
        "",
    ])
    (output / "midpoint_check.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) != 3:
        print("Usage: python check_midpoint.py <progress.json> <output_dir>", file=sys.stderr)
        return 2
    try:
        with Path(argv[1]).open(encoding="utf-8") as stream:
            progress = json.load(stream)
        result = inspect(progress)
        output = Path(argv[2])
        output.mkdir(parents=True, exist_ok=True)
        render(progress, result, output)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print(f"Midpoint check failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote midpoint report for {result['episodes']:,} episodes to {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
