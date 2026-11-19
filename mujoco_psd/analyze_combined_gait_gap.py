"""Quantify final combined-reward gait gaps from saved rollouts; no new training."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "runs"
OUT = ROOT / "combined_gait_gap"


def recurrence_errors(angles, rates, max_lag):
    """Same standardized joint-state return error as evaluate_periods.py."""
    values = np.column_stack((angles, rates)).astype(np.float64)
    floor = np.r_[np.full(angles.shape[1], 0.05), np.full(rates.shape[1], 0.5)]
    states = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), floor)
    baseline = np.sqrt(2 * np.mean(states ** 2))
    if baseline <= 0:
        raise ValueError("No joint motion in this analysis window")
    errors = []
    for lag in range(1, max_lag + 1):
        squared = np.mean((states[lag:] - states[:-lag]) ** 2, axis=1)
        errors.append(float(max(np.sqrt(half.mean()) for half in np.array_split(squared, 2)) / baseline))
    return np.asarray(errors)


def summarize(values):
    return {"mean": float(np.mean(values)), "min": float(np.min(values)),
            "max": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT,
                        help="Run directories or curated reports directory.")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="Destination for generated JSON and PNG; defaults to ignored runs/.")
    args = parser.parse_args()
    out = args.out
    result = {
        "description": "Combined-reward fixed-period gait gaps; five held-out reset seeds per L",
        "original_window": "step 40 through episode cap, excluding final post-action state",
        "late_window": "final half of each capped episode (Walker2D 200:400; Ant 100:200)",
        "threshold": 0.35,
        "target_tolerance": 0.10,
        "caveats": [
            "The 0.35 and 10% thresholds are our diagnostics, not criteria stated by the paper.",
            "Late-window normalization is refitted in that window, using the original channel floors.",
            "Late-window lag search extends through half its length, permitting fewer cycles than the original n/3 search.",
            "Late-window minima are candidate rhythms, not newly confirmed gait-period passes.",
            "A normalized return error is not a percentage error in period or a training-loss percentage.",
        ],
        "robots": {},
    }
    for robot in ("walker2d", "ant"):
        directory = args.runs_root / f"{robot}_combined_oscar_seed0"
        if (directory / "evaluation").is_dir():
            directory = directory / "evaluation"
        report = json.loads((directory / f"{robot}_figure5_evaluation.json").read_text())
        data = np.load(directory / f"{robot}_figure5_rollouts.npz")
        settings = []
        for index, setting in enumerate(report["settings"]):
            target = 2 * setting["L"]
            band = np.arange(int(np.ceil(0.9 * target)), int(np.floor(1.1 * target)) + 1)
            rows = []
            for seed_index, episode in enumerate(setting["rollouts"]):
                length = int(data["constant_lengths"][index, seed_index])
                angles = data["constant_joint_angles"][index, seed_index, :length]
                rates = data["constant_joint_velocities"][index, seed_index, :length]
                original = episode.get("gait_recurrence", episode.get("recurrence"))
                checked = recurrence_errors(angles[40:], rates[40:], int(band[-1]))
                assert np.isclose(checked[band - 1].min(), original["target_error"], rtol=0, atol=1e-10)
                start = length // 2
                limit = (length - start) // 2
                late = recurrence_errors(angles[start:], rates[start:], limit + 1)
                local_minima = [lag for lag in range(2, limit + 1)
                                if late[lag - 1] <= late[lag - 2] and late[lag - 1] < late[lag]]
                free_lag = min(local_minima, key=lambda lag: late[lag - 1]) if local_minima else None
                band_lag = int(band[np.argmin(late[band - 1])])
                rows.append({
                    "seed": episode["seed"],
                    "original_target_error": original["target_error"],
                    "original_shortest_repeat": original["shortest_repeat"],
                    "late_start_step": start,
                    "late_target_error": float(late[band_lag - 1]),
                    "late_best_target_band_lag": band_lag,
                    "late_best_unrestricted_local_minimum_lag": free_lag,
                    "late_best_unrestricted_local_minimum_error": float(late[free_lag - 1]) if free_lag else None,
                })
            settings.append({
                "L": setting["L"], "target_gait_period": target,
                "original_target_error": summarize([row["original_target_error"] for row in rows]),
                "late_target_error": summarize([row["late_target_error"] for row in rows]),
                "late_free_minimum_lags": [row["late_best_unrestricted_local_minimum_lag"] for row in rows],
                "rollouts": rows,
            })
        result["robots"][robot] = settings

    out.mkdir(parents=True, exist_ok=True)
    (out / "gait_gap.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for axis, (robot, settings) in zip(axes, result["robots"].items()):
        targets = np.asarray([row["target_gait_period"] for row in settings])
        for key, label, color, shift in (
            ("original_target_error", "Original: discard 40 steps", "#4477AA", -0.5),
            ("late_target_error", "Exploratory: final half", "#EE7733", 0.5),
        ):
            mean = np.asarray([row[key]["mean"] for row in settings])
            low = np.asarray([row[key]["min"] for row in settings])
            high = np.asarray([row[key]["max"] for row in settings])
            axis.errorbar(targets + shift, mean, yerr=np.vstack((mean - low, high - mean)),
                          marker="o", capsize=3, color=color, label=label)
        axis.axhline(0.35, linestyle="--", color="#222222", label="Our return-error cutoff: 0.35")
        axis.set_title("Walker2D" if robot == "walker2d" else "Ant")
        axis.set_xticks(targets)
        axis.set_xlabel("Requested gait period (environment steps)")
        axis.set_ylim(0, 1.2)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Best return error within ±10% of target\nMean and min–max over five reset seeds")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Combined-reward gait consistency: lower is better\nLate windows: refitted normalization, fewer cycles; exploratory", fontsize=12)
    fig.savefig(out / "gait_gap.png", dpi=160)
    plt.close(fig)
    print(json.dumps({robot: [{"L": row["L"], "target": row["target_gait_period"],
                              "original_error": row["original_target_error"]["mean"],
                              "late_error": row["late_target_error"]["mean"],
                              "late_free_lags": row["late_free_minimum_lags"]} for row in settings]
                      for robot, settings in result["robots"].items()}, indent=2))
    print(f"Saved {out / 'gait_gap.json'} and {out / 'gait_gap.png'}")


if __name__ == "__main__":
    main()
