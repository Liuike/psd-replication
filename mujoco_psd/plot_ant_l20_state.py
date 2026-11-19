"""Plot Ant joint-angle state values from a saved L=20 rollout archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def prominent_hip_peaks(values: np.ndarray, start_step: int) -> list[int]:
    """Return local maxima above the trace midpoint after the initial transient."""
    midpoint = (float(np.min(values)) + float(np.max(values))) / 2.0
    return [
        step
        for step in range(max(1, start_step), len(values) - 1)
        if values[step - 1] < values[step] >= values[step + 1]
        and values[step] > midpoint
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path, help="Path to ant_l20_rollouts.npz")
    parser.add_argument("output", type=Path, help="Destination PNG path")
    args = parser.parse_args()

    with np.load(args.rollouts, allow_pickle=False) as data:
        required = {"L", "target_gait_period", "seeds", "lengths", "joint_names", "joint_angles"}
        missing = required.difference(data.files)
        if missing:
            parser.error(f"rollout archive is missing fields: {', '.join(sorted(missing))}")

        skill_length = int(data["L"])
        period = int(data["target_gait_period"])
        seeds = data["seeds"]
        lengths = data["lengths"]
        names = data["joint_names"].astype(str)
        angles = data["joint_angles"]

        if angles.ndim != 3 or angles.shape[0] < 1 or angles.shape[2] != len(names):
            parser.error("joint_angles must have shape (episodes, steps, len(joint_names))")
        if len(lengths) != angles.shape[0] or len(seeds) != angles.shape[0]:
            parser.error("seeds, lengths, and joint_angles must contain the same episodes")
        if not 0 < int(lengths[0]) < angles.shape[1]:
            parser.error("the first rollout length is incompatible with the saved joint angles")
        if "hip_2" not in names:
            parser.error("joint_names must include hip_2 for the observed-peak annotations")

        seed = int(seeds[0])
        episode_length = int(lengths[0])
        joint_angles = np.asarray(angles[0, : episode_length + 1], dtype=float)
        timesteps = np.arange(episode_length + 1)
        peak_joint = int(np.flatnonzero(names == "hip_2")[0])
        peaks = prominent_hip_peaks(joint_angles[:, peak_joint], period)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()
    guide_steps = np.arange(0, episode_length + 1, period)

    for joint_index, ax in enumerate(axes):
        ax.plot(timesteps, joint_angles[:, joint_index], color="#2369a8", linewidth=1.6)
        for guide in guide_steps:
            ax.axvline(guide, color="#8b929a", linestyle="--", linewidth=0.85, alpha=0.65, zorder=0)
        ax.set_title(names[joint_index], loc="left", fontsize=10, pad=3)
        ax.grid(axis="y", color="#d9dde2", linewidth=0.6)
        ax.set_xlim(0, episode_length)

    hip_ax = axes[peak_joint]
    hip_values = joint_angles[:, peak_joint]
    hip_ax.scatter(
        peaks,
        hip_values[peaks],
        s=28,
        facecolors="white",
        edgecolors="#c33b35",
        linewidths=1.3,
        zorder=3,
        label="Observed hip_2 maxima",
    )
    for step in peaks:
        hip_ax.annotate(
            str(step),
            (step, hip_values[step]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            color="#a32c27",
            fontsize=8,
        )
    for ax in axes[-2:]:
        ax.set_xticks(guide_steps)
        ax.set_xlabel("Environment step")
    fig.supylabel("Joint angle (rad)")
    fig.suptitle(
        f"Ant joint-angle state values | L={skill_length} | requested period={period} steps",
        fontsize=13,
        y=0.985,
    )
    fig.text(
        0.5,
        0.95,
        f"First rollout seed={seed}; dashed: target guides; red circles: hip_2 maxima",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.92), h_pad=1.0, w_pad=1.2)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.output}")
    print(f"L={skill_length}; requested period={period}; first rollout seed={seed}")
    print(f"hip_2 maxima after the first target period: {peaks}")


if __name__ == "__main__":
    main()
