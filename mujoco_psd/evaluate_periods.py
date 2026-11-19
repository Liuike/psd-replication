"""Test whether the saved HalfCheetah actor changes its fundamental gait period."""

import argparse
import json
from pathlib import Path
from statistics import mean

import gymnasium as gym
import numpy as np
import torch

from evaluation import spectrum
from sac import SAC
from train import EMBED_DIM, ENV_ID, HORIZON, rollouts


PERIOD_SETTINGS = (8, 12, 16, 20)
# The training and diagnostic resets are sampled below 2**31.
RESET_SEEDS = tuple(range(2**31, 2**31 + 5))
SETTLE_STEPS = 40
TARGET_TOLERANCE = 0.10
PASS_ROLLOUTS = 4
RECURRENCE_THRESHOLD = 0.35


def gait_recurrence(angles, angular_velocity, target):
    # Standardize each joint channel so both positions and rates contribute
    # without amplifying nearly motionless degrees of freedom.
    values = np.column_stack((angles, angular_velocity)).astype(np.float64)
    std = values.std(axis=0)
    scale_floor = np.r_[np.full(angles.shape[1], 0.05), np.full(angular_velocity.shape[1], 0.5)]
    states = (values - values.mean(axis=0)) / np.maximum(std, scale_floor)
    unrelated_distance = np.sqrt(2 * np.mean(states**2))
    moving_joints = int(np.count_nonzero(std[:angles.shape[1]] >= 0.05))
    if moving_joints < 2:
        return {"shortest_repeat": None, "best_target_lag": None, "target_error": None,
                "moving_joints": moving_joints, "match": False, "lag_errors": []}

    # Require a close return in both halves of the available pairs.
    max_lag = len(states) // 3
    errors = []
    for lag in range(1, max_lag + 2):
        point_errors = np.mean((states[lag:] - states[:-lag])**2, axis=1)
        halves = np.array_split(point_errors, 2)
        errors.append(float(max(np.sqrt(half.mean()) for half in halves) / unrelated_distance))

    shortest = next(
        (lag for lag in range(2, max_lag + 1)
         if errors[lag - 1] <= RECURRENCE_THRESHOLD
         and errors[lag - 1] <= errors[lag - 2]
         and errors[lag - 1] < errors[lag]),
        None,
    )
    target_lags = [lag for lag in range(2, max_lag + 1)
                   if abs(lag - target) <= TARGET_TOLERANCE * target]
    best_target = min(target_lags, key=lambda lag: errors[lag - 1])
    return {
        "shortest_repeat": shortest,
        "best_target_lag": best_target,
        "target_error": errors[best_target - 1],
        "moving_joints": moving_joints,
        "match": shortest in target_lags,
        "lag_errors": errors,  # Index 0 is lag 1.
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(6, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["config"]["env"] != ENV_ID:
        raise ValueError("Expected a state-based HalfCheetah-v5 checkpoint.")
    envs = [gym.make(ENV_ID, max_episode_steps=HORIZON) for _ in RESET_SEEDS]
    try:
        env = envs[0]
        layout = env.unwrapped.observation_structure
        if layout != {"skipped_qpos": 1, "qpos": 8, "qvel": 9}:
            raise ValueError(f"Unexpected HalfCheetah observation layout: {layout}")
        agent = SAC(env.observation_space.shape[0] + EMBED_DIM, env.action_space, device)
        agent.actor.load_state_dict(checkpoint["agent"]["actor"])
        agent.actor.eval()
        settings = []
        states_by_period = []
        for L in PERIOD_SETTINGS:
            states, _, velocities = rollouts(
                envs, agent, np.full(len(RESET_SEEDS), L), None, device,
                deterministic=True, reset_seeds=RESET_SEEDS,
            )
            states_by_period.append(states)
            rows = []
            for seed, state, velocity in zip(RESET_SEEDS, states, velocities):
                observed = state[SETTLE_STEPS:HORIZON]
                # The six limb angles follow torso height and pitch. Their
                # velocities follow the three root velocities.
                angles, rates = observed[:, 2:8], observed[:, 11:17]
                row = gait_recurrence(angles, rates, target=2 * L)
                row["seed"] = seed
                row["mean_forward_velocity"] = float(velocity[SETTLE_STEPS:].mean())
                row["dominant_spectral_period"] = (
                    spectrum(angles)["dominant_period_steps"] if row["moving_joints"] >= 2 else None
                )
                rows.append(row)
            hits = sum(row["match"] for row in rows)
            settings.append({"L": L, "target": 2 * L, "matches": hits, "passes": hits >= PASS_ROLLOUTS,
                             "mean_forward_velocity": mean(row["mean_forward_velocity"] for row in rows),
                             "rollouts": rows})
    finally:
        for env in envs:
            env.close()

    passing = [row["L"] for row in settings if row["passes"]]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epochs": checkpoint["epoch"],
        "device": str(device),
        "reset_seeds": RESET_SEEDS,
        "rollout_steps": HORIZON,
        "discard_initial_steps": SETTLE_STEPS,
        "target_tolerance": TARGET_TOLERANCE,
        "recurrence_threshold": RECURRENCE_THRESHOLD,
        "required_matches_per_setting": PASS_ROLLOUTS,
        "passing_settings": passing,
        "demonstrates_two_periods": len(passing) >= 2,
        "settings": settings,
    }
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "gait_periods.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    np.savez_compressed(args.out / "rollouts.npz", L=PERIOD_SETTINGS, seed=RESET_SEEDS,
                        states=np.asarray(states_by_period))
    for row in settings:
        repeats = [run["shortest_repeat"] for run in row["rollouts"]]
        print(f"L={row['L']:2} target={row['target']:2} shortest repeats={repeats} "
              f"matches={row['matches']}/{len(RESET_SEEDS)}")
    print(f"Passing settings: {passing}")


if __name__ == "__main__":
    main()
