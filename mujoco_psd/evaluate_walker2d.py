"""Evaluate a shared-period Walker2d policy against the Figure 5 schedule.

Runs five matched, fixed-period episodes for each L in (20, 26, 33, 40), then
one continuous 1,600-step episode that switches L every 400 steps. The switch
episode uses an extended TimeLimit and is never reset at a segment boundary.
Early termination is recorded as missing schedule coverage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from evaluate_ant import HIDDEN_DIM, _actor_state, _infer_embed_dim, period_embedding
from evaluate_periods import TARGET_TOLERANCE, gait_recurrence
from evaluation import geometry as latent_geometry
from sac import SAC


ENV_ID = "Walker2d-v5"
PERIODS = (20, 26, 33, 40)
SEGMENT_STEPS = 400
SWITCH_STEPS = SEGMENT_STEPS * len(PERIODS)
ROLLOUT_STEPS = 400
SETTLE_STEPS = 40
RESET_SEEDS = tuple(range(2**31, 2**31 + 5))
# Fixed before inspecting this checkpoint's rollouts; matched across L.
SWITCH_SEED = 2**31 + 100
DEFAULT_EMBED_DIM = 8
RECURRENCE_ERROR_THRESHOLD = 0.35


def _condition(observation: np.ndarray, period: int, embed_dim: int, device):
    state = torch.as_tensor(observation, dtype=torch.float32, device=device)
    label = torch.tensor([period], dtype=torch.float32, device=device)
    return torch.cat((state, period_embedding(label, embed_dim)[0]))


def _encode_period_states(encoder, observations, periods, embed_dim: int, device):
    observations = np.asarray(observations, dtype=np.float32)
    periods = np.asarray(periods, dtype=np.float32)
    if len(observations) != len(periods):
        raise ValueError("Each encoded state must have exactly one period label.")
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(observations), 1024):
            stop = min(start + 1024, len(observations))
            states = torch.as_tensor(observations[start:stop], device=device)
            labels = torch.as_tensor(periods[start:stop], device=device)
            conditioned = torch.cat((states, period_embedding(labels, embed_dim)), dim=-1)
            encoded.append(encoder(conditioned).cpu().numpy())
    if not encoded:
        return np.empty((0, int(encoder[-1].out_features)), dtype=np.float32)
    return np.concatenate(encoded).astype(np.float32, copy=False)


def _joint_names(env) -> list[str]:
    """Name Walker2d's six hinge joints in qpos[3:]/qvel[3:] order."""
    try:
        import mujoco

        model = env.unwrapped.model
        # Walker2d has three unactuated root joints (x, z, and pitch) before
        # its six leg hinges, whose coordinates are qpos[3:] and qvel[3:].
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(3, model.njnt)
        ]
        if len(names) == 6 and all(names):
            return [str(name) for name in names]
    except (ImportError, AttributeError, TypeError):
        pass
    return [f"motor_joint_{index + 1}" for index in range(6)]


def _joint_state(env) -> tuple[np.ndarray, np.ndarray]:
    data = env.unwrapped.data
    angles = np.asarray(data.qpos[3:], dtype=np.float32).copy()
    rates = np.asarray(data.qvel[3:], dtype=np.float32).copy()
    if angles.shape != (6,) or rates.shape != (6,):
        raise ValueError(
            "Expected six Walker2d motor joints in qpos[3:] and qvel[3:]."
        )
    return angles, rates


def _circle_fit(points: np.ndarray) -> dict:
    """Fit a least-squares circle to 2D coordinates without rescaling them."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return {"status": "insufficient points", "diameter": None}
    if not np.isfinite(points).all():
        return {"status": "nonfinite points", "diameter": None}
    x, y = points.T
    design = np.column_stack((2 * x, 2 * y, np.ones(len(points))))
    rhs = x * x + y * y
    solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3:
        return {"status": "degenerate points", "diameter": None}
    center = solution[:2]
    radius_squared = float(solution[2] + center @ center)
    if radius_squared <= 0:
        return {"status": "invalid fitted radius", "diameter": None}
    radius = math.sqrt(radius_squared)
    residual = np.linalg.norm(points - center, axis=1) - radius
    return {
        "status": "fit",
        "center": center.tolist(),
        "radius": float(radius),
        "diameter": float(2 * radius),
        "radial_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "radial_residual_rms_over_radius": float(np.sqrt(np.mean(residual**2)) / radius),
    }


def _fit_latent_circle(latents: np.ndarray, target_diameter: int) -> dict:
    """Fit each rollout's best-fit 2D latent plane, then its circle diameter."""
    z = np.asarray(latents, dtype=np.float64)
    result = {
        "coordinate_space": "per-rollout best-fit plane in raw latent coordinates",
        "target_diameter": float(target_diameter),
        "diameter": None,
    }
    if z.ndim != 2 or len(z) < 3 or z.shape[1] < 2 or not np.isfinite(z).all():
        result["status"] = "insufficient or invalid latent data"
        return result
    mean = z.mean(axis=0)
    centered = z - mean
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[0] <= np.finfo(float).eps:
        result["status"] = "degenerate latent plane"
        return result
    coordinates = centered @ vectors[:2].T
    fit = _circle_fit(coordinates)
    result.update(fit)
    if result.get("diameter") is not None:
        result["absolute_diameter_error"] = abs(result["diameter"] - target_diameter)
        result["relative_diameter_error"] = abs(result["diameter"] - target_diameter) / target_diameter
    variance = singular**2
    result["plane_explained_variance_ratio"] = float(variance[:2].sum() / variance.sum())
    return result


def _empty_recurrence(angles: np.ndarray) -> dict:
    moving = int(np.count_nonzero(angles.std(axis=0) >= 0.05)) if len(angles) else 0
    return {
        "shortest_repeat": None,
        "best_target_lag": None,
        "target_error": None,
        "moving_joints": moving,
        "match": False,
        "lag_errors": [],
    }


def _recurrence_for_episode(episode: dict, period: int) -> dict:
    length = episode["length"]
    angles = episode["joint_angles"][SETTLE_STEPS:length]
    rates = episode["joint_velocities"][SETTLE_STEPS:length]
    target = 2 * period
    # gait_recurrence searches through floor(n/3); require its full +/-10%
    # target band to be represented, even for the largest L=40 target.
    max_target_lag = math.ceil(target * (1 + TARGET_TOLERANCE))
    enough = len(angles) >= 3 * max_target_lag
    if enough:
        result = gait_recurrence(angles, rates, target=target)
        status = "analyzed"
    else:
        result = _empty_recurrence(angles)
        status = (
            f"only {len(angles)} post-settle steps; need at least "
            f"{3 * max_target_lag} to cover target lag {target}"
        )
    full_episode = length == ROLLOUT_STEPS and not episode["terminated"]
    return {
        **result,
        "period_match": bool(result["match"]),
        "passed": bool(full_episode and result["match"]),
        "target_gait_period": target,
        "target_tolerance": TARGET_TOLERANCE,
        "recurrence_error_threshold": RECURRENCE_ERROR_THRESHOLD,
        "analysis_start_step": SETTLE_STEPS,
        "analysis_steps": int(len(angles)),
        "analysis_status": status,
        "full_episode": bool(full_episode),
        "episode_length": int(length),
        "terminated": bool(episode["terminated"]),
        "truncated": bool(episode["truncated"]),
    }


def _rollout(env, agent, encoder, period, seed, embed_dim, device):
    observation, _ = env.reset(seed=int(seed))
    observations = [np.asarray(observation, dtype=np.float32).copy()]
    angle, rate = _joint_state(env)
    joint_angles, joint_velocities = [angle], [rate]
    actions, rewards, x_velocities = [], [], []
    terminated = truncated = False
    for _ in range(ROLLOUT_STEPS):
        conditioned = _condition(observation, period, embed_dim, device)
        with torch.inference_mode():
            action = agent.act(conditioned, deterministic=True).cpu().numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Nonfinite Walker2d action for L={period}, seed={seed}.")
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        angle, rate = _joint_state(env)
        joint_angles.append(angle)
        joint_velocities.append(rate)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))
        x_velocities.append(float(info.get("x_velocity", np.nan)))
        if terminated or truncated:
            break
    episode = {
        "period": int(period),
        "seed": int(seed),
        "length": len(actions),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "observations": np.asarray(observations, dtype=np.float32),
        "joint_angles": np.asarray(joint_angles, dtype=np.float32),
        "joint_velocities": np.asarray(joint_velocities, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1, agent.action_dim),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "x_velocities": np.asarray(x_velocities, dtype=np.float32),
        "return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_x_velocity": float(np.nanmean(x_velocities)) if x_velocities else None,
    }
    if len(episode["observations"]) != episode["length"] + 1:
        raise RuntimeError("Walker2d rollout has inconsistent state and transition lengths.")
    episode["latents"] = _encode_period_states(
        encoder, episode["observations"], np.full(len(observations), period), embed_dim, device
    )
    episode["recurrence"] = _recurrence_for_episode(episode, period)
    available = len(episode["latents"]) - period
    if available > 0:
        z = episode["latents"]
        episode["latent_geometry"] = latent_geometry(
            z[:available], z[1:available + 1], z[period:], period
        )
    else:
        episode["latent_geometry"] = None
    repeat_lag = 2 * period
    if len(episode["latents"]) > repeat_lag:
        distances = np.linalg.norm(
            episode["latents"][repeat_lag:] - episode["latents"][:-repeat_lag], axis=1
        )
        episode["latent_2L_repeat"] = {
            "target_distance": 0.0,
            "mean_distance": float(distances.mean()),
            "std_distance": float(distances.std()),
        }
    else:
        episode["latent_2L_repeat"] = None
    episode["fitted_circle"] = _fit_latent_circle(episode["latents"], period)
    return episode


def _switch_rollout(env, agent, encoder, embed_dim, device):
    observation, _ = env.reset(seed=int(SWITCH_SEED))
    observations = [np.asarray(observation, dtype=np.float32).copy()]
    angle, rate = _joint_state(env)
    joint_angles, joint_velocities = [angle], [rate]
    actions, rewards, x_velocities, action_periods = [], [], [], []
    terminated = truncated = False
    for step in range(SWITCH_STEPS):
        period = PERIODS[step // SEGMENT_STEPS]
        conditioned = _condition(observation, period, embed_dim, device)
        with torch.inference_mode():
            action = agent.act(conditioned, deterministic=True).cpu().numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Nonfinite Walker2d action at switch step {step}.")
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        angle, rate = _joint_state(env)
        joint_angles.append(angle)
        joint_velocities.append(rate)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))
        x_velocities.append(float(info.get("x_velocity", np.nan)))
        action_periods.append(period)
        if terminated or truncated:
            break
    state_periods = np.asarray(action_periods + [action_periods[-1] if action_periods else PERIODS[0]],
                               dtype=np.int32)
    episode = {
        "seed": int(SWITCH_SEED),
        "length": len(actions),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "completed_switch_schedule": len(actions) == SWITCH_STEPS,
        "observations": np.asarray(observations, dtype=np.float32),
        "joint_angles": np.asarray(joint_angles, dtype=np.float32),
        "joint_velocities": np.asarray(joint_velocities, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1, agent.action_dim),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "x_velocities": np.asarray(x_velocities, dtype=np.float32),
        "periods": np.asarray(action_periods, dtype=np.int32),
        "state_periods": state_periods,
        "return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_x_velocity": float(np.nanmean(x_velocities)) if x_velocities else None,
    }
    if len(episode["observations"]) != episode["length"] + 1:
        raise RuntimeError("Walker2d switch has inconsistent state and transition lengths.")
    episode["latents"] = _encode_period_states(
        encoder, episode["observations"], episode["state_periods"], embed_dim, device
    )
    episode["segments"] = [
        {
            "L": period,
            "requested_steps": SEGMENT_STEPS,
            "observed_steps": int(max(0, min(episode["length"], (index + 1) * SEGMENT_STEPS)
                                       - index * SEGMENT_STEPS)),
            "complete": episode["length"] >= (index + 1) * SEGMENT_STEPS,
        }
        for index, period in enumerate(PERIODS)
    ]
    episode["ending"] = (
        "completed all 1,600 scheduled steps"
        if episode["completed_switch_schedule"] else
        "terminated before the 1,600-step schedule completed"
        if terminated else
        "truncated before the 1,600-step schedule completed"
        if truncated else
        "stopped before the 1,600-step schedule completed"
    )
    return episode


def _fit_shared_pca(switch: dict, constant: dict) -> dict:
    all_latents = [switch["latents"]]
    all_latents.extend(ep["latents"] for rows in constant.values() for ep in rows)
    samples = np.concatenate(all_latents, axis=0).astype(np.float64)
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((2, samples.shape[1]), dtype=np.float64)
    components[:min(2, len(vectors))] = vectors[:2]
    explained = np.zeros(2, dtype=np.float64)
    variance = singular**2
    total = variance.sum()
    if total > 0:
        explained[:min(2, len(variance))] = variance[:2] / total
    return {
        "mean": mean,
        "components": components,
        "explained": explained,
        "switch_coordinates": (switch["latents"] - mean) @ components.T,
        "constant_coordinates": {
            (period, index): (episode["latents"] - mean) @ components.T
            for period, rows in constant.items()
            for index, episode in enumerate(rows)
        },
    }


def _plot_fixed(out: Path, joint_names: list[str], constant: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(RESET_SEEDS)))
    fig, axes = plt.subplots(len(PERIODS), 6, figsize=(18, 11), sharex="row",
                             constrained_layout=True)
    for row, period in enumerate(PERIODS):
        for seed_index, episode in enumerate(constant[period]):
            steps = np.arange(len(episode["joint_angles"]))
            for joint, axis in enumerate(axes[row]):
                axis.plot(steps, episode["joint_angles"][:, joint], color=colors[seed_index],
                          alpha=0.84, lw=0.75)
        for joint, axis in enumerate(axes[row]):
            axis.set_title(joint_names[joint] if row == 0 else "")
            axis.grid(alpha=0.2)
            axis.set_xlim(0, ROLLOUT_STEPS)
            if joint == 0:
                axis.set_ylabel(f"L={period}\nAngle (rad)")
            if row == len(PERIODS) - 1:
                axis.set_xlabel("Environment step")
    handles = [plt.Line2D([0], [0], color=colors[i], lw=2,
                          label=f"seed {seed}") for i, seed in enumerate(RESET_SEEDS)]
    fig.legend(handles=handles, loc="upper center", ncol=len(RESET_SEEDS), frameon=False)
    fig.suptitle("Walker2d fixed-period joint angles, five matched reset seeds", y=1.02)
    path = out / "walker2d_figure5_fixed_joint_traces.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_l20(out: Path, joint_names: list[str], episodes: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(episodes)))
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for joint, axis in enumerate(axes.flat):
        for index, episode in enumerate(episodes):
            axis.plot(np.arange(len(episode["joint_angles"])), episode["joint_angles"][:, joint],
                      color=colors[index], alpha=0.84, lw=0.9,
                      label=f"seed {episode['seed']}" if joint == 0 else None)
        axis.set_title(joint_names[joint])
        axis.set_ylabel("Angle (rad)")
        axis.set_xlim(0, ROLLOUT_STEPS)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Environment step")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Walker2d constant L=20 joint angles", y=1.02)
    path = out / "walker2d_l20_joint_traces.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_switch(out: Path, joint_names: list[str], switch: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {20: "#4477AA", 26: "#EE6677", 33: "#228833", 40: "#AA3377"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    steps = np.arange(len(switch["joint_angles"]))
    for joint, axis in enumerate(axes.flat):
        axis.plot(steps, switch["joint_angles"][:, joint], color="#222222", lw=0.85)
        for index, period in enumerate(PERIODS):
            left, right = index * SEGMENT_STEPS, (index + 1) * SEGMENT_STEPS
            axis.axvspan(left, right, color=colors[period], alpha=0.09, linewidth=0)
            axis.axvline(left, color=colors[period], alpha=0.65, lw=0.8)
        if switch["length"] < SWITCH_STEPS:
            axis.axvspan(switch["length"], SWITCH_STEPS, color="#777777", alpha=0.16, hatch="///")
        axis.set_title(joint_names[joint])
        axis.set_ylabel("Angle (rad)")
        axis.set_xlim(0, SWITCH_STEPS)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Environment step")
    handles = [plt.Line2D([0], [0], color=colors[p], lw=4,
                          label=f"L={p} ({i*SEGMENT_STEPS}–{(i+1)*SEGMENT_STEPS})")
               for i, p in enumerate(PERIODS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=4, frameon=False)
    fig.suptitle(
        "Walker2d period-switch joint angles\n"
        f"Single episode: {switch['length']}/{SWITCH_STEPS} steps; "
        f"{switch['ending']}; no resets at L boundaries",
        y=0.985, fontsize=12,
    )
    fig.subplots_adjust(top=0.82, bottom=0.08, hspace=0.34, wspace=0.18)
    path = out / "walker2d_figure5_switch_joint_traces.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_shared_pca(out: Path, switch: dict, constant: dict, pca: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {20: "#4477AA", 26: "#EE6677", 33: "#228833", 40: "#AA3377"}
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for period in PERIODS:
        for index, episode in enumerate(constant[period]):
            xy = pca["constant_coordinates"][(period, index)]
            axis.scatter(xy[:, 0], xy[:, 1], s=4, color=colors[period], alpha=0.09,
                         linewidths=0, rasterized=True,
                         label=f"fixed L={period}" if index == 0 else None)
    xy = pca["switch_coordinates"]
    for index, period in enumerate(PERIODS):
        left, right = index * SEGMENT_STEPS, min((index + 1) * SEGMENT_STEPS, switch["length"] + 1)
        if right <= left:
            continue
        axis.plot(xy[left:right, 0], xy[left:right, 1], color=colors[period], lw=1.4,
                  label=f"switch L={period}")
    if len(xy):
        axis.scatter(*xy[0], marker="o", color="black", s=45, label="switch start", zorder=4)
        axis.scatter(*xy[-1], marker="X", color="black", s=50, label="switch end", zorder=4)
    axis.set_xlabel(f"PC 1 ({pca['explained'][0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC 2 ({pca['explained'][1] * 100:.1f}% variance)")
    axis.set_title("Walker2d shared PCA of period-conditioned latents")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)
    path = out / "walker2d_figure5_shared_pca.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _save_raw(out: Path, joint_names: list[str], constant: dict, switch: dict, pca: dict) -> str:
    obs_dim = constant[PERIODS[0]][0]["observations"].shape[1]
    latent_dim = constant[PERIODS[0]][0]["latents"].shape[1]
    action_dim = constant[PERIODS[0]][0]["actions"].shape[1]
    max_length = max((ep["length"] for rows in constant.values() for ep in rows), default=0)
    shape = (len(PERIODS), len(RESET_SEEDS), max_length + 1)
    observations = np.zeros((*shape, obs_dim), np.float32)
    angles = np.zeros((*shape, 6), np.float32)
    rates = np.zeros_like(angles)
    latents = np.zeros((*shape, latent_dim), np.float32)
    coordinates = np.zeros((*shape, 2), np.float32)
    actions = np.zeros((len(PERIODS), len(RESET_SEEDS), max_length, action_dim), np.float32)
    rewards = np.zeros((len(PERIODS), len(RESET_SEEDS), max_length), np.float32)
    x_velocity = np.full((len(PERIODS), len(RESET_SEEDS), max_length), np.nan, np.float32)
    state_mask = np.zeros(shape, bool)
    transition_mask = np.zeros((len(PERIODS), len(RESET_SEEDS), max_length), bool)
    lengths = np.zeros((len(PERIODS), len(RESET_SEEDS)), np.int32)
    terminated = np.zeros_like(lengths, dtype=bool)
    truncated = np.zeros_like(lengths, dtype=bool)
    for p_index, period in enumerate(PERIODS):
        for s_index, episode in enumerate(constant[period]):
            n = episode["length"]
            lengths[p_index, s_index] = n
            terminated[p_index, s_index] = episode["terminated"]
            truncated[p_index, s_index] = episode["truncated"]
            observations[p_index, s_index, :n + 1] = episode["observations"]
            angles[p_index, s_index, :n + 1] = episode["joint_angles"]
            rates[p_index, s_index, :n + 1] = episode["joint_velocities"]
            latents[p_index, s_index, :n + 1] = episode["latents"]
            coordinates[p_index, s_index, :n + 1] = pca["constant_coordinates"][(period, s_index)]
            actions[p_index, s_index, :n] = episode["actions"]
            rewards[p_index, s_index, :n] = episode["rewards"]
            x_velocity[p_index, s_index, :n] = episode["x_velocities"]
            state_mask[p_index, s_index, :n + 1] = True
            transition_mask[p_index, s_index, :n] = True
    n = switch["length"]
    np.savez_compressed(
        out / "walker2d_figure5_rollouts.npz",
        periods=np.asarray(PERIODS, np.int32),
        constant_seeds=np.asarray(RESET_SEEDS, np.uint32),
        switch_seed=np.asarray(SWITCH_SEED, np.uint32),
        switch_boundaries=np.arange(0, SWITCH_STEPS + 1, SEGMENT_STEPS, dtype=np.int32),
        constant_lengths=lengths, constant_terminated=terminated, constant_truncated=truncated,
        constant_state_mask=state_mask, constant_transition_mask=transition_mask,
        constant_observations=observations, constant_joint_angles=angles,
        constant_joint_velocities=rates, constant_latents=latents,
        constant_actions=actions, constant_rewards=rewards, constant_x_velocities=x_velocity,
        constant_shared_pca_coordinates=coordinates,
        switch_length=np.asarray(n, np.int32),
        switch_completed=np.asarray(switch["completed_switch_schedule"], bool),
        switch_terminated=np.asarray(switch["terminated"], bool),
        switch_truncated=np.asarray(switch["truncated"], bool),
        switch_observations=switch["observations"], switch_joint_angles=switch["joint_angles"],
        switch_joint_velocities=switch["joint_velocities"], switch_latents=switch["latents"],
        switch_actions=switch["actions"], switch_rewards=switch["rewards"],
        switch_x_velocities=switch["x_velocities"], switch_action_periods=switch["periods"],
        switch_state_periods=switch["state_periods"],
        switch_shared_pca_coordinates=pca["switch_coordinates"],
        pca_mean=pca["mean"].astype(np.float32),
        pca_components=pca["components"].astype(np.float32),
        pca_explained_variance_ratio=pca["explained"].astype(np.float32),
        joint_names=np.asarray(joint_names),
    )
    return "walker2d_figure5_rollouts.npz"


def _mean_metric(rows: list[dict], path: tuple[str, ...]) -> dict:
    values = []
    for row in rows:
        value = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return {
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "n": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Multi-period Walker2d checkpoint (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True, help="Directory for report and plots")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(6, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "agent", "encoder", "epoch"} <= checkpoint.keys():
        raise ValueError("Checkpoint must contain config, agent, encoder, and epoch entries.")
    config = checkpoint["config"]
    configured_periods = config.get("periods", config.get("L_values"))
    if config.get("env") != ENV_ID:
        raise ValueError(f"Expected a {ENV_ID} checkpoint, got env={config.get('env')!r}.")
    if configured_periods is None or tuple(map(int, configured_periods)) != PERIODS:
        raise ValueError(f"Expected periods {PERIODS}, got {configured_periods!r}.")
    latent_dim = int(config["latent_dim"])
    args.out.mkdir(exist_ok=True, parents=True)

    env = gym.make(ENV_ID, max_episode_steps=ROLLOUT_STEPS)
    switch_env = gym.make(ENV_ID, max_episode_steps=SWITCH_STEPS)
    try:
        observation_dim = int(env.observation_space.shape[0])
        action_state = _actor_state(checkpoint["agent"])
        embed_dim = _infer_embed_dim(action_state, observation_dim, config)
        if embed_dim <= 0:
            embed_dim = DEFAULT_EMBED_DIM
        agent = SAC(observation_dim + embed_dim, env.action_space, device)
        agent.actor.load_state_dict(action_state)
        agent.actor.eval()
        encoder = nn.Sequential(
            nn.Linear(observation_dim + embed_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, latent_dim),
        ).to(device)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder.eval()
        joint_names = _joint_names(env)

        constant = {
            period: [_rollout(env, agent, encoder, period, seed, embed_dim, device)
                     for seed in RESET_SEEDS]
            for period in PERIODS
        }
        switch = _switch_rollout(switch_env, agent, encoder, embed_dim, device)
    finally:
        env.close()
        switch_env.close()

    pca = _fit_shared_pca(switch, constant)
    charts = {
        "fixed_joint_traces": _plot_fixed(args.out, joint_names, constant),
        "l20_joint_traces": _plot_l20(args.out, joint_names, constant[20]),
        "switch_joint_traces": _plot_switch(args.out, joint_names, switch),
        "shared_pca": _plot_shared_pca(args.out, switch, constant, pca),
        "raw_rollouts": _save_raw(args.out, joint_names, constant, switch, pca),
    }

    settings = []
    for period in PERIODS:
        episodes = constant[period]
        recurrence_rows = [ep["recurrence"] for ep in episodes]
        circle_rows = [ep["fitted_circle"] for ep in episodes]
        settings.append({
            "L": period,
            "target_latent_circle_diameter": period,
            "target_gait_period": 2 * period,
            "rollouts": [
                {
                    "seed": ep["seed"],
                    "length": ep["length"],
                    "terminated": ep["terminated"],
                    "truncated": ep["truncated"],
                    "return": ep["return"],
                    "mean_reward": ep["mean_reward"],
                    "mean_x_velocity": ep["mean_x_velocity"],
                    "gait_recurrence": ep["recurrence"],
                    "latent_geometry_at_L": ep["latent_geometry"],
                    "latent_repeat_at_2L": ep["latent_2L_repeat"],
                    "fitted_latent_circle": ep["fitted_circle"],
                }
                for ep in episodes
            ],
            "recurrence_matches": int(sum(bool(row["passed"]) for row in recurrence_rows)),
            "raw_period_matches_including_incomplete": int(
                sum(bool(row["period_match"]) for row in recurrence_rows)
            ),
            "recurrence_matches_of": len(episodes),
            "mean_fitted_circle_diameter": _mean_metric(circle_rows, ("diameter",)),
            "mean_fitted_circle_relative_error": _mean_metric(circle_rows, ("relative_diameter_error",)),
            "mean_latent_L_lag_distance": _mean_metric(
                [ep["latent_geometry"] for ep in episodes if ep["latent_geometry"] is not None],
                ("lag_L_distance", "mean"),
            ),
            "mean_latent_2L_repeat_distance": _mean_metric(
                [ep["latent_2L_repeat"] for ep in episodes if ep["latent_2L_repeat"] is not None],
                ("mean_distance",),
            ),
        })
    switch_summary = {
        key: value for key, value in switch.items()
        if key not in {
            "observations", "joint_angles", "joint_velocities", "latents", "actions",
            "rewards", "x_velocities", "periods", "state_periods",
        }
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_config": config,
        "device": str(device),
        "environment": ENV_ID,
        "shared_policy_periods": PERIODS,
        "constant_rollout_steps": ROLLOUT_STEPS,
        "constant_reset_seeds": RESET_SEEDS,
        "switch_seed": SWITCH_SEED,
        "switch_schedule": [
            {"L": period, "start_step": index * SEGMENT_STEPS,
             "end_step": (index + 1) * SEGMENT_STEPS}
            for index, period in enumerate(PERIODS)
        ],
        "switch_episode": switch_summary,
        "switch_segments": switch["segments"],
        "joint_names": joint_names,
        "joint_state_source": "MuJoCo qpos[3:] and qvel[3:] (six actuated Walker2d joints)",
        "discard_initial_steps_for_recurrence": SETTLE_STEPS,
        "recurrence_tolerance": TARGET_TOLERANCE,
        "recurrence_error_threshold": RECURRENCE_ERROR_THRESHOLD,
        "settings": settings,
        "shared_pca": {
            "fit_samples": int(len(switch["latents"]) + sum(
                len(ep["latents"]) for rows in constant.values() for ep in rows
            )),
            "fit_population": "all switch and constant-period conditioned latent states",
            "explained_variance_ratio": pca["explained"].tolist(),
        },
        "artifacts": charts,
        "notes": [
            "All L settings use one shared actor and encoder from the multi-period checkpoint.",
            "Each fixed-period test starts from a fresh reset; the same five seeds are reused across L.",
            "The switch test is continuous and uses a 1,600-step TimeLimit; it is not reset at L boundaries.",
            "The gait recurrence tolerance is a diagnostic convention, not a threshold stated in the paper.",
            "Circle diameters are fitted in each rollout's unscaled best-fit latent plane and compared directly with L.",
            "An early fall is retained as an incomplete rollout; missing steps are not filled by another reset.",
        ],
    }
    report_path = args.out / "walker2d_figure5_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"1,600-step switch episode: {switch['length']}/1,600 steps ({switch['ending']}).")
    for row in settings:
        repeats = [run["gait_recurrence"]["shortest_repeat"] for run in row["rollouts"]]
        diameter = row["mean_fitted_circle_diameter"]["mean"]
        print(
            f"L={row['L']:2} target gait={row['target_gait_period']:2}; "
            f"shortest gait lags={repeats}; matches={row['recurrence_matches']}/5; "
            f"fitted latent circle diameter={diameter} (target {row['L']})"
        )
    print(f"Results: {report_path}")


if __name__ == "__main__":
    main()
