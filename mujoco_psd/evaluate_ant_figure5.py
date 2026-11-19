"""Evaluate the shared-period Ant policy and make Figure 5 diagnostics.

The evaluator runs five preselected reset seeds under each constant period and
one 800-step episode that changes period every 200 steps. The switch episode is
never reset at a boundary. If Ant terminates early, the remaining schedule is
reported as unobserved instead of silently restarting the environment.
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

from evaluate_ant import (
    ENV_ID,
    HIDDEN_DIM,
    _actor_state,
    _infer_embed_dim,
    _motor_joint_names,
    _motor_state,
    period_embedding,
)
from evaluate_periods import RECURRENCE_THRESHOLD, TARGET_TOLERANCE, gait_recurrence
from sac import SAC


PERIODS = (10, 13, 16, 20)
SEGMENT_STEPS = 200
SWITCH_STEPS = SEGMENT_STEPS * len(PERIODS)
ROLLOUT_STEPS = 200
SETTLE_STEPS = 40
RESET_SEEDS = tuple(range(2**31, 2**31 + 5))
# Chosen before seeing rollout results; this is the sole seed used for the
# 800-step Figure 5 switch plot and is distinct from the constant-L seeds.
SWITCH_SEED = 2**31 + 100


def _condition(observation: np.ndarray, period: int, embed_dim: int, device):
    state = torch.as_tensor(observation, dtype=torch.float32, device=device)
    period_tensor = torch.tensor([period], dtype=torch.float32, device=device)
    return torch.cat((state, period_embedding(period_tensor, embed_dim)[0]))


def _encode_period_states(encoder, observations, periods, embed_dim: int, device):
    """Encode state/period pairs in bounded batches, supporting changing L."""
    observations = np.asarray(observations, dtype=np.float32)
    periods = np.asarray(periods, dtype=np.float32)
    if len(observations) != len(periods):
        raise ValueError("Each encoded state must have exactly one period label.")
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(observations), 1024):
            stop = min(start + 1024, len(observations))
            states = torch.as_tensor(observations[start:stop], device=device)
            labels = torch.as_tensor(periods[start:stop], device=device)
            inputs = torch.cat((states, period_embedding(labels, embed_dim)), dim=-1)
            outputs.append(encoder(inputs).cpu().numpy())
    if not outputs:
        return np.empty((0, int(encoder[-1].out_features)), dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32, copy=False)


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
    observed_angles = episode["joint_angles"][SETTLE_STEPS:length]
    observed_rates = episode["joint_velocities"][SETTLE_STEPS:length]
    target = 2 * period
    # gait_recurrence searches only through floor(window_length / 3); ensure
    # the tolerance band around the target exists before calling it.
    minimum_target_lag = math.ceil(target * (1 - TARGET_TOLERANCE))
    enough = len(observed_angles) >= 3 * minimum_target_lag
    if enough:
        recurrence = gait_recurrence(observed_angles, observed_rates, target=target)
        reason = "analyzed"
    else:
        recurrence = _empty_recurrence(observed_angles)
        reason = (
            f"only {len(observed_angles)} post-settle steps; need at least "
            f"{3 * minimum_target_lag} to assess target lag {target}"
        )
    full_episode = length == ROLLOUT_STEPS and not episode["terminated"]
    period_match = bool(recurrence["match"])
    return {
        **recurrence,
        "period_match": period_match,
        "passed": bool(full_episode and period_match),
        "interpretation_reason": (
            "matched" if full_episode and period_match else
            "episode ended before the 200-step cap" if not full_episode else
            "no recurrence within the target tolerance"
        ),
        "target_period": target,
        "target_tolerance": TARGET_TOLERANCE,
        "analysis_start_step": SETTLE_STEPS,
        "analysis_steps": int(len(observed_angles)),
        "analysis_status": reason,
        "full_episode": bool(full_episode),
        "episode_length": int(length),
        "terminated": bool(episode["terminated"]),
        "truncated": bool(episode["truncated"]),
    }


def _root_x(env):
    """Return Ant's world x coordinate when the simulator exposes it."""
    try:
        qpos = np.asarray(env.unwrapped.data.qpos)
        if qpos.size and np.isfinite(qpos[0]):
            return float(qpos[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None


def _simulation_dt(env):
    """Get the simulated seconds per environment transition."""
    try:
        unwrapped = env.unwrapped
        dt = float(unwrapped.model.opt.timestep * unwrapped.frame_skip)
        return dt if math.isfinite(dt) and dt > 0 else None
    except (AttributeError, TypeError, ValueError):
        return None


def _x_velocity(env, info, previous_x, current_x, dt):
    """Use Gymnasium's measured velocity, falling back to root displacement."""
    try:
        velocity = float(info.get("x_velocity"))
        if math.isfinite(velocity):
            return velocity
    except (AttributeError, TypeError, ValueError):
        pass
    if previous_x is not None and current_x is not None and dt is not None:
        velocity = (current_x - previous_x) / dt
        if math.isfinite(velocity):
            return float(velocity)
    return float("nan")


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _net_displacement(positions):
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) >= 2 and np.isfinite(positions[0]) and np.isfinite(positions[-1]):
        return float(positions[-1] - positions[0])
    return None


def _run_rollout(env, agent, encoder, period, seed, embed_dim, device):
    observation, _ = env.reset(seed=int(seed))
    observations = [np.asarray(observation, dtype=np.float32).copy()]
    angle, rate = _motor_state(env)
    joint_angles = [angle]
    joint_velocities = [rate]
    actions, rewards, x_velocities = [], [], []
    initial_x = _root_x(env)
    x_positions = [np.nan if initial_x is None else initial_x]
    dt = _simulation_dt(env)
    previous_x = initial_x
    terminated = truncated = False
    for _ in range(ROLLOUT_STEPS):
        conditioned = _condition(observation, period, embed_dim, device)
        with torch.inference_mode():
            action = agent.act(conditioned, deterministic=True).cpu().numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Nonfinite Ant action for L={period}, seed={seed}.")
        observation, reward, terminated, truncated, info = env.step(action)
        current_x = _root_x(env)
        x_velocities.append(_x_velocity(env, info, previous_x, current_x, dt))
        x_positions.append(np.nan if current_x is None else current_x)
        previous_x = current_x
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        angle, rate = _motor_state(env)
        joint_angles.append(angle)
        joint_velocities.append(rate)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break

    observations = np.asarray(observations, dtype=np.float32)
    episode = {
        "period": int(period),
        "seed": int(seed),
        "length": len(actions),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "observations": observations,
        "joint_angles": np.asarray(joint_angles, dtype=np.float32),
        "joint_velocities": np.asarray(joint_velocities, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1, agent.action_dim),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "x_velocities": np.asarray(x_velocities, dtype=np.float32),
        "x_positions": np.asarray(x_positions, dtype=np.float64),
        "return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_x_velocity": _finite_mean(x_velocities),
        "net_forward_displacement": _net_displacement(x_positions),
    }
    if len(episode["observations"]) != len(episode["actions"]) + 1:
        raise RuntimeError("Ant rollout has inconsistent state and transition lengths.")
    episode["latents"] = _encode_period_states(
        encoder, observations, np.full(len(observations), period), embed_dim, device
    )
    episode["recurrence"] = _recurrence_for_episode(episode, period)
    return episode


def _run_switch(env, agent, encoder, embed_dim, device):
    observation, _ = env.reset(seed=int(SWITCH_SEED))
    observations = [np.asarray(observation, dtype=np.float32).copy()]
    angle, rate = _motor_state(env)
    joint_angles = [angle]
    joint_velocities = [rate]
    actions, rewards, x_velocities, action_periods, state_periods = [], [], [], [], []
    initial_x = _root_x(env)
    x_positions = [np.nan if initial_x is None else initial_x]
    previous_x = initial_x
    dt = _simulation_dt(env)
    terminated = truncated = False
    for step in range(SWITCH_STEPS):
        period = PERIODS[step // SEGMENT_STEPS]
        state_periods.append(period)
        conditioned = _condition(observation, period, embed_dim, device)
        with torch.inference_mode():
            action = agent.act(conditioned, deterministic=True).cpu().numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Nonfinite Ant action at switch step {step}.")
        observation, reward, terminated, truncated, info = env.step(action)
        current_x = _root_x(env)
        x_velocities.append(_x_velocity(env, info, previous_x, current_x, dt))
        x_positions.append(np.nan if current_x is None else current_x)
        previous_x = current_x
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        angle, rate = _motor_state(env)
        joint_angles.append(angle)
        joint_velocities.append(rate)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))
        action_periods.append(period)
        if terminated or truncated:
            break

    # The latent label at each state matches the period used for the action
    # from that state. The final post-action state keeps the last active L.
    if len(state_periods):
        state_periods.append(state_periods[-1])
    else:
        state_periods.append(PERIODS[0])
    observations = np.asarray(observations, dtype=np.float32)
    episode = {
        "seed": int(SWITCH_SEED),
        "length": len(actions),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "completed_800_steps": len(actions) == SWITCH_STEPS,
        "observations": observations,
        "joint_angles": np.asarray(joint_angles, dtype=np.float32),
        "joint_velocities": np.asarray(joint_velocities, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1, agent.action_dim),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "x_velocities": np.asarray(x_velocities, dtype=np.float32),
        "x_positions": np.asarray(x_positions, dtype=np.float64),
        "periods": np.asarray(action_periods, dtype=np.int32),
        "state_periods": np.asarray(state_periods, dtype=np.int32),
        "return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_x_velocity": _finite_mean(x_velocities),
        "net_forward_displacement": _net_displacement(x_positions),
    }
    episode["latents"] = _encode_period_states(
        encoder, observations, episode["state_periods"], embed_dim, device
    )
    episode["segments"] = [
        {
            "L": period,
            "requested_steps": SEGMENT_STEPS,
            "observed_steps": int(max(0, min(episode["length"], (i + 1) * SEGMENT_STEPS)
                                            - i * SEGMENT_STEPS)),
            "complete": episode["length"] >= (i + 1) * SEGMENT_STEPS,
            "mean_x_velocity": _finite_mean(
                episode["x_velocities"][i * SEGMENT_STEPS:
                                         min((i + 1) * SEGMENT_STEPS, episode["length"])]
            ),
            "net_forward_displacement": _net_displacement(
                episode["x_positions"][i * SEGMENT_STEPS:
                                        min((i + 1) * SEGMENT_STEPS,
                                            episode["length"]) + 1]
            ),
        }
        for i, period in enumerate(PERIODS)
    ]
    episode["ending"] = (
        "completed all 800 scheduled steps"
        if episode["completed_800_steps"] and not terminated
        else "environment terminated at the end of the 800-step schedule"
        if episode["completed_800_steps"] and terminated
        else "environment terminated before the 800-step schedule completed"
        if terminated
        else "episode truncated before the 800-step schedule completed"
        if truncated
        else "episode stopped before the 800-step schedule completed"
    )
    return episode


def _fit_shared_pca(switch, constant):
    all_latents = [switch["latents"]]
    all_latents.extend(episode["latents"] for rows in constant.values() for episode in rows)
    samples = np.concatenate(all_latents, axis=0).astype(np.float64)
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, singular, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((2, samples.shape[1]), dtype=np.float64)
    available = min(2, len(right_vectors))
    components[:available] = right_vectors[:available]
    variance = singular**2
    total_variance = variance.sum()
    explained = np.zeros(2, dtype=np.float64)
    if total_variance > 0:
        explained[:min(2, len(variance))] = variance[:2] / total_variance
    switch_coordinates = (switch["latents"] - mean) @ components.T
    constant_coordinates = {
        (period, index): (episode["latents"] - mean) @ components.T
        for period, rows in constant.items()
        for index, episode in enumerate(rows)
    }
    return mean, components, explained, switch_coordinates, constant_coordinates


def _plot_switch(out: Path, joint_names, switch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {10: "#4477AA", 13: "#EE6677", 16: "#228833", 20: "#AA3377"}
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
    steps = np.arange(len(switch["joint_angles"]))
    actual = switch["length"]
    status = (
        "complete" if switch["completed_800_steps"] else
        f"incomplete after {actual}/800 steps (fell/terminated)" if switch["terminated"] else
        f"incomplete after {actual}/800 steps"
    )
    for joint, axis in enumerate(axes.flat):
        axis.plot(steps, switch["joint_angles"][:, joint], color="#222222", lw=0.9)
        for index, period in enumerate(PERIODS):
            left, right = index * SEGMENT_STEPS, (index + 1) * SEGMENT_STEPS
            axis.axvspan(left, right, color=colors[period], alpha=0.09, linewidth=0)
            axis.axvline(left, color=colors[period], alpha=0.65, lw=0.8)
        axis.set_ylabel("Angle (rad)")
        axis.set_title(joint_names[joint])
        axis.grid(alpha=0.2)
        axis.set_xlim(0, SWITCH_STEPS)
        if actual < SWITCH_STEPS:
            axis.axvspan(actual, SWITCH_STEPS, color="#777777", alpha=0.16, hatch="///")
    axes[-1, 0].set_xlabel("Environment step")
    axes[-1, 1].set_xlabel("Environment step")
    handles = [plt.Line2D([0], [0], color=colors[p], lw=4, label=f"L={p} (steps {i*200}–{(i+1)*200})")
               for i, p in enumerate(PERIODS)]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        f"Ant single-episode period switches, seed={switch['seed']} — {status}\n"
        "One continuous rollout; no reset at the L boundaries",
        y=1.04,
    )
    path = out / "ant_figure5_switch_joint_traces.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_l20(out: Path, joint_names, constant_l20):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(constant_l20)))
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
    for index, episode in enumerate(constant_l20):
        steps = np.arange(len(episode["joint_angles"]))
        for joint, axis in enumerate(axes.flat):
            axis.plot(steps, episode["joint_angles"][:, joint], color=colors[index],
                      alpha=0.82, lw=0.9,
                      label=(f"seed {episode['seed']}" if episode["length"] == ROLLOUT_STEPS
                             else f"seed {episode['seed']} ({episode['length']}/200, terminated)"))
    for joint, axis in enumerate(axes.flat):
        axis.set_ylabel("Angle (rad)")
        axis.set_title(joint_names[joint])
        axis.grid(alpha=0.2)
        axis.set_xlim(0, ROLLOUT_STEPS)
    axes[-1, 0].set_xlabel("Environment step")
    axes[-1, 1].set_xlabel("Environment step")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Ant constant-L=20 joint angles, five preselected reset seeds", y=1.02)
    path = out / "ant_l20_joint_traces.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_pca(out: Path, switch, constant, switch_coordinates,
              constant_coordinates, explained):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {10: "#4477AA", 13: "#EE6677", 16: "#228833", 20: "#AA3377"}
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for period in PERIODS:
        for index, episode in enumerate(constant[period]):
            xy = constant_coordinates[(period, index)]
            axis.scatter(xy[:, 0], xy[:, 1], s=5, color=colors[period], alpha=0.10,
                         linewidths=0, rasterized=True,
                         label=f"constant L={period}" if index == 0 else None)
    for index, period in enumerate(PERIODS):
        left = index * SEGMENT_STEPS
        right = min((index + 1) * SEGMENT_STEPS, switch["length"] + 1)
        if right <= left:
            continue
        xy = switch_coordinates[left:right]
        axis.plot(xy[:, 0], xy[:, 1], color=colors[period], lw=1.5, alpha=0.9)
        axis.scatter(xy[:, 0], xy[:, 1], s=13, color=colors[period], alpha=0.85,
                     edgecolors="none",
                     label=f"switch rollout L={period}")
    if len(switch_coordinates):
        axis.scatter(*switch_coordinates[0], marker="o", color="black", s=50,
                     label="switch start", zorder=4)
        axis.scatter(*switch_coordinates[-1], marker="X", color="black", s=55,
                     label="switch end", zorder=4)
    axis.set_xlabel(f"PC 1 ({explained[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC 2 ({explained[1] * 100:.1f}% variance)")
    axis.set_title("Shared PCA of conditioned Ant latents\nconstant-L rollouts and one continuous switch rollout")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)
    path = out / "ant_figure5_shared_pca.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _save_raw_data(out, joint_names, constant, switch, pca):
    obs_dim = constant[PERIODS[0]][0]["observations"].shape[1]
    latent_dim = constant[PERIODS[0]][0]["latents"].shape[1]
    action_dim = constant[PERIODS[0]][0]["actions"].shape[1]
    max_const = max((ep["length"] for rows in constant.values() for ep in rows), default=0)
    states = np.zeros((len(PERIODS), len(RESET_SEEDS), max_const + 1, obs_dim), np.float32)
    angles = np.zeros((len(PERIODS), len(RESET_SEEDS), max_const + 1, 8), np.float32)
    rates = np.zeros_like(angles)
    latents = np.zeros((len(PERIODS), len(RESET_SEEDS), max_const + 1, latent_dim), np.float32)
    actions = np.zeros((len(PERIODS), len(RESET_SEEDS), max_const, action_dim), np.float32)
    rewards = np.zeros((len(PERIODS), len(RESET_SEEDS), max_const), np.float32)
    x_velocities = np.full((len(PERIODS), len(RESET_SEEDS), max_const), np.nan, np.float32)
    x_positions = np.full((len(PERIODS), len(RESET_SEEDS), max_const + 1), np.nan, np.float64)
    state_mask = np.zeros((len(PERIODS), len(RESET_SEEDS), max_const + 1), bool)
    transition_mask = np.zeros((len(PERIODS), len(RESET_SEEDS), max_const), bool)
    constant_coordinates = np.zeros(
        (len(PERIODS), len(RESET_SEEDS), max_const + 1, 2), np.float32
    )
    lengths = np.zeros((len(PERIODS), len(RESET_SEEDS)), np.int32)
    terminated = np.zeros_like(lengths, dtype=bool)
    truncated = np.zeros_like(lengths, dtype=bool)
    for p_index, period in enumerate(PERIODS):
        for s_index, episode in enumerate(constant[period]):
            n = episode["length"]
            lengths[p_index, s_index] = n
            terminated[p_index, s_index] = episode["terminated"]
            truncated[p_index, s_index] = episode["truncated"]
            states[p_index, s_index, :n + 1] = episode["observations"]
            angles[p_index, s_index, :n + 1] = episode["joint_angles"]
            rates[p_index, s_index, :n + 1] = episode["joint_velocities"]
            latents[p_index, s_index, :n + 1] = episode["latents"]
            actions[p_index, s_index, :n] = episode["actions"]
            rewards[p_index, s_index, :n] = episode["rewards"]
            x_velocities[p_index, s_index, :n] = episode["x_velocities"]
            x_positions[p_index, s_index, :n + 1] = episode["x_positions"]
            state_mask[p_index, s_index, :n + 1] = True
            transition_mask[p_index, s_index, :n] = True
            constant_coordinates[p_index, s_index, :n + 1] = pca["constant_coordinates"][(period, s_index)]
    switch_n = switch["length"]
    np.savez_compressed(
        out / "ant_figure5_rollouts.npz",
        periods=np.asarray(PERIODS, dtype=np.int32),
        constant_seeds=np.asarray(RESET_SEEDS, dtype=np.uint32),
        switch_seed=np.asarray(SWITCH_SEED, dtype=np.uint32),
        switch_boundaries=np.arange(0, SWITCH_STEPS + 1, SEGMENT_STEPS, dtype=np.int32),
        constant_lengths=lengths, constant_terminated=terminated, constant_truncated=truncated,
        constant_observation_mask=state_mask, constant_transition_mask=transition_mask,
        constant_observations=states, constant_joint_angles=angles,
        constant_joint_velocities=rates, constant_latents=latents,
        constant_actions=actions, constant_rewards=rewards,
        constant_x_velocities=x_velocities, constant_root_x_positions=x_positions,
        switch_length=np.asarray(switch_n, dtype=np.int32),
        switch_terminated=np.asarray(switch["terminated"]),
        switch_truncated=np.asarray(switch["truncated"]),
        switch_periods=switch["periods"], switch_state_periods=switch["state_periods"],
        switch_observations=switch["observations"],
        switch_joint_angles=switch["joint_angles"],
        switch_joint_velocities=switch["joint_velocities"],
        switch_latents=switch["latents"], switch_actions=switch["actions"],
        switch_rewards=switch["rewards"],
        switch_x_velocities=switch["x_velocities"],
        switch_root_x_positions=switch["x_positions"],
        switch_pca_coordinates=pca["switch_coordinates"],
        constant_pca_coordinates=constant_coordinates,
        pca_mean=pca["mean"].astype(np.float32),
        pca_components=pca["components"].astype(np.float32),
        pca_explained_variance_ratio=pca["explained"].astype(np.float32),
        joint_names=np.asarray(joint_names),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Multi-period Ant checkpoint (.pt)")
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
    if config.get("env") != ENV_ID:
        raise ValueError(f"Expected a {ENV_ID} checkpoint, got env={config.get('env')!r}.")
    configured_periods = config.get("periods", config.get("L_values"))
    if configured_periods is None or tuple(map(int, configured_periods)) != PERIODS:
        raise ValueError(
            f"Expected a shared-policy checkpoint trained with periods {PERIODS}; "
            f"got {configured_periods!r}."
        )
    latent_dim = int(config["latent_dim"])
    args.out.mkdir(exist_ok=True, parents=True)

    env = gym.make(ENV_ID, max_episode_steps=ROLLOUT_STEPS)
    switch_env = gym.make(ENV_ID, max_episode_steps=SWITCH_STEPS)
    try:
        observation_dim = int(env.observation_space.shape[0])
        action_state = _actor_state(checkpoint["agent"])
        embed_dim = _infer_embed_dim(action_state, observation_dim, config)
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
        joint_names = _motor_joint_names(env)

        constant = {}
        for period in PERIODS:
            constant[period] = [
                _run_rollout(env, agent, encoder, period, seed, embed_dim, device)
                for seed in RESET_SEEDS
            ]
        switch = _run_switch(switch_env, agent, encoder, embed_dim, device)
    finally:
        env.close()
        switch_env.close()

    mean, components, explained, switch_coordinates, constant_coordinates = _fit_shared_pca(
        switch, constant
    )
    pca = {
        "mean": mean,
        "components": components,
        "explained": explained,
        "switch_coordinates": switch_coordinates,
        "constant_coordinates": constant_coordinates,
    }
    switch_chart = _plot_switch(args.out, joint_names, switch)
    l20_chart = _plot_l20(args.out, joint_names, constant[20])
    pca_chart = _plot_pca(
        args.out, switch, constant, switch_coordinates, constant_coordinates, explained
    )
    _save_raw_data(args.out, joint_names, constant, switch, pca)

    settings = []
    for period in PERIODS:
        episodes = constant[period]
        recurrences = [ep["recurrence"] for ep in episodes]
        settings.append({
            "L": period,
            "target_gait_period": 2 * period,
            "rollouts": [
                {
                    "seed": ep["seed"],
                    "length": ep["length"],
                    "terminated": ep["terminated"],
                    "truncated": ep["truncated"],
                    "mean_reward": ep["mean_reward"],
                    "return": ep["return"],
                    "mean_x_velocity": ep["mean_x_velocity"],
                    "net_forward_displacement": ep["net_forward_displacement"],
                    "recurrence": ep["recurrence"],
                }
                for ep in episodes
            ],
            "recurrence_matches": int(sum(bool(row["passed"]) for row in recurrences)),
            "raw_period_matches_including_incomplete": int(sum(bool(row["period_match"]) for row in recurrences)),
            "recurrence_matches_of": len(episodes),
            "mean_x_velocity": _finite_mean([ep["mean_x_velocity"] for ep in episodes
                                              if ep["mean_x_velocity"] is not None]),
            "mean_net_forward_displacement": _finite_mean([
                ep["net_forward_displacement"] for ep in episodes
                if ep["net_forward_displacement"] is not None
            ]),
        })
    switch_summary = {
        key: value for key, value in switch.items()
        if key not in {
            "observations", "joint_angles", "joint_velocities", "latents", "actions",
            "rewards", "x_velocities", "x_positions", "periods", "state_periods",
        }
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_config": config,
        "device": str(device),
        "environment": ENV_ID,
        "environment_reward_semantics": (
            "Recorded return and mean_reward are Gymnasium Ant-v5 environment rewards; "
            "they are not the combined PSD plus forward-velocity training objective."
        ),
        "shared_policy_periods": PERIODS,
        "constant_rollout_steps": ROLLOUT_STEPS,
        "constant_reset_seeds": RESET_SEEDS,
        "switch_seed": SWITCH_SEED,
        "switch_schedule": [
            {"L": period, "start_step": i * SEGMENT_STEPS, "end_step": (i + 1) * SEGMENT_STEPS}
            for i, period in enumerate(PERIODS)
        ],
        "switch_episode": switch_summary,
        "switch_segments": switch["segments"],
        "discard_initial_steps_for_recurrence": SETTLE_STEPS,
        "recurrence_tolerance": TARGET_TOLERANCE,
        "recurrence_error_threshold": RECURRENCE_THRESHOLD,
        "settings": settings,
        "shared_pca": {
            "fit_samples": int(len(switch["latents"]) + sum(
                len(ep["latents"]) for rows in constant.values() for ep in rows
            )),
            "fit_population": "all switch and constant-L conditioned latent states",
            "explained_variance_ratio": explained.tolist(),
        },
        "artifacts": {
            "raw_rollouts": "ant_figure5_rollouts.npz",
            "switch_joint_traces": switch_chart,
            "constant_l20_joint_traces": l20_chart,
            "shared_pca": pca_chart,
        },
        "notes": [
            "The switch rollout is a single environment episode with no reset at L boundaries.",
            "Any unobserved part of an early-terminated switch schedule is shown as incomplete.",
            "Recurrence is an exploratory diagnostic; the paper does not prescribe this exact pass threshold.",
            "All periods use the same learned actor and encoder from the shared multi-period checkpoint.",
        ],
    }
    (args.out / "ant_figure5_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"800-step switch episode: {switch['length']}/800 steps ({switch['ending']}).")
    for row in settings:
        print(f"L={row['L']:2} target={row['target_gait_period']:2} gait recurrence="
              f"{row['recurrence_matches']}/{row['recurrence_matches_of']}")
    print(f"Results: {args.out / 'ant_figure5_evaluation.json'}")


if __name__ == "__main__":
    main()
