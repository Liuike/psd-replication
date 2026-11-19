"""Evaluate a fixed-L Ant policy checkpoint for its requested 40-step gait."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from evaluate_periods import gait_recurrence
from sac import SAC


ENV_ID = "Ant-v5"
L = 20
TARGET_GAIT_PERIOD = 2 * L
MAX_STEPS = 200
SETTLE_STEPS = 40
RESET_SEEDS = tuple(range(2**31, 2**31 + 5))
TARGET_TOLERANCE = 0.10
PASS_ROLLOUTS = 4
DEFAULT_EMBED_DIM = 8
HIDDEN_DIM = 1024


def period_embedding(period: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Match the sinusoidal L embedding used by the PSD training runner."""
    if embed_dim <= 0 or embed_dim % 2:
        raise ValueError("The period embedding dimension must be positive and even.")
    exponent = torch.arange(0, embed_dim, 2, device=period.device) / embed_dim
    angle = period.float().unsqueeze(-1) * 10000 ** -exponent
    return torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)


def _actor_state(agent_state):
    if not isinstance(agent_state, dict):
        raise ValueError("Checkpoint 'agent' entry must be a state dictionary.")
    return agent_state.get("actor", agent_state)


def _infer_embed_dim(actor_state, observation_dim: int, config: dict) -> int:
    configured = config.get("embed_dim", config.get("period_embedding_dim"))
    if configured is not None:
        embed_dim = int(configured)
    else:
        first_weight = actor_state.get("trunk.0.weight")
        if first_weight is None:
            raise ValueError("Cannot infer actor input width from the saved actor weights.")
        embed_dim = int(first_weight.shape[1]) - observation_dim
    if embed_dim <= 0 or embed_dim % 2:
        raise ValueError(f"Invalid period embedding dimension: {embed_dim}.")
    return embed_dim


def _motor_joint_names(env) -> list[str]:
    """Return names for the eight actuated Ant joints, with stable fallbacks."""
    try:
        import mujoco

        model = env.unwrapped.model
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(1, model.njnt)
        ]
        if len(names) == 8 and all(names):
            return [str(name) for name in names]
    except (ImportError, AttributeError, TypeError):
        pass
    return [f"motor_joint_{index + 1}" for index in range(8)]


def _motor_state(env) -> tuple[np.ndarray, np.ndarray]:
    data = env.unwrapped.data
    angles = np.asarray(data.qpos[7:], dtype=np.float32).copy()
    rates = np.asarray(data.qvel[6:], dtype=np.float32).copy()
    if angles.shape != (8,) or rates.shape != (8,):
        raise ValueError(
            "Expected eight Ant motor joint positions in qpos[7:] and velocities in qvel[6:]."
        )
    return angles, rates


def _encode_states(encoder, observations, period: int, embed_dim: int, device):
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    period_tensor = torch.full(
        (observation_tensor.shape[0],), period, dtype=torch.float32, device=device
    )
    conditioned = torch.cat((observation_tensor, period_embedding(period_tensor, embed_dim)), dim=-1)
    with torch.no_grad():
        encoded = encoder(conditioned)
    return encoded.detach().cpu().numpy().astype(np.float32, copy=False)


def _latent_geometry(latents: np.ndarray, latent_dim: int) -> dict:
    if len(latents) < 2:
        return {
            "samples": int(len(latents)),
            "latent_dim": latent_dim,
            "mean_norm": None,
            "mean_one_step_distance": None,
            "target_one_step_chord": float(L * math.sin(math.pi / (2 * L))),
            "one_step_chord_abs_error": None,
            "mean_L_lag_distance": None,
            "mean_antipodal_vector_residual": None,
            "mean_2L_lag_distance": None,
        }

    def mean_distance(lag: int) -> float | None:
        if len(latents) <= lag:
            return None
        return float(np.linalg.norm(latents[lag:] - latents[:-lag], axis=1).mean())

    step_distance = mean_distance(1)
    l_distance = mean_distance(L)
    repeat_distance = mean_distance(TARGET_GAIT_PERIOD)
    antipodal_residual = (
        float(np.linalg.norm(latents[L:] + latents[:-L], axis=1).mean())
        if len(latents) > L else None
    )
    target_chord = float(L * math.sin(math.pi / (2 * L)))
    return {
        "samples": int(len(latents)),
        "latent_dim": latent_dim,
        "mean_norm": float(np.linalg.norm(latents, axis=1).mean()),
        "mean_one_step_distance": step_distance,
        "target_one_step_chord": target_chord,
        "one_step_chord_abs_error": (
            abs(step_distance - target_chord) if step_distance is not None else None
        ),
        "mean_L_lag_distance": l_distance,
        "mean_antipodal_vector_residual": antipodal_residual,
        "mean_2L_lag_distance": repeat_distance,
    }


def _optional_plot(out: Path, names: list[str], episodes: list[dict]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    candidates = [episode for episode in episodes if len(episode["observations"]) > SETTLE_STEPS]
    if not candidates:
        return None
    episode = max(candidates, key=lambda item: item["length"])
    length = episode["length"]
    steps = np.arange(length + 1)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)

    for joint_index, joint_name in enumerate(names):
        axes[0].plot(steps, episode["joint_angles"][:, joint_index], label=joint_name, linewidth=1.0)
    axes[0].set_title(f"Ant motor joint positions, L=20, seed={episode['seed']}")
    axes[0].set_xlabel("Environment step")
    axes[0].set_ylabel("Joint angle (rad)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(alpha=0.25)

    for joint_index, joint_name in enumerate(names):
        axes[1].plot(steps, episode["joint_velocities"][:, joint_index], label=joint_name, linewidth=1.0)
    axes[1].set_title("Ant motor joint velocities")
    axes[1].set_xlabel("Environment step")
    axes[1].set_ylabel("Joint velocity (rad/s)")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(alpha=0.25)

    latents = episode["latents"].astype(np.float64)
    centered = latents - latents.mean(axis=0, keepdims=True)
    if latents.shape[1] >= 2 and len(latents) >= 2:
        u, singular, _ = np.linalg.svd(centered, full_matrices=False)
        coordinates = u[:, :2] * singular[:2]
        if coordinates.shape[1] == 1:
            coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
    else:
        coordinates = np.column_stack((centered[:, 0], np.zeros(len(centered))))
    scatter = axes[2].scatter(
        coordinates[:, 0], coordinates[:, 1], c=np.arange(len(coordinates)),
        cmap="viridis", s=15,
    )
    axes[2].plot(coordinates[:, 0], coordinates[:, 1], color="gray", alpha=0.35, linewidth=0.7)
    axes[2].scatter(*coordinates[0], marker="o", color="green", s=55, label="start", zorder=3)
    axes[2].scatter(*coordinates[-1], marker="X", color="red", s=55, label="end", zorder=3)
    axes[2].set_title("Conditioned latent trajectory (PCA)")
    axes[2].set_xlabel("Principal component 1")
    axes[2].set_ylabel("Principal component 2")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].legend(loc="best")
    fig.colorbar(scatter, ax=axes[2], label="Environment step")

    path = out / "ant_l20_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Ant training checkpoint (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for JSON and raw data")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(6, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "agent", "encoder", "epoch"} <= checkpoint.keys():
        raise ValueError("Checkpoint must contain config, agent, encoder, and epoch entries.")
    config = checkpoint["config"]
    env_id = config.get("env")
    saved_l = config.get("L", config.get("initial_l", config.get("l")))
    if env_id != ENV_ID:
        raise ValueError(f"Expected a {ENV_ID} checkpoint, got env={env_id!r}.")
    if saved_l is None or int(saved_l) != L:
        raise ValueError(f"Expected a fixed-L={L} checkpoint, got L={saved_l!r}.")
    latent_dim = int(config["latent_dim"])

    env = gym.make(ENV_ID, max_episode_steps=MAX_STEPS)
    episodes: list[dict] = []
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

        for seed in RESET_SEEDS:
            observation, _ = env.reset(seed=int(seed))
            observations = [np.asarray(observation, dtype=np.float32).copy()]
            joint_angles, joint_velocities = [], []
            angle, rate = _motor_state(env)
            joint_angles.append(angle)
            joint_velocities.append(rate)
            actions, rewards = [], []
            terminated = False
            truncated = False
            for _ in range(MAX_STEPS):
                conditioned = torch.cat((
                    torch.as_tensor(observation, dtype=torch.float32, device=device),
                    period_embedding(torch.tensor([L], dtype=torch.float32, device=device), embed_dim)[0],
                ))
                action = agent.act(conditioned, deterministic=True).detach().cpu().numpy()
                if not np.isfinite(action).all():
                    raise FloatingPointError(f"Actor produced a nonfinite action for seed {seed}.")
                observation, reward, terminated, truncated, _ = env.step(action)
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                angle, rate = _motor_state(env)
                joint_angles.append(angle)
                joint_velocities.append(rate)
                actions.append(np.asarray(action, dtype=np.float32).copy())
                rewards.append(float(reward))
                if terminated or truncated:
                    break

            observations_array = np.asarray(observations, dtype=np.float32)
            angles_array = np.asarray(joint_angles, dtype=np.float32)
            rates_array = np.asarray(joint_velocities, dtype=np.float32)
            actions_array = np.asarray(actions, dtype=np.float32)
            rewards_array = np.asarray(rewards, dtype=np.float32)
            latents_array = _encode_states(
                encoder, observations_array, L, embed_dim, device
            )
            length = len(actions_array)
            # Match the HalfCheetah diagnostic window: samples 40 through 199
            # for a full 200-step episode, excluding the final post-step state.
            observed_angles = angles_array[SETTLE_STEPS:length]
            observed_rates = rates_array[SETTLE_STEPS:length]
            enough_for_target = len(observed_angles) >= 3 * int(TARGET_GAIT_PERIOD * (1 - TARGET_TOLERANCE))
            if enough_for_target:
                recurrence = gait_recurrence(
                    observed_angles, observed_rates, target=TARGET_GAIT_PERIOD
                )
            else:
                recurrence = {
                    "shortest_repeat": None,
                    "best_target_lag": None,
                    "target_error": None,
                    "moving_joints": int(np.count_nonzero(angles_array.std(axis=0) >= 0.05)),
                    "match": False,
                    "lag_errors": [],
                }
            full_episode = length == MAX_STEPS and not terminated
            gait_match = bool(recurrence["match"])
            passed = bool(full_episode and gait_match)
            episodes.append({
                "seed": int(seed),
                "length": int(length),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "failed_fall": bool(terminated),
                "full_episode": bool(full_episode),
                "mean_reward": float(rewards_array.mean()) if len(rewards_array) else 0.0,
                "mean_return": float(rewards_array.sum()),
                "gait": {
                    **recurrence,
                    "target_period": TARGET_GAIT_PERIOD,
                    "target_tolerance": TARGET_TOLERANCE,
                    "analysis_start_step": SETTLE_STEPS,
                    "period_match": gait_match,
                    "passed": passed,
                    "reason": (
                        "matched" if passed else
                        "fell before episode cap" if terminated else
                        "episode ended before 200 steps" if not full_episode else
                        "no recurrence within the target tolerance"
                    ),
                },
                "latent_geometry": _latent_geometry(latents_array, latent_dim),
                "observations": observations_array,
                "joint_angles": angles_array,
                "joint_velocities": rates_array,
                "latents": latents_array,
                "actions": actions_array,
                "rewards": rewards_array,
            })
    finally:
        env.close()

    matches = sum(episode["gait"]["passed"] for episode in episodes)
    summaries = [{
        key: value for key, value in episode.items()
        if key not in {"observations", "joint_angles", "joint_velocities", "latents", "actions", "rewards"}
    } for episode in episodes]
    geometry_metrics = [episode["latent_geometry"] for episode in episodes]
    geometry_summary = {}
    for key in (
        "mean_norm", "mean_one_step_distance", "one_step_chord_abs_error",
        "mean_L_lag_distance", "mean_antipodal_vector_residual", "mean_2L_lag_distance",
    ):
        values = [row[key] for row in geometry_metrics if row[key] is not None and math.isfinite(row[key])]
        geometry_summary[key] = float(np.mean(values)) if values else None

    args.out.mkdir(exist_ok=True, parents=True)
    max_length = max((episode["length"] for episode in episodes), default=0)
    obs_dim = episodes[0]["observations"].shape[1]
    latent_dim_actual = episodes[0]["latents"].shape[1]
    action_dim = episodes[0]["actions"].shape[1] if max_length else 0
    raw_observations = np.zeros((len(episodes), max_length + 1, obs_dim), dtype=np.float32)
    raw_angles = np.zeros((len(episodes), max_length + 1, 8), dtype=np.float32)
    raw_rates = np.zeros_like(raw_angles)
    raw_latents = np.zeros((len(episodes), max_length + 1, latent_dim_actual), dtype=np.float32)
    raw_actions = np.zeros((len(episodes), max_length, action_dim), dtype=np.float32)
    raw_rewards = np.zeros((len(episodes), max_length), dtype=np.float32)
    observation_mask = np.zeros((len(episodes), max_length + 1), dtype=bool)
    transition_mask = np.zeros((len(episodes), max_length), dtype=bool)
    for index, episode in enumerate(episodes):
        state_count = len(episode["observations"])
        transition_count = len(episode["actions"])
        raw_observations[index, :state_count] = episode["observations"]
        raw_angles[index, :state_count] = episode["joint_angles"]
        raw_rates[index, :state_count] = episode["joint_velocities"]
        raw_latents[index, :state_count] = episode["latents"]
        raw_actions[index, :transition_count] = episode["actions"]
        raw_rewards[index, :transition_count] = episode["rewards"]
        observation_mask[index, :state_count] = True
        transition_mask[index, :transition_count] = True
    np.savez_compressed(
        args.out / "ant_l20_rollouts.npz",
        L=np.asarray(L), target_gait_period=np.asarray(TARGET_GAIT_PERIOD),
        seeds=np.asarray(RESET_SEEDS, dtype=np.uint32),
        lengths=np.asarray([episode["length"] for episode in episodes], dtype=np.int32),
        terminated=np.asarray([episode["terminated"] for episode in episodes], dtype=bool),
        truncated=np.asarray([episode["truncated"] for episode in episodes], dtype=bool),
        observation_mask=observation_mask, transition_mask=transition_mask,
        observations=raw_observations, joint_names=np.asarray(joint_names),
        joint_angles=raw_angles, joint_velocities=raw_rates, latents=raw_latents,
        actions=raw_actions, rewards=raw_rewards,
    )

    chart = _optional_plot(args.out, joint_names, episodes)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_config": config,
        "device": str(device),
        "environment": ENV_ID,
        "L": L,
        "target_gait_period": TARGET_GAIT_PERIOD,
        "rollout_cap": MAX_STEPS,
        "discard_initial_steps": SETTLE_STEPS,
        "reset_seeds": RESET_SEEDS,
        "target_tolerance": TARGET_TOLERANCE,
        "required_matches": PASS_ROLLOUTS,
        "matches": int(matches),
        "passes": bool(matches >= PASS_ROLLOUTS),
        "motor_joint_names": joint_names,
        "latent_geometry_summary": geometry_summary,
        "rollouts": summaries,
        "raw_data": "ant_l20_rollouts.npz",
        "chart": chart,
    }
    (args.out / "ant_l20_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(
        f"Ant L=20 target gait period={TARGET_GAIT_PERIOD}: "
        f"{matches}/{len(episodes)} rollouts passed (required {PASS_ROLLOUTS}/5)."
    )
    for episode in summaries:
        gait = episode["gait"]
        print(
            f"seed={episode['seed']} steps={episode['length']} "
            f"shortest_repeat={gait['shortest_repeat']} passed={gait['passed']} "
            f"reason={gait['reason']}"
        )
    print(f"Results: {args.out / 'ant_l20_evaluation.json'}")


if __name__ == "__main__":
    main()
