"""Train state-based Walker2D PSD skills at the four Figure 5 periods."""

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from sac import SAC
from train import EMBED_DIM, condition, period_embedding, psd_reward
from train_ant import EpisodeReplay, write_checkpoint


ENV_ID = "Walker2d-v5"
HORIZON = 400
PERIODS = (20, 26, 33, 40)
LATENT_DIM = 3
VELOCITY_TARGET = 1.0
EPISODES_PER_EPOCH = 8
UPDATES_PER_EPOCH = 64
REPLAY_TRANSITIONS = 500_000
ENCODER_BATCH_SIZE = 1024
POLICY_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
PSD_LAMBDA_LAG = 10.0
PSD_LAMBDA_STEP = 10.0


def parse_periods(value):
    try:
        periods = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("periods must be comma-separated integers") from exc
    if (not periods or any(period <= 0 or period >= HORIZON for period in periods)
            or len(set(periods)) != len(periods)):
        raise argparse.ArgumentTypeError(
            f"periods must be distinct integers between 1 and {HORIZON - 1}"
        )
    return periods


def chord(period):
    return period * torch.sin(torch.pi / (2 * period))


def walker_psd_objective(z, z_next, z_lag, period):
    """PSD encoder objective with the paper's Walker2D lambda1=lambda2=10."""
    one_step = (z_next - z).norm(dim=-1)
    lag = (z_lag - z).norm(dim=-1)
    center = (z_lag + z).norm(dim=-1)
    lag_slack = (period - lag).clamp(max=1e-5)
    step_slack = (chord(period) - one_step).clamp(max=1e-5)
    return (
        lag - 0.5 * center
        + PSD_LAMBDA_LAG * lag_slack
        + PSD_LAMBDA_STEP * step_slack
    ).mean()


def capped_velocity_reward(velocity):
    return torch.clamp(velocity / VELOCITY_TARGET, max=1.0)


def collect_episodes(envs, agent, rng, device, periods,
                     reward_mode="intrinsic_only"):
    """Collect at most 400 steps per env, preserving terminations for SAC."""
    seeds = rng.integers(0, 2**31 - 1, size=len(envs))
    states = [env.reset(seed=int(seed))[0] for env, seed in zip(envs, seeds)]
    period_values = rng.choice(np.asarray(periods, dtype=np.int64), size=len(envs))
    period_batch = torch.as_tensor(period_values, dtype=torch.long, device=device)
    period_features = period_embedding(period_batch)
    capture_velocity = reward_mode == "intrinsic_velocity"
    histories = [
        {"states": [state], "actions": [], "terminals": [],
         **({"velocities": []} if capture_velocity else {})}
        for state in states
    ]
    active = np.ones(len(envs), dtype=bool)

    for _ in range(HORIZON):
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            break
        observations = torch.as_tensor(
            np.stack([states[index] for index in active_indices]),
            dtype=torch.float32,
            device=device,
        )
        actions = agent.act(torch.cat(
            (observations, period_features[active_indices]), dim=-1
        )).cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("The Walker2D policy produced a nonfinite action.")
        for local_index, env_index in enumerate(active_indices):
            next_state, _, terminated, truncated, _ = envs[env_index].step(
                actions[local_index]
            )
            histories[env_index]["actions"].append(actions[local_index].copy())
            if capture_velocity:
                # Walker2d has free-root translation in qvel[0].
                histories[env_index]["velocities"].append(
                    float(envs[env_index].unwrapped.data.qvel[0])
                )
            # A Walker fall ends bootstrapping; the 400-step time limit does not.
            histories[env_index]["terminals"].append(float(terminated))
            histories[env_index]["states"].append(next_state)
            states[env_index] = next_state
            if terminated or truncated:
                active[env_index] = False

    episodes = []
    for index, history in enumerate(histories):
        episode = {
            "states": np.asarray(history["states"], dtype=np.float32),
            "actions": np.asarray(history["actions"], dtype=np.float32),
            "terminals": np.asarray(history["terminals"], dtype=np.float32),
            "period": np.asarray(int(period_values[index]), dtype=np.int64),
        }
        if capture_velocity:
            episode["velocities"] = np.asarray(
                history["velocities"], dtype=np.float32
            )
        if len(episode["actions"]) != len(episode["states"]) - 1:
            raise RuntimeError("Collected Walker2D episode has inconsistent transition lengths.")
        if len(episode["actions"]) == 0:
            raise RuntimeError("Walker2D terminated before producing a transition.")
        if not all(np.isfinite(array).all() for array in episode.values()):
            raise FloatingPointError("Walker2D produced a nonfinite trajectory.")
        episodes.append(episode)
    return episodes


def _encoder_eligible(replay, periods):
    return any(
        int(episode["period"]) in periods
        and len(episode["states"]) > int(episode["period"])
        for episode in replay.episodes
    )


def train_epoch(envs, agent, encoder, optimizer, replay, rng, device, periods,
                reward_mode="intrinsic_only"):
    collected = collect_episodes(
        envs, agent, rng, device, periods, reward_mode=reward_mode
    )
    for episode in collected:
        replay.add(episode)

    encoder_losses = []
    if _encoder_eligible(replay, periods):
        for _ in range(UPDATES_PER_EPOCH):
            s, s_next, s_lag, sampled_periods = replay.sample_encoder(
                rng, ENCODER_BATCH_SIZE, periods=periods
            )
            s, s_next, s_lag = (
                torch.as_tensor(array, dtype=torch.float32, device=device)
                for array in (s, s_next, s_lag)
            )
            period_batch = torch.as_tensor(
                sampled_periods, dtype=torch.long, device=device
            )
            encoded = encoder(condition(
                torch.cat((s, s_next, s_lag)), period_batch.repeat(3)
            ))
            z, z_next, z_lag = encoded.chunk(3)
            loss = -walker_psd_objective(z, z_next, z_lag, period_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            encoder_losses.append(loss.detach())

    sac_metrics = []
    for _ in range(UPDATES_PER_EPOCH):
        policy_sample = replay.sample_policy(
            rng, POLICY_BATCH_SIZE, periods=periods,
            with_velocity=(reward_mode == "intrinsic_velocity"),
        )
        s, action, s_next, terminal = (
            torch.as_tensor(array, dtype=torch.float32, device=device)
            for array in policy_sample[:4]
        )
        sampled_periods = policy_sample[4]
        period_batch = torch.as_tensor(
            sampled_periods, dtype=torch.long, device=device
        )
        x = condition(s, period_batch)
        next_x = condition(s_next, period_batch)
        with torch.no_grad():
            reward = psd_reward(encoder(x), encoder(next_x), period_batch)
            if reward_mode == "intrinsic_velocity":
                velocity = torch.as_tensor(
                    policy_sample[5],
                    dtype=torch.float32, device=device,
                )
                reward = reward + capped_velocity_reward(velocity)
        metrics = agent.update(x, action, next_x, reward[:, None], terminal[:, None])
        sac_metrics.append(torch.stack(tuple(metrics.values())))

    if sac_metrics:
        metric_columns = tuple(metrics)
        sac_values = torch.stack(sac_metrics)
        if not torch.isfinite(sac_values).all().item():
            raise FloatingPointError("A Walker2D SAC update produced nonfinite metrics.")
        last_metrics = dict(zip(metric_columns, sac_values[-1].tolist()))
    else:
        last_metrics = {}
    if encoder_losses:
        encoder_values = torch.stack(encoder_losses)
        if not torch.isfinite(encoder_values).all().item():
            raise FloatingPointError("A Walker2D encoder update produced nonfinite loss.")
        last_encoder_loss = float(encoder_values[-1].item())
    else:
        last_encoder_loss = None

    lengths = [len(episode["actions"]) for episode in collected]
    with torch.no_grad():
        episode_rewards = []
        for episode in collected:
            episode_period = int(episode["period"])
            period_steps = torch.full(
                (len(episode["actions"]),), episode_period,
                dtype=torch.long, device=device,
            )
            episode_states = torch.as_tensor(
                episode["states"], dtype=torch.float32, device=device
            )
            episode_rewards.append(psd_reward(
                encoder(condition(episode_states[:-1], period_steps)),
                encoder(condition(episode_states[1:], period_steps)),
                period_steps,
            ).mean().item())

    period_episodes = {
        str(period): sum(int(ep["period"]) == period for ep in collected)
        for period in periods
    }
    collection_metrics = {
        "mean_episode_steps": float(np.mean(lengths)),
        "environment_steps": int(sum(lengths)),
        "terminated_episodes": sum(bool(ep["terminals"][-1]) for ep in collected),
        "mean_intrinsic_reward": float(np.mean(episode_rewards)),
        "period_episodes": period_episodes,
    }
    if reward_mode == "intrinsic_velocity":
        collection_metrics["mean_x_velocity"] = float(np.mean([
            episode["velocities"].mean() for episode in collected
        ]))
        collection_metrics["mean_forward_reward"] = float(np.mean([
            capped_velocity_reward(torch.as_tensor(
                episode["velocities"], dtype=torch.float32
            )).mean().item()
            for episode in collected
        ]))
    metrics = {
        "encoder_loss": last_encoder_loss,
        "encoder_updates": len(encoder_losses),
        "encoder_updates_skipped": UPDATES_PER_EPOCH - len(encoder_losses),
        **last_metrics,
    }
    return metrics, collection_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, required=True,
                        help="Total epochs; each collects eight Walker2D episodes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--periods", type=parse_periods, default=PERIODS,
                        metavar="L,...", help="Periods sampled uniformly per episode.")
    parser.add_argument(
        "--reward-mode", choices=("intrinsic_only", "intrinsic_velocity"),
        default="intrinsic_only",
        help="intrinsic_velocity adds the paper's capped Walker forward-speed reward.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("Use at least one epoch and a nonnegative seed.")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "checkpoint.pt"
    if checkpoint_path.exists() != args.resume:
        parser.error("Use an unused output directory, or --resume with an existing checkpoint.")

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(6, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)
    envs = [gym.make(ENV_ID, max_episode_steps=HORIZON)
            for _ in range(EPISODES_PER_EPOCH)]
    try:
        obs_dim = envs[0].observation_space.shape[0]
        action_space = envs[0].action_space
        agent = SAC(obs_dim + EMBED_DIM, action_space, device)
        encoder = nn.Sequential(
            nn.Linear(obs_dim + EMBED_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, LATENT_DIM),
        ).to(device)
        optimizer = torch.optim.Adam(
            encoder.parameters(), lr=LEARNING_RATE, fused=device.type == "cuda"
        )
        replay = EpisodeReplay(max_transitions=REPLAY_TRANSITIONS)
        config = {
            "env": ENV_ID,
            "seed": args.seed,
            "periods": list(args.periods),
            "period_sampling": "uniform_per_episode",
            "embed_dim": EMBED_DIM,
            "latent_dim": LATENT_DIM,
            "horizon": HORIZON,
            "observation": "Gymnasium default Walker2d-v5 observation",
            "reward": args.reward_mode,
            "device_type": device.type,
            "psd_lambda_lag": PSD_LAMBDA_LAG,
            "psd_lambda_step": PSD_LAMBDA_STEP,
            "episodes_per_epoch": EPISODES_PER_EPOCH,
            "updates_per_epoch": UPDATES_PER_EPOCH,
            "encoder_batch_size": ENCODER_BATCH_SIZE,
            "policy_batch_size": POLICY_BATCH_SIZE,
        }
        if args.reward_mode == "intrinsic_velocity":
            config["forward_velocity_target"] = VELOCITY_TARGET
            config["reward_weights"] = {"psd": 1.0, "forward": 1.0}
            config["forward_velocity_source"] = "post-step MuJoCo qvel[0]"
        previous_epoch = 0
        previous_elapsed = 0.0
        history = []
        if args.resume:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if config != saved["config"] or args.epochs <= int(saved["epoch"]):
                parser.error("Resume with original configuration and a higher total epoch limit.")
            previous_epoch = int(saved["epoch"])
            previous_elapsed = float(saved["elapsed_seconds"])
            agent.load_state_dict(saved["agent"])
            encoder.load_state_dict(saved["encoder"])
            optimizer.load_state_dict(saved["optimizer"])
            replay.load_state_dict(saved["replay"])
            rng.bit_generator.state = saved["rng"]
            eval_rng.bit_generator.state = saved["eval_rng"]
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state(saved["cuda_rng"], device)
            history = saved["history"]
            del saved

        started = time.perf_counter()
        for epoch in range(previous_epoch + 1, args.epochs + 1):
            metrics, collection = train_epoch(
                envs, agent, encoder, optimizer, replay, rng, device, args.periods,
                reward_mode=args.reward_mode,
            )
            row = {
                "epoch": epoch,
                "episodes": epoch * EPISODES_PER_EPOCH,
                "transitions": replay.transitions,
                **metrics,
                **collection,
            }
            numeric_values = [
                value for value in row.values() if isinstance(value, (int, float))
            ]
            if not all(math.isfinite(value) for value in numeric_values):
                raise FloatingPointError("A Walker2D training metric became nonfinite.")
            history.append(row)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                forward_metrics = (
                    f" r_v={row['mean_forward_reward']:.3f}"
                    f" vx={row['mean_x_velocity']:.3f}"
                    if "mean_forward_reward" in row else ""
                )
                enc_loss = ("none" if row["encoder_loss"] is None
                            else f"{row['encoder_loss']:.3f}")
                print(
                    f"epoch={epoch} episodes={row['episodes']} "
                    f"steps={row['mean_episode_steps']:.1f} "
                    f"falls={row['terminated_episodes']} "
                    f"encoder_loss={enc_loss} "
                    f"enc_updates={row['encoder_updates']} "
                    f"skipped={row['encoder_updates_skipped']} "
                    f"alpha={row.get('alpha', float('nan')):.3f} "
                    f"r_psd={row['mean_intrinsic_reward']:.3f}{forward_metrics} "
                    f"elapsed_s={previous_elapsed + time.perf_counter() - started:.1f}",
                    flush=True,
                )
            if epoch % 250 == 0 or epoch == args.epochs:
                elapsed = previous_elapsed + time.perf_counter() - started
                write_checkpoint(
                    checkpoint_path, epoch=epoch, elapsed=elapsed, config=config,
                    agent=agent, encoder=encoder, optimizer=optimizer,
                    replay=replay, rng=rng, eval_rng=eval_rng, history=history,
                )
                progress = {
                    "config": config, "epoch": epoch,
                    "episodes": row["episodes"], "elapsed_seconds": elapsed,
                    "last_batch": metrics, "last_collection": collection,
                }
                progress_tmp = args.out / "progress.json.tmp"
                progress_tmp.write_text(
                    json.dumps(progress, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                progress_tmp.replace(args.out / "progress.json")

        elapsed = previous_elapsed + time.perf_counter() - started
        report = {
            "config": config,
            "epoch": args.epochs,
            "episodes": args.epochs * EPISODES_PER_EPOCH,
            "environment_steps": sum(row["environment_steps"] for row in history),
            "elapsed_seconds": elapsed,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("torch", "gymnasium", "mujoco", "numpy")
            },
            "history": history,
        }
        summary_path = args.out / "summary.json"
        summary_tmp = summary_path.with_suffix(".json.tmp")
        summary_tmp.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        summary_tmp.replace(summary_path)
        print(f"Wrote {summary_path}", flush=True)
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
