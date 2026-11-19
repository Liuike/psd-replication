"""Train Ant PSD skills at L=20 or optional per-episode periods."""

import argparse
import importlib.metadata
import json
import math
import platform
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from sac import SAC
from train import EMBED_DIM, condition, period_embedding, psd_objective, psd_reward


ENV_ID = "Ant-v5"
HORIZON = 200
PERIOD = 20
LATENT_DIM = 6
EPISODES_PER_EPOCH = 8
UPDATES_PER_EPOCH = 64
REPLAY_TRANSITIONS = 500_000
ENCODER_BATCH_SIZE = 1024
POLICY_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
VELOCITY_TARGET = 0.5


def parse_periods(value):
    """Parse a comma-separated list of positive periods for optional training."""
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


def capped_velocity_reward(velocity):
    """Paper's capped Ant x-velocity term, with target forward speed 0.5 m/s."""
    return torch.clamp(velocity / VELOCITY_TARGET, max=1.0)


class EpisodeReplay:
    """Keep whole episodes so lagged encoder samples never cross a reset."""

    def __init__(self, max_transitions=REPLAY_TRANSITIONS):
        self.max_transitions = max_transitions
        self.episodes = deque()
        self.transitions = 0

    def add(self, episode):
        self.episodes.append(episode)
        self.transitions += len(episode["actions"])
        while self.transitions > self.max_transitions and len(self.episodes) > 1:
            oldest = self.episodes.popleft()
            self.transitions -= len(oldest["actions"])

    def sample_encoder(self, rng, batch_size, period=PERIOD, periods=None):
        # Keep the fixed-L path, including its RNG calls and returned schema,
        # identical for the existing L=20 checkpoint and default experiment.
        if periods is not None:
            allowed = set(periods)
            eligible = [
                episode for episode in self.episodes
                if int(episode.get("period", PERIOD)) in allowed
                and len(episode["states"]) > int(episode.get("period", PERIOD))
            ]
            if not eligible:
                raise RuntimeError(
                    "Replay has no episode with enough states for the requested periods."
                )
            episode_periods = np.asarray(
                [int(episode.get("period", PERIOD)) for episode in eligible], dtype=np.int64
            )
            valid_starts = np.asarray([
                len(episode["states"]) - item_period
                for episode, item_period in zip(eligible, episode_periods)
            ], dtype=np.int64)
            probabilities = valid_starts / valid_starts.sum()
            indices = rng.choice(len(eligible), size=batch_size, p=probabilities)
            starts = rng.integers(valid_starts[indices])
            sampled_periods = episode_periods[indices]
            states = np.stack([
                eligible[index]["states"][start] for index, start in zip(indices, starts)
            ])
            states_next = np.stack([
                eligible[index]["states"][start + 1]
                for index, start in zip(indices, starts)
            ])
            states_lag = np.stack([
                eligible[index]["states"][start + item_period]
                for index, start, item_period in zip(indices, starts, sampled_periods)
            ])
            return states, states_next, states_lag, sampled_periods

        eligible = [episode for episode in self.episodes
                    if len(episode["states"]) > period]
        if not eligible:
            raise RuntimeError(
                f"Replay has no episode with enough states for the {period}-step encoder objective."
            )
        valid_starts = np.asarray(
            [len(episode["states"]) - period for episode in eligible], dtype=np.int64
        )
        probabilities = valid_starts / valid_starts.sum()
        indices = rng.choice(len(eligible), size=batch_size, p=probabilities)
        starts = rng.integers(valid_starts[indices])
        states = np.stack([eligible[index]["states"][start] for index, start in zip(indices, starts)])
        states_next = np.stack([
            eligible[index]["states"][start + 1] for index, start in zip(indices, starts)
        ])
        states_lag = np.stack([
            eligible[index]["states"][start + period] for index, start in zip(indices, starts)
        ])
        return states, states_next, states_lag

    def sample_policy(self, rng, batch_size, periods=None, with_velocity=False):
        if not self.episodes:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        episode_list = list(self.episodes)
        if periods is not None:
            allowed = set(periods)
            episode_list = [
                episode for episode in episode_list
                if int(episode.get("period", PERIOD)) in allowed
            ]
            if not episode_list:
                raise RuntimeError("Replay has no episode for the requested periods.")
        lengths = np.asarray([len(episode["actions"]) for episode in episode_list], dtype=np.int64)
        probabilities = lengths / lengths.sum()
        indices = rng.choice(len(episode_list), size=batch_size, p=probabilities)
        times = rng.integers(lengths[indices])
        states = np.stack([episode_list[index]["states"][time] for index, time in zip(indices, times)])
        states_next = np.stack([
            episode_list[index]["states"][time + 1] for index, time in zip(indices, times)
        ])
        actions = np.stack([
            episode_list[index]["actions"][time] for index, time in zip(indices, times)
        ])
        terminals = np.asarray([
            episode_list[index]["terminals"][time] for index, time in zip(indices, times)
        ], dtype=np.float32)
        if periods is not None:
            sampled_periods = np.asarray([
                int(episode_list[index].get("period", PERIOD)) for index in indices
            ], dtype=np.int64)
            result = (states, actions, states_next, terminals, sampled_periods)
        else:
            result = (states, actions, states_next, terminals)
        if with_velocity:
            velocities = np.asarray([
                episode_list[index]["velocities"][time]
                for index, time in zip(indices, times)
            ], dtype=np.float32)
            result += (velocities,)
        return result

    def state_dict(self):
        return {
            "max_transitions": self.max_transitions,
            "episodes": [
                {key: torch.from_numpy(array) for key, array in episode.items()}
                for episode in self.episodes
            ],
            "transitions": self.transitions,
        }

    def load_state_dict(self, saved):
        self.max_transitions = int(saved["max_transitions"])
        self.episodes = deque(
            {key: tensor.numpy() for key, tensor in episode.items()}
            for episode in saved["episodes"]
        )
        self.transitions = int(saved["transitions"])


def collect_episodes(envs, agent, rng, device, period_values=None,
                    reward_mode="intrinsic_only"):
    seeds = rng.integers(0, 2**31 - 1, size=len(envs))
    states = [env.reset(seed=int(seed))[0] for env, seed in zip(envs, seeds)]
    capture_velocity = reward_mode == "intrinsic_velocity"
    histories = [
        {"states": [state], "actions": [], "terminals": [],
         **({"velocities": []} if capture_velocity else {})}
        for state in states
    ]
    if period_values is None:
        period_batch = torch.full((len(envs),), PERIOD, dtype=torch.long, device=device)
    else:
        period_batch = torch.as_tensor(period_values, dtype=torch.long, device=device)
        if period_batch.shape != (len(envs),):
            raise ValueError("Each collected episode must have exactly one period value.")
    period_features = period_embedding(period_batch)
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
        actions = agent.act(
            torch.cat((observations, period_features[active_indices]), dim=-1)
        ).cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("The Ant policy produced a nonfinite action.")
        for local_index, env_index in enumerate(active_indices):
            next_state, _, terminated, truncated, _ = envs[env_index].step(actions[local_index])
            histories[env_index]["actions"].append(actions[local_index].copy())
            if capture_velocity:
                # Ant's free-root generalized velocity starts with world x speed.
                velocity = float(envs[env_index].unwrapped.data.qvel[0])
                histories[env_index]["velocities"].append(velocity)
            # The 200-step time limit does not stop SAC bootstrapping.
            histories[env_index]["terminals"].append(float(terminated))
            histories[env_index]["states"].append(next_state)
            states[env_index] = next_state
            if terminated or truncated:
                active[env_index] = False

    episodes = []
    for history in histories:
        episode = {
            "states": np.asarray(history["states"], dtype=np.float32),
            "actions": np.asarray(history["actions"], dtype=np.float32),
            "terminals": np.asarray(history["terminals"], dtype=np.float32),
        }
        if period_values is not None:
            episode["period"] = np.asarray(
                int(period_values[len(episodes)]), dtype=np.int64
            )
        if capture_velocity:
            episode["velocities"] = np.asarray(history["velocities"], dtype=np.float32)
        if len(episode["actions"]) != len(episode["states"]) - 1:
            raise RuntimeError("Collected Ant episode has inconsistent transition lengths.")
        if len(episode["actions"]) == 0:
            raise RuntimeError("Ant terminated before producing a transition.")
        if not all(np.isfinite(array).all() for array in episode.values()):
            raise FloatingPointError("Ant produced a nonfinite trajectory.")
        episodes.append(episode)
    return episodes


def train_epoch(envs, agent, encoder, optimizer, replay, rng, device,
                periods=None, reward_mode="intrinsic_only"):
    if periods is None:
        # Do not consume an extra random draw in the legacy fixed-L run.
        sampled_periods = None
        collected = collect_episodes(
            envs, agent, rng, device, reward_mode=reward_mode
        )
    else:
        sampled_periods = rng.choice(np.asarray(periods, dtype=np.int64), size=len(envs))
        collected = collect_episodes(
            envs, agent, rng, device, period_values=sampled_periods,
            reward_mode=reward_mode,
        )
    for episode in collected:
        replay.add(episode)

    encoder_losses = []
    for _ in range(UPDATES_PER_EPOCH):
        encoder_sample = replay.sample_encoder(
            rng, ENCODER_BATCH_SIZE, periods=periods
        )
        s, s_next, s_lag = (
            torch.as_tensor(array, dtype=torch.float32, device=device)
            for array in encoder_sample[:3]
        )
        if periods is None:
            period_batch = torch.full((len(s),), PERIOD, dtype=torch.long, device=device)
        else:
            period_batch = torch.as_tensor(encoder_sample[3], dtype=torch.long, device=device)
        encoded = encoder(condition(torch.cat((s, s_next, s_lag)), period_batch.repeat(3)))
        z, z_next, z_lag = encoded.chunk(3)
        loss = -psd_objective(z, z_next, z_lag, period_batch)
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
        if periods is None:
            period_batch = torch.full((len(s),), PERIOD, dtype=torch.long, device=device)
        else:
            period_batch = torch.as_tensor(policy_sample[4], dtype=torch.long, device=device)
        x = condition(s, period_batch)
        next_x = condition(s_next, period_batch)
        with torch.no_grad():
            reward = psd_reward(encoder(x), encoder(next_x), period_batch)
            if reward_mode == "intrinsic_velocity":
                velocity = torch.as_tensor(
                    policy_sample[-1], dtype=torch.float32, device=device
                )
                reward = reward + capped_velocity_reward(velocity)
        metrics = agent.update(
            x, action, next_x, reward[:, None], terminal[:, None]
        )
        sac_metrics.append(torch.stack(tuple(metrics.values())))

    update_values = torch.cat(
        (torch.stack(encoder_losses)[:, None], torch.stack(sac_metrics)), dim=1
    )
    if not torch.isfinite(update_values).all().item():
        raise FloatingPointError("An Ant encoder or SAC update produced nonfinite metrics.")

    lengths = [len(episode["actions"]) for episode in collected]
    terminations = sum(bool(episode["terminals"][-1]) for episode in collected)
    metrics = dict(zip(("encoder_loss", *metrics), update_values[-1].tolist()))
    with torch.no_grad():
        mean_intrinsic_reward = float(np.mean([
            psd_reward(
                encoder(condition(
                    torch.as_tensor(ep["states"][:-1], dtype=torch.float32, device=device),
                    torch.full((len(ep["actions"]),),
                               int(ep.get("period", PERIOD)),
                               dtype=torch.long, device=device),
                )),
                encoder(condition(
                    torch.as_tensor(ep["states"][1:], dtype=torch.float32, device=device),
                    torch.full((len(ep["actions"]),),
                               int(ep.get("period", PERIOD)),
                               dtype=torch.long, device=device),
                )),
                torch.full((len(ep["actions"]),),
                           int(ep.get("period", PERIOD)),
                           dtype=torch.long, device=device),
            ).mean().item()
            for ep in collected
        ]))
    collection_metrics = {
        "mean_episode_steps": float(np.mean(lengths)),
        "environment_steps": int(sum(lengths)),
        "terminated_episodes": int(terminations),
        "mean_intrinsic_reward": mean_intrinsic_reward,
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
    if periods is not None:
        collection_metrics["period_episodes"] = {
            str(period): sum(int(ep["period"]) == period for ep in collected)
            for period in periods
        }
    return metrics, collection_metrics


def write_checkpoint(path, *, epoch, elapsed, config, agent, encoder,
                     optimizer, replay, rng, eval_rng, history):
    temporary_path = path.with_suffix(".pt.tmp")
    torch.save({
        "epoch": epoch,
        "elapsed_seconds": elapsed,
        "config": config,
        "agent": agent.state_dict(),
        "encoder": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay": replay.state_dict(),
        "rng": rng.bit_generator.state,
        "eval_rng": eval_rng.bit_generator.state,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, temporary_path)
    temporary_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, required=True,
                        help="Total epoch count. Each epoch collects eight Ant episodes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--periods", type=parse_periods, default=None, metavar="L,...",
        help=("Optional comma-separated set of periods. One value is sampled uniformly "
              "per episode and held fixed for that episode; omitted means fixed L=20."),
    )
    parser.add_argument(
        "--reward-mode", choices=("intrinsic_only", "intrinsic_velocity"),
        default="intrinsic_only",
        help=("intrinsic_velocity adds the paper's capped forward-speed reward "
              "(target 0.5 m/s) to PSD reward."),
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("Use at least one epoch and a nonnegative seed.")
    if args.periods == (PERIOD,):
        # Treat the explicit fixed-L spelling as the legacy mode too.
        args.periods = None
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

    envs = [gym.make(ENV_ID, max_episode_steps=HORIZON) for _ in range(EPISODES_PER_EPOCH)]
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
        replay = EpisodeReplay()
        config = {
            "env": ENV_ID,
            "seed": args.seed,
            "L": PERIOD,
            "latent_dim": LATENT_DIM,
            "horizon": HORIZON,
            "reward": "intrinsic_only",
            "device_type": device.type,
        }
        if args.periods is not None:
            # Multi-period checkpoints record their complete sampling contract.
            config.pop("L")
            config["periods"] = list(args.periods)
            config["period_sampling"] = "uniform_per_episode"
        if args.reward_mode == "intrinsic_velocity":
            config["reward"] = args.reward_mode
            config["forward_velocity_target"] = VELOCITY_TARGET
            config["reward_weights"] = {"psd": 1.0, "forward": 1.0}
            config["forward_velocity_source"] = "post-step MuJoCo qvel[0]"
        previous_epoch = 0
        previous_elapsed = 0.0
        history = []
        if args.resume:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if config != saved["config"] or args.epochs <= int(saved["epoch"]):
                parser.error("Resume with the original configuration and a higher total epoch limit.")
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
                envs, agent, encoder, optimizer, replay, rng, device,
                periods=args.periods, reward_mode=args.reward_mode,
            )
            row = {
                "epoch": epoch,
                "episodes": epoch * EPISODES_PER_EPOCH,
                "transitions": replay.transitions,
                **metrics,
                **collection,
            }
            if not all(math.isfinite(float(value)) for value in row.values()
                       if isinstance(value, (int, float))):
                raise FloatingPointError("An Ant training metric became nonfinite.")
            history.append(row)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                forward_reward = (
                    f" r_v={collection['mean_forward_reward']:.3f}"
                    f" vx={collection['mean_x_velocity']:.3f}"
                    if "mean_forward_reward" in collection else ""
                )
                print(
                    f"epoch={epoch} episodes={row['episodes']} "
                    f"steps={row['mean_episode_steps']:.1f} "
                    f"falls={row['terminated_episodes']} "
                    f"encoder_loss={row['encoder_loss']:.3f} "
                    f"alpha={row['alpha']:.3f} "
                    f"r_psd={row['mean_intrinsic_reward']:.3f}{forward_reward}",
                    flush=True,
                )
            if epoch % 250 == 0 or epoch == args.epochs:
                elapsed = previous_elapsed + time.perf_counter() - started
                write_checkpoint(
                    checkpoint_path, epoch=epoch, elapsed=elapsed, config=config,
                    agent=agent, encoder=encoder, optimizer=optimizer,
                    replay=replay, rng=rng, eval_rng=eval_rng,
                    history=history,
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
