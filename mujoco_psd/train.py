"""State-based HalfCheetah replication of PSD with adaptive sampling."""

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from evaluation import geometry, spectrum
from sac import SAC


ENV_ID = "HalfCheetah-v5"
HORIZON = 200
EPISODES_PER_EPOCH = 8
UPDATES_PER_EPOCH = 64
REPLAY_CAPACITY = 500_000
EMBED_DIM = 8
ADAPT_EVAL_EPISODES = 5
L_FLOOR = 5


def period_embedding(period):
    exponent = torch.arange(0, EMBED_DIM, 2, device=period.device) / EMBED_DIM
    angle = period.float().unsqueeze(-1) * 10000 ** -exponent
    return torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)


def condition(state, period):
    return torch.cat((state, period_embedding(period)), dim=-1)


def chord(period):
    return period * torch.sin(torch.pi / (2 * period))


def psd_objective(z, z_next, z_lag, period):
    one_step = (z_next - z).norm(dim=-1)
    lag = (z_lag - z).norm(dim=-1)
    center = (z_lag + z).norm(dim=-1)
    lag_slack = (period - lag).clamp(max=1e-5)
    step_slack = (chord(period) - one_step).clamp(max=1e-5)
    return (lag - 0.5 * center + 5.0 * lag_slack + 5.0 * step_slack).mean()


def psd_reward(z, z_next, period):
    distance = (z_next - z).norm(dim=-1)
    return torch.exp(-10.0 * (distance - chord(period)).square())


@dataclass
class Bounds:
    low: int
    high: int
    expanded_low: bool = False
    expanded_high: bool = False

    def candidates(self):
        return np.unique(np.linspace(self.low, self.high, 4, dtype=int))

    def update(self, low_score, high_score):
        # Scores are cumulative intrinsic returns divided by the horizon.
        if low_score > 0.9 and self.low > L_FLOOR:
            self.low -= 1
            self.expanded_low = True
        elif low_score < 0.4 and self.expanded_low:
            self.low += 1

        if high_score > 0.9 and self.high < HORIZON - 1:
            self.high += 1
            self.expanded_high = True
        elif high_score < 0.4 and self.expanded_high:
            self.high -= 1

        if self.low > self.high:
            self.low = self.high


class EpisodeReplay:
    """Store complete episodes so sampled temporal triples never cross resets."""

    def __init__(self, obs_dim, action_dim):
        capacity = REPLAY_CAPACITY // HORIZON
        self.state = np.empty((capacity, HORIZON + 1, obs_dim), dtype=np.float32)
        self.action = np.empty((capacity, HORIZON, action_dim), dtype=np.float32)
        self.velocity = np.empty((capacity, HORIZON), dtype=np.float32)
        self.period = np.empty(capacity, dtype=np.int64)
        self.count = 0
        self.cursor = 0

    def add(self, states, actions, velocity, periods):
        indices = (self.cursor + np.arange(len(periods))) % len(self.period)
        self.state[indices] = states
        self.action[indices] = actions
        self.velocity[indices] = velocity
        self.period[indices] = periods
        self.cursor = (self.cursor + len(periods)) % len(self.period)
        self.count = min(self.count + len(periods), len(self.period))

    def sample(self, rng, size, *, temporal=False, period=None):
        allowed = np.arange(self.count) if period is None else np.flatnonzero(
            self.period[:self.count] == period
        )
        if not len(allowed):
            raise ValueError(f"No trajectories for L={period}")
        if temporal:
            # Weight episodes by the number of valid starting points before
            # sampling t, to give each available triple the same probability.
            lengths = HORIZON + 1 - self.period[allowed]
            episodes = rng.choice(allowed, size, p=lengths / lengths.sum())
            t = rng.integers(HORIZON + 1 - self.period[episodes])
            return (
                self.state[episodes, t], self.state[episodes, t + 1],
                self.state[episodes, t + self.period[episodes]], self.period[episodes],
            )

        episodes = rng.choice(allowed, size)
        t = rng.integers(HORIZON, size=size)
        return (
            self.state[episodes, t], self.state[episodes, t + 1],
            self.action[episodes, t], self.velocity[episodes, t], self.period[episodes],
        )

    def state_dict(self):
        return {
            "count": self.count, "cursor": self.cursor,
            **{key: torch.from_numpy(getattr(self, key)[:self.count])
               for key in ("state", "action", "velocity", "period")},
        }

    def load_state_dict(self, saved):
        self.count = saved["count"]
        self.cursor = saved["cursor"]
        for key in ("state", "action", "velocity", "period"):
            getattr(self, key)[:self.count] = saved[key].numpy()


@torch.no_grad()
def rollouts(envs, agent, periods, rng, device, *, deterministic=False, reset_seeds=None):
    seeds = rng.integers(2**31, size=len(envs)) if reset_seeds is None else reset_seeds
    if len(seeds) != len(envs):
        raise ValueError("Pass one reset seed per environment.")
    start = [env.reset(seed=int(seed))[0] for env, seed in zip(envs, seeds)]
    states = np.empty((len(envs), HORIZON + 1, len(start[0])), dtype=np.float32)
    actions = np.empty((len(envs), HORIZON, *envs[0].action_space.shape), dtype=np.float32)
    velocity = np.empty((len(envs), HORIZON), dtype=np.float32)
    states[:, 0] = start
    embed = period_embedding(torch.as_tensor(periods, device=device))
    for t in range(HORIZON):
        obs = torch.as_tensor(states[:, t], device=device)
        actions[:, t] = agent.act(torch.cat((obs, embed), dim=-1), deterministic).cpu().numpy()
        if not np.isfinite(actions[:, t]).all():
            raise FloatingPointError("The policy produced a nonfinite action.")
        for i, env in enumerate(envs):
            state, _, terminated, truncated, info = env.step(actions[i, t])
            if terminated or truncated != (t == HORIZON - 1):
                raise RuntimeError("Expected the standard fixed-horizon HalfCheetah rollout.")
            states[i, t + 1] = state
            velocity[i, t] = info["x_velocity"]
    if not np.isfinite(states).all() or not np.isfinite(velocity).all():
        raise FloatingPointError("HalfCheetah produced a nonfinite state or velocity.")
    return states, actions, velocity


@torch.no_grad()
def evaluate_boundary(envs, agent, encoder, period, rng, device):
    periods = np.full(ADAPT_EVAL_EPISODES, period)
    states, _, _ = rollouts(envs[:ADAPT_EVAL_EPISODES], agent, periods, rng, device,
                            deterministic=True)
    obs = torch.as_tensor(states, device=device)
    L = torch.full(obs.shape[:-1], period, device=device)
    z = encoder(condition(obs, L))
    return psd_reward(z[:, :-1], z[:, 1:], L[:, :-1]).mean().item()


def train_epoch(envs, agent, encoder, optimizer, replay, bounds, rng, device):
    periods = rng.choice(bounds.candidates(), size=EPISODES_PER_EPOCH)
    states, actions, velocity = rollouts(envs, agent, periods, rng, device)
    replay.add(states, actions, velocity, periods)
    encoder_losses = []
    for _ in range(UPDATES_PER_EPOCH):
        s, s_next, s_lag, L = (torch.as_tensor(a, device=device)
                               for a in replay.sample(rng, 1024, temporal=True))
        z, z_next, z_lag = encoder(condition(torch.cat((s, s_next, s_lag)), L.repeat(3))).chunk(3)
        loss = -psd_objective(z, z_next, z_lag, L)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        encoder_losses.append(loss.detach())

    sac_batches = []
    for _ in range(UPDATES_PER_EPOCH):
        s, s_next, a, vx, L = (torch.as_tensor(x, device=device) for x in replay.sample(rng, 256))
        x = condition(s, L)
        next_x = condition(s_next, L)
        with torch.no_grad():
            r_psd = psd_reward(encoder(x), encoder(next_x), L)
            r_ext = (vx / 0.5).clamp(max=1.0)
        # No true terminal states in HalfCheetah. Bootstrap through time limits.
        metrics = agent.update(x, a, next_x, (r_psd + r_ext)[:, None], terminal=None)
        sac_batches.append(torch.stack(tuple(metrics.values())))
    update_values = torch.cat((torch.stack(encoder_losses)[:, None], torch.stack(sac_batches)), dim=1)
    if not torch.isfinite(update_values).all().item():
        raise FloatingPointError("An encoder or SAC update produced nonfinite metrics.")
    return dict(zip(("encoder_loss", *metrics), update_values[-1].tolist())), float(velocity.mean())


@torch.no_grad()
def diagnostics(envs, agent, encoder, replay, rng, device):
    # Normalize per state component using random-action rollouts, as in §4.
    env = envs[0]
    random_states = []
    for _ in range(8):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        for _ in range(HORIZON):
            random_states.append(obs)
            action = rng.uniform(env.action_space.low, env.action_space.high)
            obs, *_ = env.step(action)
    random_states = np.asarray(random_states)
    results = []
    trained = np.unique(replay.period[:replay.count])
    for L in np.unique(np.r_[trained[[0, -1]], np.intersect1d(trained, [5, 8, 11, 15])]):
        s, s_next, s_lag, periods = (torch.as_tensor(x, device=device)
                                     for x in replay.sample(rng, 1000, temporal=True, period=L))
        z = encoder(condition(torch.cat((s, s_next, s_lag)), periods.repeat(3))).chunk(3)
        state, _, vx = rollouts(envs[:ADAPT_EVAL_EPISODES], agent, np.full(ADAPT_EVAL_EPISODES, L),
                                rng, device, deterministic=True)
        results.append({
            "L": int(L),
            "target_steps": int(2 * L),
            "geometry": geometry(*(part.cpu().numpy() for part in z), int(L)),
            "eval_mean_velocity": float(vx.mean()),
            "eval_spectra": [spectrum(s[:-1], random_states.mean(axis=0), random_states.std(axis=0))
                             for s in state],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, required=True, help="Total epochs, including resumed epochs.")
    parser.add_argument("--initial-l", type=int, default=10)
    parser.add_argument("--adaptive-interval", type=int, default=2000)
    parser.add_argument("--latent-dim", type=int, default=3, choices=(3, 6))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-figure9", action="store_true", help="Write a Figure 9 comparison on completion.")
    parser.add_argument("--midpoint-episodes", type=int, help="Write a health report at this episode checkpoint.")
    args = parser.parse_args()
    if args.epochs < 1 or not L_FLOOR <= args.initial_l < HORIZON:
        parser.error("Use at least one epoch and an initial L from 5 to 199.")
    if args.adaptive_interval <= 0 or args.adaptive_interval % EPISODES_PER_EPOCH:
        parser.error("The adaptive interval must be a positive multiple of 8 episodes.")
    if args.seed < 0:
        parser.error("Use a nonnegative seed.")
    if args.midpoint_episodes is not None and (
        not 0 < args.midpoint_episodes < args.epochs * EPISODES_PER_EPOCH
        or args.midpoint_episodes % args.adaptive_interval
    ):
        parser.error("The midpoint must land on an adaptive evaluation before the final episode.")

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "checkpoint.pt"
    if checkpoint.exists() != args.resume:
        parser.error("Choose an unused output directory, or --resume with an existing checkpoint.")

    start = perf_counter()
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(6, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)
    envs = [gym.make(ENV_ID, max_episode_steps=HORIZON) for _ in range(EPISODES_PER_EPOCH)]
    obs_dim = envs[0].observation_space.shape[0]
    action_space = envs[0].action_space
    agent = SAC(obs_dim + EMBED_DIM, action_space, device)
    encoder = nn.Sequential(
        nn.Linear(obs_dim + EMBED_DIM, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, args.latent_dim),
    ).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4, fused=device.type == "cuda")
    replay = EpisodeReplay(obs_dim, action_space.shape[0])
    bounds = Bounds(args.initial_l, args.initial_l)
    history = [{"episodes": 0, "low": bounds.low, "high": bounds.high}]
    config = {
        "env": ENV_ID, "seed": args.seed, "initial_l": args.initial_l,
        "latent_dim": args.latent_dim, "adaptive_interval": args.adaptive_interval,
        "device_type": device.type,
        "sac_entropy_update": "log_probability_plus_target_entropy",
    }
    previous_epoch = 0
    previous_elapsed = 0.0
    if args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if config != saved["config"] or args.epochs <= saved["epoch"]:
            parser.error("Resume with the original config and a higher total epoch limit.")
        previous_epoch = saved["epoch"]
        previous_elapsed = saved["elapsed_seconds"]
        agent.load_state_dict(saved["agent"])
        encoder.load_state_dict(saved["encoder"])
        optimizer.load_state_dict(saved["optimizer"])
        replay.load_state_dict(saved["replay"])
        bounds = Bounds(**saved["bounds"])
        history = saved["history"]
        rng.bit_generator.state = saved["rng"]
        eval_rng.bit_generator.state = saved["eval_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved

    try:
        for epoch in range(previous_epoch + 1, args.epochs + 1):
            metrics, vx = train_epoch(envs, agent, encoder, optimizer, replay, bounds, rng, device)
            episodes = epoch * EPISODES_PER_EPOCH
            if episodes % args.adaptive_interval == 0:
                before = [bounds.low, bounds.high]
                low_score = evaluate_boundary(envs, agent, encoder, bounds.low, eval_rng, device)
                high_score = evaluate_boundary(envs, agent, encoder, bounds.high, eval_rng, device)
                if not math.isfinite(low_score) or not math.isfinite(high_score):
                    raise FloatingPointError("Adaptive evaluation produced nonfinite scores.")
                bounds.update(low_score, high_score)
                history.append({
                    "episodes": episodes, "evaluated_bounds": before,
                    "low_score": low_score, "high_score": high_score,
                    "low": bounds.low, "high": bounds.high,
                })
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"epoch={epoch} episodes={episodes} L={bounds.low}..{bounds.high} "
                      f"encoder_loss={metrics['encoder_loss']:.3f} alpha={metrics['alpha']:.3f} "
                      f"velocity={vx:.3f} "
                      f"this_run_seconds={perf_counter() - start:.1f}", flush=True)

            at_midpoint = episodes == args.midpoint_episodes
            if epoch % 250 == 0 or epoch == args.epochs or at_midpoint:
                elapsed = previous_elapsed + perf_counter() - start
                tmp = checkpoint.with_suffix(".pt.tmp")
                torch.save({
                    "epoch": epoch, "elapsed_seconds": elapsed, "config": config,
                    "agent": agent.state_dict(), "encoder": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(), "replay": replay.state_dict(),
                    "bounds": vars(bounds), "history": history,
                    "rng": rng.bit_generator.state, "eval_rng": eval_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                }, tmp)
                tmp.replace(checkpoint)
                progress = {
                    "config": config, "epoch": epoch, "episodes": episodes,
                    "target_epochs": args.epochs, "elapsed_seconds": elapsed,
                    "midpoint_episodes": args.midpoint_episodes,
                    "last_batch": metrics, "bound_history": history,
                }
                tmp = args.out / "progress.json.tmp"
                tmp.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                progress_path = args.out / "progress.json"
                tmp.replace(progress_path)
                if at_midpoint:
                    result = subprocess.run([
                        sys.executable, "-B", str(Path(__file__).with_name("check_midpoint.py")),
                        str(progress_path), str(args.out / "midpoint"),
                    ], check=False)
                    if result.returncode:
                        print(f"Midpoint report failed with exit code {result.returncode}.",
                              file=sys.stderr, flush=True)

        report_rng = np.random.default_rng(args.seed + 2)
        report = {
            "config": config, "episodes": episodes, "environment_steps": episodes * HORIZON,
            "gradient_updates_per_network": args.epochs * UPDATES_PER_EPOCH,
            "elapsed_seconds": elapsed, "bound_history": history,
            "last_batch": metrics,
            "platform": platform.platform(), "device": torch.cuda.get_device_name(device)
            if device.type == "cuda" else "cpu",
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "gymnasium", "mujoco", "numpy")},
            "metrics": diagnostics(envs, agent, encoder, replay, report_rng, device),
        }
        summary = args.out / "summary.json"
        tmp = summary.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(summary)
        print(f"Wrote {summary}", flush=True)
        if args.compare_figure9:
            subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("compare_fig9.py")),
                            str(summary), str(args.out)], check=True)
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
