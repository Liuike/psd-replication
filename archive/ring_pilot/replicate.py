"""CPU toy replication of Periodic Skill Discovery, arXiv:2511.03187v3."""

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter


N = 24
SPEEDS = (1, 2, 3)
HALF_PERIODS = (4, 6, 12)
EPISODE_STEPS = 48
EPSILON = 1e-5
LAMBDA = 5.0
CENTER = 0.5
KAPPA = 10.0
GAMMA = 0.9
TEMPERATURE = 0.025
BATCH_SIZE = 64


def chord(L):
    return L * math.sin(math.pi / (2 * L))


def intrinsic(a, b, target):
    return math.exp(-KAPPA * (math.dist(a, b) - target) ** 2)


class Ring:
    """Forward gear choices. Observation IDs hide the positions and ring order."""

    def __init__(self, rng):
        self.obs = list(range(N))
        rng.shuffle(self.obs)
        self.edges = [None] * N
        for p, s in enumerate(self.obs):
            self.edges[s] = tuple(self.obs[(p + v) % N] for v in SPEEDS)

    def rollout(self, rng, choose):
        states = [rng.randrange(N)]
        actions = []
        for _ in range(EPISODE_STEPS):
            action = choose(states[-1])
            actions.append(action)
            states.append(self.edges[states[-1]][action])
        return states, actions


class Encoder:
    """One unconstrained 2D embedding table per period, trained by Adam."""

    def __init__(self, rng, L):
        self.L = L
        self.target = chord(L)
        self.z = [[rng.gauss(0, 0.4) for _ in range(2)] for _ in range(N)]
        self.m = [[0.0, 0.0] for _ in range(N)]
        self.v = [[0.0, 0.0] for _ in range(N)]
        self.updates = 0

    def update(self, triples, lr=0.02):
        """Gradient ascent for Eq. 7, with fixed multipliers (Appendix C.2)."""
        gradient = [[0.0, 0.0] for _ in range(N)]
        for a, b, c in triples:
            za, zb, zc = self.z[a], self.z[b], self.z[c]
            d = math.dist(za, zc)
            mid = math.hypot(za[0] + zc[0], za[1] + zc[1])
            step = math.dist(za, zb)
            # Choose the zero subgradient when a norm's argument is zero.
            diam_scale = (1 - LAMBDA * (self.L - d < EPSILON)) / d if d else 0
            center_scale = -CENTER / mid if mid else 0
            step_scale = -LAMBDA / step if step and self.target - step < EPSILON else 0
            for j in range(2):
                diam_g = diam_scale * (zc[j] - za[j])
                center_g = center_scale * (zc[j] + za[j])
                step_g = step_scale * (zb[j] - za[j])
                gradient[a][j] += -diam_g + center_g - step_g
                gradient[b][j] += step_g
                gradient[c][j] += diam_g + center_g

        self.updates += 1
        m_correction = 1 - 0.9**self.updates
        v_correction = 1 - 0.999**self.updates
        for s in range(N):
            for j in range(2):
                g = gradient[s][j] / len(triples)
                m = self.m[s][j] = 0.9 * self.m[s][j] + 0.1 * g
                v = self.v[s][j] = 0.999 * self.v[s][j] + 0.001 * g * g
                self.z[s][j] += lr * (m / m_correction) / (math.sqrt(v / v_correction) + 1e-8)

    def reward(self, s, next_s):
        return intrinsic(self.z[s], self.z[next_s], self.target)


def sample_triples(replay, L, rng):
    batch = []
    for _ in range(BATCH_SIZE):
        states = rng.choice(replay)
        t = rng.randrange(len(states) - L)
        batch.append((states[t], states[t + 1], states[t + L]))
    return batch


class SoftQ:
    """Tabular entropy-regularized control using only observed transitions."""

    def __init__(self):
        self.q = [[0.0] * len(SPEEDS) for _ in range(N)]
        self.replay = {}

    def collect(self, states, actions):
        # Deterministic environment: repeated observations of an edge agree.
        self.replay.update({(s, a): ns for s, a, ns in zip(states, actions, states[1:])})

    def fit(self, encoder, sweeps=10):
        # Relabel replay with the current encoder's intrinsic reward.
        transitions = [
            (s, a, ns, encoder.reward(s, ns))
            for (s, a), ns in self.replay.items()
        ]
        for _ in range(sweeps):
            values = []
            for qs in self.q:
                best = max(qs)
                value = best + TEMPERATURE * math.log(
                    sum(math.exp((q - best) / TEMPERATURE) for q in qs)
                )
                values.append(value)
            for s, a, ns, reward in transitions:
                self.q[s][a] = reward + GAMMA * values[ns]

    def choose(self, s, rng):
        qs = self.q[s]
        best = max(qs)
        return rng.choices(range(len(qs)), [math.exp((q - best) / TEMPERATURE) for q in qs])[0]

    def greedy(self, s):
        return max(range(len(SPEEDS)), key=self.q[s].__getitem__)


def fit_reference(ring, encoder, rng):
    # Oracle positive control: matching-period recordings for Eq. 7 only.
    action = SPEEDS.index(N // (2 * encoder.L))
    replay = [ring.rollout(rng, lambda _: action)[0] for _ in range(24)]
    for i in range(2200):
        encoder.update(sample_triples(replay, encoder.L, rng), lr=0.02 if i < 1600 else 0.005)


def train(ring, encoder, mode, rng, epochs):
    if mode == "frozen":
        fit_reference(ring, encoder, Random(rng.getrandbits(64)))
    elif mode == "fixed_random":
        rng.getrandbits(64)  # Align exploration with the frozen run.

    q = SoftQ()
    replay = deque(maxlen=24)
    for epoch in range(epochs):
        # Coherent action exploration is blind to L. No demonstrations enter
        # joint training or the policy replay of any condition.
        explore = epoch < 12 or rng.random() < 0.10
        if explore:
            action = rng.randrange(len(SPEEDS))
            states, actions = ring.rollout(rng, lambda _: action)
        else:
            states, actions = ring.rollout(rng, lambda s: q.choose(s, rng))
        if mode == "joint":
            replay.append(states)
            for _ in range(12):
                encoder.update(sample_triples(replay, encoder.L, rng))
        q.collect(states, actions)
        q.fit(encoder)
    q.fit(encoder, sweeps=75)
    return q


def get_cycle(ring, q, start):
    visited = {}
    states = []
    s = start
    while s not in visited:
        visited[s] = len(states)
        states.append(s)
        s = ring.edges[s][q.greedy(s)]
    return states[visited[s]:]


def recurrence(ring, q, start, lag):
    s = start
    states = [s]
    for _ in range(EPISODE_STEPS):
        s = ring.edges[s][q.greedy(s)]
        states.append(s)
    return mean(s == ns for s, ns in zip(states, states[lag:]))


def evaluate(ring, encoder, q, mode, seed):
    L = encoder.L
    cycles = [get_cycle(ring, q, s) for s in range(N)]
    counts = Counter(map(len, cycles))
    # Average within each recurrent orbit, then across all reset states.
    stats = []
    for cycle in cycles:
        z = [encoder.z[s] for s in cycle]
        n = len(z)
        stats.append((
            mean(math.dist(a, z[(i + 1) % n]) for i, a in enumerate(z)) / encoder.target,
            mean(math.dist(a, z[(i + L) % n]) for i, a in enumerate(z)) / L,
            mean(math.hypot(a[0] + z[(i + L) % n][0], a[1] + z[(i + L) % n][1])
                 for i, a in enumerate(z)) / L,
            mean(math.dist(a, z[(i + 2 * L) % n]) for i, a in enumerate(z)) / L,
            mean(encoder.reward(s, cycle[(i + 1) % n]) for i, s in enumerate(cycle)),
        ))
    step, half, midpoint, closure, reward = map(mean, zip(*stats))
    return {
        "experiment": mode, "seed": seed, "L": L,
        "dominant_period": min(counts, key=lambda p: (-counts[p], p)),
        "cycle_match": counts[2 * L] / N, "period_counts": dict(sorted(counts.items())),
        "rollout_recurrence": mean(recurrence(ring, q, s, 2 * L) for s in range(N)),
        "step_ratio": step, "half_ratio": half, "antipode_error": midpoint,
        "closure_error": closure, "intrinsic_reward": reward,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "summary.json")
    args = parser.parse_args()
    if args.epochs < 13:
        parser.error("--epochs must be at least 13")

    rows = []
    start = perf_counter()
    for seed in args.seeds:
        ring = Ring(Random(seed))
        for L in HALF_PERIODS:
            run_seed = 1009 * seed + L
            for mode in ("joint", "frozen", "fixed_random"):
                i = perf_counter()
                encoder = Encoder(Random(run_seed), L)
                q = train(ring, encoder, mode, Random(run_seed + 1), args.epochs)
                row = evaluate(ring, encoder, q, mode, seed)
                rows.append(row)
                print(
                    f"{mode:12} seed={seed} L={L:2} target={2 * L:2} "
                    f"cycle={row['dominant_period']:2} exact={row['cycle_match']:5.0%} "
                    f"step/half ratios={row['step_ratio']:.2f}/{row['half_ratio']:.2f} "
                    f"reward={row['intrinsic_reward']:.2f} ({perf_counter() - i:.1f}s)",
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "seeds": args.seeds, "epochs": args.epochs, "episode_steps": EPISODE_STEPS,
        "states": N, "speeds": SPEEDS, "elapsed_seconds": round(perf_counter() - start, 3),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
