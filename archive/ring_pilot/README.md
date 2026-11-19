# Archived toy pilot

Historical notes from the initial ring-method check. This is not a MuJoCo experiment. Current reproduction status and commands are in the [project README](../../README.md). Early follow-up proposals below are historical.

# Periodic Skill Discovery: ring method check

The ring experiment isolates PSD's encoder objective and reward. The [MuJoCo HalfCheetah runner](../../docs/HALFCHEETAH.md) follows the published continuous-control protocol and records which details had to be inferred.

This pilot implements the encoder objective and intrinsic reward from [Periodic Skill Discovery, §3.2–3.3](https://arxiv.org/html/2511.03187v3#S3). It checks whether those components recover a circular representation and whether alternating encoder and policy updates discover the requested cycle length. It runs in about **7 seconds** on the machine used for this study and needs only Python 3.10+.

The paper trains continuous-control SAC agents in MuJoCo and reports using an NVIDIA A6000, with full training typically completing within 24 hours per task ([Appendix C.2](https://arxiv.org/html/2511.03187v3#A3.SS2)). The [authors' linked repository](https://github.com/jonghaepark/psd) contained a README and license but no training implementation when checked. This pilot is a reimplementation from the equations.

## Run

From this directory:

```powershell
python replicate.py
python plot_results.py results/summary.json results/periods.svg
```

The main run uses 5 seeds, 3 target periods, and 150 episodes of 48 transitions per condition and seed–period pair. The longer check uses 600 episodes for 2 of the same seeds:

```powershell
python replicate.py --seeds 0 1 --epochs 600 --out results/longer_pilot.json
```

## What was replicated

- **Dynamics:** The agent sees only one of 24 shuffled state IDs along a ring, with no phase coordinate or time input. The actions advance 1, 2, or 3 positions. The corresponding fixed-action loops take 24, 12, or 8 steps. The agent can switch actions at each step.
- **Representation and reward:** For a target half-period `L`, learn a two-dimensional embedding from within-episode tuples `(s[t], s[t+1], s[t+L])`. Equation 7 pushes lag-`L` states toward opposite points a distance `L` apart and constrains adjacent states to the target chord `c = L sin(π / (2L))`. The embedding isn't constrained to lie on a circle by construction. Equation 8 pays `exp(-10 * (d - c)^2)`, where `d` is the learned one-step distance. No environment reward is used.
- **Control:** Use a table for the embedding and fitted soft Q-learning for the policy, replacing the paper's neural encoder and continuous SAC. The Q update relabels observed transitions with the current encoder. Train separate tables for `L = 4, 6, 12`; this isolates the objective and doesn't test generalization by a shared policy.

The main **joint** condition alternates between collecting a rollout, fitting the encoder from recent episodes, and updating the policy. Some exploration episodes hold a uniformly random action constant to cover sustained speeds; these episodes are independent of the target. **Oracle encoder** (`frozen` in the data) uses recordings from a known matching speed to fit and freeze the encoder before policy training. Those recordings are used for the encoder only. This is a positive control for the geometry and reward, with the cycle already present in the data. **Random encoder** (`fixed_random`) keeps the initial random embedding fixed and trains the same controller with its reward.

## Results

We evaluated the greedy policy from all 24 initial states per seed. The columns below count seeds for which **every** initial state eventually reached a cycle whose shortest period was exactly the requested length. The JSON also records the most common cycle, the full cycle length counts, the share of state pairs equal at offset `2L` over full rollouts, and the latent distances and reward on the recurrent cycles. A lag-`2L` recurrence alone can also count a shorter cycle that divides `2L`.

| Requested cycle `2L` | Joint training | Oracle encoder | Random encoder | Joint: most common cycle per seed | Joint: mean reward |
| --- | ---: | ---: | ---: | --- | ---: |
| 8 | 5/5 | 5/5 | 0/5 | 8, 8, 8, 8, 8 | 0.998 |
| 12 | 0/5 | 5/5 | 1/5 | 10, 10, 8, 11, 10 | 0.998 |
| 24 | 0/5 | 5/5 | 0/5 | 10, 8, 9, 10, 8 | 0.993 |

The oracle encoder reached both target latent distances to within 0.2% on average for each target, and policies recovered all three requested cycles. Joint training recovered the shortest target. For the two longer targets its average one-step distance was near ideal, but its lag-`L` distance was only **73% and 26%** of the target, respectively. The 4× training budget left both longer targets unmatched on the two seeds tested. This suggests that the local reward can be nearly saturated before the encoder learns the required long-range geometry. These toy results do not establish whether the same failure occurs with the paper's shared networks and continuous dynamics.

- [Plot](results/periods.svg)
- [Five-seed results](results/summary.json)
- [Longer check](results/longer_pilot.json)

## Smallest follow-up in a paper environment

Run one state-observation **Hopper** experiment. Use the 400-step episodes from [Appendix C.1](https://arxiv.org/html/2511.03187v3#A3.SS1), fix two targets (for example `L = 10, 20`), and condition a single compact encoder and SAC policy on `L`. Start with one seed and a hard cap (for example 100,000 environment steps or 30 minutes); treat this as a pilot budget, since we haven't timed or run it. Leave adaptive sampling, pixels, and downstream PPO out.

Replicate the diagnostics from [Figures 5–6](https://arxiv.org/html/2511.03187v3#S4): compare the one-step and lag-`L` latent distance distributions with their targets and compare joint-angle periods with `2L` on fresh rollouts. Use a spectral peak plus autocorrelation to check for harmonics, and report the surviving rollout lengths so early falls don't masquerade as periodicity. Include a fixed-random-encoder control at the same budget, and add seeds only if the pilot shows two distinct rhythms and the corresponding latent geometry.
