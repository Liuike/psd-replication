# Curated replication results

This directory contains a public, compact selection of completed experiment outputs. These results are independent reimplementations of *Periodic Skill Discovery* (Park et al., NeurIPS 2025). They are not official author checkpoints. Training checkpoints and private cluster logs are omitted. The exported JSON reports replace cluster checkpoint paths with a note; numerical results are preserved.

## Results at a glance

| Experiment | Training | Main result | Caveat |
|---|---|---|---|
| [HalfCheetah Figure 9](figure9_gpu_seed0_corrected/) | One seed; 320,000 episodes, 64 million steps; 7.52 h on RTX 4060 Ti | Learned bounds ended at Lmin=8 and plotted upper=21. The paper plot is approximately 5 and 16 at 320k episodes. | Full Figure 9 horizon, but a single seed; paper transitions were visually digitized. |
| [Walker2D, intrinsic only](walker2d_figure5_oscar_seed0/) | One training seed; 80,000 episodes; five held-out resets per L | Strict gait diagnostic matches: 0/5 at L20, 2/5 at L26, 2/5 at L33, and 4/5 at L40. | Continuous switch rollout fell at step 431/1600. The threshold is our diagnostic, not a paper criterion. |
| [Walker2D, PSD plus forward reward](walker2d_combined_oscar_seed0/) | One training seed; 80,000 episodes; five held-out resets per L | Mean evaluation speed about 1.25 m/s; continuous switch rollout completed 1600 steps. | 0/5 requested-period matches at all L under the strict diagnostic. |
| [Ant, PSD plus forward reward](ant_combined_oscar_seed0/) | One training seed; 80,000 episodes; five held-out resets per L | Mean evaluation speed about 1.51 m/s; continuous switch rollout completed 800 steps. | 0/5 requested-period matches at all L under the strict diagnostic. |
| [Walker2D and Ant gait-gap follow-up](combined_gait_gap/) | Re-analysis of the same combined-reward rollouts | Later Walker2D windows show candidate periods close to target (37–38, 48–53, 68–69, 79 steps). Ant errors remain high. | Exploratory, selected final-half windows with refitted scaling and few cycles; not confirmed gaits. |
| [Ant fixed L20 continuation](oscar_ant_l20_continuation/) | One seed; 80,000 episodes; 15.31 million steps; 3.10 h on L40S | Mean latent L-lag distance 19.86 versus target 20; 2L distance 3.57. | 0/5 matched 40-step physical recurrence under our diagnostic. |

Every experiment above has **one independently trained policy**. Five evaluation reset seeds per L describe variation in rollouts for that policy; they are not five training seeds or a cross-seed confidence interval.

## Reading the gait metrics

The requested physical period is 2L. Our evaluator compares six Walker2D motor angles and rates, or eight Ant motor angles and rates, after discarding the first 40 steps. The original diagnostic requires a short-return local minimum within ±10% of 2L and normalized state-return error no greater than 0.35. These cutoffs are our analysis choices; the paper does not state them. Thus 0/5 means none passed this conservative diagnostic, not that the paper itself would classify all trajectories as failures.

The late-window analysis uses only the final half of each rollout, refits normalization to that segment, and searches candidate lags over a wider range. It has roughly 2.5 cycles for the longest requested periods. These candidate lengths and errors can suggest later rhythm formation, but do not establish steady gaits.

The combined reward follows the paper's stated form: PSD intrinsic reward plus capped forward velocity, with unit weights; the targets are 1.0 m/s for Walker2D and 0.5 m/s for Ant. The main combined-reward jobs completed all 80,000 episodes and their continuous switch tests completed.

## Artifact index and provenance

Each folder includes selected final reports/plots, and combined reward folders include raw NPZ rollouts. JSON keeps the original measurements/configuration; only machine-specific checkpoint paths were removed. Source provenance below points to the archived original run tree. If that tree is retained locally, it lives at `local_artifacts/runs/`; the committed folder here is the portable, curated copy.

- `figure9_gpu_seed0_corrected/`: source `local_artifacts/runs/figure9_gpu_seed0_corrected/`; full progress summary, approximate Figure 9 comparison and bounds SVG. No checkpoint.
- `walker2d_figure5_oscar_seed0/`: source `local_artifacts/runs/walker2d_figure5_oscar_seed0/evaluation/`; fixed and continuous-switch traces, shared PCA, JSON and raw NPZ. No checkpoint.
- `walker2d_combined_oscar_seed0/`: source `local_artifacts/runs/walker2d_combined_oscar_seed0/evaluation/`; combined-reward evaluation traces, shared PCA, JSON and raw NPZ. No checkpoint.
- `ant_combined_oscar_seed0/`: source `local_artifacts/runs/ant_combined_oscar_seed0/evaluation/`; combined-reward switch/state plots, shared PCA, JSON and raw NPZ. No checkpoint.
- `combined_gait_gap/`: source `local_artifacts/runs/combined_gait_gap/`; reproducible numeric re-analysis JSON and comparison plot. It derives only from the combined Ant and Walker2D NPZs above.
- `oscar_ant_l20_continuation/`: source `local_artifacts/runs/oscar_ant_l20_continuation/`; completed L20 summary, five-reset evaluation, state/diagnostic plots and NPZ. No checkpoint.

The report JSON artifacts list their outputs and configuration. Training code and launch instructions are in the repository's `mujoco_psd/` folder. Environment details, inferred setup choices, and limitations are documented in the root README and `mujoco_psd/README.md`.
