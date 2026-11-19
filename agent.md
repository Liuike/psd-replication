# Replication handoff

The canonical agent instructions are in [AGENTS.md](AGENTS.md). This file is a short orientation for a researcher or coding agent joining the project.

The active implementation is in `mujoco_psd/`. We have completed a 320,000-episode corrected HalfCheetah adaptive-sampling run, an Ant L=20 intrinsic run, and 80,000-episode Walker2D intrinsic and Ant/Walker2D combined-reward runs. Combined reward adds the paper's capped forward-speed term to PSD. The latter runs achieve forward locomotion and complete their continuous L-switching schedules, but miss our original added whole-rollout physical-gait recurrence criterion. Later Walker2D rhythms have near-target timing in an exploratory reanalysis.

Start with [README.md](README.md) for commands and the results table, [methods and limitations](docs/METHODS_AND_LIMITATIONS.md) for fidelity gaps, and [published evidence](reports/README.md) for plots, JSON and raw evaluation trajectories. The original toy ring experiment is archived separately and must not be presented as a reproduction of MuJoCo results.

Important open issues are the missing adaptive curriculum in Ant/Walker2D, the shorter 80k budget versus Figure 9's roughly 320k episodes, simulator-version uncertainty, and one training seed per robot. Further experiments should isolate those factors rather than silently changing multiple settings. The physical-gait thresholds were introduced here; they are not reported by the paper. See AGENTS.md before editing, launching jobs or interpreting PCA circles as physical gait success.
