# MuJoCo PSD implementation

See the [project README](../README.md) for installation, current results, smoke commands, full run commands and Slurm instructions.

- `train.py`: HalfCheetah with adaptive period bounds.
- `train_ant.py`, `train_walker2d.py`: fixed-period or four-period shared-policy runs.
- `sac.py`: shared continuous-action SAC implementation.
- `evaluate_ant.py`, `evaluate_ant_figure5.py`, `evaluate_walker2d.py`: physical-state and latent evaluation.
- `evaluate_periods.py`, `evaluation.py`: recurrence and latent geometry helpers.
- `analyze_combined_gait_gap.py`: analysis of saved combined-reward evaluation trajectories.
- `compare_fig9.py`, `check_midpoint.py`: HalfCheetah bound comparisons and monitoring.
- `*.sbatch`: explicit, dry-runable Oscar launchers.

Published evidence lives in [`../reports/`](../reports/README.md). New runs default to ignored `runs/`; checkpoints from the original local workspace are preserved in ignored `../local_artifacts/runs/`.

Historical details: [HalfCheetah](../docs/HALFCHEETAH.md), [Ant](../docs/experiments/ANT_L20_OSCAR.md), [Walker2D](../docs/experiments/WALKER2D_OSCAR.md), [combined reward](../docs/experiments/COMBINED_FORWARD_OSCAR.md). The [methods reference](../docs/METHODS_AND_LIMITATIONS.md) distinguishes matched settings from reconstruction choices.
