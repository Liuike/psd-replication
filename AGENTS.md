# Agent instructions and project handoff

## Purpose and current state

This repository is an independent reconstruction of **Periodic Skill Discovery (PSD)**. Read `README.md`, `docs/METHODS_AND_LIMITATIONS.md`, and `reports/README.md` before interpreting results or proposing experiments. This is not the authors' implementation or an exact reproduction of every training detail.

Current main experiments are state-based MuJoCo HalfCheetah, Ant and Walker2D. The corrected HalfCheetah run uses adaptive period sampling and 320,000 episodes. Ant and Walker2D use fixed four-period schedules and 80,000 episodes per seed-0 run; their combined-reward runs have completed. Both move forward and finish the continuous switching evaluations. Neither passes the original added physical-gait recurrence diagnostic. Walker2D has near-target candidate rhythms in exploratory late windows. Do not turn those observations into claims of exact physical periodicity.

## Source and artifact layout

- `mujoco_psd/`: active trainers, SAC, evaluators and durable Slurm launchers. Modules use sibling imports; run entrypoints from the repository root with `python mujoco_psd/<script>.py`.
- `reports/`: curated, versioned evidence and raw evaluation NPZ files. Preserve original scientific values and provenance.
- `docs/`: methods, limitations, reproducibility and historical experiment notes.
- `archive/ring_pilot/`: the original toy ring check. It is not a MuJoCo experiment or substitute for the paper's locomotion tests.
- `mujoco_psd/runs/`: default location for new local outputs; ignored by Git.
- `local_artifacts/`: ignored local preservation of old runs, checkpoints and logs. It does not exist in a fresh clone. Do not depend on it for published analysis.

## Scientific invariants

1. Keep training conditions explicit: environment/version, seed, L values and sampling rule, reward mode, latent dimension, horizon, epochs and episode count.
2. The policy receives the period embedding, not an explicit clock or gait phase. PSD encoder/reward must not impose circles by construction.
3. Combined reward is `r_PSD + min(v_x / v_target, 1)`, with unit weights; targets are 0.5 m/s for Ant and 1.0 m/s for Walker2D. Preserve signed backward penalties. Gymnasium's native total environment reward is not the training objective here.
4. Distinguish true terminal falls from time limits; bootstrap through time limits. Stop evaluation rollouts at actual termination. Never concatenate resets to fabricate continuous switching.
5. Use one shared actor/encoder across L within a run. Preserve matched evaluation reset seeds and raw unscaled latent data. Use common PCA axes across a plotted switching trajectory; label fitting population and conditional-encoder changes. A correct circle diameter does not establish a correct physical gait.
6. The ±10% timing, 0.35 normalized return-error and 4/5 rollout requirements are our added diagnostics, not paper acceptance criteria. Five reset seeds are not five training seeds. Label late-window minima exploratory; shorter windows and refitted normalization affect interpretation.
7. Mark fixed-L training, the missing adaptive Ant/Walker curriculum, environment-version uncertainty, selected budgets and single training seeds as reconstruction limitations. Figure 9's roughly 320k episodes do not establish Figure 5's exact checkpoint age.
8. Earlier HalfCheetah runs with the wrong SAC entropy update are invalid evidence; use the corrected run. Never merge their metrics into valid results.

## Development and verification

Use Python 3.11 and pinned dependencies in `requirements.txt`; install the appropriate CPU/CUDA PyTorch wheel as described in the README. Preserve command-line compatibility and checkpoint config checks. Use fresh output directories when reward or other training configuration changes. Resume only a compatible checkpoint with a higher total epoch limit.

For changes to scientific code, run a bounded smoke or numerical check appropriate to the changed behavior. A two-epoch smoke checks execution, not gait convergence. For analysis-only changes, rerun against `reports/` and verify values against saved JSON/NPZ. Do not run full GPU training merely to test documentation or formatting.

Keep the primary agent responsible for requirements, integration, verification and final review. Delegate independent bounded work to `simple_worker`; use `complex_worker` for genuinely complex independent work and `reviewer` for risky changes. Workers must have explicit file ownership, avoid reverting others, and report evidence. For nontrivial divisible tasks, use an independent agent; keep short indivisible tasks on the primary agent.

## Oscar / Slurm

Follow readable variables, explicit condition/seed mapping, semantic names, shell COMMAND arrays, `set -euo pipefail`, and `DRY_RUN=1` output in the existing launchers. Validate `bash -n`, representative dry runs and `sbatch --test-only` before submission. Use project-local environment paths; keep credentials out of scripts. Record actual GPU, versions, job IDs and run names in experiment provenance. Do not submit expensive jobs unless the user's task authorizes them. Already-authorized runs do not require repeated confirmation.

## Cleanup and publishing

Keep environments, caches, checkpoints, credentials, smoke outputs and raw training histories out of Git. Preserve valuable local data rather than deleting it. Before moving or recursively removing Windows paths, verify their resolved absolute locations remain inside the intended workspace. Curated published files must use portable paths and clearly distinguish generated outputs from source. Do not rewrite historical metric values for presentation. Do not commit `.venv/`, `local_artifacts/`, `mujoco_psd/runs/`, `.pt` files or secrets.
