# Ant and Walker2D with PSD plus forward reward

Fresh seed-0 runs use the additive reward from the paper's [Appendix C.2](https://arxiv.org/html/2511.03187v3#A3.SS2):

`r_total = r_PSD + min(v_x / v_target, 1)`

The target speed is 0.5 m/s for Ant and 1.0 m/s for Walker2D. Both terms have weight one; negative forward velocity contributes a negative external reward. The forward term uses post-step MuJoCo root velocity `qvel[0]`, stored with the transition in replay. No Gymnasium survival bonus, control penalty, or native total reward is added to this objective. Checkpoints record the reward mode, numeric target, weights, and velocity source. The PSD encoder objective is unchanged.

| Robot | L values | Target gait periods | Training episode cap | Budget |
|---|---|---|---|---|
| Ant | 10, 13, 16, 20 | 20, 26, 32, 40 | 200 steps | 10,000 epochs / 80,000 episodes |
| Walker2D | 20, 26, 33, 40 | 40, 52, 66, 80 | 400 steps | 10,000 epochs / 80,000 episodes |

Each robot has one policy and encoder shared across its four periods, uniformly sampled per episode. These are fresh runs from initialization; only each combined-reward run's own pilot is resumed. Previous intrinsic-only checkpoints and outputs remain available. The architecture, modern environment versions, fixed-period sampling and training budgets follow our earlier reconstruction choices; this is not evidence that every undocumented detail of the authors' setup matches.

Oscar project: `/oscar/scratch/<oscar-user>/psd-combined-figure5`. The Python environment links to the existing Ant project environment. Each job requests one L40S, four CPU cores, 12 GB RAM and eight hours. Each robot's 50-epoch startup pilot includes evaluation; after a clean pilot, continue its own checkpoint to 10,000 epochs with `RESUME=1` and the same run name.

Example from the remote project root:

```bash
export ROBOT=ant SEED=0 EPOCHS=50 RESUME=0
export RUN_NAME=ant-figure5-intrinsic-velocity__seed0__epochs10000
bash -n mujoco_psd/combined_forward_figure5_l40s.sbatch
DRY_RUN=1 bash mujoco_psd/combined_forward_figure5_l40s.sbatch
sbatch --test-only --time=00:20:00 mujoco_psd/combined_forward_figure5_l40s.sbatch
sbatch --parsable --time=00:20:00 mujoco_psd/combined_forward_figure5_l40s.sbatch
# After successful completion of the pilot:
export EPOCHS=10000 RESUME=1 PRESERVE_PILOT=1
DRY_RUN=1 bash mujoco_psd/combined_forward_figure5_l40s.sbatch
sbatch --test-only mujoco_psd/combined_forward_figure5_l40s.sbatch
sbatch --parsable mujoco_psd/combined_forward_figure5_l40s.sbatch
```

Repeat with `ROBOT=walker2d` and the corresponding run name. Pilot outputs are preserved under `pilot_epoch0050/` before continuation. Training logs show PSD reward, forward reward and measured forward velocity. The velocity logged during training is an equal-weight mean of per-episode means.

When the pilot is still queued, the full continuation can be queued with `--dependency=afterok:PILOT_JOB_ID`. Slurm starts it only if the pilot (including its evaluation) exits successfully. `PRESERVE_PILOT=1` additionally checks the pilot summary is epoch 50 with the combined reward before archiving it. A failed pilot leaves its continuation unable to run and requires diagnosis; it cannot silently launch a full fresh run.

Evaluation uses the existing five fresh matched reset seeds per L, joint-state recurrence and shared PCA. It also measures forward velocity, so forward movement can be assessed separately from periodicity. The continuous switching test remains a single episode, with no resets at period boundaries. Falls remain incomplete trajectories. Saved environment returns are the native Gymnasium return, not the combined training objective. Our recurrence thresholds remain diagnostic choices, not criteria stated by the paper.

The initial time estimate is approximately 3–5 hours per full run after scheduling, with an eight-hour limit. This is based on the earlier runs; changed motion and episode lengths can change the runtime.

## Submissions on 2026-11-18

| Robot | Pilot job | Full job | Last checked state |
|---|---|---|---|
| Walker2D | 9284564 | 9284759 | Pilot completed (exit 0); full run queued |
| Ant | 9284587 | 9284761 | Pilot queued; full run waits on `afterok:9284587` |

Walker2D's L40S pilot completed in 1:11 including evaluation (38.8 seconds of training, 400 episodes). Its physical gait recurrence diagnostic was 0/5 for all four periods and its switch rollout fell at step 180; this is a startup integrity check, not a successful behavioral replication. Training forward velocity was still negative at epoch 50. Full training is needed before judging the new objective.

Local CUDA smoke runs exercised both combined trainers, and Walker2D checkpoint continuation. A full Ant CPU evaluator smoke verified finite speed and displacement for all 20 fixed-period rollouts, correct NPZ masks, and continuous switching without hidden resets. Reward cap values were checked at negative, zero, below-target and above-target speeds. The launcher passed shell syntax checks, both robot dry runs, and Oscar `sbatch --test-only`; the final archive and speed changes passed independent review.

Remote job logs: `psd-forward-fig5-9284759.log` and `psd-forward-fig5-9284761.log`. Source hashes are recorded in the remote project root's `source_manifest.sha256`. Final evaluations will be written beneath each semantic run directory in `mujoco_psd/runs/`, using the run names above with `walker2d` or `ant`.

## Final results

Both full jobs completed with exit code 0 at epoch 10,000 / 80,000 episodes. Walker2D job 9284759 took 3:09:09; Ant job 9284761 took 3:13:54. Walker2D completed the entire 1,600-step switching rollout, and Ant completed all 800 switching steps. All four L settings for each robot scored 0/5 under our gait-recurrence diagnostic. Missing qualifying recurrence detections do not establish an absence of all periodic motion; they show these rollouts failed our requested-period criterion.

Final evaluation JSON, raw NPZ rollouts and plots were retrieved to `runs/walker2d_combined_oscar_seed0/evaluation/` and `runs/ant_combined_oscar_seed0/evaluation/` respectively. The last training batches reported forward velocities of 1.233 m/s (Walker2D) and 1.336 m/s (Ant); use the evaluation JSON's per-rollout speed measurements when assessing held-out behavior.

## Gait-gap follow-up

`analyze_combined_gait_gap.py` reconstructs the saved target errors from raw joint traces (checked against the evaluator JSON), then separately examines the final half of each rollout. Outputs are `runs/combined_gait_gap/gait_gap.json` and `gait_gap.png`. This exploratory shorter-window analysis does not replace the original pass counts. It refits normalization in each late window and allows lag searches through half the window length, rather than a third, so fewer cycles support the long-period estimates.

Walker2D's late-window strongest recurrence minima are 37–38, 48–53, 68–69 and 79 steps for target periods 40, 52, 66 and 80 respectively. Their timing is within about 1–8% of target, but the mean best return errors near the requested periods are 0.668, 0.586, 0.476 and 0.443 versus our 0.35 consistency cutoff. Thus the zero-pass whole-rollout summary masks near-target later rhythms, while repeatability remains imperfect. Ant's corresponding late-window errors remain 0.885, 1.049, 1.000 and 0.985, with scattered unrestricted recurrence minima. A minimum is a candidate rhythm, not confirmation of a reliable gait period; recurrence error is not a percentage period error or training-loss percentage.
