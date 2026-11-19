# Walker2D periodic skill replication

This run targets the Walker2D row of [Figure 5](https://jonghaepark.github.io/psd/static/psd_paper.pdf#page=7). Train a shared actor and encoder on `L=20,26,33,40`, requesting physical gait periods of 40, 52, 66, and 80 environment steps.

Use state-based Walker2d-v5 with default unhealthy termination and a 400-step training cap. Follow Table 2 with two 1,024-unit hidden layers, 8D sinusoidal period embedding, Adam at 0.0001, 64 encoder and SAC updates per epoch, eight episodes per epoch, replay of 500,000 transitions, and batches of 1,024 for the encoder and 256 for SAC. Walker2D uses constraint multipliers of 10. The three-dimensional latent space, intrinsic-only reward, fixed four-period sampling, modern simulator version and initial 80,000-episode budget are explicit reconstruction choices where the paper leaves Figure 5 details unspecified.

The evaluation runs five matched fresh reset seeds per period. Save physical joint angles and velocities, empirical period estimates, and latent distance metrics. An additional uninterrupted 1,600-step evaluation changes `L` every 400 steps as shown in the figure. Respect falls and label incomplete trajectories. Plot joint state values over time and a common 2D PCA projection with equal axes. The optional 4/5 within-10% flag is our diagnostic threshold.

On Oscar, from the staged project root:

```bash
bash -n mujoco_psd/walker2d_figure5_l40s.sbatch
DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash mujoco_psd/walker2d_figure5_l40s.sbatch
sbatch --test-only mujoco_psd/walker2d_figure5_l40s.sbatch
sbatch --parsable mujoco_psd/walker2d_figure5_l40s.sbatch
```

The launcher uses one L40S, four CPU cores, 12 GB RAM and an eight-hour walltime. Set `EPOCHS=50` and a pilot `RUN_NAME` for the startup test, then retain that run name with `RESUME=1 EPOCHS=10000` to continue it. Checkpoints allow retries without repeating completed epochs. Evaluation and figure production run after training.

## Submitted runs (2026-11-17)

Oscar project: `/oscar/scratch/<oscar-user>/psd-walker2d-figure5`.
Run name: `walker2d-figure5-intrinsic__seed0__epochs10000`.

- Pilot job `9248636`: completed successfully on NVIDIA L40S, 50 epochs / 400 episodes / 25,720 transitions. Training took 34 seconds; the full job including evaluation took 67 seconds. The pilot checks execution and checkpoint integrity, not convergence: it did not yet recover the requested gaits.
- Full job `9248771`: submitted to resume at epoch 50 and continue to 10,000 epochs / 80,000 episodes. Budget is at most 32 million environment transitions because falls shorten episodes. Estimated training time is roughly 3–5 hours excluding queue time, with an eight-hour allocation; the short pilot makes this provisional.

Pilot summary and plots are preserved under the run directory's `pilot_epoch0050/`. Full output goes to `evaluation/`, including fixed and switched joint-state traces, shared PCA, a JSON report and raw NPZ rollouts. Job log: `walker2d-psd-fig5-9248771.log` in the Oscar project root. GPU allocation and checkpoint resume must be verified from that log after scheduling.

## Completed result

Full job `9248771` completed with exit code 0 after 3:03:38, reaching epoch 10,000 / 80,000 episodes. The five fixed-period evaluation rollouts matched the requested gait period under our diagnostic in 0/5 cases for L=20, 2/5 for L=26, 2/5 for L=33, and 4/5 for L=40. Detected shortest repeating gait lags were 52 for L=26, 65–66 for L=33, and 77–78 for L=40; missing detections are not proof of no periodic motion. Mean fitted latent circle diameters were 19.884, 25.911, 32.957 and 39.969 respectively. The continuous 1,600-step schedule terminated at step 431, so it did not reproduce the complete Figure 5 switching trajectory. This is a partial behavioral replication, despite close latent diameter agreement. Our matching rule is an evaluation choice, not a threshold claimed in the paper.

Retrieved final evaluation artifacts locally to `runs/walker2d_figure5_oscar_seed0/evaluation/`.
