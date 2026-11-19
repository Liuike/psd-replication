# Methods and limitations

This project is an independent implementation of Periodic Skill Discovery (PSD), not a reproduction of the authors' unavailable training code. It reconstructs the published equations and reported settings, then tests them in Gymnasium environments backed by MuJoCo. The paper describes PSD's theoretical latent periodicity and reports approximate physical behaviors; our additional gait thresholds are diagnostic choices.

## Method implemented

The policy and encoder are conditioned on an eight-dimensional sinusoidal embedding of the half-period parameter `L`. The target physical period is `2L`. At each training episode, `L` is sampled once and held fixed. The encoder receives the state concatenated with that embedding. Its PSD objective (Equation 7) maximizes the lag-`L` latent distance, centers opposite points, and penalizes violations of the one-step chord and lag-`L` distance constraints. The policy receives the single-transition PSD reward from Equation 8. The encoder and policy are trained together with SAC.

The implementation uses fixed PSD constraint multipliers: 5 for Ant and HalfCheetah, and 10 for Walker2D. Those values match Table 2. The paper says its fixed multiplier values worked well; choosing fixed multipliers is not itself a departure from the method. Main Table 2 settings also followed include two 1,024-unit hidden layers, 8 episodes and 64 encoder/policy updates per epoch, a 500,000-transition replay buffer, encoder/policy batch sizes of 1,024/256, learning rate `1e-4`, discount `0.99`, target smoothing `0.995`, automatic entropy tuning, `k=0.5`, `epsilon=1e-5`, and `kappa=10`. See the [paper](https://arxiv.org/html/2511.03187v3) and the [HalfCheetah runner notes](HALFCHEETAH.md).

The simulator is **MuJoCo**; **Gymnasium** supplies its environment API. The recorded environment versions are Gymnasium 1.2.1 and MuJoCo 4.0.1. We used state observations and 200-step episodes for Ant and HalfCheetah, and 400-step episodes for Walker2D, matching the paper's stated horizons. The paper does not pin an API or simulator version, so equivalence of environment versions and their defaults is unverified.

In the combined-reward Ant and Walker2D runs, SAC received

`r_total = r_PSD + min(v_x / v_target, 1)`

with unit weights and target speed 0.5 m/s for Ant and 1.0 m/s for Walker2D. Negative forward speed contributes negative reward. The measured post-step MuJoCo root velocity `qvel[0]` is stored with its transition and added to the intrinsic reward. The native Gymnasium environment reward is recorded during evaluation but is not the combined training objective. This capped velocity reward and its targets are described in Appendix C.2. The earlier intrinsic-only Ant and Walker2D runs are separate conditions and should not be compared as if they used this combined objective.

### Period schedules

| Experiment | L values trained | Requested physical periods | Training budget |
|---|---|---|---|
| HalfCheetah, Figure 9 | Adaptive range; four candidates sampled within current bounds | Varies with learned bounds | 320,000 episodes, seed 0 |
| Ant, Figure 5 follow-up | 10, 13, 16, 20, fixed and sampled uniformly per episode | 20, 26, 32, 40 steps | 80,000 episodes, seed 0 |
| Walker2D, Figure 5 follow-up | 20, 26, 33, 40, fixed and sampled uniformly per episode | 40, 52, 66, 80 steps | 80,000 episodes, seed 0 |

HalfCheetah used the paper's adaptive period-range procedure. Ant and Walker2D used four fixed target values from the beginning. This skips curriculum learning over the period boundaries. The paper documents adaptive range learning and Figure 9, but does not specify Figure 5's exact checkpoint budget or all its training choices. Thus the 80,000-episode Figure 5 runs are a reconstruction budget, and their training exposure is not known to match the authors' checkpoints. The HalfCheetah run matches Figure 9's plotted horizon, but its final measured bounds (Lmin 8, plotted upper 21) differ from approximate paper readings (5 and 16). Details are in the [combined-reward run record](experiments/COMBINED_FORWARD_OSCAR.md) and [HalfCheetah notes](HALFCHEETAH.md).

## Results and their limits

The completed combined-reward Ant and Walker2D training runs each used one training seed and 80,000 episodes. Five new environment reset seeds per period were used only for evaluation. All 20 fixed-period rollouts per robot reached their environment episode caps. Their aggregate measured forward velocities were about 1.51 m/s for Ant and 1.25 m/s for Walker2D. The continuous period-switch rollout completed all 800 Ant steps and all 1,600 Walker2D steps without a reset at a period boundary.

No fixed-period rollout passed the project's strict recurrence diagnostic: **0/5 for each L** on both robots. That is not a pass/fail criterion from the paper. Our diagnostic discards the first 40 steps, searches for a shortest joint-state recurrence within 10% of `2L`, and requires normalized return error no greater than 0.35. The paper does not define these thresholds or report this five-reset test. Its Figure 5 presents representative joint trajectories and a PCA projection; the text describes Ant at L=20 as completing approximately five 40-step cycles in 200 steps.

An exploratory analysis of only the final half of each rollout found Walker2D candidate recurrence lags of 37–38, 48–53, 68–69, and 79 steps for target periods 40, 52, 66, and 80. The mean normalized return errors in those windows were 0.668, 0.586, 0.476, and 0.443. Ant's corresponding errors were 0.885, 1.049, 1.000, and 0.985. The late-window analysis refits normalization and has as few as about 2.5 target cycles, especially limiting the longest-period estimates. These candidate lags suggest approximate Walker2D timing after startup, but do not establish stable gaits; they do not overturn the original 0/5 criterion result. They are not period-error percentages.

The PCA figures use a shared, raw unscaled latent projection across the fixed-period rollouts and the continuous switch rollout. They do not normalize the data into circles. A visually round projected trace is therefore not proof of a physically exact cycle. The paper itself says its interpolated physical trajectories do not perfectly meet `2L` periodicity (Appendix D.2). Its exact periodicity claim is for the ideal latent construction at the objective optimum, not a guarantee that every learned physical rollout repeats perfectly. The paper also notes that restricting complex agents to two latent dimensions can be unstable; our models use 6 latent dimensions for Ant and 3 for Walker2D, then apply PCA for visualization.

The main causal uncertainties are the missing adaptive curriculum in the Ant/Walker Figure 5 runs, the chosen 80,000-episode budget, unpinned environment versions, and use of one training seed per robot. These differences could contribute to the observed gait mismatch, but our experiments do not isolate their effects. The public repository linked from the paper currently contains a README and license rather than its training implementation, so differences in undocumented code details cannot be ruled out. Startup pilots verified execution and checkpoint resume only; pilot gait outcomes are not convergence results.

## Reproduction artifacts

Training, evaluation, launch, and analysis entrypoints live in [`../mujoco_psd/`](../mujoco_psd/). Run-specific reports should be interpreted with their settings and limitations above. The main pointers are [`COMBINED_FORWARD_OSCAR.md`](experiments/COMBINED_FORWARD_OSCAR.md), [`ANT_L20_OSCAR.md`](experiments/ANT_L20_OSCAR.md), [`WALKER2D_OSCAR.md`](experiments/WALKER2D_OSCAR.md), and the [HalfCheetah notes](HALFCHEETAH.md). The HalfCheetah Figure 9 comparison uses approximate values digitized from the published plot; treat pointwise differences as approximate. Published evidence is indexed in [`reports/README.md`](../reports/README.md).
