# State-based HalfCheetah replication

Historical run record. Current installation and commands are in the [project README](../README.md); the [curated final results](../reports/figure9_gpu_seed0_corrected/) are included in fresh clones. Original `runs/` paths mentioned below are now preserved locally under ignored `local_artifacts/runs/`.

This runner targets the HalfCheetah panel in Figure 9 of [Periodic Skill Discovery](https://jonghaepark.github.io/psd/static/psd_paper.pdf#page=24). That panel tracks the period bounds over roughly 320,000 training episodes. The run in `runs/figure9_gpu_seed0_corrected` starts from seed 0 with the corrected SAC entropy update. The earlier CPU smoke run and the first GPU attempt used the wrong sign in that update and their results are invalid for the Figure 9 comparison.

The runner uses continuous MuJoCo HalfCheetah with default proprioceptive observations and 200-step episodes. A shared encoder, actor, and twin critics each receive the 8-dimensional sinusoidal encoding of `L`. The encoder uses Equation 7 with both multipliers set to 5. The SAC reward is Equation 8 plus `min(x_velocity / 0.5, 1)`, as specified for this HalfCheetah experiment in Appendices C.2 and D.1. The runner follows Table 2's two 1,024-unit hidden layers, Adam at 0.0001, 0.99 discount, 0.995 target smoothing, automatic entropy tuning, eight episodes and 64 updates per epoch, 500,000-transition replay, and batches of 1,024 for the encoder and 256 for SAC. See the [paper](https://arxiv.org/html/2511.03187v3#A3) and the [SAC implementation cited in it](https://github.com/pranz24/pytorch-soft-actor-critic).

## Details to verify against the original implementation

The [authors' linked repository](https://github.com/jonghaepark/psd) provides a README and license. This independent implementation makes the following choices where the paper leaves details open or gives conflicting information.

| Detail | Runner choice and evidence |
| --- | --- |
| Environment version | Gymnasium `HalfCheetah-v5`, MuJoCo 4.0.1. The authors give the environment name and horizon without a version. |
| Starting period and bounds | Train at `L=10` initially. Figure 9 plots 10 and 11 at the start and roughly 5 and 16 at the end. Interpreting the plotted upper bound as exclusive reconciles these values with a single starting period and the evaluation settings of 5, 8, 11, and 15 in Appendix D.4. The runner stores the largest **included** period, so add one to the recorded upper bound when comparing curves under this inference. |
| Discrete period candidates | Take up to four unique integer values spaced across the range, including both trained endpoints. Round intermediate values down, which gives 5, 8, 11, 15 for the reported final range. |
| Adaptive sampling interval | 2,000 episodes from Table 2. Appendix C.2's prose gives 1,000. Evaluate deterministic policy rollouts at each trained endpoint with five resets per endpoint. |
| Latent dimension | 3. The paper lists 3 and 6 without a per-environment assignment. Both are available through `--latent-dim`. |
| Seed and initialization | Seed 0. Use PyTorch's default initialization for the encoder and the cited SAC implementation's Xavier initialization for actor and critics. Start collection from the untrained stochastic policy, following Algorithm 1. |

## Run

From the `psd_replication` directory in PowerShell. The virtual environment in this workspace already contains the CPU dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "gymnasium[mujoco]==1.2.1" "mujoco==4.0.1" "numpy==2.2.6"
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -B mujoco_psd/train.py --epochs 4 --device cpu --out mujoco_psd/runs/seed0
```

For the plotted budget use `--epochs 40000`. On a machine with a CUDA build of PyTorch, use `--device cuda` and a fresh output directory. The GPU path uses CUDA's fused implementation of Adam with the paper's learning rate, and keeps the Float32 network precision. With the same seed, period settings, latent size, and device type, use `--resume` with a higher total `--epochs` to continue a pilot. The runner saves model, optimizer, replay, random state, bounds, and a readable `progress.json` every 250 epochs and at the end of training.

Pass `--midpoint-episodes 160000` to save a health report and comparison in the run's `midpoint` directory after the halfway checkpoint. Pass `--compare-figure9` to write the full comparison and bounds plot when the run completes. To compare a partial run, pass its latest `progress.json` and a separate output directory to `compare_fig9.py`. The paper curve in those files is approximately digitized from Figure 9.

The final report contains training steps, bound history, and additional checks based on Section 4. It samples 1,000 within-episode replay triples for selected trained periods and compares one-step and lag-`L` distances with their theoretical targets. It also reports PCA/FFT peaks for five fresh rollouts per selected period. These are diagnostics for the partial run, supplementary to the bound curve in Figure 9.
