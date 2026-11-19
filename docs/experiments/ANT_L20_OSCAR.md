# Ant with fixed L=20 on Oscar

This run tests the 40-step gait reported for Ant at `L=20` in [Figure 5](https://jonghaepark.github.io/psd/static/psd_paper.pdf#page=7). It trains one seed with `L=20` throughout and evaluates the fundamental gait period on five matched, previously unseen reset seeds. A setting passes when at least four rollouts show a shortest repeat within 10% of 40 steps. An early fall counts as a failed rollout.

The run uses state observations from Ant-v5, a 200-step training horizon, the intrinsic PSD reward, and the paper's Table 2 settings. The six-dimensional encoder, simulator versions and fixed sampling condition are reconstruction choices. The default budget is 2,500 epochs, or 20,000 training episodes and at most four million environment steps. Early Ant terminations can reduce the transition count.

Run these commands on Oscar from the staged project root. The Python environment is local to this project.

```bash
export UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
uv venv --python 3.11 .oscar-venv
uv pip install --python .oscar-venv/bin/python -r mujoco_psd/requirements-oscar.txt
uv pip install --python .oscar-venv/bin/python torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

bash -n mujoco_psd/ant_l20_l40s.sbatch
DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash mujoco_psd/ant_l20_l40s.sbatch
sbatch --test-only mujoco_psd/ant_l20_l40s.sbatch
sbatch --parsable mujoco_psd/ant_l20_l40s.sbatch
```

The launcher requests one L40S in the `gpu` partition and verifies the allocated GPU at startup. The default run directory is `mujoco_psd/runs/ant-psd-intrinsic-l20__seed0__epochs2500`, and Slurm writes `ant-psd-l20-<job-id>.log` at the project root. The evaluation runs after training and saves its per-reset report and figure in the run's `evaluation` directory. To extend an existing run, keep the same `RUN_NAME` and `SEED`, set `RESUME=1`, and increase `EPOCHS`.

The initial seed-0 submission is job `9216358`, staged in `/oscar/scratch/<oscar-user>/psd-ant-l20-figure5`. Check its state with `squeue -j 9216358` or `sacct -j 9216358` after it exits.
