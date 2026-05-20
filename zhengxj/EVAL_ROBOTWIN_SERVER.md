# RoboTwin Eval Runbook (This Server)

This note summarizes the proven RoboTwin evaluation flow for this server, so other agents can continue from the same context quickly.

## 1) Shell + Conda init

In every new shell:

```bash
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2
cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM
```

## 2) Required runtime env vars for eval

Use local checkpoints and avoid runtime download:

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
```

Memory stability for this server:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Current workaround for planner issues:

```bash
export ROBOTWIN_FORCE_MPLIB=1
```

## 3) Single-task smoke eval (known-good command)

```bash
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=click_bell \
  EVALUATION.task_config=demo_randomized \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.skip_get_obs_within_replan=false \
  EVALUATION.device=cuda \
  mixed_precision=fp16 \
  gpu_id=0
```

Then scale up:

```bash
# only change this value
EVALUATION.eval_num_episodes=10
```

## 4) Run in tmux (recommended)

```bash
tmux new -s fastwam_eval

# inside tmux, run Section 1 + 2 + 3 command
```

Useful tmux commands:

```bash
# detach: Ctrl-b then d
tmux ls
tmux attach -t fastwam_eval
tmux kill-session -t fastwam_eval
```

## 5) Output locations

Eval outputs are written under:

```bash
evaluate_results/robotwin/robotwin_uncond_3cam_384/<timestamp>/
```

Key files:

- `eval_click_bell_<timestamp>.log` (full log)
- `click_bell/_result_random.txt` (success summary)
- `click_bell/episode*_success-*.mp4` (rendered videos)

## 6) Important server-specific notes

- There is a long-running keep-GPU-busy workload (`tools/scripts/keep_gpu_busy.py`) on all 8 GPUs.
- Do **not** kill those processes unless user explicitly asks.
- If eval OOM happens, first check for stale eval process still occupying VRAM:

```bash
nvidia-smi
ps -fp <pid_from_nvidia_smi>
kill <stale_eval_pid>
```

- `missing pytorch3d` appears in logs from RoboTwin camera module. In this eval path it has been non-fatal, but installing pytorch3d is still recommended for full compatibility.

## 7) Current code-side workaround applied

To make this server run more reliably in current environment:

- `third_party/RoboTwin/envs/robot/robot.py` has fallback path for `ROBOTWIN_FORCE_MPLIB=1`
- `third_party/RoboTwin/envs/robot/planner.py` has compatibility tweaks:
  - `plan_path(..., constraint_pose=None, ...)`
  - safe default when planner type is unsupported

If these files are reverted or replaced, re-check fallback behavior before running eval.

