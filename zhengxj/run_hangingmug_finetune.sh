#!/usr/bin/env bash
# Fine-tune FastWAM on the hanging_mug subtask dataset, resuming from the
# released pretrained checkpoint.
set -euo pipefail
cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh 2>/dev/null || true
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2

export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NVLink workaround: the first FT attempt on this node hit
#   "CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error"
# after ~25 min of training. Disable NCCL P2P so collectives go over PCIe
# instead of NVLink. ~5-15% slower per step but reliable.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
# Better diagnostics if anything else goes wrong.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}

NPROC=${NPROC:-8}
TASK=${TASK:-robotwin_hangingmug_subtask_ft_3cam_384_1e-4}

echo "[$(date -Iseconds)] === START fine-tune (nproc=${NPROC} task=${TASK}) ==="
bash scripts/train_zero1.sh "${NPROC}" task="${TASK}"

echo "[$(date -Iseconds)] === DONE fine-tune ==="
