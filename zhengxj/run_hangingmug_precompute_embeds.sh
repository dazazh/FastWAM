#!/usr/bin/env bash
# Precompute T5 text embeddings for the hanging_mug subtask dataset.
set -euo pipefail
cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh 2>/dev/null || true
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2

export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=${NPROC:-8}
TASK=${TASK:-robotwin_hangingmug_subtask_ft_3cam_384_1e-4}

echo "[$(date -Iseconds)] === START precompute_text_embeds (nproc=${NPROC}) ==="
torchrun --standalone --nproc_per_node="${NPROC}" scripts/precompute_text_embeds.py \
    task="${TASK}"

echo "[$(date -Iseconds)] === DONE precompute_text_embeds ==="
