#!/usr/bin/env bash
set -euo pipefail

source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam

cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/robotwin_subset4_${RUN_TS}"
mkdir -p "${LOG_DIR}"

echo "[start] $(date -Iseconds)" | tee -a "${LOG_DIR}/session.log"
echo "[info] log_dir=${LOG_DIR}" | tee -a "${LOG_DIR}/session.log"

echo "[step] precompute text embeddings" | tee -a "${LOG_DIR}/session.log"
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py \
  task=robotwin_subset4_uncond_3cam_384_1e-4 2>&1 | tee "${LOG_DIR}/precompute.log"

echo "[step] start training" | tee -a "${LOG_DIR}/session.log"
bash scripts/train_zero1.sh 8 task=robotwin_subset4_uncond_3cam_384_1e-4 \
  2>&1 | tee "${LOG_DIR}/train.log"

echo "[done] $(date -Iseconds)" | tee -a "${LOG_DIR}/session.log"
