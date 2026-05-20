#!/usr/bin/env bash
# VLM + LLM labeling pipeline for hanging_mug subtask dataset.
# Designed to be run inside a tmux session for monitoring.
set -euo pipefail

cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh 2>/dev/null || true
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2

export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-sk-dab16332be144b8193e988b4e63f4206}

VLM_WORKERS=${VLM_WORKERS:-16}
LLM_WORKERS=${LLM_WORKERS:-16}
FLUSH_EVERY=${FLUSH_EVERY:-8}

echo "[$(date -Iseconds)] === START VLM LABELING (workers=${VLM_WORKERS}) ==="
python -u scripts/label_hangingmug_subtasks_vlm.py \
    --workers "${VLM_WORKERS}" \
    --flush-every "${FLUSH_EVERY}" \
    --request-timeout 120

echo
echo "[$(date -Iseconds)] === START LLM PARAPHRASING (workers=${LLM_WORKERS}) ==="
python -u scripts/label_hangingmug_subtasks_llm.py \
    --workers "${LLM_WORKERS}" \
    --flush-every "${FLUSH_EVERY}" \
    --request-timeout 90

echo
echo "[$(date -Iseconds)] === DONE labeling pipeline ==="
