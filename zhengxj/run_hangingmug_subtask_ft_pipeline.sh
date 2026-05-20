#!/usr/bin/env bash
# End-to-end pipeline for the hanging_mug subtask fine-tuning experiment.
# Assumes that scripts/segment_hangingmug_subtasks.py and the VLM/LLM
# labeling pipeline (zhengxj/run_hangingmug_labeling.sh) have already produced:
#   - data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_segments.json
#   - data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_label_cache.json
#
# Stages run by this script:
#   1. Assemble the relabeled LeRobot dataset.
#   2. Precompute T5 text embeddings for every paraphrase.
#   3. Fine-tune the released pretrained FastWAM checkpoint.
set -euo pipefail

cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh 2>/dev/null || true
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2

NPROC=${NPROC:-8}
TASK=${TASK:-robotwin_hangingmug_subtask_ft_3cam_384_1e-4}
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/hangingmug_subtask_ft_${RUN_TS}"
mkdir -p "${LOG_DIR}"

echo "[$(date -Iseconds)] === STEP 1: build dataset ===" | tee -a "${LOG_DIR}/session.log"
python -u scripts/build_hangingmug_subtask_dataset.py --overwrite 2>&1 | tee "${LOG_DIR}/build_dataset.log"

echo "[$(date -Iseconds)] === STEP 2: precompute T5 text embeddings ===" | tee -a "${LOG_DIR}/session.log"
torchrun --standalone --nproc_per_node="${NPROC}" scripts/precompute_text_embeds.py \
    task="${TASK}" 2>&1 | tee "${LOG_DIR}/precompute.log"

echo "[$(date -Iseconds)] === STEP 3: fine-tune ===" | tee -a "${LOG_DIR}/session.log"
bash scripts/train_zero1.sh "${NPROC}" task="${TASK}" 2>&1 | tee "${LOG_DIR}/train.log"

echo "[$(date -Iseconds)] === DONE pipeline === log_dir=${LOG_DIR}" | tee -a "${LOG_DIR}/session.log"
