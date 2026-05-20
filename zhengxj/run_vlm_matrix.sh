#!/usr/bin/env bash
# Launch 4 tasks x 2 conditions (baseline + VLM) in parallel across 8 GPUs.
# Each job runs `EVAL_EPISODES` episodes of `demo_randomized` and writes its
# results under evaluate_results/robotwin/.../vlm_matrix/<task>_<cond>/.
#
# Usage:
#   export DASHSCOPE_API_KEY=...
#   bash zhengxj/run_vlm_matrix.sh
#
# Notes:
# - Designed to run inside a tmux session (`tmux new -s vlm_matrix`).
# - Joins all 8 child PIDs before exiting; status of each is printed at the end.

set -u

cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM

source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh 2>/dev/null || true
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2

export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOTWIN_FORCE_MPLIB=1

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "ERROR: DASHSCOPE_API_KEY is not set. Export it before running." >&2
  exit 2
fi

CKPT="./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
STATS="./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
EVAL_EPISODES="${EVAL_EPISODES:-100}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="evaluate_results/robotwin/vlm_matrix/${TS}"
mkdir -p "${RUN_ROOT}"
echo "Run root: ${RUN_ROOT}"

# (task, condition, gpu_id)
declare -a JOBS=(
  "hanging_mug      baseline 0"
  "hanging_mug      vlm      1"
  "open_microwave   baseline 2"
  "open_microwave   vlm      3"
  "place_can_basket baseline 4"
  "place_can_basket vlm      5"
  "turn_switch      baseline 6"
  "turn_switch      vlm      7"
)

declare -a PIDS=()
declare -a JOB_TAGS=()

for job in "${JOBS[@]}"; do
  # shellcheck disable=SC2206
  parts=(${job})
  task="${parts[0]}"
  cond="${parts[1]}"
  gpu="${parts[2]}"
  tag="${task}_${cond}"

  if [ "${cond}" = "vlm" ]; then
    vlm_mode="open_loop"
  else
    vlm_mode="off"
  fi

  out_dir="${RUN_ROOT}/${tag}"
  mkdir -p "${out_dir}"
  log_file="${out_dir}/launcher.log"

  echo "[$(date +%H:%M:%S)] launch task=${task} cond=${cond} gpu=${gpu} log=${log_file}"

  (
    python experiments/robotwin/eval_robotwin_single.py \
      task=robotwin_uncond_3cam_384_1e-4 \
      ckpt="${CKPT}" \
      EVALUATION.dataset_stats_path="${STATS}" \
      EVALUATION.task_name="${task}" \
      EVALUATION.task_config=demo_randomized \
      EVALUATION.eval_num_episodes="${EVAL_EPISODES}" \
      EVALUATION.skip_get_obs_within_replan=false \
      EVALUATION.device=cuda \
      EVALUATION.vlm_mode="${vlm_mode}" \
      EVALUATION.output_dir="./${out_dir}" \
      mixed_precision=fp16 \
      gpu_id="${gpu}" \
      > "${log_file}" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] DONE task=${task} cond=${cond} gpu=${gpu} rc=${rc}" \
      | tee -a "${RUN_ROOT}/_completion.log"
  ) &

  PIDS+=("$!")
  JOB_TAGS+=("${tag}")
  sleep 2
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting for all to finish..."
echo "Run root: ${RUN_ROOT}"
echo ""

# Track progress every 60s while jobs are still alive.
SUMMARY="${RUN_ROOT}/_status.log"
while true; do
  alive=0
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  echo "[$(date +%H:%M:%S)] alive=${alive}/${#PIDS[@]}" | tee -a "${SUMMARY}"
  if [ "${alive}" -eq 0 ]; then
    break
  fi
  sleep 60
done

echo ""
echo "=========================================="
echo "All jobs finished. Per-job status:"
echo "=========================================="
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"
  rc=$?
  echo "  ${JOB_TAGS[$i]}: rc=${rc}"
done

echo ""
echo "Run root: ${RUN_ROOT}"
echo "Result files (success rates):"
for tag in "${JOB_TAGS[@]}"; do
  result_file=$(ls "${RUN_ROOT}/${tag}"/**/_result_random.txt 2>/dev/null | head -1)
  if [ -z "${result_file}" ]; then
    result_file=$(find "${RUN_ROOT}/${tag}" -name "_result_random.txt" 2>/dev/null | head -1)
  fi
  if [ -n "${result_file}" ]; then
    rate=$(tail -1 "${result_file}")
    echo "  ${tag}: ${rate}  (${result_file})"
  else
    echo "  ${tag}: MISSING"
  fi
done
