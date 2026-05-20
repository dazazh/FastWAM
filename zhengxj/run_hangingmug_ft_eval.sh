#!/usr/bin/env bash
# Evaluate the fine-tuned hanging_mug WAM under two conditions:
#   B1-FT-raw     : FT WAM with the original raw task instruction, no VLM planner.
#                   Ablates the effect of FT alone.
#   B1-FT-cl-menu : FT WAM + closed-loop VLM (menu mode, enriched_instruction).
#                   Main bonus result.
#
# Runs both conditions in parallel on two separate GPUs.
#
# Usage:
#   export DASHSCOPE_API_KEY=...
#   FT_CKPT=./runs/robotwin_hangingmug_subtask_ft_3cam_384_1e-4/<run_id>/checkpoints/weights/step_000200.pt \
#   FT_STATS=./runs/robotwin_hangingmug_subtask_ft_3cam_384_1e-4/<run_id>/dataset_stats.json \
#   bash zhengxj/run_hangingmug_ft_eval.sh
#
# Optional env overrides:
#   EVAL_EPISODES=10
#   VLM_REPLAN_K=3
#   VLM_THINK=false
#   B1_RAW_GPU=0
#   B1_CL_GPU=1
set -u

cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh 2>/dev/null || true
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2

export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOTWIN_FORCE_MPLIB=1

if [ -z "${FT_CKPT:-}" ]; then
  echo "ERROR: FT_CKPT must be set. Example:" >&2
  echo "  FT_CKPT=./runs/.../checkpoints/<step>/model.pt bash $0" >&2
  exit 2
fi
if [ -z "${FT_STATS:-}" ]; then
  # FT_CKPT is at runs/<task>/<run_id>/checkpoints/weights/step_NNNNNN.pt
  # FT_STATS lives at runs/<task>/<run_id>/dataset_stats.json (4 levels up from the .pt).
  FT_STATS="$(dirname "$(dirname "$(dirname "$(dirname "${FT_CKPT}")")")")/dataset_stats.json"
  echo "[info] FT_STATS not set; defaulting to ${FT_STATS}"
fi

if [ ! -f "${FT_CKPT}" ]; then
  echo "ERROR: FT_CKPT does not exist: ${FT_CKPT}" >&2
  exit 2
fi
if [ ! -f "${FT_STATS}" ]; then
  echo "ERROR: FT_STATS does not exist: ${FT_STATS}" >&2
  exit 2
fi

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "ERROR: DASHSCOPE_API_KEY is not set. Export it before running." >&2
  exit 2
fi

EVAL_EPISODES="${EVAL_EPISODES:-10}"
VLM_REPLAN_K="${VLM_REPLAN_K:-3}"
VLM_THINK="${VLM_THINK:-false}"
B1_RAW_GPU="${B1_RAW_GPU:-0}"
B1_CL_GPU="${B1_CL_GPU:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
CKPT_TAG="$(basename "$(dirname "${FT_CKPT}")")"
RUN_ROOT="evaluate_results/robotwin/ft_eval/${TS}_${CKPT_TAG}"
mkdir -p "${RUN_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "FT_CKPT=${FT_CKPT}"
echo "FT_STATS=${FT_STATS}"
echo "EVAL_EPISODES=${EVAL_EPISODES}  VLM_REPLAN_K=${VLM_REPLAN_K}  VLM_THINK=${VLM_THINK}"

declare -a JOBS=(
  "b1_ft_raw     off  none  ${B1_RAW_GPU}"
  "b1_ft_cl_menu closed_loop menu ${B1_CL_GPU}"
)

declare -a PIDS=()
declare -a JOB_TAGS=()

for job in "${JOBS[@]}"; do
  # shellcheck disable=SC2206
  parts=(${job})
  tag="${parts[0]}"
  vlm_mode="${parts[1]}"
  subtask_mode="${parts[2]}"
  gpu="${parts[3]}"

  out_dir="${RUN_ROOT}/${tag}"
  mkdir -p "${out_dir}"
  log_file="${out_dir}/launcher.log"

  echo "[$(date +%H:%M:%S)] launch tag=${tag} vlm_mode=${vlm_mode} subtask_mode=${subtask_mode} gpu=${gpu} log=${log_file}"

  # Build base command. For vlm_mode=off, subtask_mode is irrelevant.
  CMD=(
    python experiments/robotwin/eval_robotwin_single.py
    task=robotwin_uncond_3cam_384_1e-4
    ckpt="${FT_CKPT}"
    EVALUATION.dataset_stats_path="${FT_STATS}"
    EVALUATION.task_name=hanging_mug
    EVALUATION.task_config=demo_randomized
    EVALUATION.eval_num_episodes="${EVAL_EPISODES}"
    EVALUATION.skip_get_obs_within_replan=false
    EVALUATION.device=cuda
    EVALUATION.vlm_mode="${vlm_mode}"
    EVALUATION.vlm_replan_every_k_chunks="${VLM_REPLAN_K}"
    EVALUATION.vlm_enable_thinking="${VLM_THINK}"
    EVALUATION.output_dir="./${out_dir}"
    mixed_precision=fp16
    gpu_id="${gpu}"
  )
  if [ "${subtask_mode}" != "none" ]; then
    CMD+=(EVALUATION.vlm_subtask_mode="${subtask_mode}")
  fi

  (
    "${CMD[@]}" > "${log_file}" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] DONE tag=${tag} gpu=${gpu} rc=${rc}" \
      | tee -a "${RUN_ROOT}/_completion.log"
  ) &

  PIDS+=("$!")
  JOB_TAGS+=("${tag}")
  sleep 3
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting for all to finish..."
echo "Run root: ${RUN_ROOT}"
echo ""

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
  result_file=$(find "${RUN_ROOT}/${tag}" -name "_result_random.txt" 2>/dev/null | head -1)
  if [ -n "${result_file}" ]; then
    rate=$(tail -1 "${result_file}")
    echo "  ${tag}: ${rate}  (${result_file})"
  else
    echo "  ${tag}: MISSING"
  fi
done
