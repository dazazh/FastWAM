#!/usr/bin/env bash
# Launch 4 tasks x 2 closed-loop conditions (menu + free_form) in parallel
# across 8 GPUs. Each job runs `EVAL_EPISODES` episodes of `demo_randomized`
# and writes its results under
# evaluate_results/robotwin/.../vlm_matrix/<task>_<cond>/.
#
# Conditions:
#   vlm_cl_menu : M3 closed-loop, picks subtask_index from
#                 experiments/robotwin/fastwam_policy/subtask_menus.json
#                 then writes a visually-grounded enriched_instruction.
#   vlm_cl_free : M3 ablation, no menu validation - VLM emits a free-form
#                 short instruction every K replan boundaries.
#
# K (chunks between VLM calls) defaults to 2 (~1 s of robot motion).
#
# Usage:
#   export DASHSCOPE_API_KEY=...
#   bash zhengxj/run_vlm_cl_matrix.sh
#
# Optional env overrides:
#   EVAL_EPISODES=10
#   VLM_REPLAN_K=2
#   VLM_MAX_CONT=4
#
# Notes:
# - Designed to run inside a tmux session (`tmux new -s vlm_cl_matrix`).
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
VLM_REPLAN_K="${VLM_REPLAN_K:-3}"
# Closed-loop runs are dominated by VLM latency. We disable Qwen thinking by
# default to bring per-call latency from ~60 s to ~5-10 s. Override with
# VLM_THINK=true if you want to re-enable thinking for an ablation.
VLM_THINK="${VLM_THINK:-false}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="evaluate_results/robotwin/vlm_matrix/${TS}_cl"
mkdir -p "${RUN_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "EVAL_EPISODES=${EVAL_EPISODES}  VLM_REPLAN_K=${VLM_REPLAN_K}  VLM_THINK=${VLM_THINK}"

# (task, condition, gpu_id)
# cond is one of {vlm_cl_menu, vlm_cl_free}; consumed as the policy subtask_mode.
declare -a JOBS=(
  "hanging_mug      vlm_cl_menu 0"
  "hanging_mug      vlm_cl_free 1"
  "open_microwave   vlm_cl_menu 2"
  "open_microwave   vlm_cl_free 3"
  "place_can_basket vlm_cl_menu 4"
  "place_can_basket vlm_cl_free 5"
  "turn_switch      vlm_cl_menu 6"
  "turn_switch      vlm_cl_free 7"
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

  case "${cond}" in
    vlm_cl_menu) subtask_mode="menu" ;;
    vlm_cl_free) subtask_mode="free_form" ;;
    *)
      echo "Unknown cond=${cond}" >&2
      exit 2
      ;;
  esac

  out_dir="${RUN_ROOT}/${tag}"
  mkdir -p "${out_dir}"
  log_file="${out_dir}/launcher.log"

  echo "[$(date +%H:%M:%S)] launch task=${task} cond=${cond} subtask_mode=${subtask_mode} gpu=${gpu} log=${log_file}"

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
      EVALUATION.vlm_mode=closed_loop \
      EVALUATION.vlm_subtask_mode="${subtask_mode}" \
      EVALUATION.vlm_replan_every_k_chunks="${VLM_REPLAN_K}" \
      EVALUATION.vlm_enable_thinking="${VLM_THINK}" \
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
