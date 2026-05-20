#!/usr/bin/env bash
# turn_switch 100-episode eval, 2 conditions in parallel on 2 GPUs.
# Mirrors zhengxj/run_vlm_matrix.sh and zhengxj/run_vlm_cl_matrix.sh patterns,
# but restricted to turn_switch (open-loop + closed-loop free_form) so all
# 100 episodes per condition stay on one GPU (matches teammates' standard
# `seed=0, eval_num_episodes=100` recipe).
#
# Conditions:
#   vlm        : M2 open-loop VLM planner. One VLM call per episode at the
#                first head-camera frame; returns a refined instruction the
#                WAM follows for the whole rollout.
#   vlm_cl_free: M3 closed-loop VLM planner, free_form subtask mode. VLM
#                emits a fresh short instruction every K replan boundaries.
#                Override with CL_MODE=menu for the menu variant.
#
# K (chunks between VLM calls) defaults to 2 (~1 s of robot motion), to match
# run_vlm_cl_matrix.sh defaults.
#
# Usage:
#   export DASHSCOPE_API_KEY=...
#   bash zhengxj/run_turn_switch_100ep_eval.sh
#
# Optional env overrides:
#   EVAL_EPISODES=100
#   CL_MODE=free_form        # or "menu"
#   VLM_REPLAN_K=2
#   VLM_THINK=false
#   OPEN_GPU=0
#   CL_GPU=1
#
# Notes:
# - Designed to run inside a tmux session (`tmux new -s turn_switch_100ep`).
# - 100 episodes per condition takes ~5-6 h sequentially on one GPU.

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
CL_MODE="${CL_MODE:-free_form}"
VLM_REPLAN_K="${VLM_REPLAN_K:-3}"
VLM_THINK="${VLM_THINK:-false}"
OPEN_GPU="${OPEN_GPU:-0}"
CL_GPU="${CL_GPU:-1}"

case "${CL_MODE}" in
  menu|free_form) ;;
  *) echo "ERROR: CL_MODE must be 'menu' or 'free_form', got '${CL_MODE}'" >&2; exit 2 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="evaluate_results/robotwin/turn_switch_100ep/${TS}"
mkdir -p "${RUN_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "EVAL_EPISODES=${EVAL_EPISODES}  CL_MODE=${CL_MODE}  VLM_REPLAN_K=${VLM_REPLAN_K}  VLM_THINK=${VLM_THINK}"

# (task, condition, gpu_id)
# cond is one of {vlm, vlm_cl_free, vlm_cl_menu}.
declare -a JOBS=(
  "turn_switch vlm           ${OPEN_GPU}"
  "turn_switch vlm_cl_${CL_MODE} ${CL_GPU}"
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
    vlm)
      vlm_mode="open_loop"
      subtask_mode=""
      ;;
    vlm_cl_menu)
      vlm_mode="closed_loop"
      subtask_mode="menu"
      ;;
    vlm_cl_free_form|vlm_cl_free)
      vlm_mode="closed_loop"
      subtask_mode="free_form"
      ;;
    *)
      echo "Unknown cond=${cond}" >&2
      exit 2
      ;;
  esac

  out_dir="${RUN_ROOT}/${tag}"
  mkdir -p "${out_dir}"
  log_file="${out_dir}/launcher.log"

  echo "[$(date +%H:%M:%S)] launch task=${task} cond=${cond} vlm_mode=${vlm_mode} subtask_mode=${subtask_mode} gpu=${gpu} log=${log_file}"

  (
    if [ -n "${subtask_mode}" ]; then
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
        EVALUATION.vlm_subtask_mode="${subtask_mode}" \
        EVALUATION.vlm_replan_every_k_chunks="${VLM_REPLAN_K}" \
        EVALUATION.vlm_enable_thinking="${VLM_THINK}" \
        EVALUATION.output_dir="./${out_dir}" \
        mixed_precision=fp16 \
        gpu_id="${gpu}" \
        > "${log_file}" 2>&1
    else
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
        EVALUATION.vlm_enable_thinking="${VLM_THINK}" \
        EVALUATION.output_dir="./${out_dir}" \
        mixed_precision=fp16 \
        gpu_id="${gpu}" \
        > "${log_file}" 2>&1
    fi
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
echo "(100 episodes per condition takes ~5-6 h on one GPU.)"
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
  sleep 300
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
echo "Result files (success rates over ${EVAL_EPISODES} episodes):"
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
