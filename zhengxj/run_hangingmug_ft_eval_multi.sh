#!/usr/bin/env bash
# Evaluate one or more FT checkpoints of the hanging_mug subtask WAM under
# two conditions, ALL IN PARALLEL across one GPU per (ckpt, condition) job:
#   b1_ft_raw     : FT WAM with the original raw task instruction, no VLM
#                   planner. Ablates the effect of FT alone -- detects whether
#                   FT caused catastrophic forgetting on raw instructions.
#                   Compare against M1 (pretrained + raw) = 20% baseline.
#   b1_ft_cl_free : FT WAM + closed-loop VLM (free_form mode). This matches
#                   our training distribution: the dataset paraphrases were
#                   generated free-form by VLM+LLM (visually grounded), not
#                   constrained to the canonical menu strings. Compare
#                   against M3-cl-free (pretrained + cl-free) = 40% baseline.
#
# GPU layout: jobs are pinned to GPUs from $GPUS (default "0 1 2 3 4 5 6 7").
# 2 conditions x N_CKPTS jobs must be <= number of GPUs (script errors out
# otherwise so it stays fully parallel).
#
# Usage:
#   export DASHSCOPE_API_KEY=...
#
#   # Auto-pick every step_*.pt in the latest FT run:
#   bash zhengxj/run_hangingmug_ft_eval_multi.sh
#
#   # Whitelist specific step numbers:
#   FT_STEPS="100 200 300" bash zhengxj/run_hangingmug_ft_eval_multi.sh
#
#   # Explicit ckpt list (overrides auto-detect):
#   FT_CKPTS="./runs/.../weights/step_000100.pt ..." \
#     bash zhengxj/run_hangingmug_ft_eval_multi.sh
#
# Optional env overrides:
#   FT_STATS                (default: auto-detect from each ckpt path)
#   EVAL_EPISODES=10
#   VLM_REPLAN_K=3
#   VLM_THINK=false
#   GPUS="0 1 2 3 4 5 6 7"  (whitespace-separated GPU ids to use)
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

# -- Resolve ckpt list ---------------------------------------------------------
if [ -z "${FT_CKPTS:-}" ]; then
  RUN_ROOT_DIR="./runs/robotwin_hangingmug_subtask_ft_3cam_384_1e-4"
  LATEST_RUN=$(ls -1dt ${RUN_ROOT_DIR}/2026-* 2>/dev/null | head -1)
  if [ -z "${LATEST_RUN}" ]; then
    echo "ERROR: no run dirs found under ${RUN_ROOT_DIR} and FT_CKPTS not set." >&2
    exit 2
  fi
  WEIGHTS_DIR="${LATEST_RUN}/checkpoints/weights"
  if [ -n "${FT_STEPS:-}" ]; then
    CANDIDATES=""
    for step in ${FT_STEPS}; do
      f="${WEIGHTS_DIR}/step_$(printf '%06d' ${step}).pt"
      if [ -f "${f}" ]; then
        CANDIDATES="${CANDIDATES} ${f}"
      else
        echo "[warn] requested step ${step} not found at ${f}, skipping"
      fi
    done
    FT_CKPTS="${CANDIDATES# }"
  else
    FT_CKPTS=$(ls -1 "${WEIGHTS_DIR}"/step_*.pt 2>/dev/null | sort | tr '\n' ' ')
  fi
  if [ -z "${FT_CKPTS}" ]; then
    echo "ERROR: no step_*.pt found in ${WEIGHTS_DIR}" >&2
    exit 2
  fi
  echo "[info] Auto-detected ckpts (latest run: ${LATEST_RUN}):"
  for f in ${FT_CKPTS}; do echo "  ${f}"; done
fi

EVAL_EPISODES="${EVAL_EPISODES:-10}"
VLM_REPLAN_K="${VLM_REPLAN_K:-3}"
VLM_THINK="${VLM_THINK:-false}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

# -- Build (ckpt, condition, gpu) job table ------------------------------------
# Read ckpts and gpus into arrays.
read -r -a CKPT_ARR <<< "${FT_CKPTS}"
read -r -a GPU_ARR  <<< "${GPUS}"
N_CKPT=${#CKPT_ARR[@]}
N_GPU=${#GPU_ARR[@]}
N_JOB=$((N_CKPT * 2))

if [ ${N_JOB} -gt ${N_GPU} ]; then
  echo "ERROR: ${N_JOB} jobs but only ${N_GPU} GPUs available." >&2
  echo "       Either reduce ckpts (FT_STEPS=...) or add more GPUs (GPUS=...)." >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
EVAL_RUN_ROOT="evaluate_results/robotwin/ft_eval/${TS}_multi"
mkdir -p "${EVAL_RUN_ROOT}"

echo "=========================================="
echo "Multi-checkpoint FT eval (FULLY PARALLEL)"
echo "=========================================="
echo "Run root:       ${EVAL_RUN_ROOT}"
echo "EVAL_EPISODES:  ${EVAL_EPISODES}"
echo "VLM_REPLAN_K:   ${VLM_REPLAN_K}"
echo "VLM_THINK:      ${VLM_THINK}"
echo "GPUS:           ${GPUS}"
echo "Ckpts:          ${N_CKPT} (${N_JOB} jobs across ${N_GPU} GPUs)"
echo "=========================================="

SUMMARY="${EVAL_RUN_ROOT}/_summary.tsv"
printf "ckpt_tag\tcondition\tgpu\tsuccess_rate\tresult_file\n" > "${SUMMARY}"

declare -a PIDS=()
declare -a JOB_DESCS=()

# 2 conditions per ckpt; assign GPUs round-robin (raw gets ckpt_i*2, cl_free gets ckpt_i*2+1).
job_idx=0
for ckpt_i in "${!CKPT_ARR[@]}"; do
  FT_CKPT="${CKPT_ARR[$ckpt_i]}"
  if [ ! -f "${FT_CKPT}" ]; then
    echo "[warn] ckpt does not exist, skipping: ${FT_CKPT}"
    continue
  fi
  FT_STATS_FOR_CKPT="${FT_STATS:-}"
  if [ -z "${FT_STATS_FOR_CKPT}" ]; then
    FT_STATS_FOR_CKPT="$(dirname "$(dirname "$(dirname "${FT_CKPT}")")")/dataset_stats.json"
  fi
  if [ ! -f "${FT_STATS_FOR_CKPT}" ]; then
    echo "[error] dataset_stats.json not found at ${FT_STATS_FOR_CKPT} (for ${FT_CKPT})" >&2
    continue
  fi

  CKPT_TAG="$(basename "${FT_CKPT}" .pt)"
  CKPT_OUT_DIR="${EVAL_RUN_ROOT}/${CKPT_TAG}"
  mkdir -p "${CKPT_OUT_DIR}"

  for cond in "b1_ft_raw off none" "b1_ft_cl_free closed_loop free_form"; do
    # shellcheck disable=SC2086
    set -- ${cond}
    tag="$1"; vlm_mode="$2"; subtask_mode="$3"
    gpu="${GPU_ARR[$job_idx]}"
    job_idx=$((job_idx + 1))

    out_dir="${CKPT_OUT_DIR}/${tag}"
    mkdir -p "${out_dir}"
    log_file="${out_dir}/launcher.log"

    echo "[$(date +%H:%M:%S)] launch ckpt=${CKPT_TAG} tag=${tag} vlm_mode=${vlm_mode} subtask_mode=${subtask_mode} gpu=${gpu}"

    CMD=(
      python experiments/robotwin/eval_robotwin_single.py
      task=robotwin_uncond_3cam_384_1e-4
      ckpt="${FT_CKPT}"
      EVALUATION.dataset_stats_path="${FT_STATS_FOR_CKPT}"
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
      echo "[$(date +%H:%M:%S)] DONE ckpt=${CKPT_TAG} tag=${tag} gpu=${gpu} rc=${rc}" \
        | tee -a "${EVAL_RUN_ROOT}/_completion.log"
    ) &

    PIDS+=("$!")
    JOB_DESCS+=("${CKPT_TAG}::${tag}::gpu${gpu}")
    sleep 2
  done
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting for all to finish..."
echo ""

STATUS_LOG="${EVAL_RUN_ROOT}/_status.log"
while true; do
  alive=0
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  echo "[$(date +%H:%M:%S)] alive=${alive}/${#PIDS[@]}" | tee -a "${STATUS_LOG}"
  if [ "${alive}" -eq 0 ]; then
    break
  fi
  sleep 60
done

echo ""
echo "=========================================="
echo "ALL JOBS DONE"
echo "=========================================="
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"
  rc=$?
  echo "  ${JOB_DESCS[$i]}: rc=${rc}"
done

echo ""
echo "Result files (success rates):"
for desc in "${JOB_DESCS[@]}"; do
  ckpt_tag="${desc%%::*}"
  rest="${desc#*::}"
  tag="${rest%%::*}"
  gpu_tag="${rest##*::}"
  result_file=$(find "${EVAL_RUN_ROOT}/${ckpt_tag}/${tag}" -name "_result_random.txt" 2>/dev/null | head -1)
  if [ -n "${result_file}" ]; then
    rate=$(tail -1 "${result_file}")
    echo "  ${ckpt_tag}::${tag} (${gpu_tag}): ${rate}  (${result_file})"
    printf "%s\t%s\t%s\t%s\t%s\n" "${ckpt_tag}" "${tag}" "${gpu_tag}" "${rate}" "${result_file}" >> "${SUMMARY}"
  else
    echo "  ${ckpt_tag}::${tag} (${gpu_tag}): MISSING"
    printf "%s\t%s\t%s\t%s\t%s\n" "${ckpt_tag}" "${tag}" "${gpu_tag}" "MISSING" "" >> "${SUMMARY}"
  fi
done

echo ""
echo "Summary file: ${SUMMARY}"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' < "${SUMMARY}"
else
  cat "${SUMMARY}"
fi
