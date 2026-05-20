cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh 2>/dev/null || true
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2

export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOTWIN_FORCE_MPLIB=1
export DASHSCOPE_API_KEY=sk-dab16332be144b8193e988b4e63f4206

OUT="./evaluate_results/robotwin/vlm_matrix/$(date +%Y%m%d_%H%M%S)_smoke/hanging_mug_vlm_cl_menu"
mkdir -p "${OUT}"

python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=hanging_mug \
  EVALUATION.task_config=demo_randomized \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.skip_get_obs_within_replan=false \
  EVALUATION.device=cuda \
  EVALUATION.vlm_mode=closed_loop \
  EVALUATION.vlm_subtask_mode=menu \
  EVALUATION.vlm_replan_every_k_chunks=3 \
  EVALUATION.vlm_enable_thinking=false \
  EVALUATION.output_dir="./evaluate_results/robotwin/vlm_matrix/$(date +%Y%m%d_%H%M%S)_smoke/hanging_mug_vlm_cl_menu" \
  mixed_precision=fp16 \
  gpu_id=0