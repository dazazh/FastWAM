# WAM-CoT @ FastWAM — Inference-time VLM Planner & Subtask-aligned FT

This document summarizes what we built for the SII WAM-CoT competition track inside this repo. Everything under `zhengxj/` plus the new scripts/configs listed below was produced for this project; the upstream FastWAM code under `src/`, `experiments/robotwin/`, and `third_party/RoboTwin/` is unchanged except for the small workarounds noted in [zhengxj/EVAL_ROBOTWIN_SERVER.md](EVAL_ROBOTWIN_SERVER.md).

Plan / motivation: [zhengxj/WAM_CoT_plan.md](WAM_CoT_plan.md).
Competition brief: [课题详情.md](../课题详情.md).

---

## 1. TL;DR

We follow the *external Semantic CoT* track of WAM-CoT: a frozen VLM acts as a high-level planner that produces visually-grounded instructions, and a low-level WAM executes them. Two complementary contributions:

- **Track A — Inference-time VLM-CoT (training-free).**
  Wraps the released pretrained FastWAM (`robotwin_uncond_3cam_384.pt`) with a Qwen3-VL-Plus planner in two modes:
  - *Open-loop (M2)*: VLM called once on the first frame; refined instruction conditions the whole rollout.
  - *Closed-loop (M3)*: VLM called every K replan boundaries with the current frame; can pick a subtask from a per-task **menu** scaffold or emit a **free-form** short instruction.
- **Track B — Subtask-aligned FT (bonus).**
  Build a subtask-relabeled `hanging_mug` LeRobot dataset (gripper-event segmentation + VLM+LLM visually-grounded paraphrases) and fine-tune the pretrained WAM on it, so the low-level policy is in-distribution for the VLM planner's short instructions.

Both tracks share the same eval harness (`experiments/robotwin/eval_robotwin_single.py`) and the same closed-loop planner code (`experiments/robotwin/fastwam_policy/vlm_planner.py`).

---

## 2. System architecture

```text
              ┌─────────────────────────────────────────────┐
              │  Qwen3-VL-Plus (DashScope OpenAI-compatible) │
              │  - open-loop:   first frame → refined instr  │
              │  - closed-loop: every K chunks → enriched    │
              │                  (menu or free_form)         │
              └────────────────────────┬────────────────────┘
                                       │ short imperative English
                                       ▼
       ┌──────────────────────────────────────────────────────┐
       │  FastWAM low-level policy (Wan2.2-TI2V-5B backbone)  │
       │  - Track A: released pretrained 50-task checkpoint   │
       │  - Track B: subtask-FT checkpoint (this work)        │
       │  - input : video (3 cam) + state(14) + text embed    │
       │  - output: action chunk(14 dof, 32 steps)            │
       └──────────────────────────────────────────────────────┘
                                       │
                                       ▼
                                   RoboTwin sim
```

Per-task subtask menus (used by closed-loop `menu` mode AND as anchor strings in Track B's training data) live in [experiments/robotwin/fastwam_policy/subtask_menus.json](../experiments/robotwin/fastwam_policy/subtask_menus.json). 4 stages for hanging_mug, 2 for open_microwave, 3 for place_can_basket, 1 for turn_switch.

---

## 3. Track A — Inference-time VLM-CoT (training-free)

### What it is

- Open-loop planner: [experiments/robotwin/fastwam_policy/vlm_planner.py](../experiments/robotwin/fastwam_policy/vlm_planner.py) `VLMPlanner`. One VLM call per episode, cached by `(task, seed)`.
- Closed-loop planner: same file, `ClosedLoopVLMPlanner`. Called every `K = vlm_replan_every_k_chunks` chunks. `subtask_mode = "menu"` returns `(subtask_index, enriched_instruction)`; `subtask_mode = "free_form"` returns a free instruction only. Cached by `(task, seed, chunk, mode)`.
- All VLM calls swallow errors and fall back to the previous / raw instruction, so eval never crashes from an API hiccup.

### How we ran it

All evals use `task_config=demo_randomized`, `instruction_type=unseen`, `seed=0` so episode seeds are `100000..100099`.

Matrix scripts (parallel across 8 GPUs):

- [zhengxj/run_vlm_matrix.sh](run_vlm_matrix.sh) — 4 tasks × {baseline, open-loop VLM}.
- [zhengxj/run_vlm_cl_matrix.sh](run_vlm_cl_matrix.sh) — 4 tasks × {closed-loop menu, closed-loop free}.
- [zhengxj/run_vlm_cl_smoke_test.sh](run_vlm_cl_smoke_test.sh) — 1-episode sanity check.
- [zhengxj/run_turn_switch_100ep_eval.sh](run_turn_switch_100ep_eval.sh) — 100-episode follow-up on turn_switch only, 2 GPUs.

Summary writer:

- [zhengxj/summarize_vlm_matrix.py](summarize_vlm_matrix.py) — collects per-task `_result_random.txt`s and renders the Markdown table in `results_vlm_training_free.md`.

### Results (10 episodes per cell unless noted)

Numbers from [zhengxj/results_vlm_training_free.md](results_vlm_training_free.md) (run root `evaluate_results/robotwin/vlm_matrix/20260519_115312`). All cells use the released pretrained WAM.

| Task | M1 baseline (raw) | M2 open-loop VLM | M3 closed-loop menu | M3 closed-loop free |
| --- | --- | --- | --- | --- |
| `hanging_mug` | 20% | 30% | 10% | 40% |
| `open_microwave` | 30% | 60% | 40% | 20% |
| `place_can_basket` | 30% | 60% | (TBD) | (TBD) |
| `turn_switch` | 40% | 40% | 70% | 70% |
| **mean** | **30%** | **47.5%** | — | — |

The 100-episode `turn_switch` follow-up (open-loop + closed-loop free) is currently running via [run_turn_switch_100ep_eval.sh](run_turn_switch_100ep_eval.sh) on GPUs 6 + 7. Results will land at `evaluate_results/robotwin/turn_switch_100ep/<TS>/`.

---

## 4. Track B — Subtask-aligned WAM fine-tuning (bonus)

### Idea

The pretrained WAM is trained on full-task instructions (e.g. *"pick the mug, rotate it, place it, hang on the rack"*). When the closed-loop VLM emits a short subtask instruction (*"pick up the black ceramic mug from the left side"*), this is mildly out-of-distribution. We construct a subtask dataset where each training frame is paired with a short visually-grounded instruction matching its current stage, and fine-tune.

### Pipeline (all data files live under `data/robotwin2.0_hangingmug_subtask_lerobot/`)

```text
1. gripper-event segmentation
   scripts/segment_hangingmug_subtasks.py
     state[:,6] (L gripper), state[:,13] (R gripper)
     -> per-episode boundaries: [0, t_L_close, t_L_open, t_R_close, t_R_open]
     -> meta/subtask_segments.json   (550 episodes, all 4-stage clean)

2. VLM labeling (Qwen3-VL-Plus via DashScope)
   scripts/label_hangingmug_subtasks_vlm.py
     for each (episode, stage) -> head-cam FIRST + LAST frames + original
     full task + canonical stage hint
     -> 1 short imperative English description, <= 400 chars
     -> meta/subtask_label_cache.json["vlm"]  (550 * 4 = 2200 calls, all OK)

3. LLM paraphrasing (Qwen3.6-Plus via DashScope)
   scripts/label_hangingmug_subtasks_llm.py
     for each VLM output -> 7 paraphrases (preserves arm + visual descriptors)
     -> meta/subtask_label_cache.json["llm"]  (2200 calls, all OK)

4. dataset assembly
   scripts/build_hangingmug_subtask_dataset.py
     for each frame in stage k of episode ep -> task_index randomly drawn
     from a pool of 9 strings:
       {1 VLM, 7 LLM paraphrases, 1 canonical anchor from subtask_menus.json}
     -> 550 episodes, 185793 frames, 17567 unique task strings.
     Videos hardlinked from data/robotwin2.0_subset4_lerobot/ to save disk.

5. precompute T5 text embeddings
   scripts/precompute_text_embeds.py (upstream)
     -> data/text_embeds_cache/robotwin_hangingmug_subtask/  (~18 GB)

6. fine-tune (ZeRO-1 deepspeed, 8 GPU)
   scripts/train_zero1.sh 8 task=robotwin_hangingmug_subtask_ft_3cam_384_1e-4
     resumes weights from checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
     -> runs/robotwin_hangingmug_subtask_ft_3cam_384_1e-4/<TS>/checkpoints/weights/step_*.pt

7. eval
   zhengxj/run_hangingmug_ft_eval_multi.sh
     for each FT ckpt:
       B1-FT-raw      (vlm_mode=off, raw instruction)         — catastrophic-forgetting ablation
       B1-FT-cl-free  (vlm_mode=closed_loop, free_form)       — main result
```

Wrapper scripts:

- [zhengxj/run_hangingmug_labeling.sh](run_hangingmug_labeling.sh) — chains step 2 and 3.
- [zhengxj/run_hangingmug_precompute_embeds.sh](run_hangingmug_precompute_embeds.sh) — step 5.
- [zhengxj/run_hangingmug_finetune.sh](run_hangingmug_finetune.sh) — step 6 (sets `NCCL_P2P_DISABLE=1` for this server's NVLink quirk).
- [zhengxj/run_hangingmug_subtask_ft_pipeline.sh](run_hangingmug_subtask_ft_pipeline.sh) — end-to-end orchestrator: build → precompute → train.

### Configs

- Dataset config: [configs/data/robotwin_hangingmug_subtask.yaml](../configs/data/robotwin_hangingmug_subtask.yaml). Uses the released pretrained model's `dataset_stats.json` so the FT model stays in the same numerical regime.
- Task config: [configs/task/robotwin_hangingmug_subtask_ft_3cam_384_1e-4.yaml](../configs/task/robotwin_hangingmug_subtask_ft_3cam_384_1e-4.yaml). `lr=1e-5`, `num_epochs=1`, `save_every=100`, `num_workers=2`, `resume=<released ckpt>`.

### Training observations (run `2026-05-20_09-52-30`)

- Throughput: ~16 s/step on 8× L20X with `NCCL_P2P_DISABLE=1` and `num_workers=2`. ETA for 1 epoch (1436 steps) ~6 h.
- Loss drops fast from the pretrained init:

| step | loss | loss_action | loss_video |
| --- | --- | --- | --- |
| 10 | 2.48 | — | — |
| 50 | 0.87 | — | — |
| 100 | 0.42 | 0.19 | 0.23 |
| 200 | ~0.28 | — | — |
| 300 | ~0.25 | — | — |
| ... | ... | ... | ... |

Checkpoints saved every 100 steps under `runs/robotwin_hangingmug_subtask_ft_3cam_384_1e-4/2026-05-20_09-52-30/checkpoints/weights/`.

### Lessons learned during FT

- **Free-tier API quota.** The first labeling attempt exhausted the qwen3-vl-plus free-tier quota after ~827 calls. Fix: top up balance / disable "Free Tier Only" mode in the DashScope console.
- **Wrong LLM model name.** `qwen3-plus` does not exist on DashScope; we use `qwen3.6-plus` for the paraphraser.
- **NVLink hardware quirk on this server.** First FT attempt crashed with `CUDA error: Invalid access of peer GPU memory over nvlink`. Workaround: `export NCCL_P2P_DISABLE=1` (routes ZeRO collectives over PCIe; ~10-15% slower per step but reliable). Already baked into `run_hangingmug_finetune.sh`.
- **OOM-killer on dataloader workers.** `num_workers=8` per rank × 8 ranks = 64 worker procs killed at step ~100 by the host OOM-killer. Dropped to `num_workers=2` (16 worker procs total). Already baked into the task config.

### Eval

[zhengxj/run_hangingmug_ft_eval_multi.sh](run_hangingmug_ft_eval_multi.sh) evaluates all (or a whitelisted set of) FT checkpoints in PARALLEL, one GPU per (ckpt, condition) job. Conditions:

| Tag | `vlm_mode` | `subtask_mode` | Comparison target |
| --- | --- | --- | --- |
| `b1_ft_raw` | `off` | — | M1 baseline = 20% on hanging_mug (catastrophic-forgetting check) |
| `b1_ft_cl_free` | `closed_loop` | `free_form` | M3 cl-free = 40% on hanging_mug (the win we're claiming) |

`free_form` is chosen over `menu` because the FT training distribution is itself free-form-VLM-generated visually-grounded sentences (the canonical menu strings are only 1 of 9 pool entries per frame, ~11%).

Single-ckpt variant: [zhengxj/run_hangingmug_ft_eval.sh](run_hangingmug_ft_eval.sh) (older 2-GPU script).

Results will be saved to `evaluate_results/robotwin/ft_eval/<TS>_multi/_summary.tsv` once the eval is run.

---

## 5. Quick-start reproduction

Server bootstrap (every shell):

```bash
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
conda activate fastwam_v2
cd /workspace/zhengxj2@xiaopeng.com/workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOTWIN_FORCE_MPLIB=1
export DASHSCOPE_API_KEY=<your-key>
```

(More environment details in [zhengxj/CONDA.md](CONDA.md) and [zhengxj/EVAL_ROBOTWIN_SERVER.md](EVAL_ROBOTWIN_SERVER.md).)

### A1. Track-A matrix eval (open-loop + closed-loop)

```bash
# 10 episodes per cell, all 4 tasks, open-loop + baseline:
tmux new -s vlm_matrix -d "bash zhengxj/run_vlm_matrix.sh"

# 10 episodes per cell, all 4 tasks, closed-loop menu + free:
tmux new -s vlm_cl_matrix -d "bash zhengxj/run_vlm_cl_matrix.sh"

# 100-episode follow-up on turn_switch only (open-loop + closed-loop free):
OPEN_GPU=6 CL_GPU=7 tmux new -s ts100 -d "bash zhengxj/run_turn_switch_100ep_eval.sh"
```

### A2. Rebuild the results markdown after evals finish

```bash
python zhengxj/summarize_vlm_matrix.py --run-root evaluate_results/robotwin/vlm_matrix/<TS> \
  --out zhengxj/results_vlm_training_free.md
```

### B1. Build the subtask dataset from scratch

```bash
# 1. Segment all 550 hanging_mug episodes (10 s).
python scripts/segment_hangingmug_subtasks.py

# 2 + 3. VLM + LLM labeling, ~15 min, ~$20 of API at current DashScope rates.
tmux new -s label -d "bash zhengxj/run_hangingmug_labeling.sh"

# 4. Assemble the LeRobot dataset (~1 min).
python scripts/build_hangingmug_subtask_dataset.py --overwrite

# 5. Precompute T5 text embeds on 8 GPUs (~15 min).
tmux new -s embeds -d "bash zhengxj/run_hangingmug_precompute_embeds.sh"

# 6. Fine-tune on 8 GPUs (~6 h for 1 epoch).
tmux new -s ft -d "bash zhengxj/run_hangingmug_finetune.sh"
```

End-to-end after step 1 has run:

```bash
bash zhengxj/run_hangingmug_subtask_ft_pipeline.sh
```

### B2. Eval all FT checkpoints in parallel (8 GPUs)

```bash
bash zhengxj/run_hangingmug_ft_eval_multi.sh
# or restrict to specific steps:
FT_STEPS="100 200 300" bash zhengxj/run_hangingmug_ft_eval_multi.sh
```

---

## 6. File layout

Files created by this project (relative to repo root):

```text
configs/
  data/robotwin_hangingmug_subtask.yaml                              # Track-B dataset config
  task/robotwin_hangingmug_subtask_ft_3cam_384_1e-4.yaml             # Track-B FT task config
scripts/
  segment_hangingmug_subtasks.py                                      # Track-B step 1
  label_hangingmug_subtasks_vlm.py                                    # Track-B step 2
  label_hangingmug_subtasks_llm.py                                    # Track-B step 3
  build_hangingmug_subtask_dataset.py                                 # Track-B step 4
data/
  robotwin2.0_hangingmug_subtask_lerobot/                             # Track-B LeRobot dataset
    meta/
      subtask_segments.json
      subtask_label_cache.json
      tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, info.json
      build_report.json
    data/chunk-000/episode_*.parquet                                  # task_index relabeled
    videos/chunk-000/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_*.mp4
  text_embeds_cache/robotwin_hangingmug_subtask/                      # ~18 GB T5 cache
zhengxj/
  README_SII.md                  # this file
  WAM_CoT_plan.md                # original plan + main hypothesis
  CONDA.md, EVAL_ROBOTWIN_SERVER.md
  run_vlm_matrix.sh              # Track-A open-loop + baseline matrix
  run_vlm_cl_matrix.sh           # Track-A closed-loop matrix
  run_vlm_cl_smoke_test.sh       # Track-A 1-episode smoke
  run_turn_switch_100ep_eval.sh  # Track-A 100-ep follow-up
  summarize_vlm_matrix.py        # Track-A result aggregator
  results_vlm_training_free.md   # Track-A results
  run_hangingmug_labeling.sh         # Track-B VLM+LLM labeling
  run_hangingmug_precompute_embeds.sh
  run_hangingmug_finetune.sh
  run_hangingmug_subtask_ft_pipeline.sh
  run_hangingmug_ft_eval.sh           # Track-B eval, single ckpt
  run_hangingmug_ft_eval_multi.sh     # Track-B eval, all ckpts in parallel
  compute_avg_steps.py
  vlm_smoke_test.py
report/
  inference_time_vlm_cot.tex
  appendix_vlm_cot_details.tex
  final_method_my_part.tex
  final_experiment_my_part.tex
  figures/...
```

Files that already existed but are central to the project:

- [experiments/robotwin/eval_robotwin_single.py](../experiments/robotwin/eval_robotwin_single.py) — the eval entrypoint we drive everything through.
- [experiments/robotwin/fastwam_policy/deploy_policy.py](../experiments/robotwin/fastwam_policy/deploy_policy.py) — wires VLM planners into the policy's `update_observation` loop.
- [experiments/robotwin/fastwam_policy/vlm_planner.py](../experiments/robotwin/fastwam_policy/vlm_planner.py) — the actual VLM/LLM call code (open-loop + closed-loop, menu + free_form).
- [experiments/robotwin/fastwam_policy/subtask_menus.json](../experiments/robotwin/fastwam_policy/subtask_menus.json) — per-task canonical stage scaffolds used by closed-loop `menu` mode AND as Track-B training anchors.
- [scripts/train.py](../scripts/train.py), [scripts/train_zero1.sh](../scripts/train_zero1.sh), [scripts/precompute_text_embeds.py](../scripts/precompute_text_embeds.py) — upstream training entrypoints.
- [scripts/create_robotwin_subset.py](../scripts/create_robotwin_subset.py) — the 4-task subset extractor that `build_hangingmug_subtask_dataset.py` is modeled after.

---

## 7. Known issues / future work

- **Track B eval not yet run.** The FT model is still training. As of the last check it had checkpoints `step_000100.pt` ... `step_000900.pt` saved. Run `bash zhengxj/run_hangingmug_ft_eval_multi.sh` once you want to land Track-B numbers.
- **Single-task FT only.** We chose `hanging_mug` because it has the most stages and the lowest baseline. The dataset/FT pipeline is fully generic — to add another task, extend `subtask_menus.json`, rerun `segment_hangingmug_subtasks.py` with a different state-indexing rule if the task isn't dual-arm, then re-run the same labeling + assembly + FT pipeline.
- **Closed-loop window contamination.** Each long episode is kept as one LeRobot episode and only the per-frame `task_index` is rewritten. Sliding training windows of size 33 can straddle stage boundaries, so ~35% of training windows have an anchor-frame instruction that does not perfectly match the action chunk's coverage. We accepted this for the 24-hour budget; cleanest fix is to re-encode 4 short `.mp4`s per original demo (1 per stage) and treat them as separate episodes.
- **Pretrained WAM closed-loop on `hanging_mug`.** Only 10-episode numbers exist for cl-menu (10%) and cl-free (40%). The 100-episode follow-up was prioritized for `turn_switch` instead; bumping `hanging_mug` to 100 episodes too would tighten Track B's comparison.
- **No latency / API-cost table.** We have per-call latencies in the closed-loop traces in `results_vlm_training_free.md` but no aggregated report. Easy to compute from `*/_completion.log` and the planner cache files.
