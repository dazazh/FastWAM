# FastWAMCoT 使用指南

## 前置准备

### 1. 预计算 Text Embeddings

```bash
python scripts/precompute_text_embeds.py \
    --dataset_dir ./data/robotwin2.0/robotwin2.0 \
    --output_dir ./data/text_embeds_cache/robotwin \
    --context_len 128
```

### 2. 预计算 VLM Features

```bash
python scripts/precompute_vlm_features.py \
    --vlm_model_path checkpoints/Qwen3-VL-2B \
    --dataset_dir ./data/robotwin2.0/robotwin2.0 \
    --output_dir ./data/vlm_features/robotwin \
    --batch_size 4
```

---

## 训练

基础命令格式：

```bash
bash scripts/train_zero1.sh <NUM_GPUS> task=robotwin_cot_3cam_384_1e-4 [overrides...]
```

### 训练模式组合


| Video DiT   | Action DiT | 命令 override                                                                |
| ----------- | ---------- | -------------------------------------------------------------------------- |
| LoRA32 (默认) | Full (默认)  | 无需额外 override                                                              |
| LoRA16      | Full       | `model.training.video_dit_mode=lora16`                                     |
| LoRA64      | Full       | `model.training.video_dit_mode=lora64`                                     |
| Full        | Full       | `model.training.video_dit_mode=full`                                       |
| LoRA32      | LoRA32     | `model.training.action_dit_mode=lora32`                                    |
| LoRA32      | LoRA16     | `model.training.action_dit_mode=lora16`                                    |
| Full        | LoRA32     | `model.training.video_dit_mode=full model.training.action_dit_mode=lora32` |


> CoT DiT 始终为 full training，不支持 LoRA。

### 训练示例

#### 默认配置（LoRA32 Video + Full Action，全部 50 任务）

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4
```

#### 4 目标任务子集训练

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 data=robotwin_cot_4tasks
```

#### LoRA32 Video + Full Action（4 任务）

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 data=robotwin_cot_4tasks \
    model.training.video_dit_mode=lora32 \
    model.training.action_dit_mode=full
```

#### Full Video + Full Action（4 任务）

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 data=robotwin_cot_4tasks \
    model.training.video_dit_mode=full \
    model.training.action_dit_mode=full
```

#### LoRA32 Video + LoRA32 Action（4 任务）

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 data=robotwin_cot_4tasks \
    model.training.video_dit_mode=lora32 \
    model.training.action_dit_mode=lora32
```

#### LoRA64 Video + Full Action（4 任务，更大 LoRA rank）

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 data=robotwin_cot_4tasks \
    model.training.video_dit_mode=lora64
```

#### Full Video + Full Action（单任务 turn_switch）

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 data=robotwin_cot_turn_switch \
    model.training.video_dit_mode=full \
    model.training.action_dit_mode=full
```

### 使用预计算 VLM Features 训练

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 data=robotwin_cot_4tasks \
    data.train.vlm_features_dir=./data/vlm_features/robotwin \
    data.val.vlm_features_dir=./data/vlm_features/robotwin \
    model.vlm_config.extract_mode=precomputed
```

### 其他训练参数

```bash
# 调整学习率
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 learning_rate=5e-5

# 调整 batch size + gradient accumulation
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 batch_size=8 gradient_accumulation_steps=2

# 从 checkpoint 恢复
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 resume=./runs/.../checkpoint_step_5000.pt

# 关闭 video loss（仅训练 action）
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 model.loss.lambda_action=1.0 model.loss.lambda_video=0.0
```

### 多机训练

```bash
# 每台机器上执行（以 2 机 8 卡为例）
NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> MASTER_PORT=29500 \
    bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 data=robotwin_cot_4tasks
```

### 故障排除：NVLink P2P 通信错误

如果训练过程中出现以下错误导致崩溃：

```
CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error
```

使用禁用 P2P 的启动脚本替代默认脚本：

```bash
bash scripts/train_zero1_no_p2p.sh 8 task=robotwin_cot_3cam_384_1e-4 \
    data=robotwin_cot_turn_switch \
    model.training.video_dit_mode=full \
    model.training.action_dit_mode=full
```

该脚本设置 `NCCL_P2P_DISABLE=1` 禁用 GPU 间 NVLink 直接内存访问，强制 NCCL 走共享内存通信。影响：step time 增加约 5-10%，不影响 loss 收敛。用法与 `train_zero1.sh` 完全一致，所有参数直接透传。

---

## 评测

训练过程中的在线评测由 `eval_every` 参数控制（默认每 500 步），自动输出：

- `val_loss`：验证集上的训练 loss
- `psnr_rollout_vs_gt` / `ssim_rollout_vs_gt`：生成视频 vs GT 的质量指标
- `action_l1` / `action_l2`：Action 预测精度

### 调整评测频率和推理步数

```bash
bash scripts/train_zero1.sh 8 task=robotwin_cot_3cam_384_1e-4 \
    eval_every=200 \
    eval_num_inference_steps=20
```

---

## 任务切换

编辑或新建 `configs/data/robotwin_cot_<name>.yaml`：

```yaml
defaults:
  - robotwin_cot
  - _self_

train:
  task_names:
    - <task_1>
    - <task_2>
  task_episodes_file: configs/data/robotwin_task_episodes.yaml

val:
  task_names:
    - <task_1>
    - <task_2>
  task_episodes_file: configs/data/robotwin_task_episodes.yaml
```

可用任务名见 `configs/data/robotwin_task_episodes.yaml`（50 个环境）。

---

## 模型架构说明

FastWAMCoT 使用 3-expert MoT（Mixture of Transformers）：


| Expert     | 参数量级            | 默认训练模式 | 说明                     |
| ---------- | --------------- | ------ | ---------------------- |
| CoT DiT    | 轻量 (512 hidden) | Full   | 接收 VLM features，提供高层语义 |
| Video DiT  | 大 (3072 hidden) | LoRA32 | Wan2.2 预训练权重，视频去噪      |
| Action DiT | 中 (1024 hidden) | Full   | Action 去噪预测            |


推理时使用 Static KV Cache：CoT+Video(f0) 预填充一次，Action 迭代去噪。