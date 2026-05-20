# Experiment Summary and Analysis

This document summarizes the experiments completed for the inference-time VLM-CoT part of the project. The goal is to record what was evaluated, the main quantitative results, and the corresponding analysis for later use in the final report or presentation.

## 1. Main Question

We study whether a pretrained FastWAM policy can benefit from inference-time semantic reasoning provided by a VLM planner.

The core hypothesis is:

> A VLM can improve WAM execution by grounding the task instruction in the current visual scene, but transmitting reasoning only through natural-language instructions may be unstable because the rewritten commands can differ from WAM's training distribution.

The WAM checkpoint is kept fixed. The VLM only changes the language instruction passed to the policy.

## 2. Evaluated Conditions

| Condition | Description |
| --- | --- |
| Baseline | Original FastWAM inference. The policy receives the raw RoboTwin instruction for the whole episode. |
| Open-Loop (OL) | The VLM observes the initial frame once and rewrites the task into a visually grounded instruction. The rewritten instruction is fixed for the whole episode. |
| Closed-Loop (Menu) | The VLM is queried periodically and must choose from a constrained per-task subtask menu. |
| Closed-Loop (Free) | The VLM is queried periodically and can freely emit short grounded instructions. |

All variants use the same pretrained WAM checkpoint and the same simulator/evaluation pipeline.

## 3. Task Suite

Experiments are run on four RoboTwin V2 tasks under randomized initializations and unseen instructions:

| Task | Step Limit | Main Challenge |
| --- | ---: | --- |
| `hanging_mug` | 900 | Multi-stage manipulation with fine rotation and hook alignment. |
| `open_microwave` | 1500 | Door handle localization, approach, grasp, and pull. |
| `place_can_basket` | 700 | Object grounding, pick-and-place, container placement. |
| `turn_switch` | 400 | Precise target actuation and switch affordance. |

Each task-condition cell uses 10 episodes.

## 4. Aggregate Results

| Condition | Mean Success Rate | Mean Execution Steps | Mean Total Time |
| --- | ---: | ---: | ---: |
| Baseline | 30.0% | 735.8 | 14.7s |
| Open-Loop (OL) | 47.5% | 602.0 | 34.1s |
| Closed-Loop (Menu) | 45.0% | 633.8 | 45.8s |
| Closed-Loop (Free) | 52.5% | 624.8 | 43.5s |

Key observations:

- All VLM-CoT variants improve mean success rate over the baseline.
- Closed-Loop (Free) achieves the best mean success rate, 52.5%.
- Open-Loop achieves the lowest mean execution steps, 602.0, but has a large one-time VLM latency.
- Baseline is fastest in wall-clock time because it does not call the VLM.

## 5. Open-Loop VLM-CoT Results

Open-loop VLM-CoT uses Qwen3-VL-Plus with thinking enabled. The VLM is called once per episode using the first frame and original instruction. The returned instruction is truncated/normalized and reused for the entire rollout.

### Success Rate

| Task | Step Limit | Baseline | Open-Loop | Delta |
| --- | ---: | ---: | ---: | ---: |
| `hanging_mug` | 900 | 20.0% (2/10) | 30.0% (3/10) | +10pp |
| `open_microwave` | 1500 | 30.0% (3/10) | 60.0% (6/10) | +30pp |
| `place_can_basket` | 700 | 30.0% (3/10) | 60.0% (6/10) | +30pp |
| `turn_switch` | 400 | 40.0% (4/10) | 40.0% (4/10) | 0pp |
| Mean | - | 30.0% | 47.5% | +17.5pp |

### Execution Steps

`avg_all` averages over all 10 episodes, with failed episodes capped at the step limit. `avg_succ` averages only successful episodes.

| Task | Step Limit | Baseline avg_all | Open-Loop avg_all | Delta | Baseline avg_succ | Open-Loop avg_succ | Delta | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hanging_mug` | 900 | 783 | 727 | -56 | 316 (2) | 323 (3) | +7 | Flat |
| `open_microwave` | 1500 | 1335 | 988 | -347 | 950 (3) | 647 (6) | -303 | Clear speedup |
| `place_can_basket` | 700 | 557 | 424 | -133 | 224 (3) | 240 (6) | +16 | More successes, similar per-success length |
| `turn_switch` | 400 | 268 | 269 | +1 | 71 (4) | 72 (4) | +1 | No effect |

### Open-Loop Analysis

`open_microwave` is the strongest open-loop result. The success rate doubles from 30% to 60%, and successful episodes become much shorter. The VLM usually adds a simple spatial grounding clause, such as indicating that the microwave is in the center. This helps the WAM approach the handle more directly.

`place_can_basket` also doubles in success rate from 30% to 60%, but the successful episode length remains similar. The VLM helps convert borderline failures into successes by grounding the can and basket locations, but it does not significantly streamline already-successful trajectories.

`hanging_mug` improves only slightly. The VLM can identify the mug and rack, but the dominant difficulty is fine-grained mug rotation and hook engagement, which language-level grounding cannot directly solve.

`turn_switch` shows no change. The task is short and dominated by precise contact/affordance. The original instruction already contains enough high-level information, so the VLM has little useful ambiguity to resolve.

### Open-Loop Cost

Open-loop uses one VLM call per episode with thinking enabled. The average VLM latency is about 22.0 seconds.

| Task | Baseline Total | Open-Loop Total | Relative Cost |
| --- | ---: | ---: | ---: |
| `hanging_mug` | 15.7s | 36.5s | 2.3x |
| `open_microwave` | 26.7s | 41.8s | 1.6x |
| `place_can_basket` | 11.1s | 30.5s | 2.7x |
| `turn_switch` | 5.4s | 27.4s | 5.1x |

The cost-benefit ratio is best for longer tasks where the VLM improves success rate substantially, such as `open_microwave`.

## 6. Closed-Loop VLM-CoT Results

Closed-loop variants re-query the VLM periodically, roughly every 72 execution steps. Thinking is disabled to reduce latency. Mean per-call latency is approximately 3.3-3.9 seconds.

Two variants are evaluated:

- `CL-menu`: VLM chooses from a constrained subtask menu.
- `CL-free`: VLM freely emits short natural-language instructions.

### Success Rate

| Task | Step Limit | Baseline | Open-Loop | CL-menu | CL-free | Best Delta vs Baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hanging_mug` | 900 | 20% | 30% | 10% | 40% | +20pp |
| `open_microwave` | 1500 | 30% | 60% | 40% | 20% | +30pp |
| `place_can_basket` | 700 | 30% | 60% | 60% | 80% | +50pp |
| `turn_switch` | 400 | 40% | 40% | 70% | 70% | +30pp |
| Mean | - | 30.0% | 47.5% | 45.0% | 52.5% | - |

### Execution Steps

| Task | Baseline avg_all | Open-Loop avg_all | CL-menu avg_all | CL-free avg_all | Baseline avg_succ | Open-Loop avg_succ | CL-menu avg_succ | CL-free avg_succ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hanging_mug` | 783 | 727 | 846 | 679 | 316 (2) | 323 (3) | 365 (1) | 349 (4) |
| `open_microwave` | 1335 | 988 | 1103 | 1310 | 949 (3) | 647 (6) | 509 (4) | 550 (2) |
| `place_can_basket` | 557 | 424 | 416 | 312 | 224 (3) | 240 (6) | 228 (6) | 215 (8) |
| `turn_switch` | 268 | 269 | 170 | 198 | 71 (4) | 72 (4) | 71 (7) | 112 (7) |

### Closed-Loop Analysis

`place_can_basket` benefits most from closed-loop free-form planning. CL-free reaches 80% success and reduces average steps to 312. The task has clear visual stage boundaries: pick the can, move to basket, place in basket. The VLM can monitor these stages and update the instruction with minimal ambiguity.

`turn_switch` benefits from closed-loop grounding but not open-loop grounding. Both closed-loop variants reach 70% success, compared with 40% for baseline and open-loop. The likely reason is that repeated visual grounding lets the VLM specify the current switch/button position more precisely.

`open_microwave` is best served by open-loop planning. Open-loop reaches 60%, while CL-free drops to 20%. Free-form closed-loop tends to over-decompose the task into low-level subtasks such as moving toward the handle, grasping, and pulling. These subtask commands may be out of the WAM training distribution. CL-menu is better than CL-free but still underperforms open-loop.

`hanging_mug` shows that menu decomposition can be harmful. CL-menu drops to 10%, while CL-free improves to 40%. The explicit menu forces stage transitions that can become wrong if the object is not in the expected state. CL-free stays closer to full-task language and benefits from repeated re-grounding.

### Closed-Loop Cost

| Task | Baseline Total | Open-Loop Total | CL-menu Total | CL-free Total |
| --- | ---: | ---: | ---: | ---: |
| `hanging_mug` | 15.7s | 36.5s | 61.6s | 47.0s |
| `open_microwave` | 26.7s | 41.8s | 75.6s | 92.2s |
| `place_can_basket` | 11.1s | 30.5s | 33.3s | 20.0s |
| `turn_switch` | 5.4s | 27.4s | 12.6s | 14.9s |

Closed-loop calls are individually cheaper than open-loop thinking calls, but multiple calls can accumulate. The best trade-off occurs when closed-loop planning both improves success rate and reduces wasted robot motion, as in `place_can_basket`.

## 7. Failure Case Analysis

Representative failure cases reveal that VLM-CoT helps instruction grounding but cannot solve all WAM limitations.

### `hanging_mug`

The policy can often localize the mug and stand and move near the target region. Failure usually occurs during fine alignment between the mug handle and peg. This requires precise end-effector pose control and small-scale geometric reasoning. Language-level grounding does not directly solve this contact-rich manipulation problem.

### `open_microwave`

The gripper can approach the handle, but the local observation becomes heavily occluded by the door and robot arm. The policy cannot reliably infer the contact state or relative pose under occlusion. This suggests a need for persistent state estimation beyond visible pixels.

### `place_can_basket`

Object recognition is not enough. After grasping the can, the policy must estimate 3D relationships among the can, basket, and gripper. Errors in depth, container boundary estimation, or release pose can still cause failure.

### `turn_switch`

The task requires affordance reasoning. Pressing a rocker-style switch near the edge is more effective than pressing the center. Failures suggest that the policy sometimes treats the task as generic contact rather than selecting the contact point that induces the desired rotation.

## 8. Main Conclusions

1. Inference-time VLM-CoT improves pretrained WAM performance without retraining.
   - Mean success improves from 30.0% to 47.5% with open-loop VLM planning.
   - Mean success improves to 52.5% with closed-loop free-form planning.

2. VLM planning helps most when the failure mode is visual-language ambiguity.
   - Strong gains appear in `open_microwave` and `place_can_basket`.
   - Little or no gain appears when low-level precision dominates, such as `turn_switch` open-loop.

3. Closed-loop planning is task-dependent.
   - It helps when stages are visually separable and the VLM can track progress.
   - It hurts when the planner over-decomposes the task into commands outside the WAM training distribution.

4. Natural language is an imperfect reasoning interface for WAM.
   - Even good VLM instructions can fail if the WAM does not understand that phrasing.
   - Frequent text replanning adds latency and may disrupt action-chunk dynamics.

5. The results motivate latent CoT / CoTDiT.
   - Discrete-language CoT is useful for grounding, but not stable enough as the only reasoning interface.
   - Future systems should inject VLM-derived visual and semantic grounding directly into the policy's latent representation.

## 9. Caveats

- Each task-condition uses 10 episodes, so per-task numbers should be treated as preliminary.
- Some successful-episode averages are based on only 1-6 successful runs.
- VLM latency depends strongly on whether thinking is enabled.
- Closed-loop results are sensitive to prompt design, replan interval, and whether the instruction is menu-constrained or free-form.
- Fine-tuning/subtask-alignment evaluation summaries were not populated in the available summary TSVs, so this document focuses on completed VLM-CoT inference-time experiments and qualitative failure analysis.

