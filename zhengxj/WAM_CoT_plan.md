# WAM-CoT 24-Hour Project Plan

## 1. Revised Core Idea

Because only about 24 hours remain, the main project is changed from training-centered subtask alignment to inference-time VLM-CoT for a pretrained FastWAM model.

We study whether a VLM can improve a pretrained WAM by acting as a high-level semantic planner:

```text
original task instruction + current image
        -> VLM reasoning / instruction refinement
        -> WAM executes with the refined instruction
```

Main hypothesis:

> Even without WAM fine-tuning, inference-time VLM-CoT may improve hard long-horizon tasks by giving clearer, more visually grounded instructions. Closed-loop VLM-CoT may further help by monitoring progress and adjusting the instruction when execution deviates.

Subtask dataset construction and WAM fine-tuning are now optional bonus experiments, not the main deliverable.

## 2. System Design

The system follows a hierarchical control structure:

```text
Task instruction + current image
        |
        v
VLM planner / monitor
        |
        v
Refined instruction / next subtask
        |
        v
FastWAM low-level controller
        |
        v
Robot action chunk
        |
        v
New observation
```

Roles:

- VLM: high-level task manager, progress monitor, and subtask generator.
- WAM: low-level controller that executes one instruction-conditioned action chunk.

We will test two inference-time planner modes:

- Open-loop VLM-CoT: VLM observes the initial frame once and outputs a clearer task instruction or a short plan. WAM then executes without further VLM intervention.
- Closed-loop VLM-CoT: VLM observes the current frame after fixed WAM execution intervals and decides whether to continue, refine the instruction, switch to the next subtask, or stop.

## 3. Models

The main experiments use the released FastWAM pretrained RoboTwin checkpoint.

### Main Model: Pretrained WAM

Use the pretrained checkpoint directly for all 4 selected hard tasks.

Purpose:

- Provides the fastest possible baseline.
- Tests whether inference-time VLM-CoT can improve a pretrained policy without additional training.
- Leaves enough time for result collection, videos, and report preparation.

### Optional Bonus Model: One-Task Subtask-FT WAM

If the main experiments finish early, construct a small subtask dataset for only 1 task and fine-tune WAM:

```text
pretrained WAM -> fine-tune on one task with subtask-relabeled instructions
```

Purpose:

- Tests the concern that pretrained WAM may not understand VLM-generated short instructions.
- Serves as a pilot result or future-work motivation.

This is not required for the main story.

## 4. Experiment Matrix

Main evaluation table:

| ID | Model | Training Data | Inference Instruction | Planner |
| --- | --- | --- | --- | --- |
| M1 | Pretrained WAM | Released 50-task training | Original full task | None |
| M2 | Pretrained WAM | Released 50-task training | VLM-refined instruction | Open-loop |
| M3 | Pretrained WAM | Released 50-task training | VLM-refined / next instruction | Closed-loop |

Most important comparisons:

- M1 vs M2: Does initial VLM reasoning / instruction refinement help the pretrained WAM?
- M2 vs M3: Does closed-loop VLM monitoring and replanning improve over one-shot open-loop prompting?
- M1 vs M3: Can VLM-CoT improve hard tasks without any WAM fine-tuning?

Expected trend:

```text
M3 may help when the task has clear stages and the VLM can judge progress.
M2 may help when the original instruction is ambiguous or visually underspecified.
M2/M3 may hurt if the VLM outputs instructions outside WAM's training distribution.
```

Optional bonus table:

| ID | Model | Training Data | Inference Instruction | Planner |
| --- | --- | --- | --- | --- |
| B1 | One-task Subtask-FT WAM | Subtask-relabeled 1-task data | VLM subtask instruction | Open-loop or closed-loop |

Use this only if time remains after the main results are collected.

## 5. Task Selection

Use only 4 RoboTwin tasks because of the two-day time limit.

Selection criteria:

- Long-horizon or multi-stage.
- Baseline WAM is not already near-perfect.
- Clear gripper interaction stages.
- Subtasks are reusable across tasks.
- Subtask instructions can be generated with simple templates.

Prefer tasks with structures like:

```text
pick object -> move object -> place object
open container -> pick object -> place object into container
pick object A -> place on/near object B
```

Avoid tasks that mainly require low-level precision without semantic stage structure.

## 6. Subtask Dataset Construction

This section is now optional. Use it only for the one-task fine-tuning pilot.

If time remains, construct weak subtask labels from original demonstrations using gripper state transitions.

Basic assumption:

- Gripper close event: grasp or interaction begins.
- Gripper open event: release or placement happens.

For a simple pick-place trajectory:

```text
start -> close_event: pick up <object>
close_event -> open_event: move/place <object> to <target>
open_event -> end: finish/release <object>
```

For better robustness, use a small temporal margin around events:

```text
pick segment: start -> close_event + k
place segment: close_event + k -> open_event + k
finish segment: open_event + k -> end
```

Important note:

Gripper state only tells us when the interaction changes. It does not tell us the object or target names. Object and target names should come from task metadata, original instruction parsing, or manually written templates for the 4 selected tasks.

Example relabeling:

```text
Original instruction:
Put the red block into the drawer.

Subtask labels:
1. pick up the red block
2. move the red block to the drawer
3. place the red block into the drawer
```

## 7. VLM Planner Design

The VLM should not generate arbitrary long language for WAM. It should output short, concrete instructions that are likely close to WAM's training distribution.

### Open-Loop Prompt

```text
You are a high-level robot task planner for a pretrained robot policy.

Given:
1. The original task instruction.
2. The initial camera image.

Your job:
- Think about the task goal and visible scene.
- Rewrite the instruction into a clearer, concrete command for the low-level WAM policy.
- Keep the final WAM instruction short.
- Do not invent objects that are not visible.
- Do not include long reasoning in the final instruction.

Return JSON only:
{
  "reasoning": "...",
  "wam_instruction": "..."
}
```

Example:

```json
{
  "reasoning": "The target object is the red block on the table, and it should be placed on the blue block.",
  "wam_instruction": "Pick up the red block and place it on the blue block."
}
```

### Closed-Loop Prompt

Use this after each fixed WAM execution interval.

```text
You are a high-level progress monitor and task planner for a pretrained robot policy.

Given:
1. The original task instruction.
2. The previous WAM instruction.
3. The current camera image.

Decide whether the robot should:
- continue the previous instruction,
- refine the instruction,
- move to the next subtask,
- stop because the task is complete.

Important rules:
- Only change the instruction if the current stage is clearly finished or the robot is clearly stuck.
- Use short, concrete WAM instructions.
- Do not output long explanations as the WAM instruction.
- Do not invent invisible objects.

Return JSON only:
{
  "stage": "...",
  "status": "continue/refine/next/done",
  "wam_instruction": "...",
  "done": false,
  "reason": "..."
}
```

Example:

```json
{
  "stage": "object_grasped",
  "status": "next",
  "wam_instruction": "Move the red block above the blue block and place it down.",
  "done": false,
  "reason": "The robot appears to have grasped the red block, so the next stage is placement."
}
```

### Closed-Loop Execution Rule

Use a fixed replan interval instead of calling the VLM every frame:

```text
Run WAM for N steps -> query VLM -> update or continue instruction
```

Recommended starting point:

- N = one WAM action chunk, or about 20-50 environment steps.
- If the VLM says `continue`, keep the previous instruction.
- If the VLM repeats the same unhelpful command several times, terminate or mark as failure.

## 8. Metrics

Main metrics:

- Task success rate.
- Stage success rate.
- Average execution steps.
- Failure type statistics.
- VLM planning latency.
- Number of VLM calls per episode.

Useful failure categories:

- VLM-refined instruction hurts compared with original instruction.
- VLM changes instruction too often.
- VLM fails to recognize current progress.
- VLM judges task complete too early.
- VLM repeats the same instruction when stuck.
- Correct subtask but WAM execution failure.
- WAM cannot follow short instruction.

## 9. Ablations

Minimum ablation under the 24-hour scope:

```text
Original instruction vs VLM-refined instruction
```

This tests whether the VLM instruction refinement itself helps or only adds noise.

Optional ablations:

- Open-loop vs closed-loop.
- VLM with reasoning field vs VLM final instruction only.
- Closed-loop with frequent replanning vs less frequent replanning.
- One-task subtask fine-tuning vs pretrained WAM.

## 10. Remaining 24-Hour Schedule

### Main Track

1. Finish baseline pretrained WAM evaluation on 4 hard tasks.
2. Finish open-loop VLM-CoT evaluation on the same 4 tasks.
3. Implement the simplest closed-loop VLM-CoT controller.
4. Evaluate closed-loop VLM-CoT on the same 4 tasks.
5. Save representative success and failure videos for all methods.
6. Collect logs:
   - original instruction,
   - VLM reasoning,
   - final WAM instruction,
   - success/failure result,
   - failure category.

### Report Track

Prepare these as early as possible:

1. System diagram.
2. Experiment matrix.
3. Success rate table.
4. Qualitative video examples.
5. Failure analysis.
6. Limitations and future work.

Suggested report figures:

   - system diagram,
   - baseline vs open-loop vs closed-loop comparison,
   - VLM prompt / output example,
   - success and failure case screenshots,
   - optional one-task fine-tuning pilot result.

### Bonus Track

Only start this if the main track has usable results:

1. Choose 1 task.
2. Build gripper-event-based subtask labels.
3. Fine-tune WAM for a small number of steps.
4. Test whether subtask fine-tuning helps WAM follow VLM-generated instructions.

## 11. Fallback Plan

If closed-loop planning is unstable:

- Present open-loop VLM-CoT as the main result.
- Use closed-loop as qualitative analysis or future work.
- Explain that frequent instruction changes can destabilize pretrained WAM.

If VLM outputs bad instructions:

- Add stronger prompt constraints.
- Force the VLM to preserve the original object and target names.
- Use a manually verified VLM instruction for each task as an upper-bound open-loop setting.

If pretrained WAM does not improve:

- Report negative results honestly.
- Emphasize failure analysis:
  - VLM instruction is out-of-distribution for WAM,
  - WAM cannot recover from low-level errors,
  - closed-loop planning needs a subtask-aligned controller.
- Use this to motivate the one-task subtask fine-tuning pilot.

## 12. Main Claim

The final presentation should emphasize:

> Under a tight compute budget, we evaluate whether a pretrained WAM can benefit from inference-time semantic CoT. Open-loop VLM-CoT improves the initial task instruction using visual context, while closed-loop VLM-CoT monitors progress and replans during execution. The results reveal when VLM planning helps, when it hurts, and why future subtask-aligned WAM fine-tuning may be necessary.
