"""Compute average per-episode step count for each (task, condition) cell.

The eval renders one frame per `task_env.take_action(...)` at 10 fps, so the
mp4 frame count == the policy's `take_action_cnt` at episode termination
(success early-exit, or step_lim cap on failure).

Usage:
    python zhengxj/compute_avg_steps.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_RESULTS_ROOT = (
    PROJECT_ROOT / "evaluate_results" / "robotwin" / "robotwin_uncond_3cam_384"
)
STEP_LIMITS = {
    "hanging_mug": 900,
    "open_microwave": 1500,
    "place_can_basket": 700,
    "turn_switch": 400,
}
TASKS = ["hanging_mug", "open_microwave", "place_can_basket", "turn_switch"]
CONDITIONS = ["baseline", "vlm"]


def _frame_count(path: Path) -> Optional[int]:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    rows: list[dict] = []
    for task in TASKS:
        step_lim = STEP_LIMITS[task]
        for cond in CONDITIONS:
            job_dir = CKPT_RESULTS_ROOT / f"{task}_{cond}" / task
            videos = sorted(job_dir.glob("episode*_randomized-*_success-*.mp4"))
            steps_success: list[int] = []
            steps_failure: list[int] = []
            for v in videos:
                n = _frame_count(v)
                if n is None:
                    continue
                if "_success-true" in v.name:
                    steps_success.append(n)
                else:
                    steps_failure.append(n)
            n_succ = len(steps_success)
            n_fail = len(steps_failure)
            n_total = n_succ + n_fail
            all_steps = steps_success + steps_failure
            avg_all = mean(all_steps) if all_steps else None
            avg_succ = mean(steps_success) if steps_success else None
            avg_fail = mean(steps_failure) if steps_failure else None
            rows.append(
                {
                    "task": task,
                    "cond": cond,
                    "step_lim": step_lim,
                    "n_succ": n_succ,
                    "n_fail": n_fail,
                    "n_total": n_total,
                    "avg_all": avg_all,
                    "avg_succ": avg_succ,
                    "avg_fail": avg_fail,
                    "steps_succ": steps_success,
                    "steps_fail": steps_failure,
                }
            )

    def fmt(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.1f}"

    print()
    print("Per-(task, condition) average execute steps (open-loop matrix)")
    print("=" * 95)
    header = (
        f"{'task':<18}{'cond':<10}{'n':>3}  "
        f"{'avg_all':>9}  {'avg_succ':>9} ({'n':>2})  "
        f"{'avg_fail':>9} ({'n':>2})  step_lim"
    )
    print(header)
    print("-" * 95)
    for r in rows:
        print(
            f"{r['task']:<18}{r['cond']:<10}{r['n_total']:>3}  "
            f"{fmt(r['avg_all']):>9}  "
            f"{fmt(r['avg_succ']):>9} ({r['n_succ']:>2})  "
            f"{fmt(r['avg_fail']):>9} ({r['n_fail']:>2})  "
            f"{r['step_lim']}"
        )
    print()

    print("Per-task summary (delta = VLM - baseline)")
    print("-" * 95)
    print(f"{'task':<18}{'baseline_avg_all':>18}  {'vlm_avg_all':>14}  {'delta':>10}    {'baseline_avg_succ':>18}  {'vlm_avg_succ':>14}  {'delta':>10}")
    by_key = {(r["task"], r["cond"]): r for r in rows}
    for task in TASKS:
        b = by_key[(task, "baseline")]
        v = by_key[(task, "vlm")]
        d_all = (
            None if b["avg_all"] is None or v["avg_all"] is None
            else v["avg_all"] - b["avg_all"]
        )
        d_succ = (
            None if b["avg_succ"] is None or v["avg_succ"] is None
            else v["avg_succ"] - b["avg_succ"]
        )
        print(
            f"{task:<18}{fmt(b['avg_all']):>18}  {fmt(v['avg_all']):>14}  "
            f"{(fmt(d_all) if d_all is None else f'{d_all:+.1f}'):>10}    "
            f"{fmt(b['avg_succ']):>18}  {fmt(v['avg_succ']):>14}  "
            f"{(fmt(d_succ) if d_succ is None else f'{d_succ:+.1f}'):>10}"
        )
    print()

    out_json = PROJECT_ROOT / "zhengxj" / "avg_steps.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Per-episode raw step counts saved to: {out_json.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
