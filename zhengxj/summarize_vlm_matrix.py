"""Aggregate results from a vlm_matrix run and emit a markdown summary.

Usage:
    python zhengxj/summarize_vlm_matrix.py [<run_root>] [--conditions a,b,c]

If ``<run_root>`` is omitted, picks the latest subdir of
``evaluate_results/robotwin/vlm_matrix/``. Conditions default to
``baseline,vlm,vlm_cl_menu,vlm_cl_free``; any missing condition is reported
as "MISSING" instead of erroring.

For closed-loop conditions (``vlm_cl_*``) the script also reads
``vlm_episode_traces.json`` and reports:
- mean VLM calls per episode
- mean consecutive-continue cap firings
- mean done-early rate
- mean status histogram across all VLM calls

Writes ``zhengxj/results_vlm_training_free.md`` and prints it to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = PROJECT_ROOT / "evaluate_results" / "robotwin" / "vlm_matrix"
CKPT_RESULTS_ROOT = (
    PROJECT_ROOT / "evaluate_results" / "robotwin" / "robotwin_uncond_3cam_384"
)
OUTPUT_MD = PROJECT_ROOT / "zhengxj" / "results_vlm_training_free.md"

TASKS = ["hanging_mug", "open_microwave", "place_can_basket", "turn_switch"]
DEFAULT_CONDITIONS = ["baseline", "vlm", "vlm_cl_menu", "vlm_cl_free"]
CLOSED_LOOP_CONDITIONS = {"vlm_cl_menu", "vlm_cl_free"}

CONDITION_LABELS = {
    "baseline": "Baseline (M1, raw)",
    "vlm": "Open-loop VLM (M2)",
    "vlm_cl_menu": "Closed-loop menu (M3-menu)",
    "vlm_cl_free": "Closed-loop free (M3-free)",
}


def _job_results_dir(tag: str, task: str) -> Path:
    """Where eval_robotwin_single.py actually wrote the per-task results.

    The wrapper constructs run_output_dir = evaluate_results/robotwin/<ckpt_tag>/<run_ts>
    where run_ts = output_dir.name (i.e. our matrix `<tag>`), and the inner
    eval files land at run_output_dir/<task>/.
    """
    return CKPT_RESULTS_ROOT / tag / task


def _pick_latest(parent: Path) -> Path:
    if not parent.is_dir():
        raise FileNotFoundError(f"No vlm_matrix runs found at {parent}")
    subs = sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subs:
        raise FileNotFoundError(f"No subdirs in {parent}")
    return subs[-1]


def _read_success_rate(result_file: Path) -> float | None:
    try:
        text = result_file.read_text(encoding="utf-8").strip().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(text):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None


def _find_result_file(job_dir: Path) -> Path | None:
    for path in job_dir.rglob("_result_random.txt"):
        return path
    return None


def _find_videos(job_dir: Path) -> tuple[list[Path], list[Path]]:
    successes: list[Path] = []
    failures: list[Path] = []
    for path in job_dir.rglob("episode*_randomized-*_success-*.mp4"):
        name = path.name
        if "success-true" in name:
            successes.append(path)
        else:
            failures.append(path)
    successes.sort()
    failures.sort()
    return successes, failures


def _load_vlm_cache(job_dir: Path) -> dict | None:
    for path in job_dir.rglob("vlm_cache.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _load_vlm_episode_traces(job_dir: Path) -> dict | None:
    for path in job_dir.rglob("vlm_episode_traces.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _summarize_closed_loop_traces(traces: dict) -> dict:
    """Aggregate per-episode trace dict into headline stats."""
    n_episodes = 0
    total_vlm_calls = 0
    total_oor = 0
    subtask_index_hist: dict[int, int] = {}
    used_fallback_count = 0
    total_latency_s = 0.0
    latency_samples = 0

    for key, entry in traces.items():
        if not isinstance(entry, dict):
            continue
        n_episodes += 1
        total_vlm_calls += int(entry.get("n_vlm_calls", 0) or 0)
        total_oor += int(entry.get("n_subtask_index_oor", 0) or 0)
        for call in entry.get("subtask_trace", []) or []:
            if not isinstance(call, dict):
                continue
            idx = call.get("subtask_index")
            if isinstance(idx, int):
                subtask_index_hist[idx] = subtask_index_hist.get(idx, 0) + 1
            if bool(call.get("used_fallback", False)):
                used_fallback_count += 1
            lat = call.get("latency_s")
            if isinstance(lat, (int, float)) and lat > 0:
                total_latency_s += float(lat)
                latency_samples += 1

    return {
        "n_episodes": n_episodes,
        "total_vlm_calls": total_vlm_calls,
        "mean_vlm_calls_per_episode": (
            total_vlm_calls / n_episodes if n_episodes else 0.0
        ),
        "total_oor": total_oor,
        "subtask_index_hist": subtask_index_hist,
        "used_fallback_count": used_fallback_count,
        "mean_latency_s": (
            total_latency_s / latency_samples if latency_samples else 0.0
        ),
    }


def _parse_args(argv: list[str]) -> tuple[Path | None, list[str]]:
    run_root: Path | None = None
    conditions = list(DEFAULT_CONDITIONS)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--conditions" and i + 1 < len(argv):
            conditions = [c.strip() for c in argv[i + 1].split(",") if c.strip()]
            i += 2
            continue
        if not a.startswith("--"):
            run_root = Path(a).resolve()
            i += 1
            continue
        i += 1
    return run_root, conditions


def main() -> None:
    run_root, conditions = _parse_args(sys.argv[1:])
    if run_root is None:
        run_root = _pick_latest(DEFAULT_PARENT)
    if not run_root.is_dir():
        raise SystemExit(f"Run root not found: {run_root}")

    rel_run = run_root.resolve().relative_to(PROJECT_ROOT)

    lines: list[str] = []
    lines.append("# VLM CoT - Results\n")
    lines.append(f"Run root: `{rel_run}`\n")
    lines.append(
        "Setup: FastWAM release ckpt `robotwin_uncond_3cam_384.pt`, "
        "`task_config=demo_randomized`, `instruction_type=unseen`. "
        "VLM = Qwen3-VL-Plus via DashScope. Open-loop (M2) calls VLM once per "
        "episode with the first head-frame; closed-loop (M3) calls VLM every "
        "K replan boundaries and either picks a subtask from a per-task menu "
        "scaffold (menu mode) or emits a free-form short instruction "
        "(free_form mode).\n"
    )
    lines.append(f"Conditions in this run: {', '.join(conditions)}\n")

    # Per-task table.
    header = ["Task"] + [CONDITION_LABELS.get(c, c) for c in conditions]
    sep = ["---"] * len(header)
    lines.append("## Success Rate\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    per_task: dict[str, dict[str, float | None]] = {}
    per_task_job_dirs: dict[str, dict[str, Path]] = {}
    cond_sums: dict[str, float] = {c: 0.0 for c in conditions}
    cond_counts: dict[str, int] = {c: 0 for c in conditions}

    for task in TASKS:
        per_task[task] = {}
        per_task_job_dirs[task] = {}
        row = [task]
        for cond in conditions:
            tag = f"{task}_{cond}"
            job_dir = _job_results_dir(tag, task)
            per_task_job_dirs[task][cond] = job_dir
            rate = None
            result_file = _find_result_file(job_dir)
            if result_file is not None:
                rate = _read_success_rate(result_file)
            per_task[task][cond] = rate
            if rate is None:
                row.append("MISSING")
            else:
                row.append(f"{rate * 100:.1f}%")
                cond_sums[cond] += rate
                cond_counts[cond] += 1
        lines.append("| " + " | ".join(row) + " |")

    mean_row = ["**mean**"]
    for cond in conditions:
        if cond_counts[cond] > 0:
            mean_row.append(f"**{(cond_sums[cond] / cond_counts[cond]) * 100:.1f}%**")
        else:
            mean_row.append("-")
    lines.append("| " + " | ".join(mean_row) + " |")

    # Per-task details
    lines.append("\n## Per-Task Details\n")
    for task in TASKS:
        lines.append(f"### {task}\n")
        for cond in conditions:
            job_dir = per_task_job_dirs[task][cond]
            result_file = _find_result_file(job_dir)
            rate = per_task[task][cond]
            successes, failures = _find_videos(job_dir)
            lines.append(f"**{cond}** ({CONDITION_LABELS.get(cond, cond)}):")
            if rate is None:
                lines.append("- result file: MISSING")
            else:
                rel_res = (
                    result_file.resolve().relative_to(PROJECT_ROOT)
                    if result_file
                    else "?"
                )
                lines.append(
                    f"- success rate: **{rate * 100:.1f}%**  "
                    f"({len(successes)} success / {len(failures)} fail) "
                    f"`{rel_res}`"
                )
            if successes:
                rel = successes[0].resolve().relative_to(PROJECT_ROOT)
                lines.append(f"- sample success video: `{rel}`")
            if failures:
                rel = failures[0].resolve().relative_to(PROJECT_ROOT)
                lines.append(f"- sample failure video: `{rel}`")

            if cond == "vlm":
                cache = _load_vlm_cache(job_dir)
                if cache:
                    lines.append("- VLM augmentations (episode_seed -> augmented_instruction):")
                    for key, entry in cache.items():
                        aug = entry.get("augmented_instruction", "")
                        raw = entry.get("raw_instruction", "")
                        latency = entry.get("latency_s")
                        lat_str = (
                            f"{latency:.1f}s"
                            if isinstance(latency, (int, float))
                            else "n/a"
                        )
                        lines.append(f"  - `{key}` (vlm_latency={lat_str})")
                        lines.append(f"    - raw: {raw}")
                        lines.append(f"    - aug: {aug}")

            if cond in CLOSED_LOOP_CONDITIONS:
                traces = _load_vlm_episode_traces(job_dir)
                if traces:
                    summary = _summarize_closed_loop_traces(traces)
                    lines.append("- closed-loop summary:")
                    lines.append(
                        f"  - episodes traced: {summary['n_episodes']}, "
                        f"total VLM calls: {summary['total_vlm_calls']} "
                        f"(mean {summary['mean_vlm_calls_per_episode']:.1f}/episode)"
                    )
                    if summary["subtask_index_hist"]:
                        sorted_hist = dict(
                            sorted(summary["subtask_index_hist"].items())
                        )
                        lines.append(
                            f"  - subtask_index histogram: {sorted_hist}"
                        )
                    lines.append(
                        f"  - out-of-range subtask_index: {summary['total_oor']}, "
                        f"used_fallback total: {summary['used_fallback_count']}, "
                        f"mean VLM latency: {summary['mean_latency_s']:.2f}s"
                    )
                    # Show one example trace
                    first_key = next(iter(traces))
                    first_trace = traces[first_key]
                    trace_calls = first_trace.get("subtask_trace") or []
                    if trace_calls:
                        lines.append(
                            f"  - example trace (`{first_key}`, "
                            f"{len(trace_calls)} VLM calls):"
                        )
                        for call in trace_calls[:8]:
                            lines.append(
                                f"    - chunk={call.get('chunk_index')} "
                                f"status={call.get('status')} "
                                f"idx={call.get('subtask_index')} "
                                f"aug={call.get('enriched_instruction')!r}"
                            )
                        if len(trace_calls) > 8:
                            lines.append(
                                f"    - ... ({len(trace_calls) - 8} more calls)"
                            )
            lines.append("")
        lines.append("")

    output = "\n".join(lines).rstrip() + "\n"
    OUTPUT_MD.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT_MD.resolve().relative_to(PROJECT_ROOT)}")
    print()
    print(output)


if __name__ == "__main__":
    main()
