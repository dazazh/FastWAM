#!/usr/bin/env python3
"""Build the relabeled hanging_mug subtask LeRobot dataset.

Inputs:
- data/robotwin2.0_subset4_lerobot/                       (source: 550 hanging_mug episodes are subset rows 0..549)
- data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_segments.json
- data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_label_cache.json
- experiments/robotwin/fastwam_policy/subtask_menus.json  (for the 4 canonical anchor strings)

Outputs (under data/robotwin2.0_hangingmug_subtask_lerobot/):
- data/chunk-000/episode_*.parquet                       (rewritten task_index column)
- videos/chunk-000/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_*.mp4
                                                          (hardlinked from the source subset)
- meta/info.json
- meta/tasks.jsonl
- meta/episodes.jsonl
- meta/episodes_stats.jsonl
- meta/build_report.json

For each frame in segment k of episode ep, task_index is randomly sampled from the pool of
9 strings = 1 VLM + 7 LLM paraphrases + 1 canonical anchor for that (ep, k). The pool is
deterministic per (ep, k) and the per-frame draw is reproducible given --seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
N_STAGES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the hanging_mug subtask LeRobot dataset.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/robotwin2.0_subset4_lerobot"),
        help="Source LeRobot dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/robotwin2.0_hangingmug_subtask_lerobot"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="Path to subtask_segments.json (defaults to <output>/meta/subtask_segments.json).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to subtask_label_cache.json (defaults to <output>/meta/subtask_label_cache.json).",
    )
    parser.add_argument(
        "--subtask-menus",
        type=Path,
        default=Path("experiments/robotwin/fastwam_policy/subtask_menus.json"),
        help="Path to subtask_menus.json (for canonical anchor strings).",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="hanging_mug",
        help="Key inside subtask_menus.json to read canonical stage strings from.",
    )
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true", help="Remove output data/, videos/ if present.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def _stats_for_int_column(values: np.ndarray) -> dict:
    n = int(values.shape[0])
    if n == 0:
        return {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [0]}
    vmin = int(values.min())
    vmax = int(values.max())
    mean = float(values.mean())
    std = float(values.std()) if n > 1 else 0.0
    return {"min": [vmin], "max": [vmax], "mean": [mean], "std": [std], "count": [n]}


def _stats_for_index(length: int, frame_start: int) -> dict:
    frame_end = frame_start + length - 1
    mean = (frame_start + frame_end) / 2.0
    std = math.sqrt((length * length - 1) / 12.0) if length > 1 else 0.0
    return {"min": [frame_start], "max": [frame_end], "mean": [mean], "std": [std], "count": [length]}


def _stats_for_episode_index(ep: int, length: int) -> dict:
    return {"min": [ep], "max": [ep], "mean": [ep], "std": [0.0], "count": [length]}


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    segments_path = (args.segments or (output / "meta" / "subtask_segments.json")).resolve()
    labels_path = (args.labels or (output / "meta" / "subtask_label_cache.json")).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments JSON not found: {segments_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels JSON not found: {labels_path}")

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    menus = json.loads(args.subtask_menus.read_text(encoding="utf-8"))
    canonical_stages: list[str] = list(menus["tasks"][args.task_name]["stages"])
    if len(canonical_stages) != N_STAGES:
        raise ValueError(
            f"Expected {N_STAGES} canonical stages for {args.task_name}, "
            f"got {len(canonical_stages)}: {canonical_stages}"
        )

    valid_episodes = sorted(int(ep) for ep, info in segments["episodes"].items() if info["ok"])
    print(f"[info] valid episodes: {len(valid_episodes)} (out of {len(segments['episodes'])})")

    # Validate that every (ep, stage) has a fully-labeled cache entry.
    missing_entries: list[str] = []
    incomplete_paraphrases: list[str] = []
    for ep in valid_episodes:
        for k in range(1, N_STAGES + 1):
            key = f"ep={ep}|stage={k}"
            entry = labels.get(key)
            if entry is None:
                missing_entries.append(key)
                continue
            vlm = entry.get("vlm") or {}
            llm = entry.get("llm") or {}
            if not vlm.get("subtask_instruction"):
                missing_entries.append(f"{key} (no vlm)")
                continue
            if not isinstance(llm.get("paraphrases"), list) or len(llm["paraphrases"]) < 1:
                incomplete_paraphrases.append(f"{key} (only {len(llm.get('paraphrases', []))} paraphrases)")
    if missing_entries:
        print(f"[warn] {len(missing_entries)} (ep, stage) entries are missing VLM labels; skipping those episodes.")
        print(f"[warn] first 5: {missing_entries[:5]}")
    if incomplete_paraphrases:
        print(f"[warn] {len(incomplete_paraphrases)} (ep, stage) entries have < 1 LLM paraphrase.")
        print(f"[warn] first 5: {incomplete_paraphrases[:5]}")

    # Drop episodes that lack any label.
    episodes_to_emit: list[int] = []
    for ep in valid_episodes:
        ok = True
        for k in range(1, N_STAGES + 1):
            entry = labels.get(f"ep={ep}|stage={k}")
            if entry is None or not (entry.get("vlm") or {}).get("subtask_instruction"):
                ok = False
                break
        if ok:
            episodes_to_emit.append(ep)
    print(f"[info] episodes to emit: {len(episodes_to_emit)}")

    # Build the global task vocabulary.
    # - First the 4 canonical anchors at indices 0..3.
    # - Then every unique VLM/LLM string in the order they appear (sorted by ep, stage).
    task_to_index: dict[str, int] = {}
    tasks_jsonl_rows: list[dict] = []

    def _add_task(s: str) -> int:
        s = s.strip()
        if not s:
            raise ValueError("empty task string")
        if s in task_to_index:
            return task_to_index[s]
        idx = len(task_to_index)
        task_to_index[s] = idx
        tasks_jsonl_rows.append({"task_index": idx, "task": s})
        return idx

    canonical_indices = [_add_task(s) for s in canonical_stages]

    # Per-(ep, stage) pool of task_index values.
    pool_per_ep_stage: dict[tuple[int, int], list[int]] = {}
    for ep in episodes_to_emit:
        for k in range(1, N_STAGES + 1):
            entry = labels[f"ep={ep}|stage={k}"]
            pool_strings = [str(entry["vlm"]["subtask_instruction"]).strip()]
            llm = entry.get("llm") or {}
            for p in (llm.get("paraphrases") or []):
                ps = str(p).strip()
                if ps:
                    pool_strings.append(ps)
            pool_strings.append(canonical_stages[k - 1])
            # De-dup within the pool (keep order).
            seen_in_pool: set[str] = set()
            ordered: list[str] = []
            for s in pool_strings:
                if s not in seen_in_pool:
                    seen_in_pool.add(s)
                    ordered.append(s)
            pool_per_ep_stage[(ep, k)] = [_add_task(s) for s in ordered]

    print(f"[info] total unique task strings: {len(tasks_jsonl_rows)}")
    print(f"[info] pool sizes per (ep, stage): "
          f"min={min(len(v) for v in pool_per_ep_stage.values())} "
          f"max={max(len(v) for v in pool_per_ep_stage.values())} "
          f"mean={np.mean([len(v) for v in pool_per_ep_stage.values()]):.2f}")

    # Prepare output directories.
    dst_meta = output / "meta"
    dst_data = output / "data"
    dst_videos = output / "videos"
    if args.overwrite:
        for d in (dst_data, dst_videos):
            if d.exists():
                shutil.rmtree(d)
    dst_meta.mkdir(parents=True, exist_ok=True)
    dst_data.mkdir(parents=True, exist_ok=True)
    dst_videos.mkdir(parents=True, exist_ok=True)

    # Rebuild source meta lookups.
    src_episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    src_episodes_stats = _read_jsonl(source / "meta" / "episodes_stats.jsonl")
    if len(src_episodes) != len(src_episodes_stats):
        raise ValueError("Mismatched lengths between episodes.jsonl and episodes_stats.jsonl in source.")
    src_ep_by_index: dict[int, dict] = {int(rec["episode_index"]): rec for rec in src_episodes}
    src_stats_by_index: dict[int, dict] = {int(rec["episode_index"]): rec for rec in src_episodes_stats}

    rng = np.random.default_rng(args.seed)
    new_episodes: list[dict] = []
    new_episodes_stats: list[dict] = []
    build_report_rows: list[dict] = []

    global_frame_start = 0
    for new_ep_idx, src_ep_idx in enumerate(episodes_to_emit):
        src_chunk = src_ep_idx // 1000
        dst_chunk = new_ep_idx // 1000
        src_parquet = source / "data" / f"chunk-{src_chunk:03d}" / f"episode_{src_ep_idx:06d}.parquet"
        dst_parquet = dst_data / f"chunk-{dst_chunk:03d}" / f"episode_{new_ep_idx:06d}.parquet"
        if not src_parquet.exists():
            raise FileNotFoundError(f"Missing source parquet: {src_parquet}")

        table = pq.read_table(src_parquet)
        num_rows = table.num_rows
        seg_info = segments["episodes"][str(src_ep_idx)]
        boundaries = seg_info["boundaries"]
        if boundaries[-1] != num_rows:
            # Stage 4 ends at t_R_open (sometimes earlier than T). We map any
            # tail frames past boundaries[-1] to stage 4.
            pass

        # Compute per-frame task_index using deterministic per-(ep, k) RNG.
        new_task_idx = np.empty(num_rows, dtype=np.int64)
        for k in range(1, N_STAGES + 1):
            start = boundaries[k - 1]
            end = boundaries[k] if k < N_STAGES else num_rows  # tail frames after stage 4 go to stage 4
            pool = pool_per_ep_stage[(src_ep_idx, k)]
            seed_str = f"{args.seed}|ep={src_ep_idx}|stage={k}"
            local_seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
            local_rng = np.random.default_rng(local_seed)
            for f in range(start, end):
                new_task_idx[f] = int(local_rng.choice(pool))
        # Tail catch-all (any rows past stage 4 boundary).
        if boundaries[-1] < num_rows:
            pool = pool_per_ep_stage[(src_ep_idx, N_STAGES)]
            tail_seed = int(hashlib.sha256(f"{args.seed}|ep={src_ep_idx}|tail".encode()).hexdigest()[:8], 16)
            tail_rng = np.random.default_rng(tail_seed)
            for f in range(boundaries[-1], num_rows):
                new_task_idx[f] = int(tail_rng.choice(pool))

        new_ep_col = pa.array([new_ep_idx] * num_rows, type=pa.int64())
        new_index_col = pa.array(range(global_frame_start, global_frame_start + num_rows), type=pa.int64())
        new_task_col = pa.array(new_task_idx.tolist(), type=pa.int64())
        # Replace columns.
        col_names = table.column_names
        ep_col_idx = col_names.index("episode_index")
        idx_col_idx = col_names.index("index")
        task_col_idx = col_names.index("task_index")
        table = table.set_column(ep_col_idx, "episode_index", new_ep_col)
        table = table.set_column(idx_col_idx, "index", new_index_col)
        table = table.set_column(task_col_idx, "task_index", new_task_col)

        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_parquet)

        # Videos: hardlink/copy the 3 cameras.
        for cam_key in CAMERA_KEYS:
            src_video = (
                source / "videos" / f"chunk-{src_chunk:03d}" / cam_key / f"episode_{src_ep_idx:06d}.mp4"
            )
            dst_video = (
                dst_videos / f"chunk-{dst_chunk:03d}" / cam_key / f"episode_{new_ep_idx:06d}.mp4"
            )
            if not src_video.exists():
                raise FileNotFoundError(f"Missing source video: {src_video}")
            _safe_link_or_copy(src_video, dst_video, args.link_mode)

        # Build per-episode meta rows.
        src_ep_meta = src_ep_by_index[src_ep_idx]
        src_ep_stats = src_stats_by_index[src_ep_idx]
        # New unique tasks for this episode.
        unique_idxs = sorted(set(int(v) for v in new_task_idx.tolist()))
        unique_strings = [tasks_jsonl_rows[i]["task"] for i in unique_idxs]
        new_episodes.append({
            "episode_index": new_ep_idx,
            "tasks": unique_strings,
            "length": num_rows,
        })

        # Rebuild stats: keep the source stats but overwrite episode_index, index, task_index.
        stats = dict(src_ep_stats["stats"])
        stats["episode_index"] = _stats_for_episode_index(new_ep_idx, num_rows)
        stats["index"] = _stats_for_index(num_rows, global_frame_start)
        stats["task_index"] = _stats_for_int_column(new_task_idx)
        new_episodes_stats.append({"episode_index": new_ep_idx, "stats": stats})

        build_report_rows.append({
            "new_ep": new_ep_idx,
            "src_ep": src_ep_idx,
            "length": num_rows,
            "boundaries": boundaries,
            "unique_task_index_count": len(unique_idxs),
        })

        global_frame_start += num_rows

    # Write tasks.jsonl (already deduplicated).
    _write_jsonl(dst_meta / "tasks.jsonl", tasks_jsonl_rows)
    _write_jsonl(dst_meta / "episodes.jsonl", new_episodes)
    _write_jsonl(dst_meta / "episodes_stats.jsonl", new_episodes_stats)

    # Build info.json from the source one.
    src_info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    info = dict(src_info)
    info["total_episodes"] = len(episodes_to_emit)
    info["total_frames"] = int(global_frame_start)
    info["total_tasks"] = len(tasks_jsonl_rows)
    info["total_videos"] = len(episodes_to_emit) * len(CAMERA_KEYS)
    info["total_chunks"] = (len(episodes_to_emit) + 1000 - 1) // 1000
    info["splits"] = {"train": f"0:{len(episodes_to_emit)}"}
    (dst_meta / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding="utf-8")

    report = {
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "link_mode": args.link_mode,
        "n_episodes_emitted": len(episodes_to_emit),
        "n_total_frames": int(global_frame_start),
        "n_unique_tasks": len(tasks_jsonl_rows),
        "canonical_anchor_indices": canonical_indices,
        "canonical_stages": canonical_stages,
        "first_5_episodes": build_report_rows[:5],
        "last_5_episodes": build_report_rows[-5:],
    }
    (dst_meta / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] dataset assembled at {output}")
    print(f"[done] episodes={len(episodes_to_emit)} frames={global_frame_start} "
          f"unique_tasks={len(tasks_jsonl_rows)}")


if __name__ == "__main__":
    main()
