#!/usr/bin/env python3
"""
Build a 4-task RoboTwin LeRobot subset with episode-level validation.

Selected tasks:
- hanging_mug:      [5500, 6050)
- open_microwave:   [9350, 9900)
- place_can_basket: [13750, 14300)
- turn_switch:      [26950, 27500)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


@dataclass(frozen=True)
class TaskRange:
    task_name: str
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start


SELECTED_TASK_RANGES: list[TaskRange] = [
    TaskRange("hanging_mug", 5500, 6050),
    TaskRange("open_microwave", 9350, 9900),
    TaskRange("place_can_basket", 13750, 14300),
    TaskRange("turn_switch", 26950, 27500),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 4-task RoboTwin LeRobot subset dataset.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/robotwin2.0/robotwin2.0"),
        help="Source LeRobot dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/robotwin2.0_subset4_lerobot"),
        help="Output subset dataset root.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to materialize files into subset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory if it already exists.",
    )
    return parser.parse_args()


def _iter_selected_episode_indices(task_ranges: Iterable[TaskRange]) -> list[tuple[int, str]]:
    selected: list[tuple[int, str]] = []
    for tr in task_ranges:
        selected.extend((idx, tr.task_name) for idx in range(tr.start, tr.end))
    selected.sort(key=lambda x: x[0])
    return selected


def _expected_task_for_episode(episode_index: int) -> str:
    for tr in SELECTED_TASK_RANGES:
        if tr.start <= episode_index < tr.end:
            return tr.task_name
    raise ValueError(f"Episode index out of selected ranges: {episode_index}")


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


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _recompute_index_stats(length: int, frame_start: int) -> dict:
    frame_end = frame_start + length - 1
    mean = (frame_start + frame_end) / 2.0
    if length <= 1:
        std = 0.0
    else:
        # std of equally spaced integers [a, a+1, ..., a+n-1].
        std = math.sqrt((length * length - 1) / 12.0)
    return {
        "min": [frame_start],
        "max": [frame_end],
        "mean": [mean],
        "std": [std],
        "count": [length],
    }


def _recompute_episode_index_stats(new_episode_index: int, length: int) -> dict:
    return {
        "min": [new_episode_index],
        "max": [new_episode_index],
        "mean": [new_episode_index],
        "std": [0.0],
        "count": [length],
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Pass --overwrite to rebuild."
            )
        shutil.rmtree(output)

    selected = _iter_selected_episode_indices(SELECTED_TASK_RANGES)
    print(f"[info] selected episodes: {len(selected)}")
    for tr in SELECTED_TASK_RANGES:
        print(f"[info]  - {tr.task_name}: [{tr.start}, {tr.end}) count={tr.count}")

    src_meta = source / "meta"
    src_data = source / "data"
    src_videos = source / "videos"
    dst_meta = output / "meta"
    dst_data = output / "data"
    dst_videos = output / "videos"

    # Load source metadata fully (size is manageable for episodes/episode_stats).
    episodes_meta = _read_jsonl(src_meta / "episodes.jsonl")
    episodes_stats_meta = _read_jsonl(src_meta / "episodes_stats.jsonl")
    if len(episodes_meta) != len(episodes_stats_meta):
        raise ValueError(
            f"Mismatched meta lengths: episodes={len(episodes_meta)} "
            f"episodes_stats={len(episodes_stats_meta)}"
        )

    # Build subset.
    new_episodes_meta: list[dict] = []
    new_episodes_stats_meta: list[dict] = []
    used_task_indices: set[int] = set()
    validation_rows: list[dict] = []

    global_frame_start = 0
    for new_episode_index, (src_episode_index, expected_task) in enumerate(selected):
        ep_meta = episodes_meta[src_episode_index].copy()
        ep_stats = episodes_stats_meta[src_episode_index].copy()
        if ep_meta.get("episode_index") != src_episode_index:
            raise ValueError(
                f"episodes.jsonl row mismatch at line {src_episode_index}: "
                f"episode_index={ep_meta.get('episode_index')}"
            )
        if ep_stats.get("episode_index") != src_episode_index:
            raise ValueError(
                f"episodes_stats.jsonl row mismatch at line {src_episode_index}: "
                f"episode_index={ep_stats.get('episode_index')}"
            )

        # Validate by episode index -> task mapping.
        mapped_task = _expected_task_for_episode(src_episode_index)
        if mapped_task != expected_task:
            raise ValueError(
                f"Internal mapping mismatch for episode {src_episode_index}: "
                f"{mapped_task} != {expected_task}"
            )

        src_chunk = src_episode_index // 1000
        dst_chunk = new_episode_index // 1000
        src_parquet = src_data / f"chunk-{src_chunk:03d}" / f"episode_{src_episode_index:06d}.parquet"
        dst_parquet = dst_data / f"chunk-{dst_chunk:03d}" / f"episode_{new_episode_index:06d}.parquet"
        if not src_parquet.exists():
            raise FileNotFoundError(f"Missing source parquet: {src_parquet}")

        table = pq.read_table(src_parquet)
        num_rows = table.num_rows
        # Update per-row episode_index and global frame index to keep subset self-consistent.
        new_episode_col = pa.array([new_episode_index] * num_rows, type=pa.int64())
        new_index_col = pa.array(range(global_frame_start, global_frame_start + num_rows), type=pa.int64())
        ep_col_idx = table.column_names.index("episode_index")
        idx_col_idx = table.column_names.index("index")
        table = table.set_column(ep_col_idx, "episode_index", new_episode_col)
        table = table.set_column(idx_col_idx, "index", new_index_col)

        used_task_indices.update(int(v) for v in table["task_index"].to_pylist())
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_parquet)

        # Videos: keep 3 camera streams.
        for cam_key in CAMERA_KEYS:
            src_video = (
                src_videos
                / f"chunk-{src_chunk:03d}"
                / cam_key
                / f"episode_{src_episode_index:06d}.mp4"
            )
            dst_video = (
                dst_videos
                / f"chunk-{dst_chunk:03d}"
                / cam_key
                / f"episode_{new_episode_index:06d}.mp4"
            )
            if not src_video.exists():
                raise FileNotFoundError(f"Missing source video: {src_video}")
            _safe_link_or_copy(src_video, dst_video, args.link_mode)

        # Rebuild per-episode metadata rows.
        ep_meta["episode_index"] = new_episode_index
        new_episodes_meta.append(ep_meta)

        ep_stats["episode_index"] = new_episode_index
        stats = ep_stats["stats"]
        stats["episode_index"] = _recompute_episode_index_stats(new_episode_index, num_rows)
        stats["index"] = _recompute_index_stats(num_rows, global_frame_start)
        new_episodes_stats_meta.append(ep_stats)

        validation_rows.append(
            {
                "src_episode_index": src_episode_index,
                "subset_episode_index": new_episode_index,
                "mapped_task": expected_task,
                "length": num_rows,
                "first_task_preview": ep_meta["tasks"][0] if ep_meta.get("tasks") else "",
            }
        )
        global_frame_start += num_rows

    # Filter tasks.jsonl to used task indices.
    filtered_tasks: list[dict] = []
    with (src_meta / "tasks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if int(rec["task_index"]) in used_task_indices:
                filtered_tasks.append(rec)
    filtered_tasks.sort(key=lambda x: int(x["task_index"]))

    missing_task_indices = sorted(used_task_indices - {int(x["task_index"]) for x in filtered_tasks})
    if missing_task_indices:
        raise ValueError(
            f"Missing task_index rows in tasks.jsonl for {len(missing_task_indices)} indices. "
            f"First few: {missing_task_indices[:10]}"
        )

    # Build info.json
    src_info = json.loads((src_meta / "info.json").read_text(encoding="utf-8"))
    dst_info = dict(src_info)
    total_episodes = len(selected)
    total_frames = global_frame_start
    total_chunks = (total_episodes + 1000 - 1) // 1000
    dst_info["total_episodes"] = total_episodes
    dst_info["total_frames"] = total_frames
    dst_info["total_tasks"] = len(filtered_tasks)
    dst_info["total_videos"] = total_episodes * len(CAMERA_KEYS)
    dst_info["total_chunks"] = total_chunks
    dst_info["splits"] = {"train": f"0:{total_episodes}"}

    # Write meta.
    dst_meta.mkdir(parents=True, exist_ok=True)
    (dst_meta / "info.json").write_text(json.dumps(dst_info, ensure_ascii=False, indent=4), encoding="utf-8")
    _write_jsonl(dst_meta / "episodes.jsonl", new_episodes_meta)
    _write_jsonl(dst_meta / "episodes_stats.jsonl", new_episodes_stats_meta)
    _write_jsonl(dst_meta / "tasks.jsonl", filtered_tasks)

    # Validation summary.
    by_task_counts: dict[str, int] = {tr.task_name: 0 for tr in SELECTED_TASK_RANGES}
    for row in validation_rows:
        by_task_counts[row["mapped_task"]] += 1

    validation = {
        "source_dataset": str(source),
        "output_dataset": str(output),
        "link_mode": args.link_mode,
        "selected_task_ranges": [
            {"task_name": tr.task_name, "start": tr.start, "end": tr.end, "count": tr.count}
            for tr in SELECTED_TASK_RANGES
        ],
        "expected_total_episodes": sum(tr.count for tr in SELECTED_TASK_RANGES),
        "actual_total_episodes": len(validation_rows),
        "actual_total_frames": total_frames,
        "by_task_episode_counts": by_task_counts,
        "sample_rows_head": validation_rows[:5],
        "sample_rows_tail": validation_rows[-5:],
    }
    (dst_meta / "subset_validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[done] subset dataset created successfully")
    print(f"[done] output: {output}")
    print(f"[done] episodes: {len(validation_rows)}, frames: {total_frames}, tasks: {len(filtered_tasks)}")
    print(f"[done] validation report: {dst_meta / 'subset_validation_report.json'}")


if __name__ == "__main__":
    main()
