#!/usr/bin/env python3
"""Segment all hanging_mug episodes by gripper transitions into 4 subtasks.

Stage definitions (canonical, matches experiments/robotwin/fastwam_policy/subtask_menus.json):
    1. left pick           [0, t_L_close)
    2. rotate + place      [t_L_close, t_L_open)
    3. right pick           [t_L_open, t_R_close)
    4. hang on rack         [t_R_close, T)

State convention (verified empirically on episode 0..549):
    state[:, 6]  -> left gripper  (1=open, 0=closed)
    state[:, 13] -> right gripper (1=open, 0=closed)

Output: <output_path> JSON of the form
    {
      "stages": ["Use the left arm ...", "Rotate the mug ...", ...],  # 4 canonical names
      "episodes": {
        "0": {"length": 337, "boundaries": [0, 54, 122, 208, 337], "ok": true},
        ...
      },
      "stats": {"total": 550, "ok": 548, "failed": 2, "failed_episode_ids": [...]}
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

LEFT_GRIPPER_IDX = 6
RIGHT_GRIPPER_IDX = 13
THR = 0.5

CANONICAL_STAGES = [
    "Use the left arm to pick up the mug from the table.",
    "Rotate the mug and place it in the center of the table.",
    "Use the right arm to pick up the mug from the center of the table.",
    "Hang the mug onto the rack with the right arm.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment hanging_mug episodes into 4 gripper-event subtasks.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/robotwin2.0_subset4_lerobot"),
        help="Source LeRobot dataset root containing the hanging_mug episodes.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=550,
        help="Number of hanging_mug episodes (subset rows 0..N-1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_segments.json"),
        help="Output path for the per-episode segment JSON.",
    )
    return parser.parse_args()


def _find_first_transition(x: np.ndarray, start: int, want_close: bool) -> Optional[int]:
    """Find first index >= start where x crosses THR in the desired direction.

    want_close=True  -> first i where x[i] < THR (gripper closes)
    want_close=False -> first i where x[i] >= THR (gripper opens)
    """
    if want_close:
        mask = x < THR
    else:
        mask = x >= THR
    idxs = np.where(mask[start:])[0]
    if len(idxs) == 0:
        return None
    return int(start + idxs[0])


def segment_episode(state: np.ndarray) -> dict:
    """Return {"boundaries": [b0,b1,b2,b3,b4], "ok": bool, "error": Optional[str]} for one episode.

    Boundaries are inclusive-start, exclusive-end indices: stage k = [b[k], b[k+1]).
    """
    T = state.shape[0]
    L = state[:, LEFT_GRIPPER_IDX]
    R = state[:, RIGHT_GRIPPER_IDX]
    if L[0] < THR or R[0] < THR:
        return {"boundaries": None, "ok": False, "error": "gripper_starts_closed"}

    t_L_close = _find_first_transition(L, start=1, want_close=True)
    if t_L_close is None:
        return {"boundaries": None, "ok": False, "error": "no_L_close"}
    t_L_open = _find_first_transition(L, start=t_L_close + 1, want_close=False)
    if t_L_open is None:
        return {"boundaries": None, "ok": False, "error": "no_L_open"}
    t_R_close = _find_first_transition(R, start=t_L_open + 1, want_close=True)
    if t_R_close is None:
        return {"boundaries": None, "ok": False, "error": "no_R_close"}
    t_R_open = _find_first_transition(R, start=t_R_close + 1, want_close=False)
    end = t_R_open if t_R_open is not None else T

    boundaries = [0, int(t_L_close), int(t_L_open), int(t_R_close), int(end)]
    if not all(boundaries[i] < boundaries[i + 1] for i in range(4)):
        return {"boundaries": None, "ok": False, "error": f"non_monotonic_boundaries:{boundaries}"}

    min_seg = min(boundaries[i + 1] - boundaries[i] for i in range(4))
    if min_seg < 2:
        return {"boundaries": None, "ok": False, "error": f"segment_too_short:{boundaries}"}

    return {"boundaries": boundaries, "ok": True, "error": None}


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    episodes_out: dict[str, dict] = {}
    failed_ids: list[int] = []
    ok_count = 0
    for ep in range(args.num_episodes):
        chunk = ep // 1000
        parquet = source / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
        if not parquet.exists():
            failed_ids.append(ep)
            episodes_out[str(ep)] = {"length": 0, "boundaries": None, "ok": False, "error": "missing_parquet"}
            continue
        table = pq.read_table(parquet, columns=["observation.state"])
        state = np.array(table["observation.state"].to_pylist())
        result = segment_episode(state)
        result["length"] = int(state.shape[0])
        episodes_out[str(ep)] = result
        if result["ok"]:
            ok_count += 1
        else:
            failed_ids.append(ep)

    stats = {
        "total": args.num_episodes,
        "ok": ok_count,
        "failed": len(failed_ids),
        "failed_episode_ids": failed_ids[:50],
        "failed_total": len(failed_ids),
    }

    payload = {
        "source": str(source),
        "stages": CANONICAL_STAGES,
        "stats": stats,
        "episodes": episodes_out,
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {output}")
    print(f"[done] total={args.num_episodes} ok={ok_count} failed={len(failed_ids)}")
    if failed_ids:
        print(f"[warn] first 10 failed ids: {failed_ids[:10]}")


if __name__ == "__main__":
    main()
