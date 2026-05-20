#!/usr/bin/env python3
"""
Upload FastWAM VLM+CoT task checkpoints to Hugging Face Hub.

Requires:
  - `pip install huggingface_hub` (repo already uses it)
  - `export HF_TOKEN=...` with write access to the target repo

Usage:
  export HF_TOKEN=hf_...
  python scripts/upload_fastwam_vlm_cot_hf.py

Optional:
  --skip-state   Do not upload the large DeepSpeed state folder (~24GB).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

REPO_ID = "Himmdy/FastWAM_VLMCoT"
RUNS = Path(__file__).resolve().parents[1] / "runs" / "robotwin_cot_3cam_384_1e-4"

TURN_SWITCH_RUN = RUNS / "2026-05-19_23-25-22_lora32V_lora32A"
TURN_SWITCH_WEIGHTS = (
    TURN_SWITCH_RUN
    / "checkpoints"
    / "weights_init_fastwam_robotwin_uncond_3cam_384"
    / "step_002120.pt"
)
TURN_SWITCH_STATE_DIR = (
    TURN_SWITCH_RUN
    / "checkpoints"
    / "state_init_fastwam_robotwin_uncond_3cam_384"
    / "step_002120"
)

PLACE_CAN_RUN = RUNS / "2026-05-20_11-30-16_lora32V_lora32A"
PLACE_CAN_WEIGHTS = (
    PLACE_CAN_RUN
    / "checkpoints"
    / "weights_init_fastwam_robotwin_uncond_3cam_384"
    / "step_000500.pt"
)


def _require_paths() -> None:
    for label, p in [
        ("turn_switch weights", TURN_SWITCH_WEIGHTS),
        ("turn_switch state dir", TURN_SWITCH_STATE_DIR),
        ("place_can_basket weights", PLACE_CAN_WEIGHTS),
    ]:
        if not p.exists():
            raise SystemExit(f"Missing {label}: {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", REPO_ID),
        help=f"Hub repo (default: {REPO_ID})",
    )
    parser.add_argument(
        "--skip-state",
        action="store_true",
        help="Skip uploading DeepSpeed trainer state (~24GB).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Error: set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) to a token with write access "
            f"to {args.repo_id}.",
            file=sys.stderr,
        )
        sys.exit(2)

    _require_paths()

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # Turn switch — model weights (MoT / LoRA etc.)
    print(f"Uploading {TURN_SWITCH_WEIGHTS.name} -> turn_switch/ ...")
    api.upload_file(
        path_or_fileobj=str(TURN_SWITCH_WEIGHTS),
        path_in_repo=f"turn_switch/{TURN_SWITCH_WEIGHTS.name}",
        repo_id=args.repo_id,
        repo_type="model",
    )

    if not args.skip_state:
        print(
            f"Uploading trainer state folder ({TURN_SWITCH_STATE_DIR}) "
            "-> turn_switch/state_step_002120/ ... (large, may take a long time)"
        )
        api.upload_folder(
            folder_path=str(TURN_SWITCH_STATE_DIR),
            path_in_repo="turn_switch/state_step_002120",
            repo_id=args.repo_id,
            repo_type="model",
        )
    else:
        print("Skipping state upload (--skip-state).")

    print(f"Uploading {PLACE_CAN_WEIGHTS.name} -> place_can_basket/ ...")
    api.upload_file(
        path_or_fileobj=str(PLACE_CAN_WEIGHTS),
        path_in_repo=f"place_can_basket/{PLACE_CAN_WEIGHTS.name}",
        repo_id=args.repo_id,
        repo_type="model",
    )

    # Dataset stats for eval (small, useful)
    ts_stats = TURN_SWITCH_RUN / "dataset_stats.json"
    pc_stats = PLACE_CAN_RUN / "dataset_stats.json"
    if ts_stats.is_file():
        api.upload_file(
            path_or_fileobj=str(ts_stats),
            path_in_repo="turn_switch/dataset_stats.json",
            repo_id=args.repo_id,
            repo_type="model",
        )
    if pc_stats.is_file():
        api.upload_file(
            path_or_fileobj=str(pc_stats),
            path_in_repo="place_can_basket/dataset_stats.json",
            repo_id=args.repo_id,
            repo_type="model",
        )

    readme = (
        "# FastWAM VLM+CoT checkpoints\n\n"
        "## turn_switch (Pretrain + CoT DiT + LoRA32)\n"
        "- `turn_switch/step_002120.pt` — model weights\n"
        "- `turn_switch/state_step_002120/` — DeepSpeed trainer state (optimizer + shards)\n"
        "- `turn_switch/dataset_stats.json` — normalization stats for eval\n\n"
        "## place_can_basket\n"
        "- `place_can_basket/step_000500.pt` — model weights\n"
        "- `place_can_basket/dataset_stats.json` — normalization stats for eval\n"
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
