#!/usr/bin/env python3
"""Call qwen3-vl-plus once per (episode, stage) to label hanging_mug subtasks.

Inputs:
- data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_segments.json
- data/robotwin2.0_subset4_lerobot/meta/episodes.jsonl (for the original task)
- data/robotwin2.0_subset4_lerobot/videos/chunk-XXX/observation.images.cam_high/episode_*.mp4

Output / cache:
- data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_label_cache.json
  Per (episode, stage) entries keyed "ep={ep}|stage={k}" (k in 1..4).

The cache is updated in-place; re-runs only call the VLM for entries
without a successful "vlm" field. Use --overwrite to redo.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import av
import numpy as np
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

SUBSET_ROOT = Path("data/robotwin2.0_subset4_lerobot")
SEG_PATH = Path("data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_segments.json")
OUT_CACHE = Path("data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_label_cache.json")

DEFAULT_MODEL = "qwen3-vl-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CANONICAL_STAGES = [
    "Use the left arm to pick up the mug from the table.",
    "Rotate the mug and place it in the center of the table.",
    "Use the right arm to pick up the mug from the center of the table.",
    "Hang the mug onto the rack with the right arm.",
]

MAX_CHARS = 400

SYSTEM_PROMPT = (
    "You label one subtask segment for a dual-arm tabletop robot. You get the "
    "FIRST and LAST head-camera frames, the original task, and a stage hint. "
    "Write a concrete visually-grounded subtask description.\n\n"
    "Rules:\n"
    "- 1-3 imperative sentences, total <= 400 chars, start with a verb.\n"
    "- Describe the relevant object(s) using visible color, material, shape, "
    "handle orientation, and position (left / center / right / edge).\n"
    "- If a target (rack rod, basket, table center) matters, describe it too.\n"
    "- State which arm if stage-specific.\n"
    "- Optionally add 1 short clause about the LAST-frame end state.\n"
    "- No markdown, no quotes, no numbered lists, no chain-of-thought.\n\n"
    'Return JSON only: {"subtask_instruction": "...", "reason": "<one short '
    'sentence>"}.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM labeling for hanging_mug subtasks.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true", help="Re-call VLM even for cached successful entries.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many (episode, stage) entries.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Subset of episode indices (default: all).")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable VLM thinking mode (slower).")
    parser.add_argument("--thinking-budget", type=int, default=512)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit without calling VLM.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _encode_rgb_as_data_url(rgb: np.ndarray) -> str:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _extract_frames(ep: int, indices: list[int]) -> dict[int, np.ndarray]:
    chunk = ep // 1000
    path = (
        SUBSET_ROOT
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.cam_high"
        / f"episode_{ep:06d}.mp4"
    )
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    max_wanted = max(wanted)
    container = av.open(str(path))
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= max_wanted:
                break
    finally:
        container.close()
    missing = wanted - out.keys()
    if missing:
        raise ValueError(f"Failed to extract frames {sorted(missing)} from {path}")
    return out


def _try_parse_json(answer: str) -> Optional[dict]:
    if not answer:
        return None
    s = answer.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip("'")
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    last_period = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
    if last_period >= int(max_chars * 0.5):
        return head[: last_period + 1].strip()
    return head.strip()


def _call_vlm(
    client: OpenAI,
    model: str,
    enable_thinking: bool,
    thinking_budget: int,
    *,
    raw_task: str,
    stage_hint: str,
    data_url_first: str,
    data_url_last: str,
    ep: int,
    stage: int,
) -> dict:
    user_text = (
        f"Original full task: {raw_task}\n"
        f"Stage hint (canonical): {stage_hint}\n"
        f"Stage index: {stage} of 4 for the hanging_mug task.\n"
        f"Episode id: {ep}.\n"
        "Two images follow: (1) FIRST frame of this segment, (2) LAST frame of this segment."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url_first}},
                {"type": "image_url", "image_url": {"url": data_url_last}},
            ],
        },
    ]
    extra_body = {"enable_thinking": bool(enable_thinking)}
    if enable_thinking:
        extra_body["thinking_budget"] = int(thinking_budget)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        extra_body=extra_body,
        stream_options={"include_usage": True},
    )
    reasoning_chunks: list[str] = []
    answer_chunks: list[str] = []
    usage = None
    for chunk in completion:
        if not chunk.choices:
            usage = chunk.usage
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_chunks.append(reasoning)
            continue
        content = getattr(delta, "content", None)
        if content:
            answer_chunks.append(content)
    answer = "".join(answer_chunks)
    usage_dict = None
    if usage is not None:
        try:
            usage_dict = usage.model_dump()
        except Exception:
            try:
                usage_dict = usage.dict()
            except Exception:
                usage_dict = None
    return {
        "answer": answer,
        "reasoning": "".join(reasoning_chunks),
        "usage": usage_dict,
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not args.dry_run and not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)

    if not SEG_PATH.exists():
        raise FileNotFoundError(f"Missing segments file: {SEG_PATH}")
    segments = json.loads(SEG_PATH.read_text(encoding="utf-8"))
    episodes_meta = _read_jsonl(SUBSET_ROOT / "meta" / "episodes.jsonl")
    ep_to_task: dict[int, str] = {}
    for rec in episodes_meta:
        ti = int(rec["episode_index"])
        tasks = rec.get("tasks") or []
        ep_to_task[ti] = str(tasks[0]) if tasks else "Hang the mug on the rack."

    # Build the work list.
    valid_episodes = [int(ep) for ep, info in segments["episodes"].items() if info["ok"]]
    valid_episodes.sort()
    if args.episodes is not None:
        chosen = sorted(set(args.episodes) & set(valid_episodes))
    else:
        chosen = valid_episodes
    print(f"[info] valid episodes: {len(valid_episodes)}; chosen: {len(chosen)}")

    cache: dict[str, dict] = {}
    if OUT_CACHE.exists():
        try:
            cache = json.loads(OUT_CACHE.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to load existing cache, starting fresh: {exc!r}")
            cache = {}
    print(f"[info] loaded {len(cache)} cached entries from {OUT_CACHE}")

    work: list[tuple[int, int, list[int]]] = []
    for ep in chosen:
        info = segments["episodes"][str(ep)]
        b = info["boundaries"]
        ep_indices: list[int] = []
        for k in range(1, 5):
            key = f"ep={ep}|stage={k}"
            entry = cache.get(key)
            if (
                not args.overwrite
                and entry is not None
                and isinstance(entry.get("vlm"), dict)
                and entry["vlm"].get("error") is None
                and entry["vlm"].get("subtask_instruction")
            ):
                continue
            first_idx = b[k - 1]
            last_idx = max(first_idx, b[k] - 1)
            ep_indices.append(k)
        if ep_indices:
            work.append((ep, ep_indices, b))

    total_calls = sum(len(stages) for _, stages, _ in work)
    if args.limit is not None:
        # Truncate by total VLM call count
        truncated: list[tuple[int, list[int], list[int]]] = []
        remaining = int(args.limit)
        for ep, stages, b in work:
            if remaining <= 0:
                break
            take = stages[:remaining]
            truncated.append((ep, take, b))
            remaining -= len(take)
        work = truncated
        total_calls = sum(len(stages) for _, stages, _ in work)

    print(f"[info] planned VLM calls: {total_calls} (over {len(work)} episodes)")
    if args.dry_run:
        for ep, stages, _ in work[:5]:
            print(f"  ep={ep} stages={stages}")
        return

    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=args.request_timeout)

    cache_lock = threading.Lock()
    flush_counter = {"n": 0}
    progress = {"done": 0, "fail": 0, "t0": time.time()}
    pbar = tqdm(total=total_calls, desc="vlm_calls", unit="call", smoothing=0.05, dynamic_ncols=True)

    def _flush() -> None:
        with cache_lock:
            _atomic_write_json(OUT_CACHE, cache)

    def _bump_pbar(success: bool, latency: float) -> None:
        with cache_lock:
            pbar.update(1)
            pbar.set_postfix(
                ok=progress["done"],
                fail=progress["fail"],
                last_s=f"{latency:.1f}",
            )

    def _process_episode(ep: int, stages: list[int], b: list[int]) -> None:
        # Precompute the required frame indices once per episode.
        needed: list[int] = []
        for k in stages:
            needed.append(b[k - 1])
            needed.append(max(b[k - 1], b[k] - 1))
        needed = sorted(set(needed))
        try:
            frames = _extract_frames(ep, needed)
        except Exception as exc:
            with cache_lock:
                for k in stages:
                    cache[f"ep={ep}|stage={k}"] = {
                        "ep": ep,
                        "stage": k,
                        "boundaries": [b[k - 1], b[k]],
                        "vlm": {
                            "subtask_instruction": "",
                            "reason": "",
                            "raw_answer": "",
                            "latency_s": 0.0,
                            "usage": None,
                            "error": f"frame_extract_failed: {exc!r}",
                        },
                    }
                progress["fail"] += len(stages)
                for _ in stages:
                    pbar.update(1)
                pbar.set_postfix(ok=progress["done"], fail=progress["fail"])
            return

        for k in stages:
            first_idx = b[k - 1]
            last_idx = max(first_idx, b[k] - 1)
            key = f"ep={ep}|stage={k}"
            stage_hint = CANONICAL_STAGES[k - 1]
            raw_task = ep_to_task.get(ep, "Hang the mug on the rack.")
            data_url_first = _encode_rgb_as_data_url(frames[first_idx])
            data_url_last = _encode_rgb_as_data_url(frames[last_idx])

            last_exc: Optional[BaseException] = None
            response: Optional[dict] = None
            t0 = time.perf_counter()
            for attempt in range(1, args.max_retries + 1):
                try:
                    response = _call_vlm(
                        client,
                        DEFAULT_MODEL,
                        args.enable_thinking,
                        args.thinking_budget,
                        raw_task=raw_task,
                        stage_hint=stage_hint,
                        data_url_first=data_url_first,
                        data_url_last=data_url_last,
                        ep=ep,
                        stage=k,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < args.max_retries:
                        time.sleep(args.retry_backoff * attempt)
            latency = time.perf_counter() - t0

            if response is None:
                entry = {
                    "ep": ep,
                    "stage": k,
                    "boundaries": [b[k - 1], b[k]],
                    "vlm": {
                        "subtask_instruction": "",
                        "reason": "",
                        "raw_answer": "",
                        "latency_s": latency,
                        "usage": None,
                        "error": f"call_failed: {last_exc!r}",
                    },
                }
                with cache_lock:
                    cache[key] = entry
                    progress["fail"] += 1
                _bump_pbar(False, latency)
                continue

            answer = response["answer"]
            parsed = _try_parse_json(answer)
            subtask_instruction = ""
            reason = ""
            err: Optional[str] = None
            if parsed is None:
                err = "non_json_answer"
                # Fall back to using the raw answer if it's short enough.
                subtask_instruction = _truncate(answer)
            else:
                subtask_instruction = _truncate(str(parsed.get("subtask_instruction", "")).strip())
                reason = str(parsed.get("reason", "")).strip()
                if not subtask_instruction:
                    err = "missing_subtask_instruction_field"
            if not subtask_instruction:
                subtask_instruction = stage_hint
                err = err or "empty_after_clean"

            entry = {
                "ep": ep,
                "stage": k,
                "boundaries": [b[k - 1], b[k]],
                "first_frame_idx": first_idx,
                "last_frame_idx": last_idx,
                "vlm": {
                    "subtask_instruction": subtask_instruction,
                    "reason": reason,
                    "raw_answer": answer,
                    "latency_s": latency,
                    "usage": response.get("usage"),
                    "error": err,
                },
            }
            with cache_lock:
                cache[key] = entry
                if err is None:
                    progress["done"] += 1
                else:
                    progress["fail"] += 1
                flush_counter["n"] += 1
                need_flush = flush_counter["n"] >= args.flush_every
                if need_flush:
                    flush_counter["n"] = 0
            if need_flush:
                _flush()
            _bump_pbar(err is None, latency)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_episode, ep, stages, b) for ep, stages, b in work]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                traceback.print_exc()
                with cache_lock:
                    progress["fail"] += 1

    pbar.close()
    _flush()
    with cache_lock:
        n_ok = sum(1 for v in cache.values() if isinstance(v.get("vlm"), dict) and v["vlm"].get("error") is None)
        n_fail = sum(1 for v in cache.values() if isinstance(v.get("vlm"), dict) and v["vlm"].get("error") is not None)
        print(f"[done] cache total entries={len(cache)} vlm_ok={n_ok} vlm_fail={n_fail}")
        print(f"[done] cache file: {OUT_CACHE}")


if __name__ == "__main__":
    main()
