"""Smoke test for Qwen3-VL-Plus via DashScope (OpenAI-compatible endpoint).

Two checks:
1. Public URL image + a Chinese question (matches the Alibaba example) to verify
   the API key and network reachability.
2. A real RoboTwin first-frame image (extracted from a local dataset video)
   + a sample task instruction, with the actual augmented-instruction prompt
   we will use in eval. Prints both `reasoning_content` and the final answer.

Run:
    export DASHSCOPE_API_KEY=...
    python zhengxj/vlm_smoke_test.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from openai import OpenAI
from PIL import Image

DEFAULT_VIDEO = (
    "data/robotwin2.0/robotwin2.0/videos/chunk-000/"
    "observation.images.cam_high/episode_000000.mp4"
)
DEFAULT_TASK_INSTRUCTION = (
    "Grab the smooth green plastic bottle and lift it with the left arm"
)

SYSTEM_PROMPT = (
    "You are a vision planner for a tabletop dual-arm robot. "
    "You will be given (a) the first camera frame of a RoboTwin episode and "
    "(b) the original task instruction. Look at the frame and rewrite the "
    "instruction into ONE short, concrete English sentence that the low-level "
    "policy can follow. The sentence must:\n"
    "- Preserve the original goal verbatim in meaning.\n"
    "- Add at most one short clause describing where the relevant object(s) "
    "currently are in the scene (e.g. \"the mug is on the left, the rack is on "
    "the right\").\n"
    "- Use simple imperative English, no markdown, no quotation marks.\n"
    "- Stay under 200 characters.\n"
    'Return strictly this JSON: {"augmented_instruction": "...", "reasoning": "..."}.'
)


def _require_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        sys.stderr.write(
            "ERROR: DASHSCOPE_API_KEY is not set. Run `export DASHSCOPE_API_KEY=...` first.\n"
        )
        sys.exit(2)
    return key


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=_require_api_key(),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _encode_rgb_as_data_url(rgb: np.ndarray) -> str:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 RGB, got shape={rgb.shape} dtype={rgb.dtype}")
    pil = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_first_frame(video_path: Path) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    frame = iio.imread(str(video_path), index=0, plugin="pyav")
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[2] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def _stream_response(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict],
    enable_thinking: bool,
    thinking_budget: int,
) -> dict:
    reasoning_chunks: list[str] = []
    answer_chunks: list[str] = []
    usage = None
    is_answering_now = False

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

    if enable_thinking:
        print("\n" + "=" * 20 + " thinking " + "=" * 20)

    for chunk in completion:
        if not chunk.choices:
            usage = chunk.usage
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="", flush=True)
            reasoning_chunks.append(reasoning)
            continue
        content = getattr(delta, "content", None) or ""
        if content and not is_answering_now:
            print("\n" + "=" * 20 + " answer " + "=" * 20)
            is_answering_now = True
        if content:
            print(content, end="", flush=True)
            answer_chunks.append(content)

    print()
    if usage is not None:
        print("\nUsage:", usage)

    return {
        "reasoning": "".join(reasoning_chunks),
        "answer": "".join(answer_chunks),
    }


def _try_parse_json(answer: str) -> dict | None:
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


def check_public_image(client: OpenAI, model: str) -> None:
    print("\n" + "#" * 60)
    print("# Check 1: public URL image (Alibaba example)")
    print("#" * 60)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_"
                            "!!6000000002727-0-tps-1024-406.jpg"
                        )
                    },
                },
                {"type": "text", "text": "Briefly describe what you see in this image."},
            ],
        }
    ]
    _stream_response(
        client,
        model=model,
        messages=messages,
        enable_thinking=False,
        thinking_budget=0,
    )


def check_local_frame(
    client: OpenAI,
    model: str,
    video_path: Path,
    instruction: str,
    enable_thinking: bool,
    thinking_budget: int,
) -> None:
    print("\n" + "#" * 60)
    print("# Check 2: real RoboTwin first frame + augmented-instruction prompt")
    print("#" * 60)
    print(f"video: {video_path}")
    print(f"raw instruction: {instruction}")

    frame = _extract_first_frame(video_path)
    print(f"frame shape: {frame.shape} dtype={frame.dtype}")
    data_url = _encode_rgb_as_data_url(frame)
    print(f"data url length (chars): {len(data_url)}")

    user_text = f"Original instruction: {instruction}\nTask name: smoke_test"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    result = _stream_response(
        client,
        model=model,
        messages=messages,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )

    parsed = _try_parse_json(result["answer"])
    print("\n" + "-" * 20 + " parsed JSON " + "-" * 20)
    if parsed is None:
        print("WARNING: failed to parse JSON from model answer; raw printed above.")
    else:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        aug = parsed.get("augmented_instruction", "")
        print(f"\naugmented_instruction length: {len(aug)} chars")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument(
        "--video",
        default=str(Path(__file__).resolve().parents[1] / DEFAULT_VIDEO),
        help="Local mp4 to pull the first frame from.",
    )
    parser.add_argument("--instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking.")
    parser.add_argument("--thinking_budget", type=int, default=8192)
    parser.add_argument(
        "--skip_check_1",
        action="store_true",
        help="Skip the public URL image check (use if outbound to img.alicdn is blocked).",
    )
    args = parser.parse_args()

    client = _make_client()

    if not args.skip_check_1:
        try:
            check_public_image(client, model=args.model)
        except Exception as exc:
            print(f"\nCheck 1 failed: {exc!r}")
            print("Continuing to check 2...")

    check_local_frame(
        client,
        model=args.model,
        video_path=Path(args.video),
        instruction=args.instruction,
        enable_thinking=args.thinking,
        thinking_budget=args.thinking_budget,
    )

    print("\nSmoke test finished.")


if __name__ == "__main__":
    main()
