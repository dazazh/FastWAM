#!/usr/bin/env python3
"""For each VLM-labeled entry in subtask_label_cache.json, ask qwen3-plus for 7 paraphrases.

Reads/writes:
- data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_label_cache.json

Re-runs only call the LLM for entries whose "llm" field is missing or errored.
"""

from __future__ import annotations

import argparse
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

from openai import OpenAI
from tqdm import tqdm

OUT_CACHE = Path("data/robotwin2.0_hangingmug_subtask_lerobot/meta/subtask_label_cache.json")

DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MAX_CHARS = 400
N_PARAPHRASES = 7

LLM_SYSTEM_PROMPT = (
    "You are a paraphrase generator for robot subtask instructions. "
    "You output JSON only. Do not include any text outside the JSON."
)


def _user_prompt(sentence: str) -> str:
    return (
        f'Sentence: "{sentence}"\n\n'
        f"Generate exactly {N_PARAPHRASES} paraphrases of the sentence above. "
        "Each paraphrase MUST:\n"
        "- Preserve the original meaning, the chosen arm, and every visual "
        "descriptor (color, material, shape, position, target).\n"
        "- Vary the wording (verbs, sentence structure, ordering of clauses).\n"
        "- Stay imperative, plain English, no markdown, no quotes.\n"
        f"- Be <= {MAX_CHARS} characters.\n\n"
        f'Return JSON only: {{"paraphrases": [{", ".join("..." for _ in range(N_PARAPHRASES))}]}}.'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM paraphrase generation for hanging_mug subtasks.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true", help="Re-call LLM even for cached successful entries.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many entries.")
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


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


def _call_llm(
    client: OpenAI,
    model: str,
    enable_thinking: bool,
    thinking_budget: int,
    sentence: str,
) -> dict:
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(sentence)},
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

    if not OUT_CACHE.exists():
        raise FileNotFoundError(f"Missing cache file: {OUT_CACHE}. Run the VLM labeling script first.")
    cache: dict[str, dict] = json.loads(OUT_CACHE.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(cache)} entries from {OUT_CACHE}")

    work: list[str] = []
    for key, entry in cache.items():
        vlm = entry.get("vlm") if isinstance(entry, dict) else None
        if not isinstance(vlm, dict) or not vlm.get("subtask_instruction"):
            continue
        llm = entry.get("llm")
        if (
            not args.overwrite
            and isinstance(llm, dict)
            and llm.get("error") is None
            and isinstance(llm.get("paraphrases"), list)
            and len(llm["paraphrases"]) >= N_PARAPHRASES
        ):
            continue
        work.append(key)
    if args.limit is not None:
        work = work[: int(args.limit)]
    print(f"[info] planned LLM calls: {len(work)}")
    if args.dry_run:
        for k in work[:5]:
            print(f"  {k} -> {cache[k]['vlm']['subtask_instruction'][:80]}")
        return

    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=args.request_timeout)
    cache_lock = threading.Lock()
    flush_counter = {"n": 0}
    progress = {"done": 0, "fail": 0, "t0": time.time()}
    pbar = tqdm(total=len(work), desc="llm_calls", unit="call", smoothing=0.05, dynamic_ncols=True)

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

    def _process(key: str) -> None:
        with cache_lock:
            entry = cache[key]
            sentence = entry["vlm"]["subtask_instruction"]
        last_exc: Optional[BaseException] = None
        response: Optional[dict] = None
        t0 = time.perf_counter()
        for attempt in range(1, args.max_retries + 1):
            try:
                response = _call_llm(
                    client,
                    DEFAULT_MODEL,
                    args.enable_thinking,
                    args.thinking_budget,
                    sentence,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt < args.max_retries:
                    time.sleep(args.retry_backoff * attempt)
        latency = time.perf_counter() - t0
        if response is None:
            with cache_lock:
                cache[key]["llm"] = {
                    "paraphrases": [],
                    "raw_answer": "",
                    "latency_s": latency,
                    "usage": None,
                    "error": f"call_failed: {last_exc!r}",
                }
                progress["fail"] += 1
            _bump_pbar(False, latency)
            return

        answer = response["answer"]
        parsed = _try_parse_json(answer)
        paraphrases: list[str] = []
        err: Optional[str] = None
        if parsed is None:
            err = "non_json_answer"
        else:
            raw_list = parsed.get("paraphrases")
            if not isinstance(raw_list, list):
                err = "missing_paraphrases_list"
            else:
                for item in raw_list:
                    s = _truncate(str(item))
                    if s:
                        paraphrases.append(s)
                if len(paraphrases) < N_PARAPHRASES:
                    err = f"got_only_{len(paraphrases)}_paraphrases"

        with cache_lock:
            cache[key]["llm"] = {
                "paraphrases": paraphrases,
                "raw_answer": answer,
                "latency_s": latency,
                "usage": response.get("usage"),
                "error": err,
            }
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
        futures = [pool.submit(_process, k) for k in work]
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
        n_ok = sum(1 for v in cache.values() if isinstance(v.get("llm"), dict) and v["llm"].get("error") is None)
        n_fail = sum(1 for v in cache.values() if isinstance(v.get("llm"), dict) and v["llm"].get("error") is not None)
        print(f"[done] llm entries ok={n_ok} fail={n_fail} (total cache={len(cache)})")
        print(f"[done] cache file: {OUT_CACHE}")


if __name__ == "__main__":
    main()
