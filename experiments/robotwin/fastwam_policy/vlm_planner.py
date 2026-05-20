"""VLM planners that wrap Qwen3-VL-Plus (DashScope OpenAI-compatible endpoint).

This module provides two planners that share the same HTTP / cache / retry
plumbing:

- ``VLMPlanner`` (open-loop): called once per episode on the first replan
  boundary. Rewrites the original instruction into a single short augmented
  instruction that the low-level WAM uses for the whole rollout.
- ``ClosedLoopVLMPlanner`` (closed-loop, menu OR free-form): called at every
  K-th replan boundary. Either picks the next subtask from a per-task menu
  scaffold and returns an enriched instruction, or (in ``free_form`` mode)
  emits a free-form short instruction without menu validation.

Design notes:
- The WAM text encoder is capped at 128 tokens (see configs/data/robotwin.yaml).
  We enforce a hard char cap to guarantee the augmented instruction fits.
- Open-loop cache key: ``f"{task}|seed={seed}"`` (one entry per episode).
- Closed-loop cache key: ``f"{task}|seed={seed}|chunk={chunk_index}|{mode}"``
  (one entry per VLM call). Lets reruns reuse paid API calls when the rollout
  is deterministic.
- Any error inside this module is non-fatal: we always return a safe fallback
  (raw instruction for open-loop, previous subtask for closed-loop) so the
  eval rollout proceeds.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a vision planner for a tabletop dual-arm robot. "
    "You will be given (a) the first camera frame of a RoboTwin episode and "
    "(b) the original task instruction. Look at the frame and rewrite the "
    "instruction into ONE short, concrete English sentence that the low-level "
    "policy can follow. The sentence must:\n"
    "- Preserve the original goal verbatim in meaning.\n"
    "- Add at most one short clause describing where the relevant object(s) "
    "currently are in the scene (e.g. \"the mug is on the left, the rack is "
    "on the right\").\n"
    "- Use simple imperative English, no markdown, no quotation marks.\n"
    "- Stay under 200 characters.\n"
    'Return strictly this JSON: {"augmented_instruction": "...", "reasoning": "..."}.'
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


def _try_parse_json(answer: str) -> Optional[Dict[str, Any]]:
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


def _truncate_to_sentence(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    last_period = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
    if last_period >= int(max_chars * 0.5):
        return head[: last_period + 1].strip()
    return head.strip()


def _extract_head_frame(observation: Dict[str, Any]) -> np.ndarray:
    if "observation" not in observation:
        raise ValueError("Observation missing 'observation' key.")
    obs = observation["observation"]
    if "head_camera" not in obs or "rgb" not in obs["head_camera"]:
        raise ValueError("Observation missing 'observation.head_camera.rgb'.")
    rgb = obs["head_camera"]["rgb"]
    if isinstance(rgb, np.ndarray):
        return rgb
    return np.asarray(rgb)


class VLMPlanner:
    """Calls Qwen3-VL-Plus once per episode and returns an augmented instruction.

    Errors are swallowed and the raw instruction is returned as a fallback so
    eval never aborts because of a flaky VLM call.
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        thinking_budget: int,
        max_chars: int,
        cache_path: Optional[Path],
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        request_timeout_s: float = 60.0,
        enable_thinking: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError(
                "VLMPlanner requires a non-empty api_key. "
                "Set DASHSCOPE_API_KEY in the eval shell."
            )

        # Local import so the rest of the policy module doesn't depend on the
        # openai package when use_vlm_planner is false.
        from openai import OpenAI

        self.model_name = str(model_name)
        self.base_url = str(base_url)
        self.thinking_budget = int(thinking_budget)
        self.max_chars = int(max_chars)
        self.max_retries = int(max(1, max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.request_timeout_s = float(request_timeout_s)
        self.enable_thinking = bool(enable_thinking)

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_s,
        )
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_cache()

        logger.info(
            "Initialized VLMPlanner | model=%s | base_url=%s | cache=%s | "
            "enable_thinking=%s | thinking_budget=%d | max_chars=%d",
            self.model_name,
            self.base_url,
            str(self._cache_path) if self._cache_path else "<none>",
            self.enable_thinking,
            self.thinking_budget,
            self.max_chars,
        )

    def _cache_key(self, task_name: str, episode_seed: int) -> str:
        return f"{task_name}|seed={int(episode_seed)}"

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            with self._cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._cache = {str(k): dict(v) for k, v in data.items()}
                logger.info("Loaded VLM cache: %d entries from %s", len(self._cache), self._cache_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load VLM cache %s: %r", self._cache_path, exc)

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._cache_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write VLM cache %s: %r", self._cache_path, exc)

    def _call_vlm(self, *, data_url: str, raw_instruction: str, task_name: str) -> Dict[str, Any]:
        user_text = (
            f"Original instruction: {raw_instruction}\n"
            f"Task name: {task_name}"
        )
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
        extra_body = {"enable_thinking": self.enable_thinking}
        if self.enable_thinking:
            extra_body["thinking_budget"] = self.thinking_budget

        completion = self._client.chat.completions.create(
            model=self.model_name,
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
        reasoning = "".join(reasoning_chunks)
        usage_dict: Optional[Dict[str, Any]] = None
        if usage is not None:
            try:
                usage_dict = usage.model_dump()  # pydantic v2
            except Exception:  # noqa: BLE001
                try:
                    usage_dict = usage.dict()  # pydantic v1
                except Exception:  # noqa: BLE001
                    usage_dict = None

        return {
            "answer": answer,
            "reasoning": reasoning,
            "usage": usage_dict,
        }

    def _call_with_retries(self, **kwargs) -> Dict[str, Any]:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_vlm(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "VLM call attempt %d/%d failed: %r",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * attempt)
        raise RuntimeError(f"VLM call failed after {self.max_retries} attempts: {last_exc!r}")

    def augment_instruction(
        self,
        *,
        observation: Dict[str, Any],
        raw_instruction: str,
        task_name: str,
        episode_seed: int,
    ) -> Dict[str, Any]:
        cache_key = self._cache_key(task_name, episode_seed)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None and cached.get("raw_instruction") == raw_instruction:
                logger.info(
                    "VLM cache HIT key=%s aug='%s'",
                    cache_key,
                    cached.get("augmented_instruction", ""),
                )
                return dict(cached)

        result: Dict[str, Any] = {
            "augmented_instruction": raw_instruction,
            "raw_instruction": raw_instruction,
            "task_name": task_name,
            "episode_seed": int(episode_seed),
            "reasoning": "",
            "raw_answer": "",
            "usage": None,
            "error": None,
            "used_fallback": False,
        }

        try:
            head_rgb = _extract_head_frame(observation)
            data_url = _encode_rgb_as_data_url(head_rgb)
            t0 = time.perf_counter()
            response = self._call_with_retries(
                data_url=data_url,
                raw_instruction=raw_instruction,
                task_name=task_name,
            )
            latency_s = time.perf_counter() - t0

            answer = response["answer"]
            result["reasoning"] = response["reasoning"]
            result["raw_answer"] = answer
            result["usage"] = response["usage"]
            result["latency_s"] = latency_s

            parsed = _try_parse_json(answer)
            if parsed is None:
                logger.warning(
                    "VLM returned non-JSON answer for task=%s seed=%d; using raw answer as instruction",
                    task_name,
                    episode_seed,
                )
                augmented = answer.strip()
            else:
                augmented = str(parsed.get("augmented_instruction", "")).strip()
                if not augmented:
                    augmented = raw_instruction
                    result["used_fallback"] = True
                    result["error"] = "missing_augmented_instruction_field"

            augmented = re.sub(r"\s+", " ", augmented).strip().strip('"').strip("'")
            if not augmented:
                augmented = raw_instruction
                result["used_fallback"] = True
                result["error"] = result["error"] or "empty_after_clean"
            augmented = _truncate_to_sentence(augmented, self.max_chars)
            result["augmented_instruction"] = augmented

            logger.info(
                "VLM augment_instruction OK task=%s seed=%d latency=%.2fs len=%d aug='%s'",
                task_name,
                episode_seed,
                latency_s,
                len(augmented),
                augmented,
            )
        except Exception as exc:  # noqa: BLE001
            result["used_fallback"] = True
            result["error"] = repr(exc)
            logger.error(
                "VLM augment_instruction FAILED task=%s seed=%d err=%r; falling back to raw instruction",
                task_name,
                episode_seed,
                exc,
            )

        with self._lock:
            self._cache[cache_key] = dict(result)
            self._save_cache()

        return result


CLOSED_LOOP_MENU_SYSTEM_PROMPT = (
    "You are a closed-loop subtask planner for a pretrained robot policy. "
    "You will see (a) the original task instruction, (b) a small fixed MENU "
    "of canonical subtasks for the current task, (c) the previous subtask "
    "the robot was executing, and (d) the current head-camera frame.\n\n"
    "Your job:\n"
    "1. Look at the image and judge what the robot has accomplished so far.\n"
    "2. Pick the most appropriate 'subtask_index' from the menu. Choose the "
    "same index as before if the stage is still in progress, or a higher "
    "index if the stage appears done. You may also step back to an earlier "
    "index if the robot needs to redo a step. You MUST choose one of the "
    "listed 1-based indices.\n"
    "3. Write an 'enriched_instruction' that follows the chosen subtask's "
    "intent. You MAY add ONE short visual-grounding clause from the image "
    "(color, material, 'on the left/right side of the table', 'with the "
    "rounded handle', 'near the top edge', etc.). Match the dataset's "
    "phrasing style: start with an imperative verb, use plain English, no "
    "markdown, no quotes, no reasoning text inside the instruction.\n"
    "4. Keep 'enriched_instruction' under 200 characters.\n\n"
    'Return strictly this JSON only (no other text):\n'
    '{\n'
    '  "subtask_index": <int from 1 to N>,\n'
    '  "enriched_instruction": "<short instruction string>",\n'
    '  "reason": "<one short sentence describing what you see>"\n'
    '}'
)


CLOSED_LOOP_FREE_SYSTEM_PROMPT = (
    "You are a closed-loop instruction refiner for a pretrained robot policy. "
    "You will see (a) the original task instruction, (b) the previous "
    "instruction the robot was executing, and (c) the current head-camera "
    "frame.\n\n"
    "Your job:\n"
    "1. Look at the image and judge what the robot has accomplished so far.\n"
    "2. Write an 'enriched_instruction' that the low-level policy should "
    "execute next. You may keep the previous instruction if the stage is "
    "still in progress, or write a new one if the stage appears done. "
    "Preserve the original goal. You MAY add visual grounding "
    "(color, material, left/right side, handle shape, etc.). Match the "
    "dataset's phrasing style: start with an imperative verb, use plain "
    "English, no markdown, no quotes, no reasoning text inside the "
    "instruction.\n"
    "3. Keep 'enriched_instruction' under 200 characters.\n\n"
    'Return strictly this JSON only (no other text):\n'
    '{\n'
    '  "enriched_instruction": "<short instruction string>",\n'
    '  "reason": "<one short sentence describing what you see>"\n'
    '}'
)


def _format_menu(stages: list[str]) -> str:
    lines = []
    for i, s in enumerate(stages, start=1):
        lines.append(f"  {i}. {s}")
    return "\n".join(lines)


def load_subtask_menus(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the per-task subtask menu JSON file produced by the project.

    Returns a dict ``{task_name: {"summary": str, "stages": list[str]}}``.
    Raises FileNotFoundError if the path is missing; raises ValueError on a
    malformed schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Subtask menu file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError(f"Subtask menu file missing 'tasks' field: {path}")
    tasks = data["tasks"]
    if not isinstance(tasks, dict):
        raise ValueError(f"Subtask menu 'tasks' must be a dict: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for task_name, entry in tasks.items():
        if not isinstance(entry, dict) or "stages" not in entry:
            raise ValueError(
                f"Subtask menu for task '{task_name}' missing 'stages' field."
            )
        stages = entry["stages"]
        if not isinstance(stages, list) or not stages:
            raise ValueError(
                f"Subtask menu for task '{task_name}' must be a non-empty list."
            )
        out[task_name] = {
            "summary": str(entry.get("summary", "")),
            "stages": [str(s).strip() for s in stages],
        }
    return out


class ClosedLoopVLMPlanner:
    """Calls Qwen3-VL-Plus at every K-th replan boundary to choose the next subtask.

    Two subtask modes:
    - ``"menu"``: VLM picks an index from a per-task subtask menu and writes
      an enriched instruction loosely based on that template (visual grounding
      allowed). Only the index is hard-validated; the enriched instruction is
      sanitized and capped.
    - ``"free_form"``: no menu; VLM emits a free-form short instruction. Same
      char cap as open-loop.

    Caching:
    - Key = ``f"{task}|seed={seed}|chunk={chunk_index}|{subtask_mode}"``.
    - Lets reruns reuse paid API calls when the rollout is deterministic.

    Errors are swallowed; the caller is expected to fall back to the previous
    subtask / instruction on `used_fallback=True`.
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        thinking_budget: int,
        max_chars: int,
        cache_path: Optional[Path],
        subtask_menus_path: Optional[Path],
        subtask_mode: str = "menu",
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        request_timeout_s: float = 60.0,
        enable_thinking: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError(
                "ClosedLoopVLMPlanner requires a non-empty api_key. "
                "Set DASHSCOPE_API_KEY in the eval shell."
            )

        subtask_mode = str(subtask_mode).lower()
        if subtask_mode not in {"menu", "free_form"}:
            raise ValueError(
                f"subtask_mode must be 'menu' or 'free_form', got: {subtask_mode!r}"
            )

        from openai import OpenAI

        self.model_name = str(model_name)
        self.base_url = str(base_url)
        self.thinking_budget = int(thinking_budget)
        self.max_chars = int(max_chars)
        self.max_retries = int(max(1, max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.request_timeout_s = float(request_timeout_s)
        self.enable_thinking = bool(enable_thinking)
        self.subtask_mode = subtask_mode

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_s,
        )
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_cache()

        self._menus: Dict[str, Dict[str, Any]] = {}
        if subtask_mode == "menu":
            if subtask_menus_path is None:
                raise ValueError(
                    "subtask_mode='menu' requires a subtask_menus_path."
                )
            self._menus = load_subtask_menus(Path(subtask_menus_path))

        logger.info(
            "Initialized ClosedLoopVLMPlanner | model=%s | mode=%s | "
            "cache=%s | menus=%s | enable_thinking=%s | thinking_budget=%d | "
            "max_chars=%d",
            self.model_name,
            self.subtask_mode,
            str(self._cache_path) if self._cache_path else "<none>",
            list(self._menus.keys()) if self._menus else "<none>",
            self.enable_thinking,
            self.thinking_budget,
            self.max_chars,
        )

    def get_menu(self, task_name: str) -> list[str]:
        if self.subtask_mode != "menu":
            return []
        entry = self._menus.get(str(task_name))
        if entry is None:
            raise KeyError(
                f"No subtask menu defined for task '{task_name}'. "
                f"Known tasks: {sorted(self._menus.keys())}"
            )
        return list(entry["stages"])

    def _cache_key(self, task_name: str, episode_seed: int, chunk_index: int) -> str:
        return (
            f"{task_name}|seed={int(episode_seed)}|chunk={int(chunk_index)}|"
            f"{self.subtask_mode}"
        )

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            with self._cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._cache = {str(k): dict(v) for k, v in data.items()}
                logger.info(
                    "Loaded closed-loop VLM cache: %d entries from %s",
                    len(self._cache),
                    self._cache_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load closed-loop VLM cache %s: %r",
                self._cache_path,
                exc,
            )

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._cache_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to write closed-loop VLM cache %s: %r",
                self._cache_path,
                exc,
            )

    def _build_user_text(
        self,
        *,
        original_instruction: str,
        task_name: str,
        chunk_index: int,
        previous: Optional[Dict[str, Any]],
        menu_stages: list[str],
    ) -> str:
        lines: list[str] = []
        lines.append(f"Original task instruction: {original_instruction}")
        lines.append(f"Task name: {task_name}")
        if self.subtask_mode == "menu":
            lines.append("")
            lines.append("Subtask menu (1-based indices, choose exactly one):")
            lines.append(_format_menu(menu_stages))
        lines.append("")
        if previous is None:
            lines.append("Previous state: (this is the first VLM decision for this episode)")
        else:
            lines.append("Previous state:")
            if self.subtask_mode == "menu":
                lines.append(
                    f"  - previous subtask_index: {previous.get('subtask_index')}"
                )
                lines.append(
                    f"  - previous subtask_string: {previous.get('subtask_string')}"
                )
            else:
                lines.append(
                    f"  - previous instruction: {previous.get('subtask_string')}"
                )
        lines.append(f"  - current chunk index (0-based): {int(chunk_index)}")
        return "\n".join(lines)

    def _call_vlm(
        self,
        *,
        data_url: str,
        original_instruction: str,
        task_name: str,
        chunk_index: int,
        previous: Optional[Dict[str, Any]],
        menu_stages: list[str],
    ) -> Dict[str, Any]:
        system_prompt = (
            CLOSED_LOOP_MENU_SYSTEM_PROMPT
            if self.subtask_mode == "menu"
            else CLOSED_LOOP_FREE_SYSTEM_PROMPT
        )
        user_text = self._build_user_text(
            original_instruction=original_instruction,
            task_name=task_name,
            chunk_index=chunk_index,
            previous=previous,
            menu_stages=menu_stages,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        extra_body = {"enable_thinking": self.enable_thinking}
        if self.enable_thinking:
            extra_body["thinking_budget"] = self.thinking_budget

        completion = self._client.chat.completions.create(
            model=self.model_name,
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
        reasoning = "".join(reasoning_chunks)
        usage_dict: Optional[Dict[str, Any]] = None
        if usage is not None:
            try:
                usage_dict = usage.model_dump()
            except Exception:  # noqa: BLE001
                try:
                    usage_dict = usage.dict()
                except Exception:  # noqa: BLE001
                    usage_dict = None

        return {
            "answer": answer,
            "reasoning": reasoning,
            "usage": usage_dict,
        }

    def _call_with_retries(self, **kwargs) -> Dict[str, Any]:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_vlm(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Closed-loop VLM call attempt %d/%d failed: %r",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * attempt)
        raise RuntimeError(
            f"Closed-loop VLM call failed after {self.max_retries} attempts: {last_exc!r}"
        )

    def decide_next_subtask(
        self,
        *,
        observation: Dict[str, Any],
        original_instruction: str,
        task_name: str,
        episode_seed: int,
        chunk_index: int,
        previous: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run one VLM decision for the current chunk boundary.

        Returns a dict with at least:
        - ``subtask_index``: int (1-based) when ``subtask_mode == "menu"``,
          else None
        - ``enriched_instruction``: sanitized instruction string
        - ``reason``: short string describing what the VLM observed
        - ``raw_answer``: VLM raw answer text
        - ``reasoning``: VLM reasoning content
        - ``latency_s``: float seconds
        - ``used_fallback``: True iff the call errored and the result is a
          safe fallback (caller should keep the previous subtask).
        - ``error``: error repr or None
        """
        menu_stages: list[str] = []
        if self.subtask_mode == "menu":
            menu_stages = self.get_menu(task_name)
        n_stages = len(menu_stages)

        cache_key = self._cache_key(task_name, episode_seed, chunk_index)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None and cached.get("original_instruction") == original_instruction:
                logger.info(
                    "Closed-loop VLM cache HIT key=%s subtask_index=%s",
                    cache_key,
                    cached.get("subtask_index"),
                )
                return dict(cached)

        prev_subtask_index = previous.get("subtask_index") if previous else None
        prev_subtask_string = previous.get("subtask_string") if previous else None
        fallback_subtask_index = (
            prev_subtask_index
            if isinstance(prev_subtask_index, int) and 1 <= prev_subtask_index <= max(1, n_stages)
            else 1
        )
        fallback_subtask_string = (
            prev_subtask_string
            if isinstance(prev_subtask_string, str) and prev_subtask_string
            else (menu_stages[fallback_subtask_index - 1] if menu_stages else original_instruction)
        )

        result: Dict[str, Any] = {
            "cache_key": cache_key,
            "task_name": task_name,
            "episode_seed": int(episode_seed),
            "chunk_index": int(chunk_index),
            "subtask_mode": self.subtask_mode,
            "original_instruction": original_instruction,
            "previous": dict(previous) if previous else None,
            "menu_stages": list(menu_stages),
            "subtask_index": fallback_subtask_index if self.subtask_mode == "menu" else None,
            "enriched_instruction": fallback_subtask_string,
            "reason": "",
            "raw_answer": "",
            "reasoning": "",
            "usage": None,
            "latency_s": 0.0,
            "used_fallback": False,
            "error": None,
        }

        try:
            head_rgb = _extract_head_frame(observation)
            data_url = _encode_rgb_as_data_url(head_rgb)
            t0 = time.perf_counter()
            response = self._call_with_retries(
                data_url=data_url,
                original_instruction=original_instruction,
                task_name=task_name,
                chunk_index=chunk_index,
                previous=previous,
                menu_stages=menu_stages,
            )
            latency_s = time.perf_counter() - t0
            result["latency_s"] = latency_s
            result["reasoning"] = response["reasoning"]
            result["raw_answer"] = response["answer"]
            result["usage"] = response["usage"]

            parsed = _try_parse_json(response["answer"])
            if parsed is None:
                logger.warning(
                    "Closed-loop VLM returned non-JSON answer for key=%s; "
                    "using fallback subtask=%s instruction=%r",
                    cache_key,
                    fallback_subtask_index,
                    fallback_subtask_string,
                )
                result["used_fallback"] = True
                result["error"] = "non_json_answer"
            else:
                result["reason"] = str(parsed.get("reason", "")).strip()[:200]

                if self.subtask_mode == "menu":
                    raw_idx = parsed.get("subtask_index")
                    try:
                        idx_int = int(raw_idx)
                    except (TypeError, ValueError):
                        idx_int = -1
                    if not (1 <= idx_int <= n_stages):
                        logger.warning(
                            "Closed-loop VLM returned out-of-range subtask_index=%r "
                            "(menu has %d stages) for key=%s; falling back to prev=%d",
                            raw_idx,
                            n_stages,
                            cache_key,
                            fallback_subtask_index,
                        )
                        idx_int = fallback_subtask_index
                        result["used_fallback"] = True
                        result["error"] = "subtask_index_oor"
                    result["subtask_index"] = idx_int

                enriched_raw = parsed.get("enriched_instruction", "")
                enriched = str(enriched_raw).strip()
                enriched = re.sub(r"\s+", " ", enriched).strip().strip('"').strip("'")
                if not enriched:
                    if self.subtask_mode == "menu":
                        enriched = menu_stages[int(result["subtask_index"]) - 1]
                    else:
                        enriched = fallback_subtask_string
                    if not result["error"]:
                        result["error"] = "empty_enriched_instruction"
                    result["used_fallback"] = True
                enriched = _truncate_to_sentence(enriched, self.max_chars)
                result["enriched_instruction"] = enriched

            logger.info(
                "Closed-loop VLM OK key=%s subtask_index=%s latency=%.2fs aug=%r",
                cache_key,
                result["subtask_index"],
                result["latency_s"],
                result["enriched_instruction"],
            )
        except Exception as exc:  # noqa: BLE001
            result["used_fallback"] = True
            result["error"] = repr(exc)
            logger.error(
                "Closed-loop VLM FAILED key=%s err=%r; falling back to "
                "subtask=%s instruction=%r",
                cache_key,
                exc,
                fallback_subtask_index,
                fallback_subtask_string,
            )

        with self._lock:
            self._cache[cache_key] = dict(result)
            self._save_cache()

        return result
