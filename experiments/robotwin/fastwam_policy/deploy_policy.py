import json
import logging
import os
import sys
import time
import inspect
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

POLICY_DIR = Path(__file__).resolve().parent
if str(POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(POLICY_DIR))
from vlm_planner import ClosedLoopVLMPlanner, VLMPlanner  # noqa: E402

VALID_VLM_MODES = {"off", "open_loop", "closed_loop"}
DEFAULT_SUBTASK_MENUS_PATH = POLICY_DIR / "subtask_menus.json"

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        vlm_mode: str = "off",
        vlm_planner: Optional["VLMPlanner"] = None,
        closed_loop_planner: Optional["ClosedLoopVLMPlanner"] = None,
        vlm_replan_every_k_chunks: int = 3,
        episode_trace_path: Optional[Path] = None,
        task_name: Optional[str] = None,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        mode = str(vlm_mode).lower()
        if mode not in VALID_VLM_MODES:
            raise ValueError(
                f"vlm_mode must be one of {sorted(VALID_VLM_MODES)}, got: {vlm_mode!r}"
            )
        self.vlm_mode = mode
        self.use_vlm_planner = self.vlm_mode != "off"
        self.vlm_planner = vlm_planner if self.vlm_mode == "open_loop" else None
        self.closed_loop_planner = (
            closed_loop_planner if self.vlm_mode == "closed_loop" else None
        )
        self.vlm_replan_every_k_chunks = int(max(1, vlm_replan_every_k_chunks))
        self.task_name = str(task_name) if task_name is not None else "unknown_task"
        self.current_instruction: Optional[str] = None

        if self.vlm_mode == "open_loop" and self.vlm_planner is None:
            raise ValueError(
                "vlm_mode='open_loop' but no vlm_planner was provided. "
                "Construct one via VLMPlanner(...) in get_model()."
            )
        if self.vlm_mode == "closed_loop" and self.closed_loop_planner is None:
            raise ValueError(
                "vlm_mode='closed_loop' but no closed_loop_planner was provided. "
                "Construct one via ClosedLoopVLMPlanner(...) in get_model()."
            )

        # Closed-loop per-episode state. Initialized at construction and
        # reset between episodes via reset(); the persisted trace is flushed
        # to disk via _flush_episode_trace().
        self.chunk_index: int = 0
        self.previous_subtask_index: Optional[int] = None
        self.previous_subtask_string: Optional[str] = None
        self.subtask_trace: List[Dict[str, Any]] = []
        self.episode_vlm_call_count: int = 0
        self.episode_subtask_index_oor_count: int = 0
        self.episode_trace_path: Optional[Path] = (
            Path(episode_trace_path) if episode_trace_path is not None else None
        )
        self._trace_lock = threading.Lock()

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | "
            "horizon=%d | replan=%d | vlm_mode=%s | K=%d | "
            "task_name=%s | trace_path=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.vlm_mode,
            self.vlm_replan_every_k_chunks,
            self.task_name,
            str(self.episode_trace_path) if self.episode_trace_path else "<none>",
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def _invoke_open_loop_vlm(
        self, observation: Dict[str, Any], raw_instruction: str
    ) -> None:
        logger.info(
            "[VLM-OL] episode=%d task=%s raw_instruction=%r",
            self.episode_count,
            self.task_name,
            raw_instruction,
        )
        vlm_result = self.vlm_planner.augment_instruction(
            observation=observation,
            raw_instruction=raw_instruction,
            task_name=self.task_name,
            episode_seed=self.episode_count,
        )
        self.current_instruction = vlm_result["augmented_instruction"]
        logger.info(
            "[VLM-OL] episode=%d augmented_instruction=%r used_fallback=%s error=%s",
            self.episode_count,
            self.current_instruction,
            vlm_result.get("used_fallback"),
            vlm_result.get("error"),
        )

    def _invoke_closed_loop_vlm(
        self, observation: Dict[str, Any], raw_instruction: str
    ) -> None:
        planner = self.closed_loop_planner
        assert planner is not None

        menu_stages: List[str] = (
            planner.get_menu(self.task_name)
            if planner.subtask_mode == "menu"
            else []
        )
        n_stages = len(menu_stages)

        previous: Optional[Dict[str, Any]] = None
        if self.previous_subtask_index is not None or self.previous_subtask_string is not None:
            previous = {
                "subtask_index": self.previous_subtask_index,
                "subtask_string": self.previous_subtask_string,
            }

        result = planner.decide_next_subtask(
            observation=observation,
            original_instruction=raw_instruction,
            task_name=self.task_name,
            episode_seed=self.episode_count,
            chunk_index=self.chunk_index,
            previous=previous,
        )
        self.episode_vlm_call_count += 1
        if result.get("error") == "subtask_index_oor":
            self.episode_subtask_index_oor_count += 1

        subtask_index = result.get("subtask_index")
        enriched: str = str(result.get("enriched_instruction") or "").strip()

        # Resolve the canonical subtask string from the menu (for trace logging).
        resolved_subtask_string: Optional[str] = None
        if planner.subtask_mode == "menu" and isinstance(subtask_index, int):
            if 1 <= subtask_index <= n_stages:
                resolved_subtask_string = menu_stages[subtask_index - 1]
        else:
            resolved_subtask_string = enriched or raw_instruction

        if not enriched:
            enriched = resolved_subtask_string or raw_instruction

        self.previous_subtask_index = (
            int(subtask_index) if isinstance(subtask_index, int) else None
        )
        self.previous_subtask_string = resolved_subtask_string
        self.current_instruction = enriched

        trace_entry = {
            "chunk_index": int(self.chunk_index),
            "subtask_index": (
                int(subtask_index) if isinstance(subtask_index, int) else None
            ),
            "subtask_string": resolved_subtask_string,
            "enriched_instruction": enriched,
            "reason": str(result.get("reason", "")),
            "raw_answer": str(result.get("raw_answer", "")),
            "latency_s": float(result.get("latency_s", 0.0)),
            "used_fallback": bool(result.get("used_fallback", False)),
            "error": result.get("error"),
        }
        self.subtask_trace.append(trace_entry)
        self._flush_episode_trace()

        logger.info(
            "[VLM-CL] episode=%d chunk=%d subtask_index=%s aug=%r",
            self.episode_count,
            self.chunk_index,
            self.previous_subtask_index,
            enriched,
        )

    def _flush_episode_trace(self) -> None:
        if self.episode_trace_path is None:
            return
        episode_payload = {
            "task_name": self.task_name,
            "episode_seed": int(self.episode_count),
            "vlm_mode": self.vlm_mode,
            "subtask_mode": (
                self.closed_loop_planner.subtask_mode
                if self.closed_loop_planner is not None
                else None
            ),
            "replan_every_k_chunks": int(self.vlm_replan_every_k_chunks),
            "n_vlm_calls": int(self.episode_vlm_call_count),
            "n_subtask_index_oor": int(self.episode_subtask_index_oor_count),
            "subtask_trace": list(self.subtask_trace),
        }
        key = f"{self.task_name}|seed={self.episode_count}|mode={self.vlm_mode}"

        with self._trace_lock:
            data: Dict[str, Any] = {}
            try:
                if self.episode_trace_path.exists():
                    with self.episode_trace_path.open("r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        data = loaded
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to read episode trace %s: %r",
                    self.episode_trace_path,
                    exc,
                )
                data = {}
            data[key] = episode_payload
            try:
                self.episode_trace_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.episode_trace_path.with_suffix(
                    self.episode_trace_path.suffix + ".tmp"
                )
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.episode_trace_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to write episode trace %s: %r",
                    self.episode_trace_path,
                    exc,
                )

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            raw_instruction = task_env.get_instruction()

            if self.vlm_mode == "open_loop":
                if self.current_instruction is None and self.vlm_planner is not None:
                    self._invoke_open_loop_vlm(observation, raw_instruction)
            elif self.vlm_mode == "closed_loop":
                if (
                    self.closed_loop_planner is not None
                    and (self.chunk_index % self.vlm_replan_every_k_chunks) == 0
                ):
                    self._invoke_closed_loop_vlm(observation, raw_instruction)

            effective_instruction = self.current_instruction or raw_instruction
            self._fill_action_queue(observation=observation, instruction=effective_instruction)
            self.chunk_index += 1

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.current_instruction = None
        self.reset_timing_rollout()

        self.chunk_index = 0
        self.previous_subtask_index = None
        self.previous_subtask_string = None
        self.subtask_trace = []
        self.episode_vlm_call_count = 0
        self.episode_subtask_index_oor_count = 0


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    # vlm_mode is the primary control; use_vlm_planner is a back-compat alias
    # (True -> "open_loop", False -> "off"). vlm_mode wins if both set.
    vlm_mode_raw = usr_args.get("vlm_mode")
    if _is_none_like(vlm_mode_raw):
        vlm_mode_raw = cfg.EVALUATION.get("vlm_mode")
    if _is_none_like(vlm_mode_raw):
        legacy_use_vlm = _parse_bool(
            usr_args.get(
                "use_vlm_planner",
                cfg.EVALUATION.get("use_vlm_planner", False),
            )
        )
        vlm_mode = "open_loop" if legacy_use_vlm else "off"
    else:
        vlm_mode = str(vlm_mode_raw).strip().lower()
    if vlm_mode not in VALID_VLM_MODES:
        raise ValueError(
            f"vlm_mode must be one of {sorted(VALID_VLM_MODES)}, got: {vlm_mode!r}"
        )

    task_name = usr_args.get("task_name")
    if _is_none_like(task_name):
        task_name = cfg.EVALUATION.get("task_name")
    task_name_str = str(task_name) if not _is_none_like(task_name) else "unknown_task"

    vlm_planner: Optional[VLMPlanner] = None
    closed_loop_planner: Optional[ClosedLoopVLMPlanner] = None
    episode_trace_path: Optional[Path] = None

    if vlm_mode != "off":
        vlm_model = str(
            usr_args.get("vlm_model", cfg.EVALUATION.get("vlm_model", "qwen3-vl-plus"))
        )
        vlm_base_url = str(
            usr_args.get(
                "vlm_base_url",
                cfg.EVALUATION.get(
                    "vlm_base_url",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )
        )
        vlm_thinking_budget = int(
            usr_args.get(
                "vlm_thinking_budget", cfg.EVALUATION.get("vlm_thinking_budget", 8192)
            )
        )
        vlm_max_chars = int(
            usr_args.get("vlm_max_chars", cfg.EVALUATION.get("vlm_max_chars", 220))
        )
        vlm_enable_thinking = _parse_bool(
            usr_args.get(
                "vlm_enable_thinking", cfg.EVALUATION.get("vlm_enable_thinking", True)
            )
        )
        eval_output_dir = usr_args.get("eval_output_dir")
        eval_output_dir_path: Optional[Path] = None
        if not _is_none_like(eval_output_dir):
            eval_output_dir_path = (
                Path(str(eval_output_dir)).expanduser().resolve()
            )
            episode_trace_path = eval_output_dir_path / "vlm_episode_traces.json"

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                f"vlm_mode={vlm_mode!r} requires DASHSCOPE_API_KEY in the "
                "environment. Run `export DASHSCOPE_API_KEY=...` before "
                "starting eval."
            )

        if vlm_mode == "open_loop":
            cache_path: Optional[Path] = None
            if eval_output_dir_path is not None:
                cache_path = eval_output_dir_path / "vlm_cache.json"
            vlm_planner = VLMPlanner(
                model_name=vlm_model,
                base_url=vlm_base_url,
                api_key=api_key,
                thinking_budget=vlm_thinking_budget,
                max_chars=vlm_max_chars,
                cache_path=cache_path,
                enable_thinking=vlm_enable_thinking,
            )
        elif vlm_mode == "closed_loop":
            subtask_mode_raw = usr_args.get(
                "vlm_subtask_mode",
                cfg.EVALUATION.get("vlm_subtask_mode", "menu"),
            )
            subtask_mode = str(subtask_mode_raw).strip().lower()
            if subtask_mode not in {"menu", "free_form"}:
                raise ValueError(
                    "vlm_subtask_mode must be 'menu' or 'free_form', got: "
                    f"{subtask_mode_raw!r}"
                )
            subtask_menus_path_raw = usr_args.get(
                "vlm_subtask_menu_path",
                cfg.EVALUATION.get("vlm_subtask_menu_path"),
            )
            if _is_none_like(subtask_menus_path_raw):
                subtask_menus_path = DEFAULT_SUBTASK_MENUS_PATH
            else:
                subtask_menus_path = Path(str(subtask_menus_path_raw)).expanduser().resolve()
            cl_cache_path: Optional[Path] = None
            if eval_output_dir_path is not None:
                cl_cache_path = eval_output_dir_path / "vlm_closed_loop_cache.json"
            closed_loop_planner = ClosedLoopVLMPlanner(
                model_name=vlm_model,
                base_url=vlm_base_url,
                api_key=api_key,
                thinking_budget=vlm_thinking_budget,
                max_chars=vlm_max_chars,
                cache_path=cl_cache_path,
                subtask_menus_path=(
                    subtask_menus_path if subtask_mode == "menu" else None
                ),
                subtask_mode=subtask_mode,
                enable_thinking=vlm_enable_thinking,
            )

    vlm_replan_every_k_chunks = int(
        usr_args.get(
            "vlm_replan_every_k_chunks",
            cfg.EVALUATION.get("vlm_replan_every_k_chunks", 3),
        )
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        vlm_mode=vlm_mode,
        vlm_planner=vlm_planner,
        closed_loop_planner=closed_loop_planner,
        vlm_replan_every_k_chunks=vlm_replan_every_k_chunks,
        episode_trace_path=episode_trace_path,
        task_name=task_name_str,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
