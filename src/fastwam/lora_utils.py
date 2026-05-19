"""LoRA utilities for per-expert training mode control in FastWAMCoT."""
from __future__ import annotations

import re
from typing import Optional

import torch.nn as nn

from .utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_LORA_TARGET_MODULES = [
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
]


def parse_train_mode(mode_str: str) -> tuple[str, int | None]:
    """Parse training mode string.

    Returns:
        ("full", None) for "full"
        ("lora", rank) for "lora{rank}" e.g. "lora32" → ("lora", 32)
    """
    mode_str = mode_str.strip().lower()
    if mode_str == "full":
        return ("full", None)
    match = re.match(r"^lora(\d+)$", mode_str)
    if match:
        rank = int(match.group(1))
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        return ("lora", rank)
    raise ValueError(
        f"Invalid training mode: '{mode_str}'. Expected 'full' or 'lora{{rank}}' (e.g. 'lora32')."
    )


def apply_lora_to_expert(
    expert: nn.Module,
    rank: int,
    target_modules: list[str] | None = None,
    lora_alpha_ratio: float = 1.0,
    lora_dropout: float = 0.0,
) -> None:
    """Inject LoRA adapters into an expert module in-place.

    Uses peft's inject_adapter_in_model to modify the expert without wrapping it,
    preserving the .blocks / .num_heads / .attn_head_dim interface required by MoT.
    """
    from peft import LoraConfig, inject_adapter_in_model

    if target_modules is None:
        target_modules = DEFAULT_LORA_TARGET_MODULES

    lora_alpha = int(rank * lora_alpha_ratio)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    inject_adapter_in_model(lora_config, expert, adapter_name="default")

    num_lora_params = sum(p.numel() for p in expert.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in expert.parameters())
    logger.info(
        "LoRA injected: rank=%d, alpha=%d, targets=%s, trainable=%d/%.1fM (%.2f%% of %.1fM total)",
        rank,
        lora_alpha,
        target_modules,
        num_lora_params,
        num_lora_params / 1e6,
        100.0 * num_lora_params / max(num_total_params, 1),
        num_total_params / 1e6,
    )


def collect_lora_state_dict(expert: nn.Module) -> dict:
    """Extract only LoRA adapter parameters from an expert's state_dict."""
    lora_state = {}
    for name, param in expert.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.data
    return lora_state


def load_lora_state_dict(expert: nn.Module, lora_state: dict, strict: bool = True) -> None:
    """Load LoRA adapter weights into an expert that already has LoRA injected."""
    current_state = expert.state_dict()
    lora_keys_in_model = {k for k in current_state if "lora_" in k}
    lora_keys_in_ckpt = set(lora_state.keys())

    if strict:
        missing = lora_keys_in_model - lora_keys_in_ckpt
        unexpected = lora_keys_in_ckpt - lora_keys_in_model
        if missing:
            raise RuntimeError(f"Missing LoRA keys in checkpoint: {sorted(missing)[:5]}...")
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys in checkpoint: {sorted(unexpected)[:5]}...")

    for key, value in lora_state.items():
        if key in current_state:
            current_state[key] = value
    expert.load_state_dict(current_state, strict=False)
