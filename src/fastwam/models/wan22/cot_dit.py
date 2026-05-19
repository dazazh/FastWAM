import re

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

from .wan_video_dit import DiTBlock, precompute_freqs_cis

logger = get_logger(__name__)


class CoTDiT(nn.Module):
    """Chain-of-Thought DiT expert for VLM feature integration via MoT.

    Receives frozen VLM last-hidden-state features, projects them through a
    learnable adapter, and participates in the MoT joint attention alongside
    Video DiT and Action DiT.  No decoder/loss -- output tokens are discarded.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers: int = 30,
        vlm_input_dim: int = 2048,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        vlm_adapter_type: str = "mlp3x_silu",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.vlm_input_dim = vlm_input_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.vlm_adapter = self._build_adapter(vlm_adapter_type, vlm_input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                attn_head_dim=attn_head_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

        self.freqs = precompute_freqs_cis(attn_head_dim, end=2048)

    @staticmethod
    def _build_adapter(adapter_type: str, in_features: int, out_features: int) -> nn.Module:
        if adapter_type == "linear":
            return nn.Linear(in_features, out_features)
        match = re.match(r"^mlp(\d+)x_silu$", adapter_type)
        if match:
            depth = int(match.group(1))
            layers = [nn.Linear(in_features, out_features)]
            for _ in range(1, depth):
                layers.append(nn.SiLU())
                layers.append(nn.Linear(out_features, out_features))
            return nn.Sequential(*layers)
        raise ValueError(f"Unknown vlm_adapter_type: {adapter_type}")

    def pre_dit(self, vlm_features: torch.Tensor) -> dict:
        """Prepare CoT tokens for MoT forward.

        Args:
            vlm_features: [B, seq_len, vlm_input_dim] from frozen VLM last hidden layer.

        Returns:
            dict with keys: tokens, freqs, t_mod (zero -- no timestep dependency).
        """
        tokens = self.vlm_adapter(vlm_features)
        seq_len = tokens.shape[1]
        freqs = self.freqs[:seq_len].unsqueeze(1).to(tokens.device)
        t_mod = torch.zeros(
            tokens.shape[0], 6, self.hidden_dim,
            device=tokens.device, dtype=tokens.dtype,
        )
        return {"tokens": tokens, "freqs": freqs, "t_mod": t_mod}
