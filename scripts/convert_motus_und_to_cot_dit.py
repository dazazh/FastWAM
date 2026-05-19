"""Convert Motus Und Expert weights to CoT DiT format.

Usage:
    python scripts/convert_motus_und_to_cot_dit.py \
        --motus_ckpt checkpoints/Motus_pretrain/mp_rank_00_model_states.pt \
        --output checkpoints/cot_dit_from_motus.pt
"""
import argparse

import torch


def convert_und_qkv_to_separate_linear(qkv_tensor: torch.Tensor):
    """Convert fused qkv [3, N, D, E] to separate q/k/v Linear weights [3072, 512].

    Motus einsum: "BTD,KNDE->KBTNE"
    - qkv[q_idx, num_heads, hidden_dim, head_dim]
    - Equivalent nn.Linear weight: qkv[q_idx].permute(0, 2, 1).reshape(out_dim, in_dim)
    """
    assert qkv_tensor.ndim == 4 and qkv_tensor.shape[0] == 3
    num_heads, hidden_dim, head_dim = qkv_tensor.shape[1], qkv_tensor.shape[2], qkv_tensor.shape[3]
    out_dim = num_heads * head_dim

    q_weight = qkv_tensor[0].permute(0, 2, 1).reshape(out_dim, hidden_dim)
    k_weight = qkv_tensor[1].permute(0, 2, 1).reshape(out_dim, hidden_dim)
    v_weight = qkv_tensor[2].permute(0, 2, 1).reshape(out_dim, hidden_dim)
    return q_weight, k_weight, v_weight


def convert_motus_und_to_cot_dit(motus_state: dict) -> dict:
    """Convert Motus und_expert state_dict to CoT DiT state_dict.

    Mapping:
        vlm_adapter.{0,2,4}.{weight,bias}  -> vlm_adapter.{0,2,4}.{weight,bias}  (direct)
        blocks.N.wan_und_qkv               -> blocks.N.self_attn.{q,k,v}.weight   (reshape)
        blocks.N.wan_und_o.weight           -> blocks.N.self_attn.o.weight         (direct)
        blocks.N.wan_und_norm_q.weight      -> blocks.N.self_attn.norm_q.weight    (direct)
        blocks.N.wan_und_norm_k.weight      -> blocks.N.self_attn.norm_k.weight    (direct)
        blocks.N.ffn.{0,2}.{weight,bias}   -> blocks.N.ffn.{0,2}.{weight,bias}    (direct)

    Not transferred (remain randomly initialized in CoT DiT):
        blocks.N.cross_attn.*   (never triggered, context=None)
        blocks.N.modulation     (zero t_mod + learned bias)
        blocks.N.norm3          (no counterpart in Motus)
        blocks.N.self_attn.{q,k,v,o}.bias  (Motus has no bias)
    """
    und_keys = {k: v for k, v in motus_state.items() if k.startswith("und_expert.")}
    if not und_keys:
        raise ValueError("No 'und_expert.*' keys found in checkpoint.")

    cot_state = {}
    transferred = 0
    skipped = 0

    for key, value in und_keys.items():
        # Strip "und_expert." prefix
        local_key = key[len("und_expert."):]

        # vlm_adapter: direct copy
        if local_key.startswith("vlm_adapter."):
            cot_state[local_key] = value.clone()
            transferred += 1
            continue

        # FFN: direct copy
        if ".ffn." in local_key:
            cot_state[local_key] = value.clone()
            transferred += 1
            continue

        # QKV: reshape from fused [3,N,D,E] to separate Linear weights
        if local_key.endswith(".wan_und_qkv"):
            block_prefix = local_key.replace(".wan_und_qkv", "")
            q_w, k_w, v_w = convert_und_qkv_to_separate_linear(value)
            cot_state[f"{block_prefix}.self_attn.q.weight"] = q_w
            cot_state[f"{block_prefix}.self_attn.k.weight"] = k_w
            cot_state[f"{block_prefix}.self_attn.v.weight"] = v_w
            transferred += 3
            continue

        # Output projection: direct copy
        if local_key.endswith(".wan_und_o.weight"):
            block_prefix = local_key.replace(".wan_und_o.weight", "")
            cot_state[f"{block_prefix}.self_attn.o.weight"] = value.clone()
            transferred += 1
            continue

        # RMSNorm Q/K: direct copy
        if local_key.endswith(".wan_und_norm_q.weight"):
            block_prefix = local_key.replace(".wan_und_norm_q.weight", "")
            cot_state[f"{block_prefix}.self_attn.norm_q.weight"] = value.clone()
            transferred += 1
            continue
        if local_key.endswith(".wan_und_norm_k.weight"):
            block_prefix = local_key.replace(".wan_und_norm_k.weight", "")
            cot_state[f"{block_prefix}.self_attn.norm_k.weight"] = value.clone()
            transferred += 1
            continue

        skipped += 1

    print(f"Conversion complete: {transferred} tensors transferred, {skipped} skipped")
    return cot_state


def main():
    parser = argparse.ArgumentParser(description="Convert Motus Und Expert to CoT DiT weights")
    parser.add_argument("--motus_ckpt", type=str, required=True,
                        help="Path to Motus checkpoint (mp_rank_00_model_states.pt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for converted CoT DiT weights")
    parser.add_argument("--verify", action="store_true",
                        help="Verify loading into a CoT DiT model after conversion")
    args = parser.parse_args()

    print(f"Loading Motus checkpoint: {args.motus_ckpt}")
    ckpt = torch.load(args.motus_ckpt, map_location="cpu")

    if "module" in ckpt:
        motus_state = ckpt["module"]
    elif "model" in ckpt:
        motus_state = ckpt["model"]
    else:
        motus_state = ckpt

    cot_state = convert_motus_und_to_cot_dit(motus_state)
    torch.save(cot_state, args.output)
    print(f"Saved converted weights to: {args.output}")

    if args.verify:
        from fastwam.models.wan22.cot_dit import CoTDiT
        cot = CoTDiT(
            hidden_dim=512, ffn_dim=2048, num_heads=24, attn_head_dim=128,
            num_layers=30, vlm_input_dim=2048, text_dim=4096, freq_dim=256,
        )
        missing, unexpected = cot.load_state_dict(cot_state, strict=False)
        print(f"\nVerification against CoTDiT(512, 30 layers):")
        print(f"  Loaded: {len(cot_state)} tensors")
        print(f"  Missing in checkpoint (will be random init): {len(missing)}")
        print(f"  Unexpected in checkpoint: {len(unexpected)}")
        if missing:
            patterns = set()
            import re
            for k in missing:
                patterns.add(re.sub(r"blocks\.\d+", "blocks.N", k))
            print(f"  Missing patterns: {sorted(patterns)}")


if __name__ == "__main__":
    main()
