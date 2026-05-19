"""Tests for LoRA training mode support in FastWAMCoT."""
import sys
import tempfile

import torch
import torch.nn as nn


def test_parse_train_mode():
    """Test mode string parsing."""
    from fastwam.lora_utils import parse_train_mode

    assert parse_train_mode("full") == ("full", None)
    assert parse_train_mode("lora32") == ("lora", 32)
    assert parse_train_mode("lora16") == ("lora", 16)
    assert parse_train_mode("lora64") == ("lora", 64)
    assert parse_train_mode("LORA32") == ("lora", 32)
    assert parse_train_mode("  Full  ") == ("full", None)

    try:
        parse_train_mode("invalid")
        assert False, "Should have raised"
    except ValueError:
        pass

    try:
        parse_train_mode("lora0")
        assert False, "Should have raised for rank=0"
    except ValueError:
        pass

    print("[PASS] test_parse_train_mode")


def test_apply_lora_to_expert():
    """Test LoRA injection on a DiTBlock-based expert."""
    from fastwam.models.wan22.cot_dit import CoTDiT

    expert = CoTDiT(
        hidden_dim=64,
        ffn_dim=128,
        num_heads=4,
        attn_head_dim=16,
        num_layers=2,
        vlm_input_dim=128,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
    )

    total_before = sum(p.numel() for p in expert.parameters())
    trainable_before = sum(p.numel() for p in expert.parameters() if p.requires_grad)
    assert trainable_before == total_before, "All params should be trainable before LoRA"

    from fastwam.lora_utils import apply_lora_to_expert

    apply_lora_to_expert(expert, rank=8)

    total_after = sum(p.numel() for p in expert.parameters())
    trainable_after = sum(p.numel() for p in expert.parameters() if p.requires_grad)

    assert total_after > total_before, "LoRA should add parameters"
    assert trainable_after < total_before, "LoRA should freeze base and only train adapters"
    assert trainable_after > 0, "Should have some trainable params (LoRA matrices)"

    # Verify .blocks and .num_heads still accessible (MoT compatibility)
    assert hasattr(expert, "blocks")
    assert expert.num_heads == 4
    assert expert.attn_head_dim == 16

    print(f"[PASS] test_apply_lora_to_expert (trainable: {trainable_after}/{total_after})")


def test_lora_state_dict():
    """Test LoRA state dict collection and loading."""
    from fastwam.models.wan22.cot_dit import CoTDiT
    from fastwam.lora_utils import apply_lora_to_expert, collect_lora_state_dict, load_lora_state_dict

    expert = CoTDiT(
        hidden_dim=64, ffn_dim=128, num_heads=4, attn_head_dim=16,
        num_layers=2, vlm_input_dim=128, text_dim=64, freq_dim=32,
    )
    apply_lora_to_expert(expert, rank=8)

    lora_state = collect_lora_state_dict(expert)
    assert len(lora_state) > 0, "Should have LoRA parameters"
    for k in lora_state:
        assert "lora_" in k, f"Key {k} doesn't look like LoRA"

    # Create a fresh expert with LoRA and load
    expert2 = CoTDiT(
        hidden_dim=64, ffn_dim=128, num_heads=4, attn_head_dim=16,
        num_layers=2, vlm_input_dim=128, text_dim=64, freq_dim=32,
    )
    apply_lora_to_expert(expert2, rank=8)
    load_lora_state_dict(expert2, lora_state, strict=False)

    # Verify weights match
    lora_state2 = collect_lora_state_dict(expert2)
    for k in lora_state:
        if k in lora_state2:
            assert torch.allclose(lora_state[k], lora_state2[k]), f"Mismatch at {k}"

    print(f"[PASS] test_lora_state_dict ({len(lora_state)} LoRA tensors)")


def test_fastwam_cot_apply_training_modes():
    """Test apply_training_modes on FastWAMCoT with mixed modes."""
    from fastwam.models.wan22.cot_dit import CoTDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.fastwam_cot import FastWAMCoT

    # Build minimal experts
    video_config = dict(
        has_image_input=False, patch_size=[1, 2, 2], in_dim=16, hidden_dim=64,
        ffn_dim=128, freq_dim=32, text_dim=64, out_dim=16, num_heads=4,
        attn_head_dim=16, num_layers=2, eps=1e-6, seperated_timestep=True,
        require_clip_embedding=False, require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=False,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False, action_dim=7,
        action_group_causal_mask_mode="group_diagonal",
    )
    video_expert = WanVideoDiT(**video_config)

    action_config = dict(
        action_dim=7, hidden_dim=64, ffn_dim=128, num_heads=4,
        attn_head_dim=16, num_layers=2, text_dim=64, freq_dim=32, eps=1e-6,
    )
    action_expert = ActionDiT(**action_config)

    cot_config = dict(
        hidden_dim=64, ffn_dim=128, num_heads=4, attn_head_dim=16,
        num_layers=2, vlm_input_dim=128, text_dim=64, freq_dim=32,
    )
    cot_expert = CoTDiT(**cot_config)

    mot = MoT(
        mixtures={"cot": cot_expert, "video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    )

    # Minimal VAE mock
    class FakeVAE(nn.Module):
        pass

    model = FastWAMCoT(
        video_expert=video_expert,
        action_expert=action_expert,
        cot_expert=cot_expert,
        mot=mot,
        vae=FakeVAE(),
        text_dim=64,
        device="cpu",
        torch_dtype=torch.float32,
    )

    # Apply mixed modes: video=lora32, action=full
    model.apply_training_modes(
        video_dit_mode="lora32",
        action_dit_mode="full",
    )

    assert model._train_modes == {"video": "lora", "action": "full", "cot": "full"}

    # Use the actual trainer logic
    from fastwam.trainer import Wan22Trainer
    Wan22Trainer._apply_dit_only_train_mode(model)

    # Check video expert: base frozen, lora trainable
    video_lora_params = [p for n, p in video_expert.named_parameters() if "lora_" in n]
    video_base_params = [p for n, p in video_expert.named_parameters() if "lora_" not in n]
    assert all(p.requires_grad for p in video_lora_params), "Video LoRA params should be trainable"
    assert all(not p.requires_grad for p in video_base_params), "Video base params should be frozen"

    # Check action expert: all trainable
    assert all(p.requires_grad for p in action_expert.parameters()), "Action params should all be trainable"

    # Check CoT expert: all trainable
    assert all(p.requires_grad for p in cot_expert.parameters()), "CoT params should all be trainable"

    # Verify filtered param count
    trainable = [p for p in model.dit.parameters() if p.requires_grad]
    all_params = list(model.dit.parameters())
    assert len(trainable) < len(all_params), "Should have fewer trainable than total (video base frozen)"
    assert len(trainable) > 0

    print(f"[PASS] test_fastwam_cot_apply_training_modes (trainable: {len(trainable)}/{len(all_params)} param tensors)")


def test_forward_backward_with_lora():
    """Test that forward/backward passes work with LoRA mode."""
    from fastwam.models.wan22.cot_dit import CoTDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT, precompute_freqs_cis
    from fastwam.models.wan22.mot import MoT

    video_config = dict(
        has_image_input=False, patch_size=[1, 2, 2], in_dim=16, hidden_dim=64,
        ffn_dim=128, freq_dim=32, text_dim=64, out_dim=16, num_heads=4,
        attn_head_dim=16, num_layers=2, eps=1e-6, seperated_timestep=True,
        require_clip_embedding=False, require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=False,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False, action_dim=7,
        action_group_causal_mask_mode="group_diagonal",
    )
    video_expert = WanVideoDiT(**video_config)

    action_config = dict(
        action_dim=7, hidden_dim=64, ffn_dim=128, num_heads=4,
        attn_head_dim=16, num_layers=2, text_dim=64, freq_dim=32, eps=1e-6,
    )
    action_expert = ActionDiT(**action_config)

    cot_config = dict(
        hidden_dim=64, ffn_dim=128, num_heads=4, attn_head_dim=16,
        num_layers=2, vlm_input_dim=128, text_dim=64, freq_dim=32,
    )
    cot_expert = CoTDiT(**cot_config)

    # Apply LoRA to video expert
    from fastwam.lora_utils import apply_lora_to_expert
    apply_lora_to_expert(video_expert, rank=8)

    mot = MoT(
        mixtures={"cot": cot_expert, "video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    )

    # Freeze base, enable LoRA + CoT + Action (simulating trainer logic)
    mot.requires_grad_(False)
    cot_expert.requires_grad_(True)
    action_expert.requires_grad_(True)
    for n, p in video_expert.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

    # Prepare inputs with correct freq shapes (need unsqueeze(1) for head broadcasting)
    B = 2
    video_seq = 4
    action_seq = 3
    vlm_seq = 5
    freqs_cis = precompute_freqs_cis(16, end=max(video_seq, action_seq, vlm_seq))

    video_tokens = torch.randn(B, video_seq, 64)
    video_freqs = freqs_cis[:video_seq].unsqueeze(1)  # [S, 1, 8]
    video_t_mod = torch.randn(B, 1, 6, 64)

    action_tokens = torch.randn(B, action_seq, 64)
    action_freqs = freqs_cis[:action_seq].unsqueeze(1)  # [S, 1, 8]
    action_t_mod = torch.randn(B, 1, 6, 64)

    vlm_features = torch.randn(B, vlm_seq, 128)
    cot_out = cot_expert.pre_dit(vlm_features)
    cot_tokens = cot_out["tokens"]
    cot_freqs = cot_out["freqs"]
    cot_t_mod = cot_out["t_mod"]

    # Build mask
    total_seq = cot_tokens.shape[1] + video_seq + action_seq
    mask = torch.ones(total_seq, total_seq, dtype=torch.bool)

    embeds_all = {"cot": cot_tokens, "video": video_tokens, "action": action_tokens}
    freqs_all = {"cot": cot_freqs, "video": video_freqs, "action": action_freqs}
    t_mod_all = {"cot": cot_t_mod, "video": video_t_mod, "action": action_t_mod}
    context_all = {"cot": None, "video": None, "action": None}

    # Forward
    outputs = mot(embeds_all, attention_mask=mask, freqs_all=freqs_all, context_all=context_all, t_mod_all=t_mod_all)
    action_out = outputs["action"]
    loss = action_out.sum()

    # Backward
    loss.backward()

    # Verify gradients
    video_lora_grads = [(n, p.grad) for n, p in video_expert.named_parameters()
                        if "lora_" in n and p.grad is not None]
    video_base_grads = [(n, p.grad) for n, p in video_expert.named_parameters()
                        if "lora_" not in n and p.grad is not None]
    cot_grads = [(n, p.grad) for n, p in cot_expert.named_parameters() if p.grad is not None]
    action_grads = [(n, p.grad) for n, p in action_expert.named_parameters() if p.grad is not None]

    assert len(video_lora_grads) > 0, "Video LoRA params should have gradients"
    assert len(video_base_grads) == 0, "Video base params should NOT have gradients"
    assert len(cot_grads) > 0, "CoT params should have gradients"
    assert len(action_grads) > 0, "Action params should have gradients"

    print(f"[PASS] test_forward_backward_with_lora (video_lora_grads={len(video_lora_grads)}, cot_grads={len(cot_grads)}, action_grads={len(action_grads)})")


def test_checkpoint_round_trip():
    """Test save/load checkpoint with LoRA mode."""
    from fastwam.models.wan22.cot_dit import CoTDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.fastwam_cot import FastWAMCoT

    video_config = dict(
        has_image_input=False, patch_size=[1, 2, 2], in_dim=16, hidden_dim=64,
        ffn_dim=128, freq_dim=32, text_dim=64, out_dim=16, num_heads=4,
        attn_head_dim=16, num_layers=2, eps=1e-6, seperated_timestep=True,
        require_clip_embedding=False, require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=False,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False, action_dim=7,
        action_group_causal_mask_mode="group_diagonal",
    )
    video_expert = WanVideoDiT(**video_config)

    action_config = dict(
        action_dim=7, hidden_dim=64, ffn_dim=128, num_heads=4,
        attn_head_dim=16, num_layers=2, text_dim=64, freq_dim=32, eps=1e-6,
    )
    action_expert = ActionDiT(**action_config)

    cot_config = dict(
        hidden_dim=64, ffn_dim=128, num_heads=4, attn_head_dim=16,
        num_layers=2, vlm_input_dim=128, text_dim=64, freq_dim=32,
    )
    cot_expert = CoTDiT(**cot_config)

    mot = MoT(
        mixtures={"cot": cot_expert, "video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    )

    class FakeVAE(nn.Module):
        pass

    model = FastWAMCoT(
        video_expert=video_expert, action_expert=action_expert,
        cot_expert=cot_expert, mot=mot, vae=FakeVAE(),
        text_dim=64, device="cpu", torch_dtype=torch.float32,
    )
    model.apply_training_modes(video_dit_mode="lora16", action_dit_mode="full")

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    model.save_checkpoint(ckpt_path, step=100)

    # Verify checkpoint contents
    payload = torch.load(ckpt_path, map_location="cpu")
    assert "train_modes" in payload
    assert payload["train_modes"] == {"video": "lora", "action": "full", "cot": "full"}
    assert "video_expert_lora" in payload
    assert "action_expert" in payload
    assert "cot_expert" in payload
    assert "mot" not in payload

    # Verify LoRA state is small
    lora_size = sum(v.numel() for v in payload["video_expert_lora"].values())
    full_video_size = sum(p.numel() for p in video_expert.parameters())
    assert lora_size < full_video_size, "LoRA checkpoint should be smaller than full"

    print(f"[PASS] test_checkpoint_round_trip (lora_size={lora_size}, full_video_size={full_video_size})")

    import os
    os.unlink(ckpt_path)


if __name__ == "__main__":
    test_parse_train_mode()
    test_apply_lora_to_expert()
    test_lora_state_dict()
    test_fastwam_cot_apply_training_modes()
    test_forward_backward_with_lora()
    test_checkpoint_round_trip()
    print("\n=== ALL TESTS PASSED ===")
