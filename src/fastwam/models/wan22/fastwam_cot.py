"""FastWAMCoT: FastWAM with Chain-of-Thought DiT expert."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .cot_dit import CoTDiT
from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .vlm_feature_extractor import VLMFeatureExtractor

logger = get_logger(__name__)


class FastWAMCoT(FastWAM):
    """FastWAM extended with a CoT DiT expert that receives VLM features."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        cot_expert: CoTDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        vlm_extractor: Optional[VLMFeatureExtractor] = None,
        vlm_extract_mode: str = "precomputed",
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        init_mode: str = "default",
        fastwam_checkpoint_path: Optional[str] = None,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        self.cot_expert = cot_expert
        self.vlm_extractor = vlm_extractor
        self.vlm_extract_mode = vlm_extract_mode
        self._train_modes: dict[str, str] | None = None

        # Store initialization information for checkpoint saving
        self.init_mode = init_mode
        self.fastwam_checkpoint_path = fastwam_checkpoint_path

    def apply_training_modes(
        self,
        video_dit_mode: str = "full",
        action_dit_mode: str = "full",
        lora_target_modules: list[str] | None = None,
        lora_alpha_ratio: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        """Apply LoRA or full training configuration to each expert.

        Must be called AFTER model construction and weight loading,
        BEFORE optimizer construction. CoT DiT is always full training.
        """
        from fastwam.lora_utils import apply_lora_to_expert, parse_train_mode

        video_type, video_rank = parse_train_mode(video_dit_mode)
        if video_type == "lora":
            apply_lora_to_expert(
                self.video_expert,
                rank=video_rank,
                target_modules=lora_target_modules,
                lora_alpha_ratio=lora_alpha_ratio,
                lora_dropout=lora_dropout,
            )
            logger.info("Video DiT: LoRA mode (rank=%d)", video_rank)
        else:
            logger.info("Video DiT: full training")

        action_type, action_rank = parse_train_mode(action_dit_mode)
        if action_type == "lora":
            apply_lora_to_expert(
                self.action_expert,
                rank=action_rank,
                target_modules=lora_target_modules,
                lora_alpha_ratio=lora_alpha_ratio,
                lora_dropout=lora_dropout,
            )
            logger.info("Action DiT: LoRA mode (rank=%d)", action_rank)
        else:
            logger.info("Action DiT: full training")

        logger.info("CoT DiT: full training (always)")
        self._train_modes = {"video": video_type, "action": action_type, "cot": "full"}

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        cot_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        cot_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        vlm_config: dict[str, Any] | None = None,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        init_mode: str = "wan_backbone",
        fastwam_checkpoint_path: str | None = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required.")
        if cot_dit_config is None:
            raise ValueError("`cot_dit_config` is required for FastWAMCoT.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        cot_expert = CoTDiT(**cot_dit_config).to(device=device, dtype=torch_dtype)

        if cot_dit_pretrained_path is not None:
            cot_state = torch.load(cot_dit_pretrained_path, map_location="cpu")
            missing, unexpected = cot_expert.load_state_dict(cot_state, strict=False)
            logger.info(
                "Loaded CoT DiT pretrained weights from %s: %d loaded, %d missing (random init), %d unexpected",
                cot_dit_pretrained_path, len(cot_state) - len(unexpected), len(missing), len(unexpected),
            )

        mot = MoT(
            mixtures={"cot": cot_expert, "video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        # Handle different initialization modes
        if init_mode == "fastwam":
            if fastwam_checkpoint_path is None:
                raise ValueError("`fastwam_checkpoint_path` is required for init_mode='fastwam'.")
            logger.info("Loading fastwam checkpoint from %s", fastwam_checkpoint_path)
            fastwam_ckpt = torch.load(fastwam_checkpoint_path, map_location="cpu")

            # Load Video and Action expert weights from fastwam checkpoint
            mot_state = mot.state_dict()
            fastwam_mot_state = {
                k: v for k, v in fastwam_ckpt["mot"].items()
                if k.startswith("mixtures.video.") or k.startswith("mixtures.action.")
            }
            for k, v in fastwam_mot_state.items():
                if k in mot_state:
                    mot_state[k] = v
            missing_keys, unexpected_keys = mot.load_state_dict(mot_state, strict=False)
            if missing_keys:
                logger.info("FastWAM checkpoint missing %d keys", len(missing_keys))
            if unexpected_keys:
                logger.warning("FastWAM checkpoint has %d unexpected keys", len(unexpected_keys))
            logger.info("Loaded Video/Action experts from fastwam checkpoint")
        elif init_mode != "wan_backbone":
            raise ValueError(f"init_mode must be 'wan_backbone' or 'fastwam', got {init_mode}")

        vlm_config = vlm_config or {}
        vlm_extract_mode = vlm_config.get("extract_mode", "precomputed")
        vlm_extractor = None
        if vlm_extract_mode == "online":
            vlm_model_path = vlm_config.get("model_path")
            if not vlm_model_path:
                raise ValueError("`vlm_config.model_path` required for online extraction.")
            vlm_extractor = VLMFeatureExtractor(
                vlm_model_path=vlm_model_path,
                device=device,
                torch_dtype=torch_dtype,
            )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            cot_expert=cot_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            vlm_extractor=vlm_extractor,
            vlm_extract_mode=vlm_extract_mode,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            init_mode=init_mode,
            fastwam_checkpoint_path=fastwam_checkpoint_path,
        )

        # Load proprio encoder from fastwam checkpoint if specified
        if init_mode == "fastwam" and fastwam_checkpoint_path is not None:
            fastwam_ckpt = torch.load(fastwam_checkpoint_path, map_location="cpu")
            if "proprio_encoder" in fastwam_ckpt and hasattr(model, "proprio_encoder"):
                model.proprio_encoder.load_state_dict(fastwam_ckpt["proprio_encoder"], strict=True)
                logger.info("Loaded proprio encoder from fastwam checkpoint")

        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        cot_seq_len: int = 0,
    ) -> torch.Tensor:
        """Build 3-expert attention mask: [CoT | Video | Action]."""
        if cot_seq_len == 0:
            return super()._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )

        total = cot_seq_len + video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        c_end = cot_seq_len
        v_end = c_end + video_seq_len
        f0_end = c_end + min(video_tokens_per_frame, video_seq_len)

        # CoT self-attention
        mask[:c_end, :c_end] = True
        # CoT -> f_0
        mask[:c_end, c_end:f0_end] = True

        # Video -> CoT
        mask[c_end:v_end, :c_end] = True
        # Video -> Video (first_frame_causal)
        mask[c_end:v_end, c_end:v_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # Action -> CoT
        mask[v_end:, :c_end] = True
        # Action -> f_0
        mask[v_end:, c_end:f0_end] = True
        # Action -> Action
        mask[v_end:, v_end:] = True

        return mask

    def _get_vlm_features(self, sample: dict) -> torch.Tensor:
        """Get VLM features from sample (precomputed) or online extraction."""
        if self.vlm_extract_mode == "precomputed":
            vlm_features = sample.get("vlm_features")
            if vlm_features is None:
                raise ValueError(
                    "vlm_extract_mode='precomputed' but sample has no 'vlm_features'."
                )
            return vlm_features.to(device=self.device, dtype=self.torch_dtype)
        else:
            if self.vlm_extractor is None:
                raise ValueError("Online extraction requires vlm_extractor to be loaded.")
            vlm_input_ids = sample.get("vlm_input_ids")
            if vlm_input_ids is None:
                raise ValueError(
                    "Online VLM extraction requires dataset to provide 'vlm_input_ids'. "
                    "Set vlm_model_path in the data config."
                )
            attention_mask = sample["vlm_attention_mask"]
            pixel_values = sample["vlm_pixel_values"].to(self.device, dtype=self.torch_dtype)
            pixel_values = pixel_values.view(-1, pixel_values.shape[-1])
            return self.vlm_extractor.extract(
                input_ids=vlm_input_ids.to(self.device),
                pixel_values=pixel_values,
                attention_mask=attention_mask.to(self.device),
                image_grid_thw=sample["vlm_image_grid_thw"].to(self.device),
            )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        # VLM features -> CoT pre_dit
        vlm_features = self._get_vlm_features(sample)
        cot_pre = self.cot_expert.pre_dit(vlm_features)
        cot_seq_len = cot_pre["tokens"].shape[1]

        # Video noising
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        # Action noising
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Pre-DiT
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        # 3-expert attention mask
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            cot_seq_len=cot_seq_len,
        )

        # MoT forward with 3 experts
        tokens_out = self.mot(
            embeds_all={
                "cot": cot_pre["tokens"],
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "cot": cot_pre["freqs"],
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "cot": None,
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "cot": cot_pre["t_mod"],
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        # Post-DiT (CoT output discarded)
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # Loss computation (same as FastWAM)
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        cot_pre: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cot_pre is None:
            cot_pre = getattr(self, "_infer_cot_pre", None)
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        cot_seq_len = cot_pre["tokens"].shape[1] if cot_pre is not None else 0
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            cot_seq_len=cot_seq_len,
        )

        embeds = {"video": video_pre["tokens"], "action": action_pre["tokens"]}
        freqs = {"video": video_pre["freqs"], "action": action_pre["freqs"]}
        ctx = {
            "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
            "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
        }
        t_mods = {"video": video_pre["t_mod"], "action": action_pre["t_mod"]}

        if cot_pre is not None:
            embeds["cot"] = cot_pre["tokens"]
            freqs["cot"] = cot_pre["freqs"]
            ctx["cot"] = None
            t_mods["cot"] = cot_pre["t_mod"]

        tokens_out = self.mot(
            embeds_all=embeds,
            attention_mask=attention_mask,
            freqs_all=freqs,
            context_all=ctx,
            t_mod_all=t_mods,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise_with_static_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        static_kv_cache: list[Dict[str, Dict[str, torch.Tensor]]],
        attention_mask: torch.Tensor,
        static_seq_lens: Dict[str, int],
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_static_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            static_kv_cache=static_kv_cache,
            attention_mask=attention_mask,
            static_seq_lens=static_seq_lens,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer(
        self,
        vlm_features: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if vlm_features is None:
            raise ValueError("FastWAMCoT.infer() requires `vlm_features`.")
        vlm_features = vlm_features.to(device=self.device, dtype=self.torch_dtype)
        if vlm_features.ndim == 2:
            vlm_features = vlm_features.unsqueeze(0)

        self._infer_cot_pre = self.cot_expert.pre_dit(vlm_features)
        self._infer_vlm_features = vlm_features
        try:
            return super().infer(**kwargs)
        finally:
            self._infer_cot_pre = None
            self._infer_vlm_features = None

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        vlm_features: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Image must be multiples of 16, got HxW=({height},{width})")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # Resolve context
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("Both `context` and `context_mask` required.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        # CoT pre_dit
        if vlm_features is None:
            vlm_features = getattr(self, "_infer_vlm_features", None)
        if vlm_features is None:
            raise ValueError("`vlm_features` is required for FastWAMCoT.infer_action().")
        vlm_features = vlm_features.to(device=self.device, dtype=self.torch_dtype)
        if vlm_features.ndim == 2:
            vlm_features = vlm_features.unsqueeze(0)
        cot_pre = self.cot_expert.pre_dit(vlm_features)
        cot_seq_len = cot_pre["tokens"].shape[1]

        # Video pre_dit (first frame only, timestep=0)
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])

        # Build full attention mask
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            cot_seq_len=cot_seq_len,
        )

        # Prefill static cache (CoT + Video jointly)
        static_mask = attention_mask[:cot_seq_len + video_seq_len, :cot_seq_len + video_seq_len]
        static_kv_cache = self.mot.prefill_static_cache(
            static_experts=["cot", "video"],
            embeds_all={"cot": cot_pre["tokens"], "video": video_pre["tokens"]},
            freqs_all={"cot": cot_pre["freqs"], "video": video_pre["freqs"]},
            t_mod_all={"cot": cot_pre["t_mod"], "video": video_pre["t_mod"]},
            context_all={
                "cot": None,
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
            },
            attention_mask=static_mask,
        )

        static_seq_lens = {"cot": cot_seq_len, "video": video_seq_len}

        # Action denoising loop
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action = self._predict_action_noise_with_static_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                static_kv_cache=static_kv_cache,
                attention_mask=attention_mask,
                static_seq_lens=static_seq_lens,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        from fastwam.lora_utils import collect_lora_state_dict

        payload = {
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "init_mode": self.init_mode,
        }
        if self.fastwam_checkpoint_path is not None:
            payload["fastwam_checkpoint_path"] = self.fastwam_checkpoint_path
        if self._train_modes is not None:
            payload["train_modes"] = self._train_modes
            # Per-expert saving based on training mode
            cot_state = {k: v for k, v in self.mot.state_dict().items() if k.startswith("mixtures.cot.")}
            payload["cot_expert"] = cot_state
            if self._train_modes["video"] == "lora":
                payload["video_expert_lora"] = collect_lora_state_dict(self.video_expert)
            else:
                video_state = {k: v for k, v in self.mot.state_dict().items() if k.startswith("mixtures.video.")}
                payload["video_expert"] = video_state
            if self._train_modes["action"] == "lora":
                payload["action_expert_lora"] = collect_lora_state_dict(self.action_expert)
            else:
                action_state = {k: v for k, v in self.mot.state_dict().items() if k.startswith("mixtures.action.")}
                payload["action_expert"] = action_state
        else:
            payload["mot"] = self.mot.state_dict()

        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)
        logger.info(
            "Saved checkpoint to %s | init_mode=%s | fastwam_ckpt=%s",
            path, self.init_mode, self.fastwam_checkpoint_path or "N/A"
        )

    def load_checkpoint(self, path, optimizer=None):
        from fastwam.lora_utils import load_lora_state_dict, apply_lora_to_expert

        payload = torch.load(path, map_location="cpu")

        if "mot" in payload:
            missing, unexpected = self.mot.load_state_dict(payload["mot"], strict=False)
            if missing:
                logger.info(f"Checkpoint missing keys (expected for new CoT expert): {len(missing)} keys")
            if unexpected:
                logger.warning(f"Checkpoint unexpected keys: {unexpected[:5]}...")
        elif "cot_expert" in payload:
            # Per-expert checkpoint format
            mot_state = self.mot.state_dict()
            # Load CoT expert (always full)
            for k, v in payload["cot_expert"].items():
                if k in mot_state:
                    mot_state[k] = v
            # Load Video expert
            if "video_expert" in payload:
                for k, v in payload["video_expert"].items():
                    if k in mot_state:
                        mot_state[k] = v
            elif "video_expert_lora" in payload:
                self._ensure_lora_injected(self.video_expert, payload["video_expert_lora"])
                load_lora_state_dict(self.video_expert, payload["video_expert_lora"], strict=False)
            # Load Action expert
            if "action_expert" in payload:
                for k, v in payload["action_expert"].items():
                    if k in mot_state:
                        mot_state[k] = v
            elif "action_expert_lora" in payload:
                self._ensure_lora_injected(self.action_expert, payload["action_expert_lora"])
                load_lora_state_dict(self.action_expert, payload["action_expert_lora"], strict=False)
            # Apply non-LoRA state
            self.mot.load_state_dict(mot_state, strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing recognized keys: {path}")

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder`; keeping current params.")
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    @staticmethod
    def _ensure_lora_injected(expert, lora_state: dict):
        """Inject LoRA into expert if not already present, inferring rank from checkpoint."""
        has_lora = any("lora_" in k for k in expert.state_dict())
        if has_lora:
            return
        first_A = next((k for k in lora_state if "lora_A" in k), None)
        if first_A is None:
            return
        rank = lora_state[first_A].shape[0]
        logger.info(f"Auto-injecting LoRA (rank={rank}) for checkpoint loading")
        from fastwam.lora_utils import apply_lora_to_expert
        apply_lora_to_expert(expert, rank=rank)
