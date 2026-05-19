from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class VLMFeatureExtractor(nn.Module):
    """Frozen Qwen3-VL feature extractor for CoT DiT.

    Loads a Qwen3-VL model, freezes all parameters, and extracts the last
    hidden layer features given image + text inputs.
    """

    def __init__(
        self,
        vlm_model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logger.info(f"Loading VLM from {vlm_model_path} ...")
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.vlm.requires_grad_(False)
        self.vlm.eval()

        self.processor = AutoProcessor.from_pretrained(
            vlm_model_path, trust_remote_code=True
        )
        self._device = device
        self._dtype = torch_dtype
        logger.info("VLM loaded and frozen.")

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @torch.no_grad()
    def extract(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Extract last hidden layer features from the VLM.

        Args:
            input_ids: [B, seq_len]
            pixel_values: [B, ...] image pixel values from processor
            attention_mask: [B, seq_len]
            image_grid_thw: [B, 3] image grid dimensions

        Returns:
            Last hidden state: [B, seq_len, vlm_dim=2048]
        """
        device = input_ids.device

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

        image_embeds, deepstack_image_embeds = self.vlm.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(device, self._dtype)

        image_mask, _ = self.vlm.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]

        position_ids, _ = self.vlm.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        vlm_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": True,
            "return_dict": True,
        }

        if visual_pos_masks is not None:
            vlm_kwargs["visual_pos_masks"] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs["deepstack_visual_embeds"] = deepstack_image_embeds

        vlm_output = self.vlm.model.language_model(**vlm_kwargs)
        return vlm_output.hidden_states[-1]

    @torch.no_grad()
    def extract_from_raw(
        self,
        images: list,
        texts: list[str],
    ) -> torch.Tensor:
        """Extract features from raw images and text strings.

        Uses apply_chat_template + process_vision_info to match Motus preprocessing.

        Args:
            images: List of PIL images.
            texts: List of text instructions.

        Returns:
            Last hidden state: [B, seq_len, vlm_dim=2048]
        """
        from qwen_vl_utils import process_vision_info

        all_inputs = []
        for img, text in zip(images, texts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            formatted_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[formatted_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            all_inputs.append(inputs)

        input_ids_list = [inp["input_ids"] for inp in all_inputs]
        max_seq_len = max(ids.shape[1] for ids in input_ids_list)
        padded_input_ids = []
        padded_attention_masks = []
        pixel_values_list = []
        image_grid_thw_list = []

        for inp in all_inputs:
            ids = inp["input_ids"]
            mask = inp["attention_mask"]
            if ids.shape[1] < max_seq_len:
                pad_size = max_seq_len - ids.shape[1]
                ids = torch.cat([ids, ids.new_zeros(1, pad_size)], dim=1)
                mask = torch.cat([mask, mask.new_zeros(1, pad_size)], dim=1)
            padded_input_ids.append(ids)
            padded_attention_masks.append(mask)
            pixel_values_list.append(inp["pixel_values"])
            image_grid_thw_list.append(inp["image_grid_thw"])

        processed = {
            "input_ids": torch.cat(padded_input_ids, dim=0).to(self._device),
            "attention_mask": torch.cat(padded_attention_masks, dim=0).to(self._device),
            "pixel_values": torch.cat(pixel_values_list, dim=0).to(self._device),
            "image_grid_thw": torch.cat(image_grid_thw_list, dim=0).to(self._device),
        }

        return self.extract(
            input_ids=processed["input_ids"],
            pixel_values=processed["pixel_values"],
            attention_mask=processed["attention_mask"],
            image_grid_thw=processed["image_grid_thw"],
        )
