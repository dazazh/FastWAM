import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        episode_indices: Optional[list] = None,
        episode_ranges: Optional[list] = None,
        task_names: Optional[list] = None,
        task_episodes_file: Optional[str] = None,
        vlm_features_dir: Optional[str] = None,
        vlm_model_path: Optional[str] = None,
        vlm_max_seq_len: int = 512,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_indices=episode_indices,
            episode_ranges=episode_ranges,
            task_names=task_names,
            task_episodes_file=task_episodes_file,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.vlm_features_dir = vlm_features_dir
        self.vlm_model_path = vlm_model_path
        self.vlm_max_seq_len = vlm_max_seq_len
        self._vlm_processor = None
        if vlm_model_path is not None and vlm_features_dir is None:
            from transformers import AutoProcessor
            self._vlm_processor = AutoProcessor.from_pretrained(
                vlm_model_path, trust_remote_code=True
            )
            logger.info(f"VLM processor loaded from {vlm_model_path} for online tokenization.")

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]

        # Prepare VLM image from concatenated multi-cam first frame
        vlm_image = None
        if self.concat_multi_camera == "robotwin" and num_cameras == 3:
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
            # Save first frame for VLM (in [0, 1] range before normalization)
            vlm_image = video[0]  # [C, 384, 320], range [0, 1]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
            # Save first frame for VLM
            vlm_image = video[0]  # [C, H, W*num_cameras], range [0, 1]
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]
            vlm_image = video[0]  # [C, H, W], range [0, 1]

        # final resize and normalization for video model
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

        if self.vlm_features_dir is not None:
            episode_idx = sample["episode_index"]
            if hasattr(episode_idx, "item"):
                episode_idx = episode_idx.item()
            feat_path = os.path.join(self.vlm_features_dir, f"episode_{episode_idx:06d}.pt")
            data["vlm_features"] = torch.load(feat_path, map_location="cpu", weights_only=True)
        elif self._vlm_processor is not None:
            vlm_tok = self._tokenize_vlm(vlm_image, instruction)
            data.update(vlm_tok)

        return data

    def _tokenize_vlm(self, vlm_image: torch.Tensor, instruction: str) -> dict:
        """Tokenize concatenated multi-cam first frame + instruction for VLM.

        NOTE: Returns unpadded VLM tokens - padding is done in collate_fn.

        Args:
            vlm_image: [C, H, W] tensor in [0, 1] range, first frame of concatenated video
            instruction: text instruction
        """
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        # Convert tensor [C, H, W] to PIL Image
        frame_np = (vlm_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(frame_np)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        formatted_text = self._vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._vlm_processor(
            text=[formatted_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,  # No padding here - will be done in collate_fn
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)  # [seq_len]
        attention_mask = inputs["attention_mask"].squeeze(0)  # [seq_len]
        pixel_values = inputs["pixel_values"]  # [num_patches, patch_dim]
        image_grid_thw = inputs["image_grid_thw"]  # [num_images, 3]

        # No padding at sample level - return actual length
        return {
            "vlm_input_ids": input_ids,
            "vlm_attention_mask": attention_mask,
            "vlm_pixel_values": pixel_values.squeeze(0) if pixel_values.ndim == 3 else pixel_values,
            "vlm_image_grid_thw": image_grid_thw.squeeze(0) if image_grid_thw.ndim == 2 else image_grid_thw,
            "vlm_seq_len": input_ids.shape[0],  # Actual sequence length
        }

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate function for batch-wise VLM token padding."""
        keys = batch[0].keys()

        # Handle non-VLM tensors with default stacking
        result = {}
        vlm_keys = ["vlm_input_ids", "vlm_attention_mask", "vlm_pixel_values", "vlm_image_grid_thw", "vlm_seq_len"]
        has_vlm = any(k in batch[0] for k in vlm_keys)

        if has_vlm:
            # Get max sequence length in this batch
            max_seq_len = max(item.get("vlm_seq_len", 0) for item in batch)
            # Cap at vlm_max_seq_len to prevent OOM
            max_seq_len = min(max_seq_len, 512)

            # Pad VLM tokens batch-wise
            vlm_input_ids_list = []
            vlm_attention_mask_list = []
            for item in batch:
                ids = item["vlm_input_ids"]
                mask = item["vlm_attention_mask"]
                seq_len = ids.shape[0]

                if seq_len < max_seq_len:
                    pad_size = max_seq_len - seq_len
                    ids = torch.cat([ids, ids.new_zeros(pad_size)])
                    mask = torch.cat([mask, mask.new_zeros(pad_size)])
                elif seq_len > max_seq_len:
                    ids = ids[:max_seq_len]
                    mask = mask[:max_seq_len]

                vlm_input_ids_list.append(ids)
                vlm_attention_mask_list.append(mask)

            result["vlm_input_ids"] = torch.stack(vlm_input_ids_list, dim=0)
            result["vlm_attention_mask"] = torch.stack(vlm_attention_mask_list, dim=0)
            result["vlm_pixel_values"] = torch.cat([item["vlm_pixel_values"].unsqueeze(0) for item in batch], dim=0)
            result["vlm_image_grid_thw"] = torch.cat([item["vlm_image_grid_thw"].unsqueeze(0) for item in batch], dim=0)
        else:
            # No VLM tokens in this batch (e.g., precomputed features)
            pass

        # Stack other tensors
        for key in keys:
            if key in vlm_keys:
                continue
            if key == "prompt":
                result[key] = [item[key] for item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                if batch[0][key].ndim == 0:
                    result[key] = torch.stack([item[key] for item in batch])
                else:
                    # Check if all items have the same shape
                    first_shape = batch[0][key].shape
                    if all(item[key].shape == first_shape for item in batch):
                        result[key] = torch.stack([item[key] for item in batch])
                    else:
                        # Pad variable-length tensors if needed
                        result[key] = torch.nn.utils.rnn.pad_sequence(
                            [item[key] for item in batch], batch_first=True, padding_value=0
                        )
            else:
                result[key] = [item[key] for item in batch]

        return result
