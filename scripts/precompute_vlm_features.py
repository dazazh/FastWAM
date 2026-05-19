"""Precompute VLM features for FastWAMCoT training.

Usage:
    python scripts/precompute_vlm_features.py \
        --vlm_model_path Qwen/Qwen3-VL-2B \
        --dataset_dir /path/to/dataset \
        --output_dir /path/to/vlm_features \
        --batch_size 4
"""
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VLM features for FastWAMCoT")
    parser.add_argument("--vlm_model_path", type=str, required=True,
                        help="Path or HuggingFace model ID for Qwen3-VL-2B")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root directory of the training dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save precomputed VLM features")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for VLM inference")
    parser.add_argument("--image_size", type=int, nargs=2, default=[480, 640],
                        help="Image resize dimensions [H, W]")
    parser.add_argument("--prompt_template", type=str,
                        default="Describe what is happening in this image and what actions should be taken.",
                        help="Text prompt template for VLM")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    from fastwam.models.wan22.vlm_feature_extractor import VLMFeatureExtractor

    extractor = VLMFeatureExtractor(
        vlm_model_path=args.vlm_model_path,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    dataset_dir = Path(args.dataset_dir)
    episodes = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not episodes:
        episodes = [dataset_dir]

    print(f"Found {len(episodes)} episode(s) to process")
    print(f"Output dir: {args.output_dir}")

    total_processed = 0
    for episode_dir in tqdm(episodes, desc="Episodes"):
        episode_name = episode_dir.name

        image_files = sorted(episode_dir.glob("*.png")) + sorted(episode_dir.glob("*.jpg"))
        if not image_files:
            frame_dir = episode_dir / "frames"
            if frame_dir.exists():
                image_files = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))
        if not image_files:
            continue

        first_frame_path = image_files[0]

        from PIL import Image
        image = Image.open(first_frame_path).convert("RGB")
        if args.image_size:
            image = image.resize((args.image_size[1], args.image_size[0]))

        text_file = episode_dir / "instruction.txt"
        if text_file.exists():
            prompt = text_file.read_text().strip()
        else:
            prompt = args.prompt_template

        with torch.no_grad():
            features = extractor.extract_from_raw(
                images=[image],
                texts=[prompt],
            )

        output_path = Path(args.output_dir) / f"{episode_name}.pt"
        torch.save(features[0].cpu(), output_path)
        total_processed += 1

    print(f"Done. Processed {total_processed} episodes. Features saved to {args.output_dir}")


if __name__ == "__main__":
    main()
