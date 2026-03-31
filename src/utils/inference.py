"""
Inference script for single image or batch folder prediction.

Usage:
    python -m src.utils.inference \
        --checkpoint checkpoints/best.pt \
        --input path/to/image_or_folder \
        [--top_k 3]
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.cnn import build_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model and config from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def build_inference_transform(image_size: list[int]) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: str,
    transform: A.Compose,
    class_names: list[str],
    device: torch.device,
    top_k: int = 3,
) -> dict:
    """Run inference on a single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    top_probs, top_indices = probs.topk(min(top_k, len(class_names)))

    return {
        "image": image_path,
        "predictions": [
            {
                "rank": i + 1,
                "species": class_names[idx.item()],
                "confidence": round(prob.item(), 4),
            }
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
        ],
    }


def predict_folder(
    model: torch.nn.Module,
    folder: str,
    transform: A.Compose,
    class_names: list[str],
    device: torch.device,
    top_k: int = 1,
) -> list[dict]:
    """Run inference on all images in a folder."""
    folder_path = Path(folder)
    image_paths = list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.png"))
    results = []
    for img_path in sorted(image_paths):
        try:
            result = predict_image(
                model, str(img_path), transform, class_names, device, top_k
            )
            results.append(result)
            pred = result["predictions"][0]
            print(f"{img_path.name:<40} → {pred['species']} ({pred['confidence']:.2%})")
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Wildlife species inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--input", required=True, help="Image file or folder")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K predictions")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    class_names = sorted(cfg["data"]["species_classes"])
    transform = build_inference_transform(cfg["data"]["image_size"])

    input_path = Path(args.input)
    if input_path.is_dir():
        results = predict_folder(model, str(input_path), transform, class_names, device, args.top_k)
    else:
        result = predict_image(model, str(input_path), transform, class_names, device, args.top_k)
        results = [result]
        for pred in result["predictions"]:
            print(f"  #{pred['rank']} {pred['species']}: {pred['confidence']:.2%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
