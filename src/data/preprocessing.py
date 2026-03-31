"""
Preprocessing pipeline for camera trap images.

Expected input: data/raw/<species_name>/<image>.jpg
Output:         data/splits/{train,val,test}/<species_name>/<image>.jpg
"""

import os
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def is_valid_image(path: Path) -> bool:
    """Check if an image can be read and is not corrupted."""
    try:
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception:
        return False


def remove_empty_frames(image_path: Path, threshold: float = 0.02) -> bool:
    """
    Heuristic to detect 'empty' camera trap frames (no animal present).
    Uses variance of pixel intensities — very uniform images are likely empty.
    Returns True if the frame likely contains an animal.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    variance = np.var(img / 255.0)
    return variance > threshold


def split_dataset(
    raw_dir: str,
    splits_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    filter_empty: bool = True,
) -> dict:
    """
    Split raw per-class image folders into train/val/test subsets.

    Args:
        raw_dir:      Path to raw data (subfolders = class names).
        splits_dir:   Output directory for split subsets.
        train_ratio:  Fraction of data for training.
        val_ratio:    Fraction of data for validation.
        seed:         Random seed for reproducibility.
        filter_empty: Whether to skip likely-empty frames.

    Returns:
        Dictionary with per-split, per-class image counts.
    """
    random.seed(seed)
    raw_path = Path(raw_dir)
    splits_path = Path(splits_dir)
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    stats = {"train": {}, "val": {}, "test": {}}

    species_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    if not species_dirs:
        raise FileNotFoundError(f"No class subdirectories found in {raw_dir}")

    for species_dir in tqdm(species_dirs, desc="Processing species"):
        species = species_dir.name
        images = [
            p for p in species_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            and is_valid_image(p)
        ]

        if filter_empty:
            images = [p for p in images if remove_empty_frames(p)]

        if not images:
            print(f"[WARNING] No valid images found for species: {species}")
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train: n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest_dir = splits_path / split_name / species
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)
            stats[split_name][species] = len(split_images)

    return stats


def print_split_summary(stats: dict) -> None:
    """Print a summary table of the dataset split."""
    print("\n=== Dataset Split Summary ===")
    all_species = sorted(
        set(s for split in stats.values() for s in split.keys())
    )
    header = f"{'Species':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}"
    print(header)
    print("-" * len(header))
    for species in all_species:
        tr = stats["train"].get(species, 0)
        vl = stats["val"].get(species, 0)
        te = stats["test"].get(species, 0)
        print(f"{species:<20} {tr:>8} {vl:>8} {te:>8} {tr+vl+te:>8}")
    print()


if __name__ == "__main__":
    cfg = load_config()
    stats = split_dataset(
        raw_dir=cfg["data"]["raw_dir"],
        splits_dir=cfg["data"]["splits_dir"],
        train_ratio=cfg["training"]["split_ratios"]["train"],
        val_ratio=cfg["training"]["split_ratios"]["val"],
        seed=cfg["project"]["seed"],
    )
    print_split_summary(stats)
