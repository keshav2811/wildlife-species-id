"""
PyTorch Dataset and DataLoader factories for wildlife camera trap images.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class WildlifeDataset(Dataset):
    """
    ImageFolder-style dataset for wildlife species images.

    Directory layout:
        root/
            species_A/img1.jpg
            species_A/img2.jpg
            species_B/img1.jpg
            ...
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        classes: Optional[list] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        # Discover classes
        if classes:
            self.classes = sorted(classes)
        else:
            self.classes = sorted(
                [d.name for d in self.root.iterdir() if d.is_dir()]
            )

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples: list[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = cv2.imread(str(img_path))  # type: ignore
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency weights for handling class imbalance."""
        counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts
        weights[counts == 0] = 0.0
        return weights


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(cfg: dict, split: str) -> A.Compose:
    """Build albumentations transform pipeline from config."""
    img_h, img_w = cfg["data"]["image_size"]
    norm = cfg["augmentation"][split if split == "train" else "val"]["normalize"]
    mean = norm["mean"]
    std = norm["std"]

    if split == "train":
        aug_cfg = cfg["augmentation"]["train"]
        transforms = [A.Resize(img_h, img_w)]

        if aug_cfg.get("random_crop"):
            transforms.append(A.RandomResizedCrop(size=(img_h, img_w), scale=(0.7, 1.0)))

        if aug_cfg.get("horizontal_flip"):
            transforms.append(A.HorizontalFlip(p=0.5))

        if aug_cfg.get("rotation_limit"):
            transforms.append(
                A.Rotate(limit=aug_cfg["rotation_limit"], p=0.4)
            )

        if aug_cfg.get("brightness_contrast"):
            transforms.append(A.RandomBrightnessContrast(p=0.4))

        if aug_cfg.get("hue_saturation"):
            transforms.append(A.HueSaturationValue(p=0.3))

        if aug_cfg.get("gaussian_noise"):
            transforms.append(A.GaussNoise(p=0.2))

        # Camera-trap specific: simulate night-vision / IR images
        transforms += [
    A.RandomGamma(p=0.3),
    A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(16, 48), hole_width_range=(16, 48), p=0.3),
    A.MotionBlur(p=0.2),
    A.RandomFog(p=0.2),
]
    else:
        transforms = [A.Resize(img_h, img_w)]

    transforms += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: dict,
    splits_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    splits_dir = splits_dir or cfg["data"]["splits_dir"]
    classes = cfg["data"].get("species_classes")
    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    pin = cfg["data"]["pin_memory"]

    train_ds = WildlifeDataset(
        root="data/raw",
        transform=build_transforms(cfg, "train"),
        classes=classes,
    )
    val_ds = WildlifeDataset(
        root="data/raw",
        transform=build_transforms(cfg, "val"),
        classes=classes,
    )
    test_ds = WildlifeDataset(
        root="data/raw",
        transform=build_transforms(cfg, "val"),
        classes=classes,
    )

    # Weighted sampler to handle class imbalance in training
    class_weights = train_ds.get_class_weights()
    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin
    )

    return train_loader, val_loader, test_loader
