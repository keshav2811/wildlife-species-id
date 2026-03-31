"""
Evaluation utilities: classification report, confusion matrix, Grad-CAM.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Full evaluation on the test set
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    results_dir: str = "results",
) -> dict:
    """
    Run inference on loader, compute and save metrics + confusion matrix.

    Returns:
        dict with keys: accuracy, report, confusion_matrix
    """
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy: {acc:.4f}\n")
    print(report)

    # Save report
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Plot & save confusion matrix
    _plot_confusion_matrix(cm, class_names, results_path / "confusion_matrix.png")

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Grad-CAM visualisation
# ---------------------------------------------------------------------------

def visualize_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    labels: list[int],
    class_names: list[str],
    target_layer: nn.Module,
    save_dir: str = "results/gradcam",
    n_images: int = 8,
):
    """
    Generate and save Grad-CAM heatmaps for a batch of images.

    Args:
        model:        Trained model.
        images:       Batch tensor (B, C, H, W), values in [0,1] after un-normalizing.
        labels:       Ground-truth class indices.
        class_names:  List of class name strings.
        target_layer: The conv layer to hook (e.g. model.backbone.layer4[-1]).
        save_dir:     Where to save the overlay images.
        n_images:     How many images to process.
    """
    if not GRADCAM_AVAILABLE:
        print("[WARNING] grad-cam not installed. Skipping Grad-CAM visualisation.")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model=model, target_layers=[target_layer])

    for i in range(min(n_images, len(images))):
        input_tensor = images[i].unsqueeze(0)
        target = [ClassifierOutputTarget(labels[i])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]

        # Convert tensor to RGB numpy for overlay
        rgb = images[i].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title(f"GT: {class_names[labels[i]]}")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path / f"gradcam_{i:03d}.png", dpi=120)
        plt.close()

    print(f"Grad-CAM images saved → {save_dir}")
