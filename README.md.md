# Wildlife Species Identification from Camera Trap Images

A full-fledged deep learning pipeline for identifying wildlife species in camera trap images using convolutional neural networks (CNNs) and transfer learning with PyTorch.

---

## Project Structure

```
wildlife_species_id/
├── configs/
│   └── config.yaml          # All hyperparameters and paths
├── data/
│   ├── raw/                 # Raw images: one subfolder per species
│   ├── processed/           # (optional) pre-processed images
│   └── splits/
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── data/
│   │   ├── dataset.py       # WildlifeDataset, DataLoader factory, augmentations
│   │   └── preprocessing.py # Train/val/test split + empty frame filtering
│   ├── models/
│   │   └── cnn.py           # CustomCNN, WildlifeClassifier (ResNet/EfficientNet/ViT), factory
│   ├── training/
│   │   └── trainer.py       # Trainer: AMP, early stopping, LR scheduling, checkpointing
│   └── utils/
│       ├── evaluate.py      # Metrics, confusion matrix, Grad-CAM
│       └── inference.py     # Single-image and batch folder prediction
├── tests/
│   └── test_model.py        # Pytest unit tests
├── notebooks/               # Jupyter notebooks for exploration
├── checkpoints/             # Saved model checkpoints (.pt)
├── results/                 # Metrics, plots, TensorBoard logs
├── train.py                 # Main training entry point
├── setup.py
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Or install as an editable package:
pip install -e .
```

### 2. Prepare data

Organise your camera trap images as:
```
data/raw/
    elephant/  img001.jpg  img002.jpg ...
    lion/      img001.jpg  ...
    leopard/   ...
```
Then run the preprocessing script to create train/val/test splits:
```bash
python -m src.data.preprocessing
```

### 3. Train

```bash
# Default (ResNet-50 with pretrained weights):
python train.py

# Switch architecture:
python train.py --architecture efficientnet_b3

# Override epochs and batch size:
python train.py --epochs 30 --batch_size 64
```

### 4. Monitor with TensorBoard

```bash
tensorboard --logdir results/tensorboard
```

### 5. Inference

```bash
# Single image (top-3 predictions):
python -m src.utils.inference --checkpoint checkpoints/best.pt --input path/to/image.jpg --top_k 3

# Entire folder:
python -m src.utils.inference --checkpoint checkpoints/best.pt --input path/to/folder/ --output results/predictions.json
```

---

## Model Architectures

| Key               | Description                                      |
|-------------------|--------------------------------------------------|
| `custom_cnn`      | 5-block CNN built from scratch (lightweight)     |
| `resnet50`        | ResNet-50 with ImageNet pre-training (default)   |
| `efficientnet_b3` | EfficientNet-B3 with ImageNet pre-training       |
| `vit_b_16`        | Vision Transformer ViT-B/16 via timm             |

Switch architectures in `configs/config.yaml` → `model.architecture`.

---

## Key Features

- **Camera trap–specific augmentations**: gamma correction, coarse dropout, IR/night-vision simulation
- **Class imbalance handling**: `WeightedRandomSampler` + label smoothing in loss
- **Mixed precision training** (AMP) for faster GPU training
- **Cosine LR schedule** with linear warm-up
- **Early stopping** with configurable patience
- **Top-K checkpoint saving**
- **Grad-CAM visualisations** for model interpretability
- **Confusion matrix + per-class classification report**

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All settings live in `configs/config.yaml`. Key sections:

- `data`: image size, workers, species class list
- `augmentation`: per-split augmentation flags
- `model`: architecture, pretrained, dropout, num_classes
- `training`: epochs, batch_size, optimizer, scheduler, early stopping
- `logging`: checkpoint dir, tensorboard dir, save_top_k
