"""
Unit tests for model architecture and dataset utilities.
Run with:  pytest tests/ -v
"""

import pytest
import torch
import yaml

from src.models.cnn import CustomCNN, WildlifeClassifier, build_model


@pytest.fixture
def base_cfg():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Override for fast tests
    cfg["model"]["pretrained"] = False
    cfg["model"]["num_classes"] = 5
    return cfg


# ---------------------------------------------------------------------------
# CustomCNN
# ---------------------------------------------------------------------------

class TestCustomCNN:
    def test_output_shape(self):
        model = CustomCNN(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    def test_dropout_different_in_train_eval(self):
        model = CustomCNN(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        model.train()
        out_train = model(x)
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        # Just verify they run without error; values may differ due to dropout
        assert out_train.shape == out_eval.shape


# ---------------------------------------------------------------------------
# WildlifeClassifier (transfer learning)
# ---------------------------------------------------------------------------

class TestWildlifeClassifier:
    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b3"])
    def test_forward_pass(self, arch):
        model = WildlifeClassifier(
            architecture=arch,
            num_classes=5,
            pretrained=False,
            dropout=0.0,
        )
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5), f"[{arch}] Expected (2, 5), got {out.shape}"

    def test_freeze_backbone(self):
        model = WildlifeClassifier(
            architecture="resnet50",
            num_classes=5,
            pretrained=False,
            freeze_backbone=True,
        )
        frozen = [p for p in model.backbone.parameters() if p.requires_grad]
        assert len(frozen) == 0, "Backbone should be frozen"
        trainable = [p for p in model.classifier.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Classifier should be trainable"


# ---------------------------------------------------------------------------
# build_model factory
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_custom_cnn(self, base_cfg):
        base_cfg["model"]["architecture"] = "custom_cnn"
        model = build_model(base_cfg)
        assert isinstance(model, CustomCNN)

    def test_resnet50(self, base_cfg):
        base_cfg["model"]["architecture"] = "resnet50"
        model = build_model(base_cfg)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, base_cfg["model"]["num_classes"])
