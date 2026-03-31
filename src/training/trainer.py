"""
Trainer: orchestrates training loop, validation, early stopping, and checkpointing.
"""

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """Full training/validation loop with AMP, LR scheduling, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        tr_cfg = cfg["training"]
        log_cfg = cfg["logging"]

        self.epochs = tr_cfg["epochs"]
        self.grad_clip = tr_cfg.get("gradient_clip", 1.0)
        self.use_amp = tr_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.log_interval = log_cfg.get("log_interval", 10)

        # Paths
        self.ckpt_dir = Path(log_cfg["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(log_cfg["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer
        self.optimizer = self._build_optimizer(tr_cfg)

        # Scheduler
        self.scheduler = self._build_scheduler(tr_cfg)

        # AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=tr_cfg.get("early_stopping_patience", 10)
        )

        # TensorBoard
        tb_dir = log_cfg.get("tensorboard_dir", "results/tensorboard")
        self.writer = SummaryWriter(log_dir=tb_dir)

        self.history: list[dict] = []
        self.best_val_acc = 0.0
        self.best_ckpt_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_optimizer(self, tr_cfg: dict) -> torch.optim.Optimizer:
        opt_name = tr_cfg.get("optimizer", "adamw").lower()
        lr = tr_cfg["learning_rate"]
        wd = tr_cfg.get("weight_decay", 1e-4)
        if opt_name == "adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            return AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def _build_scheduler(self, tr_cfg: dict):
        sched_name = tr_cfg.get("scheduler", "cosine").lower()
        warmup_epochs = tr_cfg.get("warmup_epochs", 0)

        if sched_name == "cosine":
            main_sched = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup_epochs,
                eta_min=1e-6,
            )
        elif sched_name == "step":
            main_sched = StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:  # plateau
            return ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5
            )

        if warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, main_sched],
                milestones=[warmup_epochs],
            )
        return main_sched

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        for step, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            if step % self.log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [val]", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        return total_loss / total, correct / total

    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_acc: float) -> Path:
        path = self.ckpt_dir / f"epoch_{epoch+1:03d}_acc{val_acc:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "cfg": self.cfg,
            },
            path,
        )
        return path

    def _cleanup_old_checkpoints(self, keep: int = 3) -> None:
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        for ckpt in ckpts[:-keep]:
            ckpt.unlink(missing_ok=True)

    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """Run the full training loop. Returns training history."""
        print(f"\nTraining on {self.device} | AMP={self.use_amp}")
        print(f"Epochs: {self.epochs} | Architecture: {self.cfg['model']['architecture']}\n")

        for epoch in range(self.epochs):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._val_epoch(epoch)

            # LR scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            self.writer.add_scalars(
                "Loss", {"train": train_loss, "val": val_loss}, epoch
            )
            self.writer.add_scalars(
                "Accuracy", {"train": train_acc, "val": val_acc}, epoch
            )
            self.writer.add_scalar("LR", lr, epoch)

            record = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "lr": lr,
                "elapsed_s": round(elapsed, 1),
            }
            self.history.append(record)

            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.1f}s"
            )

            # Checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_ckpt_path = self._save_checkpoint(epoch, val_acc)
                self._cleanup_old_checkpoints(
                    keep=self.cfg["logging"].get("save_top_k", 3)
                )
                print(f"  ✓ Best model saved → {self.best_ckpt_path.name}")

            # Early stopping
            if self.early_stopping.step(val_acc):
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

        self.writer.close()

        # Save history
        history_path = self.results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best val acc: {self.best_val_acc:.4f}")
        return {"history": self.history, "best_val_acc": self.best_val_acc}
