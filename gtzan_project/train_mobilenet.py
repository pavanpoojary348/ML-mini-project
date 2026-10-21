"""Training Script for MobileNetV2 Transfer Learning on GTZAN Spectrograms.

Implements two-stage training:
- Stage 1: Feature extractor frozen, training newly initialized classifier head.
- Stage 2: Fine-tuning top inverted residual blocks with reduced learning rate
  and Cosine Annealing schedule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from gtzan_project.dataset import build_dataloaders
from gtzan_project.models import MobileNetV2Classifier


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, float(epoch_acc)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    eval_loss = running_loss / len(loader.dataset)
    eval_acc = float(accuracy_score(all_targets, all_preds))
    eval_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    return eval_loss, eval_acc, eval_f1


def plot_training_history(history: Dict[str, List[float]], output_path: Path, title: str):
    """Plots and saves train/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    ax1.set_title(f"{title} - Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#16a34a", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", color="#ea580c", linewidth=2, linestyle="--")
    ax2.set_title(f"{title} - Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_mobilenet(
    data_dir: Path,
    output_dir: Path,
    warmup_epochs: int = 10,
    finetune_epochs: int = 20,
    batch_size: int = 16,
    lr_head: float = 8e-4,
    lr_finetune: float = 1e-4,
    random_seed: int = 42,
) -> Dict:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MobileNetV2] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        random_seed=random_seed,
    )

    model = MobileNetV2Classifier(num_classes=10, dropout_rate=0.3)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = output_dir / "best_mobilenet_v2.pt"

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    start_time = time.time()
    current_epoch = 0

    # ------------------ STAGE 1: WARMUP HEAD ------------------
    print("\n>>> Stage 1: Classifier Head Warmup (Feature Backbone Frozen) <<<")
    model.freeze_backbone()
    optimizer_head = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_head,
        weight_decay=1e-4,
    )

    for epoch in range(1, warmup_epochs + 1):
        current_epoch += 1
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_head, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"[Stage 1] Epoch {current_epoch:02d} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% - F1: {val_f1*100:.2f}%"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = current_epoch
            torch.save(
                {
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "model_type": "mobilenet_v2",
                },
                checkpoint_path,
            )

    # ------------------ STAGE 2: FINE-TUNING ------------------
    print("\n>>> Stage 2: Fine-Tuning Top Backbone Blocks + Classifier Head <<<")
    model.unfreeze_backbone(unfreeze_from_block=12)

    # Differential learning rates: lower for backbone, slightly higher for head
    backbone_params = [p for i, block in enumerate(model.base_model.features) if i >= 12 for p in block.parameters()]
    head_params = list(model.base_model.classifier.parameters())

    optimizer_ft = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_finetune * 0.5},
            {"params": head_params, "lr": lr_finetune},
        ],
        weight_decay=1e-4,
    )
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs, eta_min=1e-6)

    for epoch in range(1, finetune_epochs + 1):
        current_epoch += 1
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler_ft.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"[Stage 2] Epoch {current_epoch:02d} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% - F1: {val_f1*100:.2f}%"
        )

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = current_epoch
            torch.save(
                {
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "model_type": "mobilenet_v2",
                },
                checkpoint_path,
            )

    elapsed_time = time.time() - start_time
    print(
        f"\n[MobileNetV2] Training Complete in {elapsed_time:.1f}s! "
        f"Best Val Acc: {best_val_acc*100:.2f}%, Best Val F1: {best_val_f1*100:.2f}% at Epoch {best_epoch}"
    )

    plot_training_history(
        history,
        output_dir / "curves_mobilenet_v2.png",
        title="Training Curves (MobileNetV2 Transfer Learning)",
    )

    with open(output_dir / "history_mobilenet_v2.json", "w") as f:
        json.dump(
            {
                "model_type": "mobilenet_v2",
                "warmup_epochs": warmup_epochs,
                "finetune_epochs": finetune_epochs,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_f1": best_val_f1,
                "elapsed_time_sec": elapsed_time,
                "history": history,
            },
            f,
            indent=2,
        )

    return {
        "model_type": "mobilenet_v2",
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "checkpoint": str(checkpoint_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on GTZAN spectrograms.")
    parser.add_argument("--data-dir", type=str, default="data/gtzan/Data")
    parser.add_argument("--output-dir", type=str, default="results_gtzan")
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=8e-4)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    args = parser.parse_args()

    train_mobilenet(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
    )
