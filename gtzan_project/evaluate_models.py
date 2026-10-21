"""Comprehensive Evaluation and Ensembling Script for GTZAN Genre Classification.

Evaluates all trained models on the held-out test split (100 unseen images):
1. PaperCNN (Baseline)
2. EnhancedSECNN (Our architectural improvement with SE-attention & BatchNorm)
3. MobileNetV2 (Transfer Learning)
4. Ensemble Blending (Hybrid Fusion)

Produces:
- Confusion Matrix Heatmaps for all models
- Performance Comparison Charts (Accuracy & Macro-F1)
- Classification Reports with precision, recall, and F1 per genre
- Detailed test_metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from gtzan_project.dataset import GENRES, build_dataloaders
from gtzan_project.models import EnhancedSECNN, MobileNetV2Classifier, PaperCNN


def get_model_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns probabilities (N, C), predicted class indices (N,), and true targets (N,)."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.extend(targets.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    preds_arr = np.argmax(probs_arr, axis=1)
    targets_arr = np.array(all_targets)
    return probs_arr, preds_arr, targets_arr


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path):
    """Plots and saves a professional confusion matrix heatmap."""
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=GENRES,
        yticklabels=GENRES,
        cbar=True,
        linewidths=0.5,
    )
    plt.title(f"Confusion Matrix - {title}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Genre", fontsize=11, labelpad=8)
    plt.ylabel("True Genre", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_comparison_chart(results: Dict[str, Dict], output_path: Path):
    """Plots bar chart comparing Accuracy and Macro-F1 across all models."""
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in model_names]
    f1_scores = [results[m]["macro_f1"] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Test Accuracy (%)", color="#2563eb", alpha=0.9)
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="Test Macro-F1 (%)", color="#16a34a", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("GTZAN Genre Classification: Model Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, fontweight="medium")
    ax.set_ylim([0, 105])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)

    # Attach values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_all(data_dir: Path, output_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    _, _, test_loader = build_dataloaders(data_dir=data_dir, batch_size=16)

    models_to_eval = {}

    # Check for PaperCNN checkpoint
    paper_ckpt = output_dir / "best_paper_cnn.pt"
    if paper_ckpt.exists():
        m_paper = PaperCNN(num_classes=10).to(device)
        ckpt = torch.load(paper_ckpt, map_location=device)
        m_paper.load_state_dict(ckpt["model_state_dict"])
        models_to_eval["Paper CNN Baseline"] = m_paper

    # Check for EnhancedSECNN checkpoint
    secnn_ckpt = output_dir / "best_enhanced_secnn.pt"
    if secnn_ckpt.exists():
        m_secnn = EnhancedSECNN(num_classes=10).to(device)
        ckpt = torch.load(secnn_ckpt, map_location=device)
        m_secnn.load_state_dict(ckpt["model_state_dict"])
        models_to_eval["Enhanced SE-CNN"] = m_secnn

    # Check for MobileNetV2 checkpoint
    mobilenet_ckpt = output_dir / "best_mobilenet_v2.pt"
    if mobilenet_ckpt.exists():
        m_mobile = MobileNetV2Classifier(num_classes=10).to(device)
        ckpt = torch.load(mobilenet_ckpt, map_location=device)
        m_mobile.load_state_dict(ckpt["model_state_dict"])
        models_to_eval["MobileNetV2"] = m_mobile

    if not models_to_eval:
        raise RuntimeError(f"No trained checkpoints found in {output_dir}")

    results = {}
    probs_dict = {}
    ground_truth = None

    for name, model in models_to_eval.items():
        probs, preds, targets = get_model_predictions(model, test_loader, device)
        probs_dict[name] = probs
        ground_truth = targets

        acc = float(np.mean(preds == targets))
        macro_f1 = float(f1_score(targets, preds, average="macro"))
        weighted_f1 = float(f1_score(targets, preds, average="weighted"))
        cm = confusion_matrix(targets, preds)
        rep = classification_report(targets, preds, target_names=GENRES, output_dict=True)

        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classification_report": rep,
        }

        # Plot confusion matrix
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        plot_confusion_matrix(cm, name, output_dir / f"cm_{safe_name}.png")

    # If both EnhancedSECNN and MobileNetV2 exist, compute Ensemble
    if "Enhanced SE-CNN" in probs_dict and "MobileNetV2" in probs_dict:
        print("\nComputing Hybrid Ensemble (Enhanced SE-CNN + MobileNetV2)...")
        # Soft voting weighted combination
        ensemble_probs = 0.40 * probs_dict["Enhanced SE-CNN"] + 0.60 * probs_dict["MobileNetV2"]
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        ens_acc = float(np.mean(ensemble_preds == ground_truth))
        ens_macro_f1 = float(f1_score(ground_truth, ensemble_preds, average="macro"))
        ens_weighted_f1 = float(f1_score(ground_truth, ensemble_preds, average="weighted"))
        ens_cm = confusion_matrix(ground_truth, ensemble_preds)
        ens_rep = classification_report(ground_truth, ensemble_preds, target_names=GENRES, output_dict=True)

        results["Ensemble (SE-CNN + MobileNetV2)"] = {
            "accuracy": ens_acc,
            "macro_f1": ens_macro_f1,
            "weighted_f1": ens_weighted_f1,
            "classification_report": ens_rep,
        }
        plot_confusion_matrix(ens_cm, "Ensemble (SE-CNN + MobileNetV2)", output_dir / "cm_ensemble.png")

    # Save comparison chart
    plot_comparison_chart(results, output_dir / "models_comparison_chart.png")

    # Save summary JSON
    with open(output_dir / "test_metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary leaderboard
    print("\n" + "=" * 65)
    print("           HELD-OUT TEST SET EVALUATION LEADERBOARD")
    print("=" * 65)
    print(f"{'Model':<35} | {'Test Acc':<10} | {'Macro F1':<10}")
    print("-" * 65)
    for model_name, metrics in results.items():
        print(f"{model_name:<35} | {metrics['accuracy']*100:6.2f}%    | {metrics['macro_f1']*100:6.2f}%")
    print("=" * 65 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/gtzan/Data")
    parser.add_argument("--output-dir", type=str, default="results_gtzan")
    args = parser.parse_args()

    evaluate_all(Path(args.data_dir), Path(args.output_dir))
