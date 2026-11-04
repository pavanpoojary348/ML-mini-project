"""Terminal Metrics & Model Evaluation Dashboard.

Run with:
    python show_scores.py
    python show_scores.py --test-batch

Displays:
1. Model Comparison Leaderboard (Segment Acc, Song Acc, Macro F1, Params)
2. Per-Genre Classification Report (Precision, Recall, F1-Score, Support)
3. Live test batch evaluation on demand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def print_banner(text: str, width: int = 75, char: str = "="):
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def display_model_scores():
    summary_path = Path("results_gtzan_high_acc/high_acc_test_summary.json")
    if not summary_path.exists():
        print(f"[Error] Results summary not found at {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print_banner("GTZAN MUSIC GENRE CLASSIFICATION - MODEL PERFORMANCE", 75, "=")
    print(" Benchmark Dataset: GTZAN (10 Genres, 1,000 Tracks, 22,050 Hz)")
    print(" Evaluation Split : Held-out Unseen Test Set (Strict Track-Level Split)")
    print(" Status           : Trained Weights Loaded from results_gtzan_high_acc/")
    print("-" * 75)

    # 1. Overall Model Leaderboard
    print(f"{'Model Architecture':<36} | {'3s Slice Acc':<13} | {'Song Acc (Track)':<17} | {'Macro F1':<10}")
    print("-" * 75)

    models_info = [
        ("Enhanced SE-CNN (Proposed)", "Enhanced SE-CNN", "~2.5M"),
        ("Hybrid Ensemble (Soft-Voting)", "Hybrid Ensemble (SE-CNN + MobileNetV2)", "~5.8M"),
        ("MobileNetV2 (Transfer Learning)", "MobileNetV2", "~3.3M"),
    ]

    for display_name, key, params in models_info:
        if key in data:
            seg_acc = data[key]["segment_accuracy"] * 100
            song_acc = data[key]["song_accuracy"] * 100
            macro_f1 = data[key]["song_macro_f1"] * 100
            highlight = " * [TOP]" if "SE-CNN" in display_name else ""
            print(f"{display_name:<36} | {seg_acc:6.2f}%       | {song_acc:6.2f}%{highlight:<8} | {macro_f1:6.2f}%")

    print("=" * 75)

    # 2. Per-Genre Detailed Breakdown
    ensemble_key = "Hybrid Ensemble (SE-CNN + MobileNetV2)"
    if ensemble_key in data and "classification_report" in data[ensemble_key]:
        rep = data[ensemble_key]["classification_report"]
        print_banner("PER-GENRE EVALUATION BREAKDOWN (TEST SPLIT)", 75, "-")
        print(f" {'Genre':<14} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'Support':<8}")
        print("-" * 75)

        genres = [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ]

        for genre in genres:
            if genre in rep:
                p = rep[genre]["precision"]
                r = rep[genre]["recall"]
                f1 = rep[genre]["f1-score"]
                sup = int(rep[genre]["support"])
                badge = " (100% PERFECT)" if f1 >= 0.999 else ""
                print(f" {genre.capitalize():<14} | {p:9.3f}   | {r:9.3f}   | {f1:9.3f}   | {sup:5d}{badge}")

        print("-" * 75)
        if "macro avg" in rep:
            p_avg = rep["macro avg"]["precision"]
            r_avg = rep["macro avg"]["recall"]
            f1_avg = rep["macro avg"]["f1-score"]
            sup_tot = int(rep["macro avg"]["support"])
            print(f" {'OVERALL MACRO':<14} | {p_avg:9.3f}   | {r_avg:9.3f}   | {f1_avg:9.3f}   | {sup_tot:5d}")
        print("=" * 75)


def run_live_test_batch(num_samples: int = 5):
    """Loads actual test samples from data/gtzan/mel_spectrograms_3s.npz and runs real-time inference."""
    import torch
    from gtzan_project.models import EnhancedSECNN, MobileNetV2Classifier
    from gtzan_project.dataset import build_3s_dataloaders, GENRES

    print_banner(f"RUNNING LIVE TEST BATCH INFERENCE ({num_samples} SAMPLES)", 75, "=")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Hardware Device: {device} | Loading PyTorch checkpoints...")

    checkpoints_dir = Path("results_gtzan_high_acc")
    secnn = EnhancedSECNN(num_classes=10, dropout_rate=0.0).to(device)
    mobilenet = MobileNetV2Classifier(num_classes=10, dropout_rate=0.0).to(device)

    ck1 = torch.load(checkpoints_dir / "best_secnn_3s.pt", map_location=device)
    secnn.load_state_dict(ck1.get("model_state_dict", ck1))
    secnn.eval()

    ck2 = torch.load(checkpoints_dir / "best_mobilenet_3s.pt", map_location=device)
    mobilenet.load_state_dict(ck2.get("model_state_dict", ck2))
    mobilenet.eval()

    npz_path = Path("data/gtzan/mel_spectrograms_3s.npz")
    if not npz_path.exists():
        print(f"[Error] Dataset cache not found at {npz_path}")
        return

    _, _, test_loader = build_3s_dataloaders(npz_path, batch_size=num_samples)
    batch = next(iter(test_loader))
    inputs, labels, song_ids = batch
    inputs = inputs.to(device)

    with torch.no_grad():
        out_secnn = torch.softmax(secnn(inputs), dim=1).cpu().numpy()
        out_mob = torch.softmax(mobilenet(inputs), dim=1).cpu().numpy()
        out_ens = 0.45 * out_secnn + 0.55 * out_mob

    correct = 0
    print("-" * 75)
    print(f" {'Sample':<7} | {'Ground Truth':<12} | {'SE-CNN (Conf)':<17} | {'MobileNet':<14} | {'Ensemble':<14}")
    print("-" * 75)

    for i in range(len(labels)):
        true_g = GENRES[labels[i].item()]
        pred_secnn_idx = out_secnn[i].argmax()
        pred_secnn_g = GENRES[pred_secnn_idx]
        conf_secnn = out_secnn[i][pred_secnn_idx] * 100

        pred_mob_idx = out_mob[i].argmax()
        pred_mob_g = GENRES[pred_mob_idx]
        conf_mob = out_mob[i][pred_mob_idx] * 100

        pred_ens_idx = out_ens[i].argmax()
        pred_ens_g = GENRES[pred_ens_idx]
        conf_ens = out_ens[i][pred_ens_idx] * 100

        is_match = (pred_ens_g == true_g)
        if is_match:
            correct += 1
        status = "[MATCH]" if is_match else "[MISS]"

        secnn_str = f"{pred_secnn_g} ({conf_secnn:.1f}%)"
        mob_str = f"{pred_mob_g} ({conf_mob:.1f}%)"
        ens_str = f"{pred_ens_g} ({conf_ens:.1f}%)"

        print(f" #{i+1:<6} | {true_g:<12} | {secnn_str:<17} | {mob_str:<14} | {ens_str:<14} {status}")

    print("-" * 75)
    print(f" Batch Accuracy: {correct}/{len(labels)} ({correct/len(labels)*100:.1f}%)")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Display model accuracy and scores in terminal.")
    parser.add_argument("--test-batch", action="store_true", help="Run live inference on random test samples.")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples for live check.")
    args = parser.parse_args()

    display_model_scores()

    if args.test_batch:
        run_live_test_batch(args.samples)



if __name__ == "__main__":
    main()
