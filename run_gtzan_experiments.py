"""Master Execution Script for GTZAN Genre Classification Project.

Executes the complete experimental pipeline:
1. Trains the Baseline PaperCNN (from the ICCCNT 2024 paper)
2. Trains our Proposed EnhancedSECNN (with Channel Attention & Regularization)
3. Trains MobileNetV2 (Transfer Learning with 2-Phase Fine-Tuning)
4. Evaluates all models on the held-out test split (100 samples)
5. Computes the Hybrid Ensemble (CNN + MobileNetV2)
6. Generates confusion matrices, accuracy curves, and performance reports.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from gtzan_project.evaluate_models import evaluate_all
from gtzan_project.train_cnn import train_model
from gtzan_project.train_mobilenet import train_mobilenet


def main():
    parser = argparse.ArgumentParser(description="Run complete GTZAN Genre Classification pipeline.")
    parser.add_argument("--data-dir", type=str, default="data/gtzan/Data")
    parser.add_argument("--output-dir", type=str, default="results_gtzan")
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=18)
    parser.add_argument("--skip-baseline", action="store_true", help="Skip training the paper baseline CNN")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      MUSIC GENRE CLASSIFICATION - GTZAN AUDIO PIPELINE")
    print("=" * 70)
    print(f"Dataset directory : {data_path}")
    print(f"Results directory : {output_path}\n")

    start_all = time.time()

    # Step 1: Baseline Paper CNN
    if not args.skip_baseline:
        print("\n" + "=" * 50)
        print(" [1/3] Training Baseline PaperCNN (ICCCNT 2024 Reference)")
        print("=" * 50)
        train_model(
            model_type="paper_cnn",
            data_dir=data_path,
            output_dir=output_path,
            epochs=20,
            batch_size=16,
            learning_rate=3e-4,
        )

    # Step 2: Enhanced SECNN (Our proposed architectural improvement)
    print("\n" + "=" * 50)
    print(" [2/3] Training Proposed EnhancedSECNN (Attention + Residual)")
    print("=" * 50)
    train_model(
        model_type="enhanced_secnn",
        data_dir=data_path,
        output_dir=output_path,
        epochs=args.cnn_epochs,
        batch_size=16,
        learning_rate=4e-4,
    )

    # Step 3: MobileNetV2 Transfer Learning
    print("\n" + "=" * 50)
    print(" [3/3] Training MobileNetV2 (2-Phase Transfer Learning)")
    print("=" * 50)
    train_mobilenet(
        data_dir=data_path,
        output_dir=output_path,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=16,
        lr_head=8e-4,
        lr_finetune=1e-4,
    )

    # Step 4: Held-out Test Evaluation & Ensembling
    print("\n" + "=" * 50)
    print(" [Evaluation] Evaluating on Held-out Test Set & Ensembling")
    print("=" * 50)
    results = evaluate_all(data_path, output_path)

    total_time = time.time() - start_all
    print(f"\n>>> Full Pipeline Completed in {total_time/60:.2f} minutes! <<<")
    return results


if __name__ == "__main__":
    main()
