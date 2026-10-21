"""Inference Script for Music Genre Classification.

Classifies any given audio track (.wav, .mp3) or spectrogram image (.png)
into one of the 10 GTZAN genres using the trained Hybrid Ensemble model.

Usage:
  python predict_genre.py --input "path/to/song.wav"
  python predict_genre.py --input "path/to/spectrogram.png"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gtzan_project.dataset import GENRES, NORM_MEAN, NORM_STD
from gtzan_project.models import EnhancedSECNN, MobileNetV2Classifier


def load_models(
    weights_dir: Path, device: torch.device
) -> Tuple[EnhancedSECNN, MobileNetV2Classifier]:
    """Loads trained weights for Enhanced SE-CNN and MobileNetV2."""
    secnn = EnhancedSECNN(num_classes=10, dropout_rate=0.0).to(device)
    mobilenet = MobileNetV2Classifier(num_classes=10, dropout_rate=0.0).to(device)

    secnn_path = weights_dir / "best_secnn_3s.pt"
    mobilenet_path = weights_dir / "best_mobilenet_3s.pt"

    if secnn_path.exists():
        ckpt = torch.load(secnn_path, map_location=device)
        secnn.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"[Warning] SECNN checkpoint not found at {secnn_path}")

    if mobilenet_path.exists():
        ckpt = torch.load(mobilenet_path, map_location=device)
        mobilenet.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"[Warning] MobileNetV2 checkpoint not found at {mobilenet_path}")

    secnn.eval()
    mobilenet.eval()
    return secnn, mobilenet


def predict_audio(
    audio_path: Path,
    secnn: EnhancedSECNN,
    mobilenet: MobileNetV2Classifier,
    device: torch.device,
) -> Dict[str, float]:
    """Extracts 3-second chunks from audio and outputs ensemble probabilities."""
    import soundfile as sf
    import torchaudio

    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    target_sr = 22050
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        data = resampler(torch.from_numpy(data)).numpy()

    chunk_samples = int(target_sr * 3.0)
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    n_chunks = max(1, len(data) // chunk_samples)
    all_chunk_probs = []

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * chunk_samples
            chunk = data[start : start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="wrap")

            chunk_t = torch.from_numpy(chunk).unsqueeze(0)
            log_mel = amp_to_db(mel_tf(chunk_t)).squeeze(0).numpy()

            # Global standardize (mean=4.88, std=15.84 from dataset statistics)
            norm_spec = (log_mel - 4.88) / 15.84
            spec_t = torch.from_numpy(norm_spec).unsqueeze(0).float()
            spec_t = F.interpolate(spec_t.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
            spec_t = spec_t.repeat(3, 1, 1).unsqueeze(0).to(device)

            p_secnn = F.softmax(secnn(spec_t), dim=1).cpu().numpy()[0]
            p_mobile = F.softmax(mobilenet(spec_t), dim=1).cpu().numpy()[0]
            p_ens = 0.45 * p_secnn + 0.55 * p_mobile
            all_chunk_probs.append(p_ens)

    avg_probs = np.mean(all_chunk_probs, axis=0)
    return {genre: float(avg_probs[idx]) for idx, genre in enumerate(GENRES)}


def predict_image(
    image_path: Path,
    secnn: EnhancedSECNN,
    mobilenet: MobileNetV2Classifier,
    device: torch.device,
) -> Dict[str, float]:
    """Predicts genre for a 224x224 RGB spectrogram image."""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        img_arr[:, :, c] = (img_arr[:, :, c] - NORM_MEAN[c]) / NORM_STD[c]
    tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        p_secnn = F.softmax(secnn(tensor), dim=1).cpu().numpy()[0]
        p_mobile = F.softmax(mobilenet(tensor), dim=1).cpu().numpy()[0]
        p_ens = 0.45 * p_secnn + 0.55 * p_mobile

    return {genre: float(p_ens[idx]) for idx, genre in enumerate(GENRES)}


def main():
    parser = argparse.ArgumentParser(description="Predict music genre for an audio file or spectrogram.")
    parser.add_argument("--input", type=str, required=True, help="Path to audio file (.wav, .mp3) or spectrogram (.png)")
    parser.add_argument("--weights-dir", type=str, default="results_gtzan_high_acc")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    secnn, mobilenet = load_models(Path(args.weights_dir), device)

    suffix = input_path.suffix.lower()
    if suffix in [".wav", ".mp3", ".ogg", ".flac"]:
        probabilities = predict_audio(input_path, secnn, mobilenet, device)
    elif suffix in [".png", ".jpg", ".jpeg"]:
        probabilities = predict_image(input_path, secnn, mobilenet, device)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    sorted_genres = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_genre, top_prob = sorted_genres[0]

    print("\n" + "=" * 50)
    print(f"File: {input_path.name}")
    print(f"Top Predicted Genre : {top_genre.upper()} ({top_prob * 100:.2f}%)")
    print("=" * 50)
    print("Full Probability Breakdown:")
    for genre, prob in sorted_genres:
        bar_len = int(prob * 30)
        bar_str = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {genre:<12} | {bar_str} | {prob * 100:5.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
