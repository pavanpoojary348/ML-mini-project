"""Transfer-learn FMA Small genre classification with official pretrained BYOL-A embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO = Path.home() / "Downloads" / "fma_small" / "fma_small"
DEFAULT_METADATA = Path.home() / "Downloads" / "fma_metadata" / "fma_metadata"
DEFAULT_BYOL_REPO = ROOT / "work" / "byol-a"
WEIGHTS_NAME = "AudioNTT2020-BYOLA-64x96d2048.pth"
SAMPLE_RATE, UNIT_SEC, N_FFT, HOP, N_MELS = 16000, 0.95, 1024, 160, 64
CLIP = int(SAMPLE_RATE * UNIT_SEC)
CROPS_PER_TRACK, FRAMES = 6, 96


def read_labels(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        h1, h2 = next(reader), next(reader)
        col = {(a, b): i for i, (a, b) in enumerate(zip(h1, h2))}
        out = []
        for row in reader:
            if not row or len(row) <= max(col.values()):
                continue
            if row[col[("set", "subset")]] != "small" or not row[col[("track", "genre_top")]]:
                continue
            out.append({"track_id": int(row[0]), "split": row[col[("set", "split")]],
                        "genre": row[col[("track", "genre_top")]]})
    return sorted(out, key=lambda x: x["track_id"])


def load_or_create_cache(rows: list[dict], audio_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path, status_path = cache_dir / "byola_logmel.npy", cache_dir / "byola_status.npy"
    shape = (len(rows), CROPS_PER_TRACK, N_MELS, FRAMES)
    if path.exists() and status_path.exists():
        cache = np.load(path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        if cache.shape != shape or status.shape != (len(rows),):
            del cache, status
            path.unlink(missing_ok=True); status_path.unlink(missing_ok=True)
    if not path.exists():
        cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        status = np.lib.format.open_memmap(status_path, mode="w+", dtype=np.bool_, shape=(len(rows),))
        status[:] = False
    else:
        cache = np.load(path, mmap_mode="r+"); status = np.load(status_path, mmap_mode="r+")
    mel_filter = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=60, fmax=7800)
    bad = []
    todo = int((~np.asarray(status)).sum())
    print(f"Preparing BYOL-A log-mels for {todo} remaining tracks; checkpoints resume automatically.", flush=True)
    for i, row in enumerate(tqdm(rows, desc="Audio feature extraction")):
        if status[i]:
            continue
        tid = row["track_id"]
        audio_path = audio_dir / f"{tid // 1000:03d}" / f"{tid:06d}.mp3"
        try:
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=30.0)
            if len(audio) < CLIP:
                raise ValueError("audio shorter than the 0.95-second model input")
            for k in range(CROPS_PER_TRACK):
                center = int((k + 0.5) * len(audio) / CROPS_PER_TRACK)
                start = max(0, min(len(audio) - CLIP, center - CLIP // 2))
                clip = audio[start:start + CLIP]
                stft = librosa.stft(clip, n_fft=N_FFT, hop_length=HOP)
                power = np.abs(stft) ** 2
                logmel = np.log(mel_filter @ (power + np.finfo(float).eps))
                # Match the reference input shape [64, 96].
                frames = min(FRAMES, logmel.shape[1])
                cache[i, k, :, :frames] = logmel[:, :frames].astype(np.float16)
                if frames < FRAMES:
                    cache[i, k, :, frames:] = logmel[:, -1:].astype(np.float16)
            status[i] = True
        except Exception as exc:
            bad.append({"track_id": tid, "error": str(exc)})
        if (i + 1) % 100 == 0:
            cache.flush(); status.flush()
            print(f"Processed {i + 1}/{len(rows)} tracks; cached {int(status.sum())}.", flush=True)
    cache.flush(); status.flush()
    if bad:
        (cache_dir / "decode_errors.json").write_text(json.dumps(bad, indent=2), encoding="utf-8")
    return cache, status


class GenreMLP(nn.Module):
    def __init__(self, input_dim=2048, class_count=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.45),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, class_count),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--byol-repo", type=Path, default=DEFAULT_BYOL_REPO)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "byola_results")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_labels(args.metadata_dir / "tracks.csv")
    cache, status = load_or_create_cache(rows, args.audio_dir, args.output_dir / "cache")
    usable = [i for i, ok in enumerate(status) if ok]
    rows = [rows[i] for i in usable]
    raw = np.asarray(cache[usable], dtype=np.float32)
    classes = sorted({r["genre"] for r in rows})
    split_ids = {s: np.asarray([i for i, r in enumerate(rows) if r["split"] == s], dtype=int)
                 for s in ("training", "validation", "test")}
    train_ids, val_ids, test_ids = split_ids["training"], split_ids["validation"], split_ids["test"]
    # BYOL-A's own evaluation recipe estimates log-mel mean/std from training audio only.
    training_rows = raw[train_ids].reshape(-1, N_MELS, FRAMES)
    mean, std = float(training_rows.mean()), float(training_rows.std())
    del training_rows
    raw = (raw - mean) / max(std, 1e-6)

    # Import the official BYOL-A architecture from its public reference repository.
    sys.path.insert(0, str(args.byol_repo))
    from byol_a.models import AudioNTT2020
    weights = args.byol_repo / "pretrained_weights" / WEIGHTS_NAME
    if not weights.exists():
        raise FileNotFoundError(f"Official BYOL-A weights not found: {weights}; see README for setup.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AudioNTT2020(n_mels=N_MELS, d=2048).to(device)
    encoder.load_weight(weights, device)
    encoder.eval()
    flat = torch.from_numpy(raw.reshape(-1, 1, N_MELS, FRAMES))
    embeddings = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(flat), 256), desc="BYOL-A embeddings"):
            embeddings.append(encoder(flat[start:start + 256].to(device)).float().cpu().numpy())
    X = np.concatenate(embeddings).reshape(len(rows), CROPS_PER_TRACK, 2048).mean(axis=1)
    np.save(args.output_dir / "byola_embeddings.npy", X.astype(np.float32))
    y = np.asarray([r["genre"] for r in rows])
    labels_to_id = {name: i for i, name in enumerate(classes)}
    y_id = np.asarray([labels_to_id[name] for name in y], dtype=np.int64)

    # Scale using training only; select best epoch using validation macro-F1.
    scaler = StandardScaler().fit(X[train_ids])
    Xs = scaler.transform(X).astype(np.float32)
    torch.manual_seed(42); np.random.seed(42)
    model = GenreMLP(class_count=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.from_numpy(Xs[train_ids]), torch.from_numpy(y_id[train_ids]))
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    val_x = torch.from_numpy(Xs[val_ids]).to(device)
    val_y = y[val_ids]
    best_score, best_epoch, best_val_acc, patience = -1.0, 0, 0.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_pred = model(val_x).argmax(dim=1).cpu().numpy()
        val_acc = float(accuracy_score(val_y, np.asarray(classes)[val_pred]))
        val_f1 = float(f1_score(val_y, np.asarray(classes)[val_pred], average="macro"))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": val_acc,
                        "validation_macro_f1": val_f1})
        print(f"Epoch {epoch}/{args.epochs}: loss={np.mean(losses):.4f}, val_acc={val_acc:.4f}, val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_score:
            best_score, best_epoch = val_f1, epoch
            best_val_acc = val_acc
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                break
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)

    # Refit from scratch on train+validation for the selected epoch count.
    fit_ids = np.concatenate((train_ids, val_ids))
    scaler = StandardScaler().fit(X[fit_ids]); Xs = scaler.transform(X).astype(np.float32)
    model = GenreMLP(class_count=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    train_ds = TensorDataset(torch.from_numpy(Xs[fit_ids]), torch.from_numpy(y_id[fit_ids]))
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    for epoch in range(best_epoch):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
    model.eval(); test_x = torch.from_numpy(Xs[test_ids]).to(device)
    with torch.inference_mode():
        test_pred_id = model(test_x).argmax(dim=1).cpu().numpy()
    test_pred = np.asarray(classes)[test_pred_id]
    metrics = {
        "model": "BYOL-A AudioNTT2020 pretrained embeddings + MLP",
        "dataset": "FMA Small; official train/validation/test splits",
        "track_count": len(rows), "train_track_count": len(train_ids), "validation_track_count": len(val_ids),
        "test_track_count": len(test_ids), "crops_per_track": CROPS_PER_TRACK,
        "normalization_mean_std_fit_on": "training audio only", "validation_accuracy_at_selected_epoch": best_val_acc,
        "best_epoch": best_epoch, "validation_macro_f1_at_selected_epoch": best_score,
        "test_accuracy": float(accuracy_score(y[test_ids], test_pred)),
        "test_macro_f1": float(f1_score(y[test_ids], test_pred, average="macro")),
        "classes": classes,
        "test_classification_report": classification_report(y[test_ids], test_pred, output_dict=True, zero_division=0),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(y[test_ids], test_pred, labels=classes)
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["true/predicted", *classes])
        for genre, line in zip(classes, cm): writer.writerow([genre, *line.tolist()])
    torch.save({"state_dict": model.state_dict(), "classes": classes, "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_, "encoder": WEIGHTS_NAME}, args.output_dir / "byola_mlp.pt")
    print(f"BYOL-A + MLP test accuracy: {metrics['test_accuracy']:.2%}; macro-F1: {metrics['test_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
