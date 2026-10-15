"""Train a song-level CNN directly on FMA Small MP3 audio."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TARGET_FRAMES = 1292
SEGMENT_FRAMES = 215
SEGMENTS_PER_TRACK = 6
TRAIN_SEGMENTS_PER_TRACK = 3

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO = Path.home() / "Downloads" / "fma_small" / "fma_small"
DEFAULT_METADATA = Path.home() / "Downloads" / "fma_metadata" / "fma_metadata"


def read_small_labels(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header_a = next(reader)
        header_b = next(reader)
        columns = {
            (a, b): i for i, (a, b) in enumerate(zip(header_a, header_b))
        }
        try:
            subset_i = columns[("set", "subset")]
            split_i = columns[("set", "split")]
            genre_i = columns[("track", "genre_top")]
        except KeyError as exc:
            raise RuntimeError(f"Required FMA metadata column is missing: {exc}") from exc
        rows = []
        for values in reader:
            if not values or len(values) <= max(subset_i, split_i, genre_i):
                continue
            if values[subset_i] != "small" or values[genre_i] == "":
                continue
            rows.append(
                {
                    "track_id": int(values[0]),
                    "split": values[split_i],
                    "genre": values[genre_i],
                }
            )
    return sorted(rows, key=lambda row: row["track_id"])


def build_cache(rows: list[dict], audio_dir: Path, cache_dir: Path) -> tuple[Path, list[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "fma_small_logmel.npy"
    status_file = cache_dir / "cache_status.npy"
    shape = (len(rows), N_MELS, TARGET_FRAMES)

    if cache_file.exists() and status_file.exists():
        cache = np.load(cache_file, mmap_mode="r+")
        status = np.load(status_file, mmap_mode="r+")
        if cache.shape != shape or status.shape != (len(rows),):
            del cache, status
            cache_file.unlink(missing_ok=True)
            status_file.unlink(missing_ok=True)
            cache = np.lib.format.open_memmap(cache_file, mode="w+", dtype=np.float16, shape=shape)
            status = np.lib.format.open_memmap(status_file, mode="w+", dtype=np.bool_, shape=(len(rows),))
            status[:] = False
    else:
        cache = np.lib.format.open_memmap(cache_file, mode="w+", dtype=np.float16, shape=shape)
        status = np.lib.format.open_memmap(status_file, mode="w+", dtype=np.bool_, shape=(len(rows),))
        status[:] = False

    bad = []
    remaining = int((~np.asarray(status)).sum())
    print(f"Preparing log-mel cache for {remaining} tracks (resumes completed tracks).", flush=True)
    for i, row in enumerate(tqdm(rows, desc="Audio preprocessing")):
        if status[i]:
            continue
        track_id = row["track_id"]
        audio_path = audio_dir / f"{track_id // 1000:03d}" / f"{track_id:06d}.mp3"
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=30.0)
            if audio.size < SAMPLE_RATE:
                raise ValueError("audio is shorter than one second")
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmin=30,
                fmax=SAMPLE_RATE / 2,
                power=2.0,
            )
            db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
            scaled = np.clip((db + 80.0) / 80.0, 0.0, 1.0)
            frames = min(scaled.shape[1], TARGET_FRAMES)
            cache[i, :, :frames] = scaled[:, :frames].astype(np.float16)
            if frames < TARGET_FRAMES:
                cache[i, :, frames:] = scaled[:, max(0, frames - 1):frames].astype(np.float16)
            status[i] = True
        except Exception as exc:
            bad.append({"track_id": track_id, "error": str(exc)})
        if (i + 1) % 100 == 0:
            cache.flush()
            status.flush()
            print(f"Processed {i + 1}/{len(rows)} tracks; cached {int(status.sum())}.", flush=True)
    cache.flush()
    status.flush()
    if bad:
        with (cache_dir / "decode_errors.json").open("w", encoding="utf-8") as stream:
            json.dump(bad, stream, indent=2)
        print(f"Warning: {len(bad)} audio files could not be decoded; details saved in decode_errors.json", flush=True)
    usable = [{**row, "cache_index": i} for i, row in enumerate(rows) if status[i]]
    print(f"Cached {len(usable)} of {len(rows)} tracks.", flush=True)
    return cache_file, usable


class TrackSegments(Dataset):
    def __init__(self, cache_path: Path, rows: list[dict], class_to_id: dict[str, int], train: bool):
        self.cache_path = cache_path
        self.rows = rows
        self.class_to_id = class_to_id
        self.train = train
        self.cache = None
        self.selected = []
        self.set_epoch(0)

    def open_cache(self):
        if self.cache is None:
            self.cache = np.load(self.cache_path, mmap_mode="r")

    def set_epoch(self, epoch: int):
        rng = np.random.default_rng(9000 + epoch)
        self.selected = []
        count = TRAIN_SEGMENTS_PER_TRACK if self.train else SEGMENTS_PER_TRACK
        for track_row in self.rows:
            if self.train:
                segments = sorted(rng.choice(SEGMENTS_PER_TRACK, size=count, replace=False).tolist())
            else:
                segments = list(range(SEGMENTS_PER_TRACK))
            for segment in segments:
                self.selected.append((track_row["cache_index"], segment, self.class_to_id[track_row["genre"]], track_row["track_id"]))

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, index: int):
        self.open_cache()
        cache_index, segment, label, track_id = self.selected[index]
        start = segment * SEGMENT_FRAMES
        stop = start + SEGMENT_FRAMES
        spec = np.asarray(self.cache[cache_index, :, start:stop], dtype=np.float32).copy()
        return torch.from_numpy(spec).unsqueeze(0), label, track_id


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class AudioGenreCNN(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, 0.08),
            ConvBlock(32, 64, 0.12),
            ConvBlock(64, 128, 0.18),
            ConvBlock(128, 192, 0.22),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, class_count),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(specs: torch.Tensor) -> torch.Tensor:
    shift = random.randint(-7, 7)
    if shift:
        specs = torch.roll(specs, shifts=shift, dims=-1)
    if random.random() < 0.8:
        width = random.randint(4, 14)
        start = random.randint(0, specs.shape[-1] - width)
        specs[:, :, :, start : start + width] = 0
    if random.random() < 0.8:
        width = random.randint(4, 14)
        start = random.randint(0, specs.shape[-2] - width)
        specs[:, :, start : start + width, :] = 0
    return specs


@torch.no_grad()
def evaluate(model, loader, device, expected_track_ids: list[int], id_to_label: dict[int, int]):
    model.eval()
    total = np.zeros((len(expected_track_ids), model.classifier[-1].out_features), dtype=np.float64)
    counts = np.zeros(len(expected_track_ids), dtype=np.int32)
    id_to_row = {track_id: i for i, track_id in enumerate(expected_track_ids)}
    for specs, _, track_ids in loader:
        specs = specs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probs = torch.softmax(model(specs), dim=1).float().cpu().numpy()
        for row, track_id in enumerate(track_ids.tolist()):
            out_i = id_to_row[track_id]
            total[out_i] += probs[row]
            counts[out_i] += 1
    averaged = total / np.maximum(counts[:, None], 1)
    y_true = np.asarray([id_to_label[x] for x in expected_track_ids])
    y_pred = averaged.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "work" / "fma_mel_cache")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "audio_results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if not (args.audio_dir.exists() and (args.metadata_dir / "tracks.csv").exists()):
        raise FileNotFoundError("FMA Small audio or tracks.csv not found. Check --audio-dir and --metadata-dir.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    rows = read_small_labels(args.metadata_dir / "tracks.csv")
    classes = sorted({row["genre"] for row in rows})
    class_to_id = {genre: index for index, genre in enumerate(classes)}
    cache_path, usable = build_cache(rows, args.audio_dir, args.cache_dir)
    train_rows = [row for row in usable if row["split"] == "training"]
    valid_rows = [row for row in usable if row["split"] == "validation"]
    test_rows = [row for row in usable if row["split"] == "test"]
    id_to_label = {row["track_id"]: class_to_id[row["genre"]] for row in usable}
    print(f"Tracks with audio: train={len(train_rows)}, validation={len(valid_rows)}, test={len(test_rows)}; classes={classes}.", flush=True)

    train_data = TrackSegments(cache_path, train_rows, class_to_id, train=True)
    valid_data = TrackSegments(cache_path, valid_rows, class_to_id, train=False)
    test_data = TrackSegments(cache_path, test_rows, class_to_id, train=False)
    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=use_pin)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin)

    model = AudioGenreCNN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, patience = -1.0, 0, 0
    history = []
    checkpoint = args.output_dir / "best_audio_cnn.pt"
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        model.train()
        loss_sum, seen = 0.0, 0
        for specs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            specs = augment(specs.to(device, non_blocking=True))
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(specs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * targets.size(0)
            seen += targets.size(0)

        validation = evaluate(model, valid_loader, device, [r["track_id"] for r in valid_rows], id_to_label)
        scheduler.step()
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch}: loss={epoch_row['train_loss']:.4f}, "
            f"val_accuracy={validation['accuracy']:.4f}, val_macro_f1={validation['macro_f1']:.4f}",
            flush=True,
        )
        if validation["macro_f1"] > best_score:
            best_score, best_epoch, patience = validation["macro_f1"], epoch, 0
            torch.save(
                {"state_dict": model.state_dict(), "classes": classes, "config": {
                    "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
                    "n_mels": N_MELS, "segment_frames": SEGMENT_FRAMES,
                }, "best_validation": epoch_row},
                checkpoint,
            )
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.", flush=True)
                break

    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    final_test = evaluate(model, test_loader, device, [r["track_id"] for r in test_rows], id_to_label)
    matrix = confusion_matrix(final_test["y_true"], final_test["y_pred"], labels=np.arange(len(classes)))
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *classes])
        for genre, row in zip(classes, matrix):
            writer.writerow([genre, *row.tolist()])
    report = classification_report(final_test["y_true"], final_test["y_pred"], target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        "model": "AudioGenreCNN",
        "dataset": "FMA Small",
        "input": "5-second log-mel spectrogram crops from the 30-second MP3 tracks",
        "best_epoch": best_epoch,
        "train_track_count": len(train_rows),
        "validation_track_count": len(valid_rows),
        "test_track_count": len(test_rows),
        "segments_per_training_track_per_epoch": TRAIN_SEGMENTS_PER_TRACK,
        "validation_accuracy_at_best_epoch": saved["best_validation"]["validation_accuracy"],
        "validation_macro_f1_at_best_epoch": saved["best_validation"]["validation_macro_f1"],
        "test_accuracy": final_test["accuracy"],
        "test_macro_f1": final_test["macro_f1"],
        "classes": classes,
        "test_classification_report": report,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(f"\nAudio CNN test accuracy: {final_test['accuracy']:.2%}; macro-F1: {final_test['macro_f1']:.4f}.", flush=True)
    print(f"Saved model and results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
