"""Extract MERT music embeddings from FMA Small and train a genre classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO = Path.home() / "Downloads" / "fma_small" / "fma_small"
DEFAULT_METADATA = Path.home() / "Downloads" / "fma_metadata" / "fma_metadata"
MODEL_ID = "m-a-p/MERT-v1-95M"
SAMPLE_RATE = 24000
CLIP_SECONDS = 5
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS


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


def extract(audio_dir: Path, rows: list[dict], out_dir: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path, status_path = out_dir / "mert_embeddings.npy", out_dir / "mert_status.npy"
    shape = (len(rows), 768)
    if cache_path.exists() and status_path.exists():
        cache = np.load(cache_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        if cache.shape != shape or status.shape != (len(rows),):
            del cache, status
            cache_path.unlink(missing_ok=True)
            status_path.unlink(missing_ok=True)
    if not cache_path.exists():
        cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=shape)
        status = np.lib.format.open_memmap(status_path, mode="w+", dtype=np.bool_, shape=(len(rows),))
        status[:] = False
    else:
        cache = np.load(cache_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")

    print(f"Loading pretrained {MODEL_ID} on {device}...", flush=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
    bad = []
    for i, row in enumerate(tqdm(rows, desc="MERT embeddings")):
        if status[i]:
            continue
        tid = row["track_id"]
        path = audio_dir / f"{tid // 1000:03d}" / f"{tid:06d}.mp3"
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=30.0)
            if audio.size < CLIP_SAMPLES:
                raise ValueError("audio shorter than 5 seconds")
            # Use six non-overlapping 5-second crops; MERT was pretrained with a 5-second context.
            crops = [audio[j * CLIP_SAMPLES:(j + 1) * CLIP_SAMPLES] for j in range(6)]
            batch = processor(crops, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(**inputs, output_hidden_states=True)
                # Average the last four representation layers, then average across time and song crops.
                layers = torch.stack(output.hidden_states[-4:]).float().mean(dim=0)
                embedding = layers.mean(dim=1).mean(dim=0)
            cache[i] = embedding.float().cpu().numpy().astype(np.float16)
            status[i] = True
        except Exception as exc:
            bad.append({"track_id": tid, "error": str(exc)})
        if (i + 1) % 50 == 0:
            cache.flush(); status.flush()
            print(f"Processed {i + 1}/{len(rows)} tracks; cached {int(status.sum())}.", flush=True)
    cache.flush(); status.flush()
    if bad:
        (out_dir / "decode_errors.json").write_text(json.dumps(bad, indent=2), encoding="utf-8")
    del model, processor
    return np.asarray(cache), np.asarray(status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "mert_results")
    args = parser.parse_args()
    rows = read_labels(args.metadata_dir / "tracks.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings, status = extract(args.audio_dir, rows, args.output_dir, device)
    usable = [i for i, ok in enumerate(status) if ok]
    rows = [rows[i] for i in usable]
    X = embeddings[usable].astype(np.float32)
    classes = sorted({r["genre"] for r in rows})
    splits = {s: np.asarray([i for i, r in enumerate(rows) if r["split"] == s], dtype=int)
              for s in ("training", "validation", "test")}
    train_i, val_i, test_i = splits["training"], splits["validation"], splits["test"]
    y = np.asarray([r["genre"] for r in rows])
    scores = {}
    candidates = {}
    for c in (0.1, 0.3, 1, 3, 10):
        clf = make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", class_weight="balanced"))
        clf.fit(X[train_i], y[train_i])
        pred = clf.predict(X[val_i])
        scores[str(c)] = {"accuracy": float(accuracy_score(y[val_i], pred)),
                          "macro_f1": float(f1_score(y[val_i], pred, average="macro"))}
        candidates[c] = clf
        print(f"C={c}: validation accuracy={scores[str(c)]['accuracy']:.4f}, macro-F1={scores[str(c)]['macro_f1']:.4f}", flush=True)
    best_c = max(candidates, key=lambda c: (scores[str(c)]["macro_f1"], scores[str(c)]["accuracy"]))
    best = candidates[best_c]
    best.fit(X[np.concatenate((train_i, val_i))], y[np.concatenate((train_i, val_i))])
    pred = best.predict(X[test_i])
    metrics = {
        "model": MODEL_ID, "embedding": "mean of last four hidden layers, temporal and six 5-second crop mean",
        "track_count": len(rows), "train_count": len(train_i), "validation_count": len(val_i), "test_count": len(test_i),
        "validation_candidates": scores, "selected_C": best_c,
        "test_accuracy": float(accuracy_score(y[test_i], pred)),
        "test_macro_f1": float(f1_score(y[test_i], pred, average="macro")),
        "classes": classes,
        "test_classification_report": classification_report(y[test_i], pred, output_dict=True, zero_division=0),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(y[test_i], pred, labels=classes)
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["true/predicted", *classes])
        for genre, line in zip(classes, cm): writer.writerow([genre, *line.tolist()])
    import joblib
    joblib.dump({"model": best, "classes": classes, "backbone": MODEL_ID}, args.output_dir / "mert_svm.joblib")
    print(f"MERT-SVM test accuracy: {metrics['test_accuracy']:.2%}; macro-F1: {metrics['test_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
