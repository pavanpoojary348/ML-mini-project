"""Preprocess GTZAN audio tracks into 3-second Log-Mel Spectrogram segments.

Each 30-second audio track is segmented into 10 non-overlapping 3-second chunks.
For each chunk, computes a 128-band Log-Mel Spectrogram (dB scale) at 22,050 Hz.
Saves the preprocessed data, genre labels, and song IDs into a compact .npz file
for ultra-fast training and song-level ensemble evaluation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from gtzan_project.dataset import GENRE_TO_IDX, GENRES

SAMPLE_RATE = 22050
DURATION_PER_CHUNK = 3.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * DURATION_PER_CHUNK)  # 66,150 samples
N_CHUNKS_PER_SONG = 10
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def extract_chunks_mels(
    audio_path: Path,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
) -> Tuple[np.ndarray, bool]:
    """Reads audio file, splits into 10 3-second chunks, and computes log-mel spectrograms."""
    try:
        data, sr = sf.read(str(audio_path), dtype="float32")
    except Exception:
        return np.empty((0,)), False

    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if sample rate differs
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        data = resampler(torch.from_numpy(data)).numpy()

    # Audio length check
    if len(data) < CHUNK_SAMPLES:
        return np.empty((0,)), False

    chunks_mels = []
    for i in range(N_CHUNKS_PER_SONG):
        start = i * CHUNK_SAMPLES
        end = start + CHUNK_SAMPLES
        if end > len(data):
            chunk = np.pad(data[start:], (0, end - len(data)), mode="wrap")
        else:
            chunk = data[start:end]

        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)  # (1, 66150)
        mel_spec = mel_transform(chunk_tensor)               # (1, 128, 130)
        log_mel = amplitude_to_db(mel_spec).squeeze(0).numpy() # (128, 130)
        chunks_mels.append(log_mel.astype(np.float32))

    return np.stack(chunks_mels, axis=0), True  # (10, 128, 130)


def build_3s_mels_dataset(
    audio_root: Path,
    output_npz_path: Path,
) -> Dict[str, int]:
    """Scans genres_original, extracts 3-second log-mel spectrograms, and saves as NPZ."""
    print("=" * 65)
    print("  Extracting 3-Second Log-Mel Spectrograms from GTZAN Audio")
    print("=" * 65)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    all_specs: List[np.ndarray] = []
    all_labels: List[int] = []
    all_song_ids: List[str] = []

    valid_songs_count = 0
    corrupted_count = 0
    t0 = time.time()

    for genre in GENRES:
        genre_dir = audio_root / genre
        if not genre_dir.exists():
            continue
        genre_idx = GENRE_TO_IDX[genre]
        audio_files = sorted(genre_dir.glob("*.wav"))

        for audio_file in tqdm(audio_files, desc=f"Processing {genre:<10}"):
            song_id = f"{genre}.{audio_file.stem.split('.')[-1]}"
            mels, success = extract_chunks_mels(audio_file, mel_transform, amplitude_to_db)
            if not success:
                corrupted_count += 1
                continue

            # mels shape: (10, 128, 130)
            for chunk_idx in range(mels.shape[0]):
                all_specs.append(mels[chunk_idx])
                all_labels.append(genre_idx)
                all_song_ids.append(f"{song_id}_{chunk_idx}")

            valid_songs_count += 1

    specs_arr = np.stack(all_specs, axis=0)  # (N, 128, 130)
    labels_arr = np.array(all_labels, dtype=np.int64)
    song_ids_arr = np.array(all_song_ids)

    print(f"\nExtraction completed in {time.time()-t0:.1f}s.")
    print(f"Valid songs processed : {valid_songs_count} (Corrupted/skipped: {corrupted_count})")
    print(f"Total 3-sec spectrograms : {specs_arr.shape[0]}")
    print(f"Spectrogram shape per chunk : {specs_arr.shape[1:]}")

    # Normalize globally (zero mean, unit variance)
    mean_val = float(specs_arr.mean())
    std_val = float(specs_arr.std())
    print(f"Global dataset stats: mean={mean_val:.2f}, std={std_val:.2f}")

    np.savez_compressed(
        output_npz_path,
        specs=specs_arr.astype(np.float32),
        labels=labels_arr,
        song_ids=song_ids_arr,
        mean=mean_val,
        std=std_val,
    )
    print(f"Saved dataset cache to: {output_npz_path} ({output_npz_path.stat().st_size / 1e6:.1f} MB)")

    return {
        "valid_songs": valid_songs_count,
        "corrupted_songs": corrupted_count,
        "total_chunks": int(specs_arr.shape[0]),
    }


if __name__ == "__main__":
    audio_dir = Path("data/gtzan/Data/genres_original")
    out_file = Path("data/gtzan/mel_spectrograms_3s.npz")
    build_3s_mels_dataset(audio_dir, out_file)
