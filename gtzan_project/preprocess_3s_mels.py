"""Preprocess GTZAN audio tracks into 3-second Log-Mel Spectrogram segments.

Audio Signal Processing Workflow:
1. Resamples raw audio to standard 22,050 Hz mono.
2. Segments each 30-second track into 10 non-overlapping 3-second slices (66,150 samples).
3. Applies Short-Time Fourier Transform (STFT) with n_fft=2048 and hop_length=512.
4. Projects frequencies onto 128 Mel frequency filter banks.
5. Converts power amplitudes to logarithmic decibel scale (dB) with top_db=80.
6. Saves processed arrays into a compressed NPZ file for ultra-fast training.
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


# ==============================================================================
# SECTION 1: AUDIO SIGNAL PROCESSING CONSTANTS
# ==============================================================================
# - SAMPLE_RATE: 22,050 Hz is the standard Nyquist rate for music (covers up to 11 kHz).
# - DURATION_PER_CHUNK: 3.0 seconds provides sufficient temporal context for rhythm.
# - CHUNK_SAMPLES: 22050 * 3.0 = 66,150 audio sample points per segment.
# - N_FFT: 2048-point Fast Fourier Transform gives high frequency resolution.
# - HOP_LENGTH: 512 samples between consecutive Fourier frames (gives 130 time frames).
# - N_MELS: 128 Mel bands map linear Hertz to the non-linear human auditory pitch scale.
# ==============================================================================

SAMPLE_RATE = 22050
DURATION_PER_CHUNK = 3.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * DURATION_PER_CHUNK)  # 66,150 samples
N_CHUNKS_PER_SONG = 10
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


# ==============================================================================
# SECTION 2: AUDIO SLICING & LOG-MEL EXTRACTION
# ==============================================================================
# What it does:
#   1. Reads audio file using soundfile.
#   2. Downmixes stereo to mono by averaging channels.
#   3. Resamples to 22,050 Hz if needed.
#   4. Loops 10 times to cut 10 consecutive 3-second chunks.
#   5. Applies MelSpectrogram and AmplitudeToDB to generate (128, 130) matrices.
# ==============================================================================

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

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if original sample rate differs from 22,050 Hz
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        data = resampler(torch.from_numpy(data)).numpy()

    # Minimum audio length safety check
    if len(data) < CHUNK_SAMPLES:
        return np.empty((0,)), False

    chunks_mels = []
    # Extract 10 consecutive non-overlapping 3-second slices
    for i in range(N_CHUNKS_PER_SONG):
        start = i * CHUNK_SAMPLES
        end = start + CHUNK_SAMPLES
        if end > len(data):
            chunk = np.pad(data[start:], (0, end - len(data)), mode="wrap")
        else:
            chunk = data[start:end]

        # Convert chunk to tensor: (1, 66150)
        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
        # Compute Mel Spectrogram: (1, 128, 130)
        mel_spec = mel_transform(chunk_tensor)
        # Convert to Decibels (dB scale)
        log_mel = amplitude_to_db(mel_spec).squeeze(0).numpy()
        chunks_mels.append(log_mel.astype(np.float32))

    return np.stack(chunks_mels, axis=0), True  # Shape: (10, 128, 130)


# ==============================================================================
# SECTION 3: BATCH DATASET PREPARATION & NPZ CACHING
# ==============================================================================
# What it does:
#   1. Scans all 10 genre folders in `genres_original`.
#   2. Iterates over 1,000 `.wav` files and tracks progress with tqdm.
#   3. Generates 9,990 total 3-second spectrogram slices.
#   4. Computes global dataset mean and standard deviation.
#   5. Saves all arrays into a compressed `.npz` archive (~160 MB).
# ==============================================================================

def build_3s_mels_dataset(
    audio_root: Path,
    output_npz_path: Path,
) -> Dict[str, int]:
    """Scans genres_original, extracts 3-second log-mel spectrograms, and saves as NPZ."""
    print("=" * 65)
    print("  Extracting 3-Second Log-Mel Spectrograms from GTZAN Audio")
    print("=" * 65)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Torchaudio Mel transform
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

    # Iterate through all 10 genre directories
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

            # Store each 3s slice with its corresponding song ID and genre label
            for chunk_idx in range(mels.shape[0]):
                all_specs.append(mels[chunk_idx])
                all_labels.append(genre_idx)
                all_song_ids.append(f"{song_id}_{chunk_idx}")

            valid_songs_count += 1

    specs_arr = np.stack(all_specs, axis=0)  # Shape: (9990, 128, 130)
    labels_arr = np.array(all_labels, dtype=np.int64)
    song_ids_arr = np.array(all_song_ids)

    print(f"\nExtraction completed in {time.time()-t0:.1f}s.")
    print(f"Valid songs processed : {valid_songs_count} (Corrupted/skipped: {corrupted_count})")
    print(f"Total 3-sec spectrograms : {specs_arr.shape[0]}")
    print(f"Spectrogram shape per chunk : {specs_arr.shape[1:]}")

    # Compute global normalization constants across the whole dataset
    mean_val = float(specs_arr.mean())
    std_val = float(specs_arr.std())
    print(f"Global dataset stats: mean={mean_val:.2f}, std={std_val:.2f}")

    # Save to compressed NPZ
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
