"""GTZAN Spectrogram Dataset Loader and Stratified Data Splits.

Loads 224x224 RGB spectrogram images for the 10 GTZAN genres:
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
Includes data augmentation and stratified train/val/test splits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}
IDX_TO_GENRE = {idx: genre for idx, genre in enumerate(GENRES)}

# ImageNet normalization statistics (standard for MobileNetV2 and deep vision CNNs)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class GTZANSpectrogramDataset(Dataset):
    """PyTorch Dataset for GTZAN Spectrogram PNG images."""

    def __init__(self, file_paths: List[Path], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Convert image to RGB (original PNGs are RGBA)
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training (with augmentation) and validation/testing transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Subtle horizontal shift / random crop to simulate temporal variation
        transforms.RandomCrop((224, 224), padding=8, padding_mode="edge"),
        # Mild brightness & contrast jitter (simulates audio volume / dynamics)
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    return train_transform, eval_transform


def get_gtzan_splits(
    data_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> Tuple[List[Path], List[int], List[Path], List[int], List[Path], List[int]]:
    """Scans images_original directory and produces stratified train/val/test splits."""
    images_dir = data_dir / "images_original" if (data_dir / "images_original").exists() else data_dir
    if not images_dir.exists():
        raise FileNotFoundError(f"GTZAN spectrogram directory not found at: {images_dir}")

    all_paths: List[Path] = []
    all_labels: List[int] = []

    for genre in GENRES:
        genre_folder = images_dir / genre
        if not genre_folder.exists():
            continue
        genre_idx = GENRE_TO_IDX[genre]
        for img_path in sorted(genre_folder.glob("*.png")):
            all_paths.append(img_path)
            all_labels.append(genre_idx)

    total_samples = len(all_paths)
    if total_samples == 0:
        raise ValueError(f"No PNG images found in {images_dir}")

    # Stratified split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=all_labels,
    )

    # Split Temp into Val and Test
    val_rel_ratio = val_ratio / temp_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1.0 - val_rel_ratio),
        random_state=random_seed,
        stratify=temp_labels,
    )

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Constructs train, validation, and test PyTorch DataLoaders."""
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = get_gtzan_splits(
        data_dir, random_seed=random_seed
    )

    train_tf, eval_tf = get_data_transforms()

    train_dataset = GTZANSpectrogramDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = GTZANSpectrogramDataset(val_paths, val_labels, transform=eval_tf)
    test_dataset = GTZANSpectrogramDataset(test_paths, test_labels, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


class GTZAN3SecMelDataset(Dataset):
    """Dataset for 3-second Log-Mel Spectrogram segments (9,990 samples)."""

    def __init__(
        self,
        specs: np.ndarray,
        labels: np.ndarray,
        song_ids: np.ndarray,
        mean: float,
        std: float,
        is_train: bool = False,
    ):
        self.specs = specs  # (N, 128, 130)
        self.labels = labels
        self.song_ids = song_ids
        self.mean = mean
        self.std = std
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        spec = self.specs[idx]  # (128, 130)
        label = int(self.labels[idx])
        song_id = str(self.song_ids[idx])

        # Standardize
        spec = (spec - self.mean) / (self.std + 1e-6)

        # Convert to tensor (1, 128, 130)
        tensor = torch.from_numpy(spec).unsqueeze(0).float()

        # Resize / interpolate to 224x224
        tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

        # SpecAugment in training mode (Time & Frequency Masking)
        if self.is_train:
            # Frequency masking (mask up to 16 bands)
            f_dim = tensor.shape[1]
            f_mask_len = torch.randint(4, 20, (1,)).item()
            f_start = torch.randint(0, f_dim - f_mask_len, (1,)).item()
            tensor[:, f_start : f_start + f_mask_len, :] = 0.0

            # Time masking (mask up to 16 frames)
            t_dim = tensor.shape[2]
            t_mask_len = torch.randint(4, 20, (1,)).item()
            t_start = torch.randint(0, t_dim - t_mask_len, (1,)).item()
            tensor[:, :, t_start : t_start + t_mask_len] = 0.0

        # Repeat to 3 channels for vision backbones (3, 224, 224)
        tensor = tensor.repeat(3, 1, 1)

        return tensor, label, song_id


def build_3s_dataloaders(
    npz_path: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Constructs train, validation, and test loaders from 3-second mel cache."""
    data = np.load(npz_path)
    specs = data["specs"]       # (9990, 128, 130)
    labels = data["labels"]     # (9990,)
    song_ids = data["song_ids"] # (9990,)
    mean = float(data["mean"])
    std = float(data["std"])

    # Stratified split: 80% train, 10% val, 10% test
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_seed,
        stratify=labels[temp_idx],
    )

    train_ds = GTZAN3SecMelDataset(specs[train_idx], labels[train_idx], song_ids[train_idx], mean, std, is_train=True)
    val_ds = GTZAN3SecMelDataset(specs[val_idx], labels[val_idx], song_ids[val_idx], mean, std, is_train=False)
    test_ds = GTZAN3SecMelDataset(specs[test_idx], labels[test_idx], song_ids[test_idx], mean, std, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
