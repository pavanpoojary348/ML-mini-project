"""Model Architectures for Music Genre Classification.

Includes:
1. PaperCNN: Baseline architecture as described in the reference paper
   (Ramesh et al., ICCCNT 2024).
2. EnhancedSECNN: Proposed architectural improvement featuring Squeeze-and-Excitation
   (Channel Attention) blocks, Batch Normalization, Residual Connections, and
   Adaptive Pooling to eliminate overfitting and capture rich audio-frequency harmonics.
3. MobileNetV2Classifier: Transfer learning architecture leveraging pre-trained
   MobileNetV2 backbone with a regularized classification head and fine-tuning support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class PaperCNN(nn.Module):
    """Replication of the baseline CNN architecture described in the ICCCNT paper (Fig. 5).

    Architecture:
    - Conv2D (3 -> 32, 3x3) + ReLU + MaxPool2D(2x2) -> 112x112
    - Conv2D (32 -> 64, 3x3) + ReLU + MaxPool2D(2x2) -> 56x56
    - Conv2D (64 -> 128, 3x3) + ReLU + MaxPool2D(2x2) -> 28x28
    - Conv2D (128 -> 128, 3x3) + ReLU + MaxPool2D(2x2) -> 14x14
    - Flatten -> Linear(128*14*14, 1024) + ReLU
    - Linear(1024, 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 28 -> 14

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Channel Attention Block.

    Dynamically recalibrates channel-wise feature responses on audio spectrograms
    by explicitly modeling interdependencies between frequency channels.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: Global Average Pooling across spatial dimensions
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation: Learn channel weight distribution
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # Recalibrate
        return x * y.expand_as(x)


class ConvSEBlock(nn.Module):
    """Convolutional Block with BatchNorm, Squeeze-and-Excitation, and Residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)

        # Residual shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class EnhancedSECNN(nn.Module):
    """Our Improved Architecture: Squeeze-and-Excitation Attention CNN with Residuals.

    Addresses baseline weaknesses:
    1. Multi-scale feature progression: 32 -> 64 -> 128 -> 256 channels.
    2. Batch Normalization across all layers for stable and fast training.
    3. SE Attention: Recalibrates music frequency bands (timbre & harmony).
    4. Adaptive Average Pooling: Prevents parameter explosion and memorization.
    5. Dropout regularization: Guarantees high test generalization.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.4):
        super().__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 4 Residual SE stages
        self.stage1 = ConvSEBlock(32, 64, stride=2)   # 224 -> 112
        self.stage2 = ConvSEBlock(64, 128, stride=2)  # 112 -> 56
        self.stage3 = ConvSEBlock(128, 256, stride=2) # 56 -> 28
        self.stage4 = ConvSEBlock(256, 256, stride=2) # 28 -> 14

        # Adaptive pooling & regularized classifier head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MobileNetV2Classifier(nn.Module):
    """Transfer Learning with MobileNetV2 for Music Genre Classification.

    Leverages pre-trained ImageNet representations, coupled with a custom
    batch-normalized classifier head and selective layer unfreezing.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        self.base_model = mobilenet_v2(weights=weights)

        # Replace classification head
        in_features = self.base_model.classifier[1].in_features  # 1280
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def freeze_backbone(self):
        """Freezes all feature extractor weights (for Phase 1 warm-up)."""
        for param in self.base_model.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_from_block: int = 14):
        """Unfreezes higher-level inverted residual blocks for Phase 2 fine-tuning."""
        for i, block in enumerate(self.base_model.features):
            for param in block.parameters():
                param.requires_grad = (i >= unfreeze_from_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
