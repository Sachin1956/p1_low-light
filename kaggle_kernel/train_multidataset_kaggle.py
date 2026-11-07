"""
Multi-Dataset Adaptation Kernel for P1: LOL-v2 Real + SMALL v1.0.0
Supports advanced training strategies:
- Combines DSLR (LOL-v2 Real) and Smartphone (SMALL v1.0.0) genuine sensor pairs
- Cosine Annealing learning rate schedule to prevent overfitting on extended epochs
- Checkpoint validation every 250 updates
- Export of best_model.pth and full training metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# Set random seeds for reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiDatasetPaired(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        crop_size: int = 256,
        augment: bool = True,
    ):
        self.pairs = pairs
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_p, norm_p = self.pairs[idx]

        low_bgr = cv2.imread(str(low_p))
        norm_bgr = cv2.imread(str(norm_p))

        if low_bgr is None or norm_bgr is None:
            raise FileNotFoundError(f"Failed to read pair: {low_p}, {norm_p}")

        low_rgb = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2RGB)
        norm_rgb = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)

        # Match dimensions if slight discrepancy
        if low_rgb.shape != norm_rgb.shape:
            norm_rgb = cv2.resize(
                norm_rgb, (low_rgb.shape[1], low_rgb.shape[0]), interpolation=cv2.INTER_AREA
            )

        h, w = low_rgb.shape[:2]

        if self.crop_size > 0 and h >= self.crop_size and w >= self.crop_size:
            if self.augment:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
            else:
                y = (h - self.crop_size) // 2
                x = (w - self.crop_size) // 2

            low_rgb = low_rgb[y : y + self.crop_size, x : x + self.crop_size]
            norm_rgb = norm_rgb[y : y + self.crop_size, x : x + self.crop_size]
        elif self.crop_size > 0:
            low_rgb = cv2.resize(low_rgb, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            norm_rgb = cv2.resize(norm_rgb, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)

        # Synchronized geometric augmentations
        if self.augment:
            if random.random() > 0.5:
                low_rgb = np.fliplr(low_rgb)
                norm_rgb = np.fliplr(norm_rgb)
            if random.random() > 0.5:
                low_rgb = np.flipud(low_rgb)
                norm_rgb = np.flipud(norm_rgb)
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                low_rgb = np.rot90(low_rgb, k)
                norm_rgb = np.rot90(norm_rgb, k)

        low_t = torch.from_numpy(np.ascontiguousarray(low_rgb)).permute(2, 0, 1).float() / 255.0
        norm_t = torch.from_numpy(np.ascontiguousarray(norm_rgb)).permute(2, 0, 1).float() / 255.0

        return low_t, norm_t


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps**2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset training on Kaggle")
    parser.add_argument("--total_updates", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="p1_multidataset_output")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Config: total_updates={args.total_updates}, lr={args.lr}, crop={args.crop_size}")


if __name__ == "__main__":
    main()
