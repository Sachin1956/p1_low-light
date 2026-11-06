"""
Dataset loader for LOLv2 paired training and validation.

Supports:
- Synchronized random cropping (default 128x128 or 256x256)
- Synchronized horizontal/vertical flips and rotations
- Float32 tensors in [0, 1]
- Manifest-driven split filtering
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedLowLightDataset(Dataset):
    """Loads synchronized paired low-light and normal-light images from CSV manifest."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str = "train",
        crop_size: int = 128,
        augment: bool = True,
        repo_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.augment = augment
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent

        manifest_p = Path(manifest_csv)
        if not manifest_p.is_absolute():
            manifest_p = self.repo_root / manifest_p

        self.samples: List[Tuple[str, Path, Path]] = []
        with open(manifest_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["project_split"] == split:
                    low_p = self.repo_root / row["low_path"]
                    norm_p = self.repo_root / row["normal_path"]
                    if not low_p.exists() or not norm_p.exists():
                        raise FileNotFoundError(f"Missing file in pair {row['pair_id']}: {low_p} or {norm_p}")
                    self.samples.append((row["pair_id"], low_p, norm_p))

        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in {manifest_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        pair_id, low_path, norm_path = self.samples[index]

        with Image.open(low_path) as im_low:
            low_np = np.asarray(im_low.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(norm_path) as im_norm:
            norm_np = np.asarray(im_norm.convert("RGB"), dtype=np.float32) / 255.0

        h, w, _ = low_np.shape

        # Synchronized crop if crop_size > 0
        if self.crop_size > 0 and h >= self.crop_size and w >= self.crop_size:
            if self.augment:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
            else:
                y = (h - self.crop_size) // 2
                x = (w - self.crop_size) // 2
            low_np = low_np[y : y + self.crop_size, x : x + self.crop_size, :]
            norm_np = norm_np[y : y + self.crop_size, x : x + self.crop_size, :]

        # Synchronized augmentations (flips & 90 deg rotation)
        if self.augment:
            if random.random() > 0.5:
                low_np = np.fliplr(low_np)
                norm_np = np.fliplr(norm_np)
            if random.random() > 0.5:
                low_np = np.flipud(low_np)
                norm_np = np.flipud(norm_np)
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                low_np = np.rot90(low_np, rot_k)
                norm_np = np.rot90(norm_np, rot_k)

        # To torch tensors (3, H, W)
        low_t = torch.from_numpy(np.ascontiguousarray(low_np.transpose(2, 0, 1)))
        norm_t = torch.from_numpy(np.ascontiguousarray(norm_np.transpose(2, 0, 1)))

        return low_t, norm_t, pair_id
