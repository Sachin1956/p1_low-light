"""
Create verified paired train / val / test manifests for LOL-v2 Real.

Fixes the upstream bugs identified in data_manifests/LOLv2-Real.csv:
1. Strips numeric prefixes properly so lowXXXXX and normalXXXXX map to the exact same pair_id.
2. Keeps low and normal paired together in the exact same project split (train vs val).
3. Preserves official Test partition untouched.
4. Uses seed 1956 and 20% validation fraction from official training pairs.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "data" / "LOLv2-real" / "LOL-v2" / "Real_captured"
    if not data_root.exists():
        raise FileNotFoundError(f"LOL-v2 Real_captured folder not found at {data_root}")

    # Discover pairs in Train
    train_low_dir = data_root / "Train" / "Low"
    train_norm_dir = data_root / "Train" / "Normal"
    test_low_dir = data_root / "Test" / "Low"
    test_norm_dir = data_root / "Test" / "Normal"

    train_low_files = sorted(train_low_dir.glob("*.png"))
    test_low_files = sorted(test_low_dir.glob("*.png"))

    print(f"Found {len(train_low_files)} training low files and {len(test_low_files)} test low files.")

    # Match train pairs
    train_pairs = []
    for low_p in train_low_files:
        num = re.findall(r"\d+", low_p.stem)[0]
        norm_p = train_norm_dir / f"normal{num}.png"
        if not norm_p.exists():
            # Try alternate naming
            norm_matches = list(train_norm_dir.glob(f"*{num}*.png"))
            if not norm_matches:
                raise FileNotFoundError(f"No matching normal pair found for {low_p}")
            norm_p = norm_matches[0]
        train_pairs.append((num, low_p, norm_p))

    # Match test pairs
    test_pairs = []
    for low_p in test_low_files:
        num = re.findall(r"\d+", low_p.stem)[0]
        norm_p = test_norm_dir / f"normal{num}.png"
        if not norm_p.exists():
            norm_matches = list(test_norm_dir.glob(f"*{num}*.png"))
            if not norm_matches:
                raise FileNotFoundError(f"No matching normal test pair for {low_p}")
            norm_p = norm_matches[0]
        test_pairs.append((num, low_p, norm_p))

    # Partition train pairs into train (80%) and validation (20%) using seed 1956
    rng = random.Random(1956)
    train_pair_indices = list(range(len(train_pairs)))
    rng.shuffle(train_pair_indices)

    n_val = round(len(train_pairs) * 0.20)
    val_indices = set(train_pair_indices[:n_val])

    rows = []
    dataset_name = "LOLv2-real"

    # Write train and val rows
    for idx, (num, low_p, norm_p) in enumerate(train_pairs):
        split = "validation" if idx in val_indices else "train"
        rows.append({
            "pair_id": num,
            "dataset": dataset_name,
            "low_path": str(low_p.relative_to(repo_root)).replace("\\", "/"),
            "normal_path": str(norm_p.relative_to(repo_root)).replace("\\", "/"),
            "official_partition": "train",
            "project_split": split,
            "seed": "1956",
        })

    # Write test rows
    for num, low_p, norm_p in test_pairs:
        rows.append({
            "pair_id": num,
            "dataset": dataset_name,
            "low_path": str(low_p.relative_to(repo_root)).replace("\\", "/"),
            "normal_path": str(norm_p.relative_to(repo_root)).replace("\\", "/"),
            "official_partition": "test",
            "project_split": "test",
            "seed": "1956",
        })

    # Save to data_manifests/LOLv2-Real-paired.csv
    out_csv = repo_root / "data_manifests" / "LOLv2-Real-paired.csv"
    fieldnames = ["pair_id", "dataset", "low_path", "normal_path", "official_partition", "project_split", "seed"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    splits_count = {}
    for r in rows:
        splits_count[r["project_split"]] = splits_count.get(r["project_split"], 0) + 1

    print(f"Wrote {len(rows)} verified paired rows to {out_csv}.")
    print(f"Splits summary: {splits_count}")


if __name__ == "__main__":
    main()
